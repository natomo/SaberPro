"""Métricas internas de validación de clústeres, ablation study y
validación de representatividad muestral (200k vs 452k)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def dunn_index(X_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Índice de Dunn: razón entre separación mínima inter-clúster y
    diámetro máximo intra-clúster. Complementa silhouette/CH/DB.

    Args:
        X_np: Embedding usado para el clustering.
        labels_np: Etiquetas de clúster (-1 se excluye como ruido).

    Returns:
        Valor del índice de Dunn, o NaN si hay menos de 2 clústeres válidos.
    """
    unique = np.unique(labels_np)
    unique = unique[unique != -1]
    if len(unique) < 2:
        return np.nan
    clusters = [X_np[labels_np == l] for l in unique]
    centroids = [c.mean(axis=0) for c in clusters]
    min_inter = min(
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    max_intra = max(
        (np.linalg.norm(c - c.mean(axis=0), axis=1).max() * 2 if len(c) > 1 else 0)
        for c in clusters
    ) or 1
    return min_inter / max_intra


def calcular_metricas(
    X_np: np.ndarray,
    labels_np: np.ndarray,
    inertia: float | None = None,
    bic: float | None = None,
) -> dict | None:
    """Calcula el set completo de métricas internas usado en Fase 1 y Fase 2.

    Args:
        X_np: Embedding usado para el clustering.
        labels_np: Etiquetas de clúster asignadas.
        inertia: Inertia de KMeans/MiniBatchKMeans, si aplica.
        bic: BIC de GaussianMixture, si aplica.

    Returns:
        Diccionario con silhouette, calinski, davies_bouldin, dunn, inertia,
        bic, n_clusters y n_ruido. None si hay menos de 2 clústeres válidos
        (las métricas no están definidas en ese caso).
    """
    mask = labels_np != -1
    X_eval, lbl_eval = X_np[mask], labels_np[mask]
    if len(set(lbl_eval)) < 2:
        return None
    sample = min(10_000, len(lbl_eval))
    sil = silhouette_score(X_eval, lbl_eval, sample_size=sample, random_state=42)
    return {
        "silhouette": round(sil, 4),
        "calinski": round(calinski_harabasz_score(X_eval, lbl_eval), 2),
        "davies_bouldin": round(davies_bouldin_score(X_eval, lbl_eval), 4),
        "dunn": round(dunn_index(X_eval, lbl_eval), 4),
        "inertia": round(inertia, 2) if inertia is not None else None,
        "bic": round(bic, 2) if bic is not None else None,
        "n_clusters": int(len(set(lbl_eval))),
        "n_ruido": int((labels_np == -1).sum()),
    }


def ablation_study(X_full: np.ndarray, labels_final: np.ndarray, m_final: dict, best_ndim: int):
    """Ablation study: compara clustering sin reducción, con PCA y con UMAP.

    Necesario para justificar metodológicamente el uso de UMAP en el artículo.

    Args:
        X_full: Matriz de features completa (sin reducir), Fase 2.
        labels_final: Etiquetas del modelo final entrenado sobre el embedding UMAP.
        m_final: Métricas ya calculadas del modelo final (sobre el espacio UMAP).
        best_ndim: Dimensión del embedding UMAP usada para el modelo final.

    Returns:
        DataFrame con una fila por método (sin reducción, PCA, UMAP, UMAP
        validado en espacio original).
    """
    from sklearn.cluster import MiniBatchKMeans

    from .dimensionality import reduce_pca

    best_k = m_final["n_clusters"]

    print("⏳ KMeans directo sobre X (sin reducción)...")
    km_raw = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init="auto", batch_size=10_000)
    labels_raw = km_raw.fit_predict(X_full)
    m_raw = calcular_metricas(X_full, labels_raw)

    print("⏳ PCA + KMeans...")
    X_pca, pca = reduce_pca(X_full, n_components=0.80)
    n_pca = X_pca.shape[1]
    print(f"   PCA → {n_pca} componentes ({pca.explained_variance_ratio_.sum() * 100:.1f}% varianza)")
    km_pca = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init="auto", batch_size=10_000)
    labels_pca = km_pca.fit_predict(X_pca)
    m_pca = calcular_metricas(X_pca, labels_pca)

    print("⏳ Validando clusters UMAP en espacio original...")
    m_orig = calcular_metricas(X_full, labels_final)

    df_ablation = pd.DataFrame(
        [
            {"metodo": "Sin reducción", "dims": X_full.shape[1], **m_raw},
            {"metodo": f"PCA {n_pca} comps", "dims": n_pca, **m_pca},
            {"metodo": f"UMAP {best_ndim}D", "dims": best_ndim, **m_final},
            {"metodo": "UMAP (valid. orig)", "dims": X_full.shape[1], **m_orig},
        ]
    )
    return df_ablation


# Variables clave para comparar muestra (200k) vs población (452k)
VARS_VALIDAR_REPRESENTATIVIDAD = {
    "FAMI_ESTRATOVIVIENDA": "ordinal",
    "ESTU_VALORMATRICULAUNIVERSIDAD": "ordinal",
    "FAMI_EDUCACIONPADRE": "ordinal",
    "FAMI_EDUCACIONMADRE": "ordinal",
    "ESTU_HORASSEMANATRABAJA": "ordinal",
    "MOD_RAZONA_CUANTITAT_PUNT": "continua",
    "MOD_LECTURA_CRITICA_PUNT": "continua",
    "MOD_COMPETEN_CIUDADA_PUNT": "continua",
    "MOD_INGLES_PUNT": "continua",
    "MOD_COMUNI_ESCRITA_PUNT": "continua",
}


def validar_representatividad(
    df_poblacion: pd.DataFrame,
    df_muestra: pd.DataFrame,
    vars_validar: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compara distribuciones muestra (200k) vs población (452k) con test KS.

    Criterio de representatividad: diferencia de medias < 2% Y estadístico
    KS < 0.05 → ✅. Diferencia entre 2% y 5% → ⚠️. Diferencia >= 5% → ❌.

    Args:
        df_poblacion: DataFrame completo (df_limpio, 452k filas).
        df_muestra: DataFrame muestreado usado en Fase 1 (df_filtrado_200k).
        vars_validar: Diccionario {columna: tipo}. Por defecto usa
            VARS_VALIDAR_REPRESENTATIVIDAD.

    Returns:
        DataFrame con una fila por variable: medias, diferencia %, estadístico
        KS, p-valor y símbolo de representatividad.
    """
    if vars_validar is None:
        vars_validar = VARS_VALIDAR_REPRESENTATIVIDAD

    resultados = []
    for col in vars_validar:
        if col not in df_poblacion.columns or col not in df_muestra.columns:
            continue

        pob = pd.to_numeric(df_poblacion[col], errors="coerce").dropna()
        mues = pd.to_numeric(df_muestra[col], errors="coerce").dropna()

        media_pob = pob.mean()
        media_mues = mues.mean()
        dif_pct = abs(media_pob - media_mues) / abs(media_pob) * 100 if media_pob != 0 else 0

        ks_stat, ks_p = ks_2samp(
            pob.sample(min(10_000, len(pob)), random_state=42),
            mues.sample(min(10_000, len(mues)), random_state=42),
        )

        representa = "✅" if dif_pct < 2 and ks_stat < 0.05 else ("⚠️" if dif_pct < 5 else "❌")

        resultados.append(
            {
                "variable": col,
                "media_poblacion": round(media_pob, 3),
                "media_muestra": round(media_mues, 3),
                "diferencia_pct": round(dif_pct, 2),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalor": round(ks_p, 4),
                "representa": representa,
            }
        )

    return pd.DataFrame(resultados)
