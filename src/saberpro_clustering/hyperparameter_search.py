"""Fase 1 — búsqueda de hiperparámetros sobre la muestra de 200k.

Evalúa combinaciones de algoritmo + dimensión UMAP + parámetros propios
de cada algoritmo, para elegir la configuración que se entrena sobre
los 452k estudiantes completos en Fase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .dimensionality import reducir_umap
from .evaluation import calcular_metricas


def evaluar_kmeans(X: np.ndarray, dims=range(2, 6), ks=range(2, 9)) -> pd.DataFrame:
    """Grid search de MiniBatchKMeans sobre distintas dimensiones UMAP y K.

    Args:
        X: Matriz de features preprocesadas (muestra de 200k).
        dims: Rango de dimensiones UMAP a evaluar.
        ks: Rango de número de clústeres a evaluar.

    Returns:
        DataFrame con una fila por combinación (n_dim, k) y sus métricas.
    """
    resultados = []
    for n_dim in dims:
        print(f"\n--- UMAP {n_dim}D ---")
        X_umap, _ = reducir_umap(X, n_components=n_dim)
        print(f"{'K':>3} | {'Silhouette':>10} | {'CH':>12} | {'Davies-B':>10} | {'Dunn':>8} | {'Inertia':>12}")
        print("-" * 65)
        for k in ks:
            try:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=10_000)
                labels = km.fit_predict(X_umap)
                m = calcular_metricas(X_umap, labels, inertia=km.inertia_)
                if m is None:
                    continue
                print(
                    f"K={k:>2} | {m['silhouette']:>10.4f} | {m['calinski']:>12.2f} | "
                    f"{m['davies_bouldin']:>10.4f} | {m['dunn']:>8.4f} | {m['inertia']:>12.2f}"
                )
                resultados.append({"algoritmo": "KMeans", "n_dim": n_dim, "k": k, **m})
            except Exception as e:
                print(f"K={k:>2} → Error: {e}")
    return pd.DataFrame(resultados)


def evaluar_gmm(X: np.ndarray, dims=range(2, 6), ks=range(2, 9)) -> pd.DataFrame:
    """Grid search de Gaussian Mixture Models sobre dimensiones UMAP y K.

    Args:
        X: Matriz de features preprocesadas (muestra de 200k).
        dims: Rango de dimensiones UMAP a evaluar.
        ks: Rango de número de componentes a evaluar.

    Returns:
        DataFrame con una fila por combinación (n_dim, k) y sus métricas.
    """
    resultados = []
    for n_dim in dims:
        print(f"\n--- UMAP {n_dim}D ---")
        X_umap, _ = reducir_umap(X, n_components=n_dim)
        print(f"{'K':>3} | {'Silhouette':>10} | {'CH':>12} | {'Davies-B':>10} | {'Dunn':>8} | {'BIC':>14}")
        print("-" * 70)
        for k in ks:
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full", n_init=1)
                labels = gmm.fit_predict(X_umap)
                bic = gmm.bic(X_umap)
                m = calcular_metricas(X_umap, labels, bic=bic)
                if m is None:
                    continue
                print(
                    f"K={k:>2} | {m['silhouette']:>10.4f} | {m['calinski']:>12.2f} | "
                    f"{m['davies_bouldin']:>10.4f} | {m['dunn']:>8.4f} | {m['bic']:>14.2f}"
                )
                resultados.append({"algoritmo": "GMM", "n_dim": n_dim, "k": k, **m})
            except Exception as e:
                print(f"K={k:>2} → Error: {e}")
    return pd.DataFrame(resultados)


def evaluar_dbscan(X: np.ndarray, n_dim: int = 2, eps_vals=None, min_samples_vals=None):
    """Grid search de DBSCAN. Estima eps automáticamente vía k-distance plot
    si no se especifica.

    Args:
        X: Matriz de features preprocesadas (muestra de 200k).
        n_dim: Dimensión UMAP a usar.
        eps_vals: Lista de valores de eps a evaluar (None = estimar automáticamente).
        min_samples_vals: Lista de min_samples a evaluar (default [5, 10, 20]).

    Returns:
        Tupla (DataFrame de resultados, embedding UMAP usado).
    """
    import matplotlib.pyplot as plt

    X_umap, _ = reducir_umap(X, n_components=n_dim)
    if eps_vals is None:
        nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        nn.fit(X_umap)
        dists, _ = nn.kneighbors(X_umap)
        k_dists = np.sort(dists[:, -1])
        eps_vals = [round(np.percentile(k_dists, p), 3) for p in [10, 25, 50]]
        print(f"eps estimados: {eps_vals}")
        plt.figure(figsize=(7, 3), dpi=120)
        plt.plot(k_dists, linewidth=0.8)
        for eps in eps_vals:
            plt.axhline(eps, linestyle="--", linewidth=0.8, label=f"eps={eps}")
        plt.xlabel("Puntos ordenados")
        plt.ylabel("5-NN distancia")
        plt.title("K-distance plot")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    if min_samples_vals is None:
        min_samples_vals = [5, 10, 20]
    resultados = []
    print(f"\n--- DBSCAN sobre UMAP {n_dim}D ---")
    print(f"{'eps':>8} | {'min_s':>6} | {'n_cls':>6} | {'ruido%':>7} | {'Silhouette':>10} | {'CH':>12} | {'Davies-B':>10}")
    print("-" * 72)
    for eps in eps_vals:
        for min_s in min_samples_vals:
            try:
                labels = DBSCAN(eps=eps, min_samples=min_s, n_jobs=-1).fit_predict(X_umap)
                noise_pct = (labels == -1).mean() * 100
                m = calcular_metricas(X_umap, labels)
                if m is None:
                    print(f"{eps:>8.3f} | {min_s:>6} | {'<2':>6} | {noise_pct:>6.1f}% | —")
                    continue
                print(
                    f"{eps:>8.3f} | {min_s:>6} | {m['n_clusters']:>6} | {noise_pct:>6.1f}% | "
                    f"{m['silhouette']:>10.4f} | {m['calinski']:>12.2f} | {m['davies_bouldin']:>10.4f}"
                )
                resultados.append(
                    {
                        "algoritmo": "DBSCAN",
                        "n_dim": n_dim,
                        "eps": eps,
                        "min_samples": min_s,
                        "noise_pct": round(noise_pct, 2),
                        **m,
                    }
                )
            except Exception as e:
                print(f"{eps:>8.3f} | {min_s:>6} → Error: {e}")
    return pd.DataFrame(resultados), X_umap


def evaluar_hdbscan(X: np.ndarray, dims=range(2, 6), min_cluster_vals=None, min_samples_vals=None) -> pd.DataFrame:
    """Grid search de HDBSCAN sobre dimensiones UMAP, min_cluster_size y min_samples.

    Args:
        X: Matriz de features preprocesadas (muestra de 200k).
        dims: Rango de dimensiones UMAP a evaluar.
        min_cluster_vals: Valores de min_cluster_size a evaluar (default [5, 10, 15]).
        min_samples_vals: Valores de min_samples a evaluar (default [5, 10]).

    Returns:
        DataFrame con una fila por combinación y sus métricas.
    """
    import hdbscan

    if min_cluster_vals is None:
        min_cluster_vals = [5, 10, 15]
    if min_samples_vals is None:
        min_samples_vals = [5, 10]
    resultados = []
    for n_dim in dims:
        print(f"\n--- UMAP {n_dim}D ---")
        X_umap, _ = reducir_umap(X, n_components=n_dim)
        print(
            f"{'mc':>4} | {'ms':>4} | {'n_cls':>6} | {'ruido%':>7} | "
            f"{'Silhouette':>10} | {'CH':>12} | {'Davies-B':>10}"
        )
        print("-" * 65)
        for mc in min_cluster_vals:
            for ms in min_samples_vals:
                try:
                    labels = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=ms).fit_predict(X_umap)
                    noise_pct = (labels == -1).mean() * 100
                    m = calcular_metricas(X_umap, labels)
                    if m is None:
                        print(f"{mc:>4} | {ms:>4} | {'<2':>6} | {noise_pct:>6.1f}% | —")
                        continue
                    print(
                        f"{mc:>4} | {ms:>4} | {m['n_clusters']:>6} | {noise_pct:>6.1f}% | "
                        f"{m['silhouette']:>10.4f} | {m['calinski']:>12.2f} | {m['davies_bouldin']:>10.4f}"
                    )
                    resultados.append(
                        {
                            "algoritmo": "HDBSCAN",
                            "n_dim": n_dim,
                            "min_cluster_size": mc,
                            "min_samples": ms,
                            "noise_pct": round(noise_pct, 2),
                            **m,
                        }
                    )
                except Exception as e:
                    print(f"{mc:>4} | {ms:>4} → Error: {e}")
    return pd.DataFrame(resultados)


def resumen_fase1(df_res_km, df_res_gmm, df_res_db, df_res_hdb) -> pd.DataFrame:
    """Concatena los resultados de los 4 algoritmos y ordena por silhouette.

    Args:
        df_res_km, df_res_gmm, df_res_db, df_res_hdb: DataFrames de cada
            función evaluar_*.

    Returns:
        DataFrame combinado, ordenado descendente por silhouette.
    """
    df_fase1 = pd.concat([df_res_km, df_res_gmm, df_res_db, df_res_hdb], ignore_index=True)
    return df_fase1.sort_values("silhouette", ascending=False).reset_index(drop=True)
