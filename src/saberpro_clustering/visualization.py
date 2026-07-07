"""Visualizaciones de embeddings y caracterización de clústeres para el artículo.

Incluye las tres figuras usadas en el pipeline original:
    - plot_clusters_2d: scatter del embedding UMAP coloreado por clúster.
    - plot_heatmap_caracterizacion: heatmap de z-scores + significancia
      estadística (Kruskal-Wallis / chi-cuadrado) por grupo de variables.
    - plot_geo_heatmap: concentración de clústeres por departamento.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def plot_clusters_2d(
    X_umap_2d: np.ndarray,
    labels: np.ndarray,
    titulo: str = "Clustering sobre UMAP 2D",
    nombres: dict | None = None,
    colores: dict | None = None,
    guardar: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot del embedding UMAP 2D coloreado por etiqueta de clúster.

    Args:
        X_umap_2d: Embedding 2D (salida de reducir_umap).
        labels: Etiquetas de clúster (-1 = ruido).
        titulo: Título del gráfico.
        nombres: Diccionario {id_cluster: nombre_legible}. Por defecto usa
            "Cluster {id}" y "Ruido" para -1.
        colores: Diccionario {id_cluster: color}. Por defecto usa tab10.
        guardar: Ruta donde guardar la figura (PNG, 200 dpi). None = no guardar.

    Returns:
        Figura de matplotlib.
    """
    unique_ids = sorted(set(labels))
    if nombres is None:
        nombres = {c: ("Ruido" if c == -1 else f"Cluster {c}") for c in unique_ids}
    if colores is None:
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique_ids), 1)))
        colores = {c: ("gray" if c == -1 else cmap[i]) for i, c in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for cid in sorted(unique_ids, key=lambda c: -(labels == c).sum()):
        idx = labels == cid
        ax.scatter(
            X_umap_2d[idx, 0],
            X_umap_2d[idx, 1],
            c=[colores[cid]],
            s=4,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            label=f"{nombres[cid]} (n={idx.sum():,})",
        )
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(markerscale=2, fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"✅ Guardado: {guardar}")

    return fig


# ── Configuración por defecto del heatmap de caracterización ────────────
GRUPOS_HEATMAP = [
    (
        "Puntajes académicos",
        [
            "MOD_RAZONA_CUANTITAT_PUNT", "MOD_LECTURA_CRITICA_PUNT",
            "MOD_COMPETEN_CIUDADA_PUNT", "MOD_INGLES_PUNT", "MOD_COMUNI_ESCRITA_PUNT",
        ],
        "quant",
    ),
    (
        "Perfil socioeconómico",
        [
            "FAMI_ESTRATOVIVIENDA", "ESTU_VALORMATRICULAUNIVERSIDAD",
            "FAMI_EDUCACIONPADRE", "FAMI_EDUCACIONMADRE", "ESTU_HORASSEMANATRABAJA",
        ],
        "quant",
    ),
    (
        "Financiación matrícula",
        [
            "ESTU_PAGOMATRICULABECA", "ESTU_PAGOMATRICULACREDITO",
            "ESTU_PAGOMATRICULAPADRES", "ESTU_PAGOMATRICULAPROPIO",
        ],
        "bin",
    ),
    (
        "Recursos del hogar",
        [
            "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR",
            "FAMI_TIENEAUTOMOVIL", "FAMI_TIENELAVADORA",
        ],
        "bin",
    ),
]

ETIQUETAS = {
    "MOD_RAZONA_CUANTITAT_PUNT": "Razonamiento Cuant.",
    "MOD_LECTURA_CRITICA_PUNT": "Lectura Crítica",
    "MOD_COMPETEN_CIUDADA_PUNT": "Comp. Ciudadanas",
    "MOD_INGLES_PUNT": "Inglés",
    "MOD_COMUNI_ESCRITA_PUNT": "Com. Escrita",
    "FAMI_ESTRATOVIVIENDA": "Estrato",
    "FAMI_EDUCACIONPADRE": "Educ. Padre",
    "FAMI_EDUCACIONMADRE": "Educ. Madre",
    "ESTU_VALORMATRICULAUNIVERSIDAD": "Valor Matrícula",
    "ESTU_HORASSEMANATRABAJA": "Horas trabajo/sem.",
    "ESTU_PAGOMATRICULABECA": "Beca",
    "ESTU_PAGOMATRICULACREDITO": "Crédito",
    "ESTU_PAGOMATRICULAPADRES": "Paga padres",
    "ESTU_PAGOMATRICULAPROPIO": "Paga propio",
    "FAMI_TIENEINTERNET": "Internet",
    "FAMI_TIENECOMPUTADOR": "Computador",
    "FAMI_TIENEAUTOMOVIL": "Automóvil",
    "FAMI_TIENELAVADORA": "Lavadora",
}


def calcular_heatmap_caracterizacion(
    df_work: pd.DataFrame,
    cluster_column: str = "_cluster",
    grupos: list | None = None,
    etiquetas: dict | None = None,
) -> pd.DataFrame:
    """Calcula z-scores y significancia (Kruskal-Wallis / chi-cuadrado) por
    variable y clúster, insumo para plot_heatmap_caracterizacion.

    Variables 'quant' (puntajes, estrato, educación) usan Kruskal-Wallis
    sobre las medias por clúster. Variables 'bin' (SI/NO) se normalizan a
    0/1 y usan chi-cuadrado de independencia.

    Args:
        df_work: DataFrame con la columna de clúster y las variables originales.
        cluster_column: Nombre de la columna de etiqueta de clúster.
        grupos: Lista de tuplas (nombre_grupo, columnas, tipo). Por defecto
            usa GRUPOS_HEATMAP.
        etiquetas: Diccionario {columna: etiqueta_legible}. Por defecto usa
            ETIQUETAS.

    Returns:
        DataFrame con columnas [grupo, etiqueta, col, tipo, z_<cluster>..., plog]
        donde plog es -log10(p-valor) (300.0 cuando p se trunca a 0.0 exacto).
    """
    from scipy.stats import chi2_contingency, kruskal

    if grupos is None:
        grupos = GRUPOS_HEATMAP
    if etiquetas is None:
        etiquetas = ETIQUETAS

    clusters_ids = sorted(c for c in df_work[cluster_column].unique() if c != -1)

    rows = []
    for grupo, cols, tipo in grupos:
        for col in cols:
            if col not in df_work.columns:
                continue
            label = etiquetas.get(col, col)

            if tipo == "quant":
                serie = pd.to_numeric(df_work[col], errors="coerce")
                gvals = [serie[df_work[cluster_column] == c].dropna().values for c in clusters_ids]
                try:
                    _, p = kruskal(*gvals)
                except Exception:
                    p = np.nan
                medias = np.array([v.mean() if len(v) > 0 else np.nan for v in gvals])
            else:  # binaria
                serie = (
                    df_work[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"si": 1, "no": 0, "s": 1, "n": 0, "1": 1, "0": 0, "yes": 1})
                )
                tabla = pd.crosstab(df_work[cluster_column], serie)
                if tabla.shape[1] < 2:
                    p = np.nan
                else:
                    try:
                        _, p, _, _ = chi2_contingency(tabla)
                    except Exception:
                        p = np.nan
                gvals = [serie[df_work[cluster_column] == c].dropna().values for c in clusters_ids]
                medias = np.array([v.mean() if len(v) > 0 else np.nan for v in gvals])

            gm, gs = np.nanmean(medias), np.nanstd(medias)
            zs = (medias - gm) / gs if gs > 0 else medias * 0

            if p is None or np.isnan(p):
                plog = 0.0
            elif p == 0.0:
                # p tan pequeño que Python lo trunca a 0 — con n grande casi
                # todas las diferencias son altamente significativas
                plog = 300.0
            else:
                plog = float(-np.log10(p))

            rows.append((grupo, label, col, tipo, *zs, plog))

    col_names = ["grupo", "etiqueta", "col", "tipo"] + [f"z_{c}" for c in clusters_ids] + ["plog"]
    return pd.DataFrame(rows, columns=col_names)


def plot_heatmap_caracterizacion(
    df_heat: pd.DataFrame,
    clusters_ids: list,
    nombres_clusters: dict | None = None,
    guardar: str | Path | None = "figura_heatmap_clusters.png",
) -> plt.Figure:
    """Heatmap de dos paneles: z-scores por clúster (izquierda) y
    significancia estadística -log10(p) (derecha).

    Args:
        df_heat: Salida de calcular_heatmap_caracterizacion.
        clusters_ids: Lista ordenada de IDs de clúster (columnas del heatmap).
        nombres_clusters: Diccionario {id: nombre legible}. Por defecto "C{id}".
        guardar: Ruta donde guardar la figura, o None para no guardar.

    Returns:
        Figura de matplotlib.
    """
    if nombres_clusters is None:
        nombres_clusters = {c: f"C{c}" for c in clusters_ids}

    n_rows = len(df_heat)
    n_cls = len(clusters_ids)

    fig, (ax_h, ax_p) = plt.subplots(
        1, 2, figsize=(7.5, n_rows * 0.32 + 1.2), dpi=150,
        gridspec_kw={"width_ratios": [n_cls, 0.5], "wspace": 0.1},
    )

    z_mat = df_heat[[f"z_{c}" for c in clusters_ids]].values.astype(float)
    p_mat = df_heat[["plog"]].values.astype(float)
    vmax = max(np.nanmax(np.abs(z_mat)), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax_h.imshow(z_mat, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax_p.imshow(
        p_mat, aspect="auto", cmap="RdPu", vmin=0, vmax=max(float(p_mat.max()), 2.0),
        interpolation="nearest",
    )

    ax_h.set_yticks(range(n_rows))
    ax_h.set_yticklabels(df_heat["etiqueta"], fontsize=7)
    ax_h.set_xticks(range(n_cls))
    ax_h.set_xticklabels([nombres_clusters.get(c, f"C{c}") for c in clusters_ids], fontsize=8, fontweight="bold")
    ax_h.set_xlabel("Clusters", fontsize=8)
    ax_p.set_xticks([0])
    ax_p.set_xticklabels(["-log₁₀(p)"], fontsize=7, rotation=45)
    ax_p.set_yticks([])

    # Líneas separadoras blancas entre grupos de variables
    grupo_sizes = df_heat.groupby("grupo", sort=False).size().tolist()
    acum = 0
    for gs in grupo_sizes[:-1]:
        acum += gs
        ax_h.axhline(acum - 0.5, color="white", linewidth=1.5)
        ax_p.axhline(acum - 0.5, color="white", linewidth=1.5)

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"✅ Guardado: {guardar}")

    return fig
