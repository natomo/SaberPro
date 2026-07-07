"""Análisis geográfico post-clustering: concentración de clústeres por
departamento colombiano."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Códigos DANE de departamento -> nombre
DEPTOS = {
    5: "Antioquia", 8: "Atlántico", 11: "Bogotá D.C.", 13: "Bolívar",
    15: "Boyacá", 17: "Caldas", 18: "Caquetá", 19: "Cauca",
    20: "Cesar", 23: "Córdoba", 25: "Cundinamarca", 27: "Chocó",
    41: "Huila", 44: "La Guajira", 47: "Magdalena", 50: "Meta",
    52: "Nariño", 54: "Norte de Santander", 63: "Quindío",
    66: "Risaralda", 68: "Santander", 70: "Sucre", 73: "Tolima",
    76: "Valle del Cauca", 81: "Arauca", 85: "Casanare",
    86: "Putumayo", 88: "San Andrés", 91: "Amazonas",
    94: "Guainía", 95: "Guaviare", 97: "Vaupés", 99: "Vichada",
}


def analizar_geografia(
    df: pd.DataFrame,
    cluster_column: str = "_cluster",
    col_geo: str = "ESTU_COD_DEPTO_PRESENTACION",
    nombres_clusters: dict | None = None,
) -> pd.DataFrame | None:
    """Calcula la tabla de concentración de clústeres por departamento (%).

    Args:
        df: DataFrame con la columna de clúster y el código de departamento.
        cluster_column: Nombre de la columna de etiqueta de clúster.
        col_geo: Nombre de la columna con el código DANE de departamento.
        nombres_clusters: Diccionario {id: nombre legible}. Por defecto "C{id}".

    Returns:
        DataFrame (departamentos x clústeres) con % de cada clúster dentro
        de cada departamento, más columna n_total, ordenado por tamaño de
        departamento. None si col_geo no está en el DataFrame.
    """
    if col_geo not in df.columns:
        print(f"⚠️  '{col_geo}' no está en el DataFrame.")
        return None

    if nombres_clusters is None:
        nombres_clusters = {c: f"C{c}" for c in sorted(df[cluster_column].unique())}

    df_geo = df[[col_geo, cluster_column]].copy()
    df_geo["depto"] = df_geo[col_geo].map(DEPTOS).fillna("Otro")

    tabla = pd.crosstab(df_geo["depto"], df_geo[cluster_column], normalize="index").round(3) * 100
    tabla.columns = [nombres_clusters.get(c, f"C{c}") for c in tabla.columns]
    tabla["n_total"] = df_geo.groupby("depto").size()
    tabla = tabla.sort_values("n_total", ascending=False)

    return tabla


def plot_geo_heatmap(
    tabla: pd.DataFrame,
    guardar: str | Path | None = "figura_geo_clusters.png",
) -> plt.Figure:
    """Heatmap de concentración de clústeres por departamento.

    Args:
        tabla: Salida de analizar_geografia.
        guardar: Ruta donde guardar la figura, o None para no guardar.

    Returns:
        Figura de matplotlib.
    """
    z_cols = [c for c in tabla.columns if c != "n_total"]
    fig, ax = plt.subplots(figsize=(len(z_cols) * 1.4 + 2, len(tabla) * 0.35 + 1.2), dpi=130)
    im = ax.imshow(tabla[z_cols].values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(len(z_cols)))
    ax.set_xticklabels(z_cols, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(tabla)))
    ax.set_yticklabels(tabla.index, fontsize=7)
    ax.set_title("Concentración de clusters por departamento (%)", fontsize=10)
    for row in range(len(tabla)):
        for col in range(len(z_cols)):
            val = tabla[z_cols].values[row, col]
            ax.text(col, row, f"{val:.0f}", ha="center", va="center", fontsize=6, color="white" if val > 60 else "black")
    plt.colorbar(im, ax=ax, label="% en cluster", shrink=0.6)
    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"✅ Guardado: {guardar}")

    return fig


def top_departamentos_por_cluster(tabla: pd.DataFrame, n: int = 5) -> None:
    """Imprime el top-N de departamentos con mayor concentración por clúster.

    Args:
        tabla: Salida de analizar_geografia.
        n: Número de departamentos a mostrar por clúster.
    """
    z_cols = [c for c in tabla.columns if c != "n_total"]
    print(f"\n=== Top {n} departamentos por cluster ===")
    for c in z_cols:
        top = tabla[c].sort_values(ascending=False).head(n)
        print(f"\n  {c}:")
        for depto, pct in top.items():
            print(f"    {depto:<25} {pct:.1f}%  (n={tabla.loc[depto, 'n_total']:,})")
