#!/usr/bin/env python
"""Fase 2 — entrena el modelo final sobre los 452,020 estudiantes completos
y genera todos los artefactos para el artículo:

    - Modelo, reducer UMAP, scaler y encoder (joblib) + dataset con clústeres
    - Ablation study (justifica el uso de UMAP frente a PCA / sin reducción)
    - Figura de embedding UMAP 2D coloreado por clúster
    - Heatmap de caracterización (z-scores + significancia estadística)
    - Análisis geográfico (concentración de clústeres por departamento)
    - Validación de representatividad de la muestra de Fase 1 (200k vs 452k)

Uso:
    python scripts/run_fase2_modelo_final.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from saberpro_clustering.clustering import entrenar_modelo_final  # noqa: E402
from saberpro_clustering.config import load_config  # noqa: E402
from saberpro_clustering.dimensionality import reducir_umap  # noqa: E402
from saberpro_clustering.evaluation import (  # noqa: E402
    ablation_study,
    calcular_metricas,
    validar_representatividad,
)
from saberpro_clustering.geo import (  # noqa: E402
    analizar_geografia,
    plot_geo_heatmap,
    top_departamentos_por_cluster,
)
from saberpro_clustering.preprocessing import load_raw_data, preprocesar_saber_pro  # noqa: E402
from saberpro_clustering.visualization import (  # noqa: E402
    calcular_heatmap_caracterizacion,
    plot_clusters_2d,
    plot_heatmap_caracterizacion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: modelo final sobre 452k")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    f2 = config["fase2"]
    figures_dir = Path(config["output"]["figures_dir"])
    results_dir = Path(config["output"]["results_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {f2['best_algo']} | K={f2['best_k']} | UMAP {f2['best_ndim']}D")

    print(f"\n[1/7] Cargando y preprocesando los 452k estudiantes completos ...")
    df = load_raw_data(config["data"]["input_path"])
    df_limpio_full, df_filtrado_full, X_full, _, encoder, scaler = preprocesar_saber_pro(
        df, sample_n=None
    )
    print(f"      Shape X_full: {X_full.shape}")

    print(f"\n[2/7] Calculando UMAP {f2['best_ndim']}D sobre {X_full.shape[0]:,} datos ...")
    X_umap_full, reducer = reducir_umap(X_full, n_components=f2["best_ndim"])
    print(f"      ✅ UMAP listo — shape: {X_umap_full.shape}")

    if f2["best_ndim"] == 2:
        X_umap_viz = X_umap_full
    else:
        print("      ⏳ Calculando UMAP 2D adicional solo para visualización ...")
        X_umap_viz, _ = reducir_umap(X_full, n_components=2, min_dist=0.3)

    print(f"\n[3/7] Entrenando {f2['best_algo']} ...")
    modelo, labels_final = entrenar_modelo_final(
        X_umap_full,
        best_algo=f2["best_algo"],
        best_k=f2["best_k"],
        dbscan_eps=f2["dbscan_eps"],
        dbscan_min_samples=f2["dbscan_min_samples"],
        hdbscan_mc=f2["hdbscan_min_cluster_size"],
        hdbscan_ms=f2["hdbscan_min_samples"],
    )
    m_final = calcular_metricas(X_umap_full, labels_final)
    print(f"      📊 Métricas en espacio UMAP {f2['best_ndim']}D:")
    for key, val in m_final.items():
        if val is not None:
            print(f"         {key}: {val}")

    df_filtrado_full["_cluster"] = labels_final
    nombres_clusters = {i: f"Cluster {i}" for i in range(f2["best_k"])}

    print("\n[4/7] Guardando modelo, reducer y datasets ...")
    save_path = Path(config["data"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    if modelo is not None:
        joblib.dump(modelo, save_path / f"{f2['best_algo'].lower()}_k{f2['best_k']}.pkl")
    joblib.dump(reducer, save_path / "umap_reducer.pkl")
    np.save(save_path / "X_umap_full.npy", X_umap_full)
    np.save(save_path / "labels_final.npy", labels_final)
    joblib.dump(scaler, save_path / "scaler.pkl")
    joblib.dump(encoder, save_path / "encoder.pkl")
    df_filtrado_full.to_parquet(save_path / "df_clustered.parquet", index=False)
    for f in os.listdir(save_path):
        size = os.path.getsize(save_path / f) / 1e6
        print(f"      {f:<35} {size:.1f} MB")

    print("\n[5/7] Ablation study (justificación de UMAP) ...")
    df_ablation = ablation_study(X_full, labels_final, m_final, f2["best_ndim"])
    print(df_ablation.to_string(index=False))
    df_ablation.to_csv(results_dir / "ablation_study.csv", index=False)

    print("\n[6/7] Generando figuras (embedding, heatmap, geografía) ...")
    plot_clusters_2d(
        X_umap_viz, labels_final,
        titulo=f"{f2['best_algo']} (K={f2['best_k']}) — UMAP {f2['best_ndim']}D — {X_full.shape[0]:,} datos",
        nombres=nombres_clusters,
        guardar=figures_dir / f"embedding_{f2['best_algo'].lower()}_k{f2['best_k']}.png",
    )

    df_heat = calcular_heatmap_caracterizacion(df_filtrado_full, cluster_column="_cluster")
    clusters_ids = sorted(c for c in df_filtrado_full["_cluster"].unique() if c != -1)
    plot_heatmap_caracterizacion(
        df_heat, clusters_ids, nombres_clusters,
        guardar=figures_dir / "heatmap_clusters.png",
    )

    col_geo = config["data"]["col_geo"]
    tabla_geo = analizar_geografia(df_filtrado_full, cluster_column="_cluster", col_geo=col_geo, nombres_clusters=nombres_clusters)
    if tabla_geo is not None:
        plot_geo_heatmap(tabla_geo, guardar=figures_dir / "geo_clusters.png")
        top_departamentos_por_cluster(tabla_geo)

    print("\n[7/7] Validando representatividad de la muestra de Fase 1 (200k vs 452k) ...")
    _, df_filtrado_200k, *_ = preprocesar_saber_pro(df, sample_n=config["fase1"]["sample_n"])
    df_rep = validar_representatividad(df_limpio_full, df_filtrado_200k)
    print(df_rep.to_string(index=False))
    df_rep.to_csv(results_dir / "validacion_representatividad.csv", index=False)

    print(f"\n✅ Fase 2 completa. Resultados en {results_dir}/, figuras en {figures_dir}/, modelo en {save_path}/")


if __name__ == "__main__":
    main()
