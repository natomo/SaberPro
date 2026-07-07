#!/usr/bin/env python
"""Fase 1 — búsqueda de hiperparámetros sobre una muestra de 200k estudiantes.

Evalúa KMeans, GMM, DBSCAN y HDBSCAN sobre distintas dimensiones UMAP y
guarda el resumen ordenado por silhouette score. Usa el TOP de este
resultado para fijar fase2.best_algo / best_k / best_ndim en config.yaml
antes de correr run_fase2_modelo_final.py.

Uso:
    python scripts/run_fase1_busqueda.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saberpro_clustering.config import load_config  # noqa: E402
from saberpro_clustering.hyperparameter_search import (  # noqa: E402
    evaluar_dbscan,
    evaluar_gmm,
    evaluar_hdbscan,
    evaluar_kmeans,
    resumen_fase1,
)
from saberpro_clustering.preprocessing import load_raw_data, preprocesar_saber_pro  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: búsqueda de hiperparámetros")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    f1 = config["fase1"]

    print(f"[1/3] Cargando datos desde {config['data']['input_path']} ...")
    df = load_raw_data(config["data"]["input_path"])

    print(f"[2/3] Preprocesando muestra de {f1['sample_n']:,} estudiantes ...")
    _, df_filtrado_200k, X_200k, *_ = preprocesar_saber_pro(df, sample_n=f1["sample_n"])
    print(f"      Shape X_200k: {X_200k.shape}")

    dims = range(min(f1["dims_umap"]), max(f1["dims_umap"]) + 1)
    ks = range(min(f1["ks"]), max(f1["ks"]) + 1)

    print("\n[3/3] Ejecutando búsqueda de hiperparámetros ...")
    print("=== Fase 1: KMeans ===")
    df_res_km = evaluar_kmeans(X_200k, dims=dims, ks=ks)

    print("\n=== Fase 1: GMM ===")
    df_res_gmm = evaluar_gmm(X_200k, dims=dims, ks=ks)

    print("\n=== Fase 1: DBSCAN ===")
    df_res_db, _ = evaluar_dbscan(
        X_200k,
        n_dim=f1["dbscan"]["n_dim"],
        min_samples_vals=f1["dbscan"]["min_samples_vals"],
    )

    print("\n=== Fase 1: HDBSCAN ===")
    df_res_hdb = evaluar_hdbscan(
        X_200k,
        dims=dims,
        min_cluster_vals=f1["hdbscan"]["min_cluster_vals"],
        min_samples_vals=f1["hdbscan"]["min_samples_vals"],
    )

    df_fase1 = resumen_fase1(df_res_km, df_res_gmm, df_res_db, df_res_hdb)

    cols_show = ["algoritmo", "n_dim", "n_clusters", "silhouette", "calinski", "davies_bouldin", "dunn", "noise_pct"]
    cols_show = [c for c in cols_show if c in df_fase1.columns]
    print("\n=== TOP 15 configuraciones por Silhouette ===")
    print(df_fase1[cols_show].head(15).to_string(index=False))

    output_path = Path(f1["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_fase1.to_csv(output_path, index=False)
    print(f"\n✅ Guardado: {output_path}")
    print("\n👉 Actualiza config.yaml -> fase2 con el algoritmo/K ganador antes de correr Fase 2.")


if __name__ == "__main__":
    main()
