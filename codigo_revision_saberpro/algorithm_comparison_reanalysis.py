# -*- coding: utf-8 -*-
"""R2-3: comparacion de algoritmos de clustering sobre espacios de busqueda
COMPARABLES.

El Revisor 2 senala 3 problemas de la Fase 1 original (confirmados al
inspeccionar el notebook):
  1) DBSCAN solo se probo en 2D (evaluar_dbscan(X_200k, n_dim=2)), mientras
     que KMeans/GMM/HDBSCAN se probaron en 2-5D.
  2) HDBSCAN solo probo min_cluster_size en {5, 10, 15}, muy pequeno respecto
     a las 200,000 observaciones de la Fase 1 (0.0025%-0.0075% de N).
  3) GMM solo se corrio con covariance_type='full' y n_init=1; no se reporto
     ni se probo sensibilidad a estas decisiones.

Este script reproduce EXACTAMENTE la metodologia original de Fase 1
(preprocesar_saber_pro(sample_n=200_000, random_state=42) + reducir_umap con
el mismo n_neighbors=10 y la misma submuestra de ajuste fija de 80,000,
sembrada con random_state=42 dentro de reducir_umap, igual que el notebook
original) y luego:
  A) Corre DBSCAN en 2D-5D (antes solo 2D), con el mismo procedimiento de
     estimacion de eps vía k-distance y min_samples=[5,10,20] usado
     originalmente.
  B) Amplia el barrido de HDBSCAN con min_cluster_size mas grandes
     (50, 100, 250, 500, 1000), ademas de repetir los valores pequenos
     originales (5, 10, 15) para dims 2-5, como control de consistencia.
  C) En el punto operativo elegido en el manuscrito (K=8, 2D), corre GMM con
     las 4 covariance_type disponibles en sklearn y n_init en {1, 10}, y lo
     mismo en K=2 (el K con mejor Silhouette bruto) como contraste.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap.umap_ as umap_cpu
import hdbscan

import sys
sys.path.insert(0, "/home/claude")
from preprocess import preprocesar_saber_pro  # noqa: E402

OUT_DIR = "/home/claude/algo_comparison_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "progress.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def guardar(resultados, path):
    json.dump(resultados, open(path, "w"), indent=2, default=_json_default)


def reducir_umap(X, n_components=2, random_state=42, **kwargs):
    """Copia EXACTA de la funcion del notebook original (incluida la
    submuestra de ajuste, siempre sembrada con 42 independientemente de
    random_state -- asi se genero resultados_fase1_2.csv)."""
    reducer = umap_cpu.UMAP(n_components=n_components, random_state=random_state,
                             n_neighbors=10, low_memory=True, n_jobs=-1, **kwargs)
    if X.shape[0] > 100_000:
        rng = np.random.default_rng(42)
        idx_fit = rng.choice(X.shape[0], size=80_000, replace=False)
        reducer.fit(X[idx_fit])
        return reducer.transform(X)
    return reducer.fit_transform(X)


def dunn_index(X_np, labels_np):
    unique = np.unique(labels_np)
    unique = unique[unique != -1]
    if len(unique) < 2:
        return np.nan
    clusters = [X_np[labels_np == l] for l in unique]
    centroids = [c.mean(axis=0) for c in clusters]
    min_inter = min(np.linalg.norm(centroids[i] - centroids[j])
                     for i in range(len(centroids)) for j in range(i + 1, len(centroids)))
    max_intra = max((np.linalg.norm(c - c.mean(axis=0), axis=1).max() * 2 if len(c) > 1 else 0)
                     for c in clusters) or 1
    return min_inter / max_intra


def calcular_metricas(X_np, labels_np, inertia=None, bic=None):
    mask = labels_np != -1
    X_eval, lbl_eval = X_np[mask], labels_np[mask]
    if len(set(lbl_eval)) < 2:
        return None
    sample = min(10_000, len(lbl_eval))
    sil = silhouette_score(X_eval, lbl_eval, sample_size=sample, random_state=42)
    return {
        'silhouette': round(sil, 4),
        'calinski': round(calinski_harabasz_score(X_eval, lbl_eval), 2),
        'davies_bouldin': round(davies_bouldin_score(X_eval, lbl_eval), 4),
        'dunn': round(dunn_index(X_eval, lbl_eval), 4),
        'bic': round(bic, 2) if bic is not None else None,
        'n_clusters': int(len(set(lbl_eval))),
        'n_ruido': int((labels_np == -1).sum()),
    }


def main():
    t0 = time.time()
    resultados_path = os.path.join(OUT_DIR, "resultados.json")
    if os.path.exists(resultados_path):
        resultados = json.load(open(resultados_path))
    else:
        resultados = {"dbscan": [], "hdbscan": [], "gmm": [], "dims_done": []}

    log("Preprocesando X_200k (misma muestra que Fase 1 original: sample_n=200000, seed=42)...")
    df = pd.read_csv('/mnt/user-data/uploads/df_maestra.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^Column1')]
    _, _, X_200k, _, _, _ = preprocesar_saber_pro(df, sample_n=200_000, random_state=42)
    log(f"X_200k shape: {X_200k.shape}")

    dims = [2, 3, 4, 5]
    embeddings_path = os.path.join(OUT_DIR, "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    for n_dim in dims:
        emb_file = os.path.join(embeddings_path, f"umap_{n_dim}d.npy")
        if os.path.exists(emb_file):
            X_umap = np.load(emb_file)
            log(f"[{n_dim}D] embedding cacheado, shape={X_umap.shape}")
        else:
            t_d = time.time()
            log(f"[{n_dim}D] ajustando UMAP...")
            X_umap = reducir_umap(X_200k, n_components=n_dim)
            np.save(emb_file, X_umap)
            log(f"[{n_dim}D] listo en {time.time()-t_d:.1f}s")

        # ---- A) DBSCAN en 2D-5D (antes solo 2D) ----
        dims_dbscan_done = resultados.get("_dims_dbscan_done", [])
        if n_dim not in dims_dbscan_done:
            log(f"[{n_dim}D] DBSCAN: estimando eps via k-distance...")
            nn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X_umap)
            dists, _ = nn.kneighbors(X_umap)
            k_dists = np.sort(dists[:, -1])
            eps_vals = [round(np.percentile(k_dists, p), 3) for p in [10, 25, 50]]
            log(f"[{n_dim}D] eps estimados: {eps_vals}")
            for eps in eps_vals:
                for min_s in [5, 10, 20]:
                    try:
                        labels = DBSCAN(eps=eps, min_samples=min_s, n_jobs=-1).fit_predict(X_umap)
                        noise_pct = (labels == -1).mean() * 100
                        m = calcular_metricas(X_umap, labels)
                        row = {'algoritmo': 'DBSCAN', 'n_dim': n_dim, 'eps': eps,
                               'min_samples': min_s, 'noise_pct': round(noise_pct, 2)}
                        if m:
                            row.update(m)
                        resultados["dbscan"].append(row)
                        log(f"[{n_dim}D] DBSCAN eps={eps} min_s={min_s} -> "
                            f"n_cls={row.get('n_clusters')} ruido={noise_pct:.1f}%")
                    except Exception as e:
                        log(f"[{n_dim}D] DBSCAN eps={eps} min_s={min_s} ERROR: {e}")
            dims_dbscan_done.append(n_dim)
            resultados["_dims_dbscan_done"] = dims_dbscan_done
            guardar(resultados, resultados_path)

        # ---- B) HDBSCAN ampliado (chico + grande) ----
        dims_hdb_done = resultados.get("_dims_hdb_done", [])
        if n_dim not in dims_hdb_done:
            min_cluster_vals = [5, 10, 15, 50, 100, 250, 500, 1000]
            for mc in min_cluster_vals:
                for ms in [5, 10]:
                    try:
                        labels = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=ms).fit_predict(X_umap)
                        noise_pct = (labels == -1).mean() * 100
                        m = calcular_metricas(X_umap, labels)
                        row = {'algoritmo': 'HDBSCAN', 'n_dim': n_dim, 'min_cluster_size': mc,
                               'min_samples': ms, 'noise_pct': round(noise_pct, 2)}
                        if m:
                            row.update(m)
                        resultados["hdbscan"].append(row)
                        log(f"[{n_dim}D] HDBSCAN mc={mc} ms={ms} -> "
                            f"n_cls={row.get('n_clusters')} ruido={noise_pct:.1f}%")
                    except Exception as e:
                        log(f"[{n_dim}D] HDBSCAN mc={mc} ms={ms} ERROR: {e}")
            dims_hdb_done.append(n_dim)
            resultados["_dims_hdb_done"] = dims_hdb_done
            guardar(resultados, resultados_path)

    # ---- C) GMM: sensibilidad a covariance_type / n_init en el punto operativo ----
    if not resultados.get("_gmm_done"):
        X_2d = np.load(os.path.join(embeddings_path, "umap_2d.npy"))
        for k in [2, 8]:
            for cov in ['full', 'tied', 'diag', 'spherical']:
                for n_init in [1, 10]:
                    try:
                        gmm = GaussianMixture(n_components=k, random_state=42,
                                               covariance_type=cov, n_init=n_init)
                        labels = gmm.fit_predict(X_2d)
                        bic = gmm.bic(X_2d)
                        m = calcular_metricas(X_2d, labels, bic=bic)
                        row = {'algoritmo': 'GMM', 'n_dim': 2, 'k': k,
                               'covariance_type': cov, 'n_init': n_init}
                        if m:
                            row.update(m)
                        resultados["gmm"].append(row)
                        log(f"GMM k={k} cov={cov} n_init={n_init} -> "
                            f"sil={m['silhouette'] if m else None} bic={bic:.1f}")
                    except Exception as e:
                        log(f"GMM k={k} cov={cov} n_init={n_init} ERROR: {e}")
        resultados["_gmm_done"] = True
        guardar(resultados, resultados_path)

    log(f"TODO LISTO. Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
