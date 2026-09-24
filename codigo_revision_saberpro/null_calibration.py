# -*- coding: utf-8 -*-
"""R2-1: calibrar el pipeline completo (UMAP+K-Means, K=8) contra datasets
nulos (cada columna de X permutada de forma independiente, lo que rompe la
estructura multivariada pero conserva la distribución marginal de cada
variable). Si la separación real no supera claramente a la de los datos
nulos, la separación observada podría deberse al procedimiento de reducción
+ selección de modelo más que a estructura real en los datos.
"""
import json
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/null_calib_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

K = 8
N_EVAL = 50_000
N_FIT = 80_000
N_REAL_REPS = 5
N_NULL_REPS = 10


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def metricas(emb, labels):
    sample = min(10_000, len(labels))
    sil = silhouette_score(emb, labels, sample_size=sample, random_state=42)
    ch = calinski_harabasz_score(emb, labels)
    db = davies_bouldin_score(emb, labels)
    return {"silhouette": round(sil, 4), "calinski": round(ch, 2), "davies_bouldin": round(db, 4)}


def permutar_columnas(X, seed):
    """Permuta cada columna de X de forma independiente (rompe covarianza,
    conserva la distribución marginal de cada variable)."""
    rng = np.random.default_rng(seed)
    X_perm = np.empty_like(X)
    for j in range(X.shape[1]):
        X_perm[:, j] = rng.permutation(X[:, j])
    return X_perm


def correr_pipeline(X_source, n_total, seed, fixed_eval_idx=None):
    """Corre UMAP+KMeans una vez sobre X_source y devuelve métricas."""
    rng_fit = np.random.default_rng(seed=seed)
    idx_fit = rng_fit.choice(n_total, size=N_FIT, replace=False)
    X_fit = X_source[idx_fit]

    if fixed_eval_idx is None:
        rng_eval = np.random.default_rng(seed=seed + 1000)
        idx_eval = rng_eval.choice(n_total, size=N_EVAL, replace=False)
    else:
        idx_eval = fixed_eval_idx
    X_eval = X_source[idx_eval]

    reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                             low_memory=True, n_jobs=-1)
    reducer.fit(X_fit)
    emb_eval = reducer.transform(X_eval)

    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=10_000)
    labels = km.fit_predict(emb_eval)

    return metricas(emb_eval, labels)


def main():
    t0 = time.time()
    log("Cargando X_full...")
    X_full = np.load("/home/claude/X_full.npy")
    n_total = X_full.shape[0]

    rng_eval_real = np.random.default_rng(seed=0)
    idx_eval_real = rng_eval_real.choice(n_total, size=N_EVAL, replace=False)

    results_path = os.path.join(OUT_DIR, "resultados.json")
    if os.path.exists(results_path):
        results = json.load(open(results_path))
    else:
        results = {"real": [], "null": []}

    # ---- Réplicas sobre datos REALES ----
    while len(results["real"]) < N_REAL_REPS:
        r = len(results["real"])
        seed = 200 + r
        t_r = time.time()
        m = correr_pipeline(X_full, n_total, seed, fixed_eval_idx=idx_eval_real)
        results["real"].append(m)
        json.dump(results, open(results_path, "w"), indent=2)
        log(f"[REAL] Repetición {r+1}/{N_REAL_REPS} en {time.time()-t_r:.1f}s -> {m}")

    # ---- Réplicas sobre datos NULOS (columnas permutadas) ----
    while len(results["null"]) < N_NULL_REPS:
        r = len(results["null"])
        seed = 300 + r
        t_r = time.time()
        X_null = permutar_columnas(X_full, seed=seed + 5000)
        m = correr_pipeline(X_null, n_total, seed, fixed_eval_idx=None)
        results["null"].append(m)
        json.dump(results, open(results_path, "w"), indent=2)
        log(f"[NULL] Repetición {r+1}/{N_NULL_REPS} en {time.time()-t_r:.1f}s -> {m}")

    # ---- Resumen ----
    real_sil = np.array([m["silhouette"] for m in results["real"]])
    null_sil = np.array([m["silhouette"] for m in results["null"]])
    real_db = np.array([m["davies_bouldin"] for m in results["real"]])
    null_db = np.array([m["davies_bouldin"] for m in results["null"]])
    real_ch = np.array([m["calinski"] for m in results["real"]])
    null_ch = np.array([m["calinski"] for m in results["null"]])

    p_sil = float((null_sil >= real_sil.mean()).mean())
    p_ch = float((null_ch >= real_ch.mean()).mean())

    summary = {
        "K": K, "N_EVAL": N_EVAL, "N_FIT": N_FIT,
        "N_REAL_REPS": N_REAL_REPS, "N_NULL_REPS": N_NULL_REPS,
        "real_silhouette_mean": float(real_sil.mean()), "real_silhouette_sd": float(real_sil.std()),
        "null_silhouette_mean": float(null_sil.mean()), "null_silhouette_sd": float(null_sil.std()),
        "real_davies_bouldin_mean": float(real_db.mean()), "null_davies_bouldin_mean": float(null_db.mean()),
        "real_calinski_mean": float(real_ch.mean()), "null_calinski_mean": float(null_ch.mean()),
        "p_empirico_silhouette": p_sil,
        "p_empirico_calinski": p_ch,
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    log(f"RESUMEN: Silhouette real={real_sil.mean():.4f}±{real_sil.std():.4f}  "
        f"null={null_sil.mean():.4f}±{null_sil.std():.4f}  p≈{p_sil:.3f}")
    log(f"RESUMEN: Davies-Bouldin real={real_db.mean():.4f}  null={null_db.mean():.4f} "
        f"(menor=mejor)")
    log(f"RESUMEN: Calinski-Harabasz real={real_ch.mean():.1f}  null={null_ch.mean():.1f}  p≈{p_ch:.3f}")
    log("LISTO.")


if __name__ == "__main__":
    main()
