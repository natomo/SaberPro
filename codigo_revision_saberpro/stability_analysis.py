# -*- coding: utf-8 -*-
"""Analisis de estabilidad del pipeline COMPLETO (UMAP + K-Means), respondiendo
a E2 (editor) y R2-2 (Revisor 2): variar semillas de UMAP Y submuestras de
ajuste de 80,000 obs. de forma independiente en cada repeticion, no solo la
inicializacion de K-Means con el embedding UMAP fijo.

Protocolo:
  - K = 8 (igual al manuscrito publicado; el notebook original traia
    BEST_K = 6 en un comentario que parece una config de prueba desactualizada
    -- se usa K=8 porque es lo que reportan el manuscrito y ambos revisores).
  - Conjunto de evaluacion FIJO: 50,000 filas (semilla 0), constante en todas
    las repeticiones, para poder comparar asignaciones de cluster punto a punto.
  - En cada repeticion r = 1..R:
      1) Muestra de ajuste de 80,000 filas, extraida de forma INDEPENDIENTE
         (semilla distinta por repeticion) del conjunto completo.
      2) UMAP.fit() sobre esa muestra con random_state = r (semilla distinta).
      3) transform() del conjunto de evaluacion fijo -> embedding 2D.
      4) K-Means (K=8, random_state = r) sobre el embedding del conjunto de
         evaluacion.
      5) Se guardan las etiquetas de cluster del conjunto de evaluacion.
  - Al final: ARI por pares entre repeticiones (distribucion completa),
    matriz de consenso/co-clustering sobre una submuestra, y estabilidad
    por cluster (alineando con el metodo hungaro contra una repeticion de
    referencia).
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/stability_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

K = 8
N_EVAL = 50_000
N_FIT = 80_000
N_REPS = 15


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    t_start = time.time()
    log("Cargando X_full...")
    X_full = np.load("/home/claude/X_full.npy")
    n_total = X_full.shape[0]
    log(f"X_full shape: {X_full.shape}")

    # Conjunto de evaluación fijo (constante en todas las repeticiones)
    rng_eval = np.random.default_rng(seed=0)
    idx_eval = rng_eval.choice(n_total, size=N_EVAL, replace=False)
    np.save(os.path.join(OUT_DIR, "idx_eval.npy"), idx_eval)
    X_eval = X_full[idx_eval]

    labels_path = os.path.join(OUT_DIR, "labels_all.npy")
    done_path = os.path.join(OUT_DIR, "done_reps.json")

    if os.path.exists(labels_path) and os.path.exists(done_path):
        labels_all = np.load(labels_path)
        done_reps = json.load(open(done_path))
        log(f"Reanudando: {len(done_reps)} repeticiones ya completas")
    else:
        labels_all = np.full((N_REPS, N_EVAL), -1, dtype=int)
        done_reps = []

    for r in range(N_REPS):
        if r in done_reps:
            continue
        t_r = time.time()
        seed = 100 + r  # semilla distinta por repetición (tanto para muestreo como UMAP/KMeans)

        rng_fit = np.random.default_rng(seed=seed)
        idx_fit = rng_fit.choice(n_total, size=N_FIT, replace=False)
        X_fit = X_full[idx_fit]

        reducer = umap_cpu.UMAP(
            n_components=2, random_state=seed, n_neighbors=10,
            low_memory=True, n_jobs=-1,
        )
        reducer.fit(X_fit)
        emb_eval = reducer.transform(X_eval)

        km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto",
                              batch_size=10_000)
        labels = km.fit_predict(emb_eval)

        labels_all[r] = labels
        done_reps.append(r)
        np.save(labels_path, labels_all)
        json.dump(done_reps, open(done_path, "w"))
        log(f"Repetición {r+1}/{N_REPS} lista en {time.time()-t_r:.1f}s "
            f"(seed={seed}, n_clusters_encontrados={len(set(labels))})")

    log(f"TODAS las repeticiones completas. Tiempo total: {time.time()-t_start:.1f}s")

    # ---------------- Análisis post-hoc ----------------
    log("Calculando ARI por pares...")
    aris = []
    for i in range(N_REPS):
        for j in range(i + 1, N_REPS):
            ari = adjusted_rand_score(labels_all[i], labels_all[j])
            aris.append(ari)
    aris = np.array(aris)
    log(f"ARI: media={aris.mean():.4f}  DE={aris.std():.4f}  "
        f"min={aris.min():.4f}  max={aris.max():.4f}  n_pares={len(aris)}")
    np.save(os.path.join(OUT_DIR, "ari_pairs.npy"), aris)

    # Alinear todas las repeticiones a la repetición 0 (referencia) vía Hungarian
    log("Alineando clusters contra repetición de referencia (Hungarian)...")
    ref = labels_all[0]
    aligned = np.zeros_like(labels_all)
    aligned[0] = ref
    for r in range(1, N_REPS):
        cost = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                cost[a, b] = -np.sum((ref == a) & (labels_all[r] == b))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {b: a for a, b in zip(row_ind, col_ind)}
        aligned[r] = np.array([mapping.get(lab, -1) for lab in labels_all[r]])
    np.save(os.path.join(OUT_DIR, "labels_aligned.npy"), aligned)

    # Estabilidad por cluster: para cada punto, tasa de acuerdo con la referencia
    # a través de las repeticiones alineadas (excluyendo la propia referencia)
    log("Calculando estabilidad por clúster...")
    agree = (aligned[1:] == ref[None, :]).mean(axis=0)  # tasa de acuerdo por punto
    cluster_stability = {}
    for c in range(K):
        mask = ref == c
        cluster_stability[c] = {
            "n_puntos": int(mask.sum()),
            "tasa_acuerdo_media": float(agree[mask].mean()) if mask.sum() > 0 else None,
            "tasa_acuerdo_sd": float(agree[mask].std()) if mask.sum() > 0 else None,
        }
    json.dump(cluster_stability, open(os.path.join(OUT_DIR, "cluster_stability.json"), "w"), indent=2)
    for c, st in cluster_stability.items():
        log(f"  Cluster {c}: n={st['n_puntos']:,}  acuerdo medio={st['tasa_acuerdo_media']:.3f}"
            if st['tasa_acuerdo_media'] is not None else f"  Cluster {c}: vacío")

    # Matriz de consenso/co-clustering sobre una submuestra de 1500 puntos
    log("Calculando matriz de consenso (submuestra 1500 puntos)...")
    rng_co = np.random.default_rng(seed=7)
    idx_co = rng_co.choice(N_EVAL, size=1500, replace=False)
    sub = aligned[:, idx_co]  # (N_REPS, 1500)
    co_matrix = np.zeros((1500, 1500), dtype=np.float32)
    for r in range(N_REPS):
        lab = sub[r]
        same = (lab[:, None] == lab[None, :])
        co_matrix += same
    co_matrix /= N_REPS
    np.save(os.path.join(OUT_DIR, "co_matrix.npy"), co_matrix)
    np.save(os.path.join(OUT_DIR, "idx_co_within_eval.npy"), idx_co)
    # también guardamos las etiquetas de referencia de esa submuestra para ordenar el heatmap
    np.save(os.path.join(OUT_DIR, "ref_labels_co.npy"), ref[idx_co])

    summary = {
        "K": K,
        "N_EVAL": N_EVAL,
        "N_FIT": N_FIT,
        "N_REPS": N_REPS,
        "ari_mean": float(aris.mean()),
        "ari_sd": float(aris.std()),
        "ari_min": float(aris.min()),
        "ari_max": float(aris.max()),
        "cluster_stability": cluster_stability,
        "tiempo_total_seg": time.time() - t_start,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    log("LISTO. Resumen guardado en summary.json")


if __name__ == "__main__":
    main()
