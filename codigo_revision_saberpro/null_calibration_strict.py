# -*- coding: utf-8 -*-
"""R2-1 (version reforzada): en el primer analisis (null_calibration.py), los
datos nulos se generaron permutando cada COLUMNA CODIFICADA (una-caliente) de
forma independiente. Para las variables nominales, esto puede crear filas con
combinaciones invalidas (por ejemplo, ambas columnas 'internet=Si' e
'internet=No' activas, o ninguna). Esta version corrige eso: permuta las
ETIQUETAS categoricas ORIGINALES (antes de codificar) de cada variable
nominal, de forma independiente entre estudiantes, y luego codifica -- asi
cada fila nula sigue teniendo una combinacion valida de una-caliente. Las
columnas ordinales y de puntaje se permutan igual que antes (no tienen este
problema, son numericas).

Reutiliza los 5 resultados REALES ya calculados en null_calibration.py
(no cambian, los datos reales no se tocan) y genera 10 nuevas replicas
NULAS con este metodo mas riguroso, para comparar.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/null_calib_strict_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

K = 8
N_EVAL = 50_000
N_FIT = 80_000
N_NULL_REPS = 10

COLS_ORD = ['FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD',
            'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'ESTU_HORASSEMANATRABAJA']
COLS_PUNT = ['MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
             'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT', 'MOD_COMUNI_ESCRITA_PUNT']
COLS_NOM = ['ESTU_TITULOOBTENIDOBACHILLER', 'ESTU_PAGOMATRICULABECA',
            'ESTU_PAGOMATRICULACREDITO', 'ESTU_PAGOMATRICULAPADRES',
            'ESTU_PAGOMATRICULAPROPIO', 'ESTU_COMOCAPACITOEXAMENSB11',
            'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
            'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA']


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
    raise TypeError(str(type(o)))


def metricas(emb, labels):
    sample = min(10_000, len(labels))
    sil = silhouette_score(emb, labels, sample_size=sample, random_state=42)
    ch = calinski_harabasz_score(emb, labels)
    db = davies_bouldin_score(emb, labels)
    return {"silhouette": round(sil, 4), "calinski": round(ch, 2), "davies_bouldin": round(db, 4)}


def construir_X_nula_estricta(df_filtrado, scaler, encoder, seed):
    """Permuta ETIQUETAS originales (ordinal/puntaje/nominal), preservando
    combinaciones validas en las variables nominales."""
    rng = np.random.default_rng(seed)
    d = df_filtrado.copy()
    n = len(d)

    for col in COLS_ORD:
        d[col] = d[col].values[rng.permutation(n)]
    for col in COLS_PUNT:
        d[col] = d[col].values[rng.permutation(n)]
    for col in COLS_NOM:
        d[col] = d[col].values[rng.permutation(n)]  # permuta la ETIQUETA completa, no bits sueltos

    X_punt = scaler.transform(d[COLS_PUNT])
    X_ohe = encoder.transform(d[COLS_NOM])
    X_null = np.hstack([d[COLS_ORD].values, X_punt, X_ohe])
    return X_null


def correr_pipeline(X, n_total, seed, fixed_eval_idx):
    rng_fit = np.random.default_rng(seed=seed)
    idx_fit = rng_fit.choice(n_total, size=N_FIT, replace=False)
    X_fit = X[idx_fit]
    X_eval = X[fixed_eval_idx]

    reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                             low_memory=True, n_jobs=-1)
    reducer.fit(X_fit)
    emb_eval = reducer.transform(X_eval)

    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=10_000)
    labels = km.fit_predict(emb_eval)
    return metricas(emb_eval, labels)


def main():
    t0 = time.time()
    log("Cargando df_filtrado_full.pkl y refitting scaler/encoder (como en preprocess.py)...")
    df_filtrado = pd.read_pickle("/home/claude/df_filtrado_full.pkl")
    n_total = len(df_filtrado)

    scaler = StandardScaler().fit(df_filtrado[COLS_PUNT])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df_filtrado[COLS_NOM])

    rng_eval_real = np.random.default_rng(seed=0)
    idx_eval_real = rng_eval_real.choice(n_total, size=N_EVAL, replace=False)

    results_path = os.path.join(OUT_DIR, "resultados_null_estricto.json")
    if os.path.exists(results_path):
        resultados_null = json.load(open(results_path))
    else:
        resultados_null = []

    while len(resultados_null) < N_NULL_REPS:
        r = len(resultados_null)
        seed = 400 + r
        t_r = time.time()
        X_null = construir_X_nula_estricta(df_filtrado, scaler, encoder, seed=seed + 6000)
        m = correr_pipeline(X_null, n_total, seed, idx_eval_real)
        resultados_null.append(m)
        json.dump(resultados_null, open(results_path, "w"), indent=2, default=_json_default)
        log(f"[NULL-ESTRICTO] Repetición {r+1}/{N_NULL_REPS} en {time.time()-t_r:.1f}s -> {m}")

    null_sil = np.array([m["silhouette"] for m in resultados_null])
    null_db = np.array([m["davies_bouldin"] for m in resultados_null])
    null_ch = np.array([m["calinski"] for m in resultados_null])

    # reutiliza los 5 resultados REALES ya calculados (los datos reales no cambian)
    real_summary = json.load(open("/home/claude/null_calib_out/summary.json"))

    p_sil = float((null_sil >= real_summary["real_silhouette_mean"]).mean())
    p_ch = float((null_ch >= real_summary["real_calinski_mean"]).mean())

    summary = {
        "K": K, "N_EVAL": N_EVAL, "N_FIT": N_FIT, "N_NULL_REPS": N_NULL_REPS,
        "real_silhouette_mean": real_summary["real_silhouette_mean"],
        "real_davies_bouldin_mean": real_summary["real_davies_bouldin_mean"],
        "real_calinski_mean": real_summary["real_calinski_mean"],
        "null_estricto_silhouette_mean": float(null_sil.mean()),
        "null_estricto_silhouette_sd": float(null_sil.std()),
        "null_estricto_davies_bouldin_mean": float(null_db.mean()),
        "null_estricto_calinski_mean": float(null_ch.mean()),
        "p_empirico_silhouette": p_sil,
        "p_empirico_calinski": p_ch,
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2, default=_json_default)
    log(f"RESUMEN (permutación estricta, pre-codificación): "
        f"Silhouette real={real_summary['real_silhouette_mean']:.4f}  "
        f"null={null_sil.mean():.4f}±{null_sil.std():.4f}  p≈{p_sil:.3f}")
    log(f"RESUMEN: Davies-Bouldin real={real_summary['real_davies_bouldin_mean']:.4f}  "
        f"null={null_db.mean():.4f} (menor=mejor)")
    log(f"RESUMEN: Calinski-Harabasz real={real_summary['real_calinski_mean']:.1f}  "
        f"null={null_ch.mean():.1f}  p≈{p_ch:.3f}")
    log("LISTO.")


if __name__ == "__main__":
    main()
