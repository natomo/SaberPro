# -*- coding: utf-8 -*-
"""R2-11: caracterizar el Clúster 7 (manuscrito) -- pequeño (~1-1.5% de la
muestra) y con dos "islas" desconectadas en la representación UMAP.
Evaluar si refleja subpoblaciones reproducibles, valores extremos, o
errores de preprocesamiento/puntuación; caracterizar las dos islas por
separado.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/cluster7_out"
import os
os.makedirs(OUT_DIR, exist_ok=True)

COLS_ORD = ['FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD',
            'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'ESTU_HORASSEMANATRABAJA']
COLS_PUNT = ['MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
             'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT', 'MOD_COMUNI_ESCRITA_PUNT',
             'PUNT_GLOBAL']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def main():
    t0 = time.time()
    X_full = np.load("/home/claude/X_full.npy")
    df_filtrado_full = pd.read_pickle("/home/claude/df_filtrado_full.pkl")
    idx_eval = np.load("/home/claude/stability_out/idx_eval.npy")
    labels_pub = np.load("/home/claude/stability_out/labels_all.npy")[0]  # seed=100, rep 0
    n_total = X_full.shape[0]

    sizes = np.bincount(labels_pub)
    cluster_pequeno = int(np.argmin(sizes))
    log(f"Tamaños de clúster: {sizes}  -> clúster más pequeño = {cluster_pequeno} "
        f"(n={sizes[cluster_pequeno]}, {sizes[cluster_pequeno]/len(labels_pub)*100:.2f}% del eval set)")

    # ---- reproducir el embedding 2D (seed=100, igual que stability_analysis.py rep 0) ----
    log("Reproduciendo embedding UMAP (seed=100)...")
    seed = 100
    rng_fit = np.random.default_rng(seed=seed)
    idx_fit = rng_fit.choice(n_total, size=80_000, replace=False)
    reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                             low_memory=True, n_jobs=-1)
    reducer.fit(X_full[idx_fit])
    emb_eval = reducer.transform(X_full[idx_eval])
    log(f"Embedding listo. shape={emb_eval.shape}")

    mask_c = labels_pub == cluster_pequeno
    emb_c = emb_eval[mask_c]
    idx_eval_c = idx_eval[mask_c]  # índices originales (en X_full/df_filtrado_full)

    # ---- separar las dos "islas" via DBSCAN sobre las coordenadas UMAP del clúster ----
    log("Separando las dos islas (DBSCAN sobre coordenadas UMAP del clúster)...")
    db = DBSCAN(eps=0.5, min_samples=5).fit(emb_c)
    islas = db.labels_
    vals, counts = np.unique(islas, return_counts=True)
    log(f"Islas encontradas (DBSCAN): {dict(zip(vals.tolist(), counts.tolist()))}")

    # si DBSCAN no separa limpio en 2, usar KMeans(k=2) como respaldo
    if len(vals[vals != -1]) != 2:
        log("DBSCAN no dio exactamente 2 islas -> usando KMeans(k=2) de respaldo")
        km2 = MiniBatchKMeans(n_clusters=2, random_state=42, n_init="auto")
        islas = km2.fit_predict(emb_c)

    # ---- caracterizar cada isla ----
    log("Caracterizando cada isla...")
    df_c = df_filtrado_full.iloc[idx_eval_c].copy()
    df_c['isla'] = islas
    df_c['umap_x'] = emb_c[:, 0]
    df_c['umap_y'] = emb_c[:, 1]

    df_global = df_filtrado_full  # referencia (toda la muestra, 452,020)

    perfil = {}
    for isla_id in sorted(df_c['isla'].unique()):
        if isla_id == -1:
            continue
        sub = df_c[df_c['isla'] == isla_id]
        perfil[int(isla_id)] = {
            'n': int(len(sub)),
            'pct_del_cluster': float(len(sub) / len(df_c) * 100),
            'medias_puntaje': {c: float(sub[c].mean()) for c in COLS_PUNT},
            'medias_puntaje_global_referencia': {c: float(df_global[c].mean()) for c in COLS_PUNT},
            'medias_ordinales': {c: float(sub[c].mean()) for c in COLS_ORD},
            'medias_ordinales_global_referencia': {c: float(df_global[c].mean()) for c in COLS_ORD},
            'n_instituciones_distintas': int(sub['INST_COD_INSTITUCION'].nunique()),
            'institucion_top1_pct': float(sub['INST_COD_INSTITUCION'].value_counts(normalize=True).iloc[0] * 100),
            'departamentos_top3': sub['ESTU_COD_DEPTO_PRESENTACION'].value_counts().head(3).to_dict(),
            'internet_si_pct': float((sub['FAMI_TIENEINTERNET'] == 'Si').mean() * 100),
            'beca_si_pct': float((sub['ESTU_PAGOMATRICULABECA'] == 'Si').mean() * 100),
        }
        log(f"Isla {isla_id}: n={len(sub)} ({len(sub)/len(df_c)*100:.1f}% del clúster)  "
            f"PUNT_GLOBAL medio={sub['PUNT_GLOBAL'].mean():.1f} (ref global={df_global['PUNT_GLOBAL'].mean():.1f})  "
            f"instituciones distintas={sub['INST_COD_INSTITUCION'].nunique()}  "
            f"internet={  (sub['FAMI_TIENEINTERNET']=='Si').mean()*100:.1f}%")

    # ---- reproducibilidad del clúster pequeño a través de las 15 repeticiones ya calculadas ----
    cs = json.load(open("/home/claude/stability_out/cluster_stability.json"))
    reproducibilidad = cs.get(str(cluster_pequeno))

    resultados = {
        "cluster_pequeno_id": cluster_pequeno,
        "n_cluster": int(sizes[cluster_pequeno]),
        "pct_del_eval": float(sizes[cluster_pequeno] / len(labels_pub) * 100),
        "reproducibilidad_15_repeticiones": reproducibilidad,
        "perfil_por_isla": perfil,
    }
    json.dump(resultados, open(f"{OUT_DIR}/resultados.json", "w"), indent=2, default=_json_default)
    np.save(f"{OUT_DIR}/emb_eval.npy", emb_eval)
    np.save(f"{OUT_DIR}/labels_pub_eval.npy", labels_pub)
    np.save(f"{OUT_DIR}/emb_c.npy", emb_c)
    np.save(f"{OUT_DIR}/islas.npy", islas)
    log(f"LISTO. Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
