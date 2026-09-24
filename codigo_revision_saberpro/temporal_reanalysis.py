# -*- coding: utf-8 -*-
"""R2-8: analisis temporal. El manuscrito combina 4 convocatorias (2021-2 a
2023-1) sin documentar comparabilidad entre periodos. El Revisor 2 pide:
  (a) documentar comparabilidad de escalas/distribuciones entre periodos,
  (b) reportar composicion de clusteres por convocatoria,
  (c) ajustar el pipeline POR SEPARADO por periodo, como prueba de
      replicacion temporal (perfiles estables vs. especificos de cohorte).

Periodos disponibles: 20212 (n=150,444), 20222 (n=75,694), 20225 (n=103,777),
20231 (n=122,105).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/temporal_out"
os.makedirs(OUT_DIR, exist_ok=True)


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
    periodos = df_filtrado_full["PERIODO"].values

    # ---- (a) Comparabilidad de escalas entre periodos ----
    log("=== (a) Comparabilidad de puntajes entre periodos ===")
    cols_punt = ['MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
                 'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT',
                 'MOD_COMUNI_ESCRITA_PUNT', 'PUNT_GLOBAL']
    comparabilidad = {}
    for col in cols_punt:
        stats = df_filtrado_full.groupby('PERIODO')[col].agg(['mean', 'std', 'count'])
        comparabilidad[col] = stats.to_dict(orient='index')
        log(f"{col}:\n{stats}")

    # ---- (b) Composicion de clusteres publicados por periodo ----
    log("=== (b) Composicion del clustering publicado, por periodo ===")
    idx_eval = np.load("/home/claude/stability_out/idx_eval.npy")
    labels_pub = np.load("/home/claude/stability_out/labels_all.npy")[0]
    periodos_eval = periodos[idx_eval]
    df_comp = pd.DataFrame({'cluster': labels_pub, 'periodo': periodos_eval})
    tabla_comp = pd.crosstab(df_comp['periodo'], df_comp['cluster'], normalize='index') * 100
    log(f"\n{tabla_comp.round(1)}")
    tabla_comp.to_csv(f"{OUT_DIR}/composicion_cluster_por_periodo.csv")

    # ---- (c) Pipeline ajustado POR SEPARADO en cada periodo ----
    log("=== (c) Pipeline independiente por periodo ===")
    K = 8
    resultados_periodo = {}
    centroides_por_periodo = {}
    tam_por_periodo = {}
    labels_by_period = {}

    for p in sorted(pd.unique(periodos)):
        t_p = time.time()
        mask_p = periodos == p
        idx_p = np.where(mask_p)[0]
        X_p = X_full[idx_p]
        n_p = X_p.shape[0]
        log(f"[{p}] n={n_p:,}  ajustando UMAP...")

        seed = 700 + list(sorted(pd.unique(periodos))).index(p)
        reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                                 low_memory=True, n_jobs=-1)
        if n_p > 100_000:
            rng = np.random.default_rng(seed)
            idx_fit = rng.choice(n_p, size=80_000, replace=False)
            reducer.fit(X_p[idx_fit])
            emb_p = reducer.transform(X_p)
        else:
            emb_p = reducer.fit_transform(X_p)

        km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=10_000)
        labels_p = km.fit_predict(emb_p)
        labels_by_period[str(p)] = labels_p
        tam_por_periodo[str(p)] = (np.bincount(labels_p, minlength=K) / n_p * 100).tolist()

        # centroides en el espacio original de 32 features (estandarizadas/OHE), para
        # poder comparar perfiles entre periodos (las coordenadas UMAP de cada periodo
        # viven en espacios distintos y no son comparables directamente)
        centroides = np.array([X_p[labels_p == c].mean(axis=0) if (labels_p == c).sum() > 0
                                else np.full(X_p.shape[1], np.nan) for c in range(K)])
        centroides_por_periodo[str(p)] = centroides

        log(f"[{p}] listo en {time.time()-t_p:.1f}s. Tamaños (%): "
            f"{np.round(tam_por_periodo[str(p)], 1)}")

    np.savez(f"{OUT_DIR}/labels_by_period.npz", **{k: v for k, v in labels_by_period.items()})

    # ---- Emparejar clusteres entre periodos via centroides (Hungarian) ----
    log("=== Emparejando perfiles entre periodos (distancia euclidiana de centroides) ===")
    periodos_lista = sorted(pd.unique(periodos))
    ref_p = str(periodos_lista[0])  # 20212, el periodo mas grande, como referencia
    ref_cent = centroides_por_periodo[ref_p]

    emparejamientos = {}
    for p in periodos_lista[1:]:
        p = str(p)
        cent_p = centroides_por_periodo[p]
        cost = np.linalg.norm(ref_cent[:, None, :] - cent_p[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        corrs = []
        for a, b in zip(row_ind, col_ind):
            c = np.corrcoef(ref_cent[a], cent_p[b])[0, 1]
            corrs.append(float(c))
        emparejamientos[p] = {
            "match_ref_to_p": {int(a): int(b) for a, b in zip(row_ind, col_ind)},
            "correlacion_centroides_emparejados": corrs,
            "correlacion_media": float(np.mean(corrs)),
        }
        log(f"[{ref_p} vs {p}] correlación media de centroides emparejados = "
            f"{np.mean(corrs):.3f}  (por clúster: {[round(c,2) for c in corrs]})")

    resultados = {
        "comparabilidad_puntajes": comparabilidad,
        "tamanos_cluster_por_periodo_pct": tam_por_periodo,
        "emparejamientos_vs_referencia": emparejamientos,
        "periodo_referencia": ref_p,
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(resultados, open(f"{OUT_DIR}/resultados.json", "w"), indent=2, default=_json_default)
    log("LISTO.")


if __name__ == "__main__":
    main()
