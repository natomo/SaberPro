# -*- coding: utf-8 -*-
"""R2-10: los estudiantes estan anidados en programas e instituciones -- los
perfiles podrian ser una configuracion institucional, no de estudiante.

(a) Composicion institucional/de programa por clúster: índice de
    concentracion (HHI) de instituciones y de programas dentro de cada
    clúster publicado, comparado con el HHI de referencia (toda la muestra).
(b) Analisis de sensibilidad leave-institution-out / leave-program-out: para
    las 5 instituciones y los 5 programas mas grandes, se reajusta el
    pipeline EXCLUYENDO cada uno (uno a la vez) y se compara -- sobre los
    MISMOS estudiantes restantes -- la particion resultante contra la
    particion publicada (ARI), para ver si excluir una sola institucion o
    programa grande cambia sustancialmente la estructura para el resto.
"""
import json
import os
import time
import re

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/institutional_out"
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


def hhi(shares):
    return float(np.sum(np.square(shares)))


def normalizar_programa(s):
    if pd.isna(s):
        return s
    s = s.upper().strip()
    reemplazos = str.maketrans('ÁÉÍÓÚ', 'AEIOU')
    return s.translate(reemplazos)


def main():
    t0 = time.time()
    X_full = np.load("/home/claude/X_full.npy")
    df_filtrado_full = pd.read_pickle("/home/claude/df_filtrado_full.pkl")
    n_total = X_full.shape[0]
    df_filtrado_full['PROGRAMA_NORM'] = df_filtrado_full['ESTU_PRGM_ACADEMICO'].apply(normalizar_programa)

    idx_eval = np.load("/home/claude/stability_out/idx_eval.npy")
    labels_pub = np.load("/home/claude/stability_out/labels_all.npy")[0]

    # ---- (a) Composicion institucional / de programa por clúster ----
    log("=== (a) Concentración institucional/de programa por clúster ===")
    df_eval = pd.DataFrame({
        'cluster': labels_pub,
        'institucion': df_filtrado_full['INST_COD_INSTITUCION'].values[idx_eval],
        'programa': df_filtrado_full['PROGRAMA_NORM'].values[idx_eval],
    })
    hhi_ref_inst = hhi(df_eval['institucion'].value_counts(normalize=True).values)
    hhi_ref_prog = hhi(df_eval['programa'].value_counts(normalize=True).values)
    log(f"HHI de referencia (toda la muestra de evaluación): instituciones={hhi_ref_inst:.4f}  "
        f"programas={hhi_ref_prog:.4f}")

    composicion = {}
    for c in sorted(df_eval['cluster'].unique()):
        sub = df_eval[df_eval['cluster'] == c]
        inst_shares = sub['institucion'].value_counts(normalize=True)
        prog_shares = sub['programa'].value_counts(normalize=True)
        composicion[int(c)] = {
            'n': int(len(sub)),
            'n_instituciones_distintas': int(sub['institucion'].nunique()),
            'share_institucion_top1': float(inst_shares.iloc[0]),
            'hhi_instituciones': hhi(inst_shares.values),
            'n_programas_distintos': int(sub['programa'].nunique()),
            'share_programa_top1': float(prog_shares.iloc[0]),
            'hhi_programas': hhi(prog_shares.values),
        }
        log(f"Clúster {c}: n={len(sub):,}  instituciones distintas={sub['institucion'].nunique()}  "
            f"HHI_inst={composicion[int(c)]['hhi_instituciones']:.4f} (ref {hhi_ref_inst:.4f})  "
            f"top1_inst={inst_shares.iloc[0]*100:.1f}%  |  "
            f"programas distintos={sub['programa'].nunique()}  "
            f"HHI_prog={composicion[int(c)]['hhi_programas']:.4f} (ref {hhi_ref_prog:.4f})")

    # ---- (b) Leave-institution-out / leave-program-out ----
    log("=== (b) Sensibilidad: excluir instituciones/programas grandes, uno a la vez ===")
    top_inst = df_filtrado_full['INST_COD_INSTITUCION'].value_counts().head(5).index.tolist()
    top_prog = df_filtrado_full['PROGRAMA_NORM'].value_counts().head(5).index.tolist()
    log(f"Top 5 instituciones a excluir: {top_inst}")
    log(f"Top 5 programas a excluir: {top_prog}")

    resultados_sens = {"instituciones": {}, "programas": {}}

    def correr_leave_out(mask_excluir, seed):
        """Reajusta el pipeline excluyendo las filas en mask_excluir, y
        devuelve el ARI entre la particion resultante (para los estudiantes
        del idx_eval que NO pertenecen al grupo excluido) y la particion
        publicada para esos mismos estudiantes."""
        idx_incluidos = np.where(~mask_excluir)[0]
        X_incl = X_full[idx_incluidos]
        n_incl = X_incl.shape[0]

        rng = np.random.default_rng(seed)
        idx_fit_local = rng.choice(n_incl, size=min(80_000, n_incl), replace=False)
        reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                                 low_memory=True, n_jobs=-1)
        reducer.fit(X_incl[idx_fit_local])

        # evaluamos sobre los mismos estudiantes de idx_eval que sobreviven la exclusión
        mask_eval_incl = ~mask_excluir[idx_eval]
        idx_eval_incl = idx_eval[mask_eval_incl]
        emb_eval = reducer.transform(X_full[idx_eval_incl])
        km = MiniBatchKMeans(n_clusters=8, random_state=seed, n_init="auto", batch_size=10_000)
        labels_new = km.fit_predict(emb_eval)

        labels_pub_sub = labels_pub[mask_eval_incl]
        ari = adjusted_rand_score(labels_pub_sub, labels_new)
        return ari, int(mask_eval_incl.sum()), int(mask_excluir.sum())

    for i, inst in enumerate(top_inst):
        t_i = time.time()
        mask_excl = (df_filtrado_full['INST_COD_INSTITUCION'] == inst).values
        ari, n_eval_incl, n_excl = correr_leave_out(mask_excl, seed=800 + i)
        resultados_sens["instituciones"][str(inst)] = {
            "n_excluidos_del_total": n_excl,
            "n_eval_restante": n_eval_incl,
            "ari_vs_publicado": ari,
        }
        log(f"[excluir institución {inst}] n_excluidos={n_excl:,}  ARI vs. publicado={ari:.4f}  "
            f"({time.time()-t_i:.1f}s)")
        json.dump(resultados_sens, open(f"{OUT_DIR}/resultados_sensibilidad.json", "w"),
                   indent=2, default=_json_default)

    for i, prog in enumerate(top_prog):
        t_i = time.time()
        mask_excl = (df_filtrado_full['PROGRAMA_NORM'] == prog).values
        ari, n_eval_incl, n_excl = correr_leave_out(mask_excl, seed=850 + i)
        resultados_sens["programas"][str(prog)] = {
            "n_excluidos_del_total": n_excl,
            "n_eval_restante": n_eval_incl,
            "ari_vs_publicado": ari,
        }
        log(f"[excluir programa {prog}] n_excluidos={n_excl:,}  ARI vs. publicado={ari:.4f}  "
            f"({time.time()-t_i:.1f}s)")
        json.dump(resultados_sens, open(f"{OUT_DIR}/resultados_sensibilidad.json", "w"),
                   indent=2, default=_json_default)

    resultados = {
        "hhi_referencia": {"instituciones": hhi_ref_inst, "programas": hhi_ref_prog},
        "composicion_por_cluster": composicion,
        "sensibilidad_leave_out": resultados_sens,
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(resultados, open(f"{OUT_DIR}/resultados.json", "w"), indent=2, default=_json_default)
    log("LISTO.")


if __name__ == "__main__":
    main()
