# -*- coding: utf-8 -*-
"""R2-4: reanalisis de datos faltantes en educacion de los padres.

El manuscrito original imputa FAMI_EDUCACIONPADRE / FAMI_EDUCACIONMADRE
(~22-23% de valores faltantes cada una) con un unico valor de mediana por
columna. El Revisor 2 pide repetir el analisis con (a) indicadores de valor
faltante y/o (b) imputacion multiple, y comparar las asignaciones de cluster
resultantes contra la particion original.

Metodologia:
  - Partimos de X_full.npy (32 columnas, imputacion baseline = mediana),
    construido por preprocess.py, y reconstruimos la mascara de valores
    faltantes originales de las dos columnas de educacion aplicando el mismo
    mapeo ordinal sobre el CSV crudo y el mismo filtro de filas (dropna en
    columnas de puntaje) que usa preprocesar_saber_pro().
  - Variante A (indicadores de faltante): mismas 32 columnas + 2 columnas
    binarias (1 = el valor de esa fila fue imputado).
  - Variante B (imputacion multiple / MICE): se reemplazan los valores
    imputados por mediana en las 2 columnas de educacion por una imputacion
    IterativeImputer (BayesianRidge, sample_posterior=True) usando como
    predictores las otras 3 variables ordinales y las 5 de puntaje. Se genera
    una imputacion distinta (semilla distinta) por cada repeticion del
    pipeline.
  - Para 3 semillas independientes (misma muestra de ajuste de 80,000 y mismo
    conjunto de evaluacion de 50,000 en cada semilla, para aislar el efecto
    del tratamiento de datos faltantes de la variabilidad ordinaria del
    pipeline), se corre UMAP+KMeans(K=8) sobre: baseline (mediana), variante A
    (indicadores) y variante B (MICE). Se calcula el ARI entre cada variante y
    el baseline de la MISMA semilla (alineando clusters con el metodo
    hungaro), y se reporta media/DE sobre las 3 semillas.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import adjusted_rand_score
import umap.umap_ as umap_cpu

OUT_DIR = "/home/claude/missing_data_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

K = 8
N_FIT = 80_000
N_EVAL = 50_000
SEEDS = [500, 501, 502]

MAPA_EDUC = {
    'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
    'Secundaria (Bachillerato) incompleta': 3,
    'Secundaria (Bachillerato) completa': 4,
    'Técnica o tecnológica incompleta': 5,
    'Técnica o tecnológica completa': 6,
    'Educación profesional incompleta': 7,
    'EDUCACIÓN PROFESIONAL COMPLETA': 8, 'POSTGRADO': 9,
}
COLUMNAS_PUNTAJE = [
    'MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
    'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT', 'MOD_COMUNI_ESCRITA_PUNT']
COLUMNAS_ORDINALES = [
    'FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD',
    'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'ESTU_HORASSEMANATRABAJA']


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def construir_mascara_faltantes():
    """Reproduce el mapeo + filtro de filas de preprocesar_saber_pro() para
    obtener, en el MISMO orden que X_full.npy, la mascara original (antes de
    imputar) de FAMI_EDUCACIONPADRE / FAMI_EDUCACIONMADRE."""
    df = pd.read_csv('/mnt/user-data/uploads/df_maestra.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^Column1')]

    df['FAMI_EDUCACIONPADRE'] = df['FAMI_EDUCACIONPADRE'].map(MAPA_EDUC)
    df['FAMI_EDUCACIONMADRE'] = df['FAMI_EDUCACIONMADRE'].map(MAPA_EDUC)
    falta_padre = df['FAMI_EDUCACIONPADRE'].isna().to_numpy()
    falta_madre = df['FAMI_EDUCACIONMADRE'].isna().to_numpy()

    # mismo filtro de filas que preprocesar_saber_pro(): requiere puntajes no-nulos
    puntajes = df[COLUMNAS_PUNTAJE].apply(pd.to_numeric, errors='coerce')
    filas_validas = puntajes.notna().all(axis=1).to_numpy()

    return falta_padre[filas_validas], falta_madre[filas_validas], df.loc[filas_validas].reset_index(drop=True)


def construir_ordinales_base(df_filtrado):
    """Reconstruye las 5 columnas ordinales EXACTAMENTE como preprocess.py
    (mismo mapeo + imputacion por mediana), para verificar consistencia con
    X_full.npy y como insumo de las variantes A/B."""
    mapa_bano = {'1': 1, '2': 2, '3 o 4': 3, '5 o 6': 5, 'MAS DE 6': 6, 'NINGUNA': 0}
    mapa_estrato = {'Sin estrato': 0, 'Estrato 1': 1, 'Estrato 2': 2,
                    'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6}
    mapa_valormatricula = {
        'Sin costo': 0, 'Menos de 500 mil': 1,
        'Entre 500 mil y menos de 1 millón': 2,
        'Entre 1 millón y menos de 2.5 millones': 3,
        'Entre 2.5 millones y menos de 4 millones': 4,
        'Entre 4 millones y menos de 5.5 millones': 5,
        'Entre 5.5 millones y menos de 7 millones': 6,
        'Más de 7 millones': 7}
    mapeo_horas = {'0': 0, 'Menos de 10 horas': 1, 'Entre 11 y 20 horas': 2,
                   'Entre 21 y 30 horas': 3, 'Más de 30 horas': 4}

    d = df_filtrado.copy()
    d['FAMI_ESTRATOVIVIENDA'] = d['FAMI_ESTRATOVIVIENDA'].map(mapa_estrato)
    d['ESTU_VALORMATRICULAUNIVERSIDAD'] = d['ESTU_VALORMATRICULAUNIVERSIDAD'].map(mapa_valormatricula)
    d['ESTU_HORASSEMANATRABAJA'] = d['ESTU_HORASSEMANATRABAJA'].map(mapeo_horas)
    # FAMI_EDUCACIONPADRE / MADRE ya vienen mapeadas (con NaN) desde construir_mascara_faltantes()

    for col in COLUMNAS_ORDINALES:
        d[col] = d[col].fillna(d[col].median())

    return d[COLUMNAS_ORDINALES].values.astype(float)


def imputar_mice(df_filtrado, falta_padre, falta_madre, seed):
    """Imputacion multiple (MICE) de las 2 columnas de educacion, usando como
    predictores las otras 3 ordinales + las 5 de puntaje. Devuelve las 5
    columnas ordinales completas (mismo orden que COLUMNAS_ORDINALES)."""
    mapa_estrato = {'Sin estrato': 0, 'Estrato 1': 1, 'Estrato 2': 2,
                    'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6}
    mapa_valormatricula = {
        'Sin costo': 0, 'Menos de 500 mil': 1,
        'Entre 500 mil y menos de 1 millón': 2,
        'Entre 1 millón y menos de 2.5 millones': 3,
        'Entre 2.5 millones y menos de 4 millones': 4,
        'Entre 4 millones y menos de 5.5 millones': 5,
        'Entre 5.5 millones y menos de 7 millones': 6,
        'Más de 7 millones': 7}
    mapeo_horas = {'0': 0, 'Menos de 10 horas': 1, 'Entre 11 y 20 horas': 2,
                   'Entre 21 y 30 horas': 3, 'Más de 30 horas': 4}

    d = df_filtrado.copy()
    d['FAMI_ESTRATOVIVIENDA'] = d['FAMI_ESTRATOVIVIENDA'].map(mapa_estrato)
    d['ESTU_VALORMATRICULAUNIVERSIDAD'] = d['ESTU_VALORMATRICULAUNIVERSIDAD'].map(mapa_valormatricula)
    d['ESTU_HORASSEMANATRABAJA'] = d['ESTU_HORASSEMANATRABAJA'].map(mapeo_horas)
    for col in ['FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD', 'ESTU_HORASSEMANATRABAJA']:
        d[col] = d[col].fillna(d[col].median())

    puntajes = d[COLUMNAS_PUNTAJE].apply(pd.to_numeric, errors='coerce')
    for col in COLUMNAS_PUNTAJE:
        puntajes[col] = puntajes[col].fillna(puntajes[col].mean())

    predictores = pd.concat([
        d[['FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD', 'ESTU_HORASSEMANATRABAJA']],
        puntajes,
        d[['FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE']],  # con NaN -> se imputan
    ], axis=1)

    imputer = IterativeImputer(
        estimator=BayesianRidge(), sample_posterior=True,
        max_iter=8, random_state=seed, min_value=0, max_value=9,
    )
    completo = imputer.fit_transform(predictores)
    padre_imp = np.clip(np.round(completo[:, -2]), 0, 9)
    madre_imp = np.clip(np.round(completo[:, -1]), 0, 9)

    ord_out = np.column_stack([
        d['FAMI_ESTRATOVIVIENDA'].values,
        d['ESTU_VALORMATRICULAUNIVERSIDAD'].values,
        padre_imp,
        madre_imp,
        d['ESTU_HORASSEMANATRABAJA'].values,
    ]).astype(float)
    return ord_out


def correr_pipeline(X, n_total, seed, idx_fit, idx_eval):
    X_fit = X[idx_fit]
    X_eval = X[idx_eval]
    reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                             low_memory=True, n_jobs=-1)
    reducer.fit(X_fit)
    emb_eval = reducer.transform(X_eval)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=10_000)
    labels = km.fit_predict(emb_eval)
    return labels


def ari_alineado(labels_ref, labels_otro):
    """ARI simple (no requiere alineacion de etiquetas: adjusted_rand_score ya
    es invariante a la permutacion de etiquetas)."""
    return adjusted_rand_score(labels_ref, labels_otro)


def main():
    t0 = time.time()
    log("Cargando X_full.npy...")
    X_full = np.load("/home/claude/X_full.npy")
    n_total = X_full.shape[0]

    log("Reconstruyendo mascara de valores faltantes (educacion padre/madre)...")
    falta_padre, falta_madre, df_filtrado = construir_mascara_faltantes()
    assert len(falta_padre) == n_total, f"{len(falta_padre)} vs {n_total}"
    log(f"% faltante padre={falta_padre.mean()*100:.1f}%  madre={falta_madre.mean()*100:.1f}%")

    # sanity check: la reconstruccion de las 5 columnas ordinales (mediana)
    # debe coincidir con las primeras 5 columnas de X_full
    ord_check = construir_ordinales_base(df_filtrado.assign(
        FAMI_EDUCACIONPADRE=np.where(falta_padre, np.nan, df_filtrado['FAMI_EDUCACIONPADRE'].map(MAPA_EDUC)),
        FAMI_EDUCACIONMADRE=np.where(falta_madre, np.nan, df_filtrado['FAMI_EDUCACIONMADRE'].map(MAPA_EDUC)),
    ))
    diff = np.abs(ord_check - X_full[:, :5]).max()
    log(f"Verificacion baseline (debe ser ~0): max|diff| = {diff:.4f}")

    # ---- Variante A: indicadores de faltante ----
    X_indicador = np.hstack([X_full, falta_padre.reshape(-1, 1).astype(float),
                              falta_madre.reshape(-1, 1).astype(float)])
    log(f"X_indicador shape: {X_indicador.shape}")

    resultados = {"seeds": SEEDS, "ari_indicador": [], "ari_mice": []}
    results_path = os.path.join(OUT_DIR, "resultados.json")

    for seed in SEEDS:
        t_s = time.time()
        rng_fit = np.random.default_rng(seed=seed)
        idx_fit = rng_fit.choice(n_total, size=N_FIT, replace=False)
        rng_eval = np.random.default_rng(seed=seed + 1000)
        idx_eval = rng_eval.choice(n_total, size=N_EVAL, replace=False)

        log(f"[seed={seed}] pipeline baseline (mediana)...")
        labels_base = correr_pipeline(X_full, n_total, seed, idx_fit, idx_eval)

        log(f"[seed={seed}] pipeline variante A (indicadores)...")
        labels_ind = correr_pipeline(X_indicador, n_total, seed, idx_fit, idx_eval)
        ari_ind = ari_alineado(labels_base, labels_ind)

        log(f"[seed={seed}] imputacion MICE...")
        ord_mice = imputar_mice(df_filtrado, falta_padre, falta_madre, seed=seed)
        X_mice = np.hstack([ord_mice, X_full[:, 5:]])
        log(f"[seed={seed}] pipeline variante B (MICE)...")
        labels_mice = correr_pipeline(X_mice, n_total, seed, idx_fit, idx_eval)
        ari_mice = ari_alineado(labels_base, labels_mice)

        resultados["ari_indicador"].append(ari_ind)
        resultados["ari_mice"].append(ari_mice)
        json.dump(resultados, open(results_path, "w"), indent=2)
        log(f"[seed={seed}] ARI base-vs-indicador={ari_ind:.4f}  base-vs-MICE={ari_mice:.4f}  "
            f"({time.time()-t_s:.1f}s)")

    ari_ind_arr = np.array(resultados["ari_indicador"])
    ari_mice_arr = np.array(resultados["ari_mice"])
    summary = {
        "K": K, "N_FIT": N_FIT, "N_EVAL": N_EVAL, "seeds": SEEDS,
        "pct_faltante_padre": float(falta_padre.mean() * 100),
        "pct_faltante_madre": float(falta_madre.mean() * 100),
        "ari_indicador_mean": float(ari_ind_arr.mean()), "ari_indicador_sd": float(ari_ind_arr.std()),
        "ari_mice_mean": float(ari_mice_arr.mean()), "ari_mice_sd": float(ari_mice_arr.std()),
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    log(f"RESUMEN: ARI base-vs-indicador = {ari_ind_arr.mean():.4f} ± {ari_ind_arr.std():.4f}")
    log(f"RESUMEN: ARI base-vs-MICE      = {ari_mice_arr.mean():.4f} ± {ari_mice_arr.std():.4f}")
    log("LISTO.")


if __name__ == "__main__":
    main()
