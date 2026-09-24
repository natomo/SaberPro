# -*- coding: utf-8 -*-
"""R2-6: la pertenencia al clúster aporta informacion mas alla de la
combinacion lineal de los insumos?

El Revisor 2 pide predecir un resultado (puntaje, idealmente graduacion)
condicionado en las covariables, comparando el modelo con y sin la
pertenencia al clúster. No tenemos variable de graduacion en este dataset,
asi que usamos PUNT_GLOBAL (que SI esta en df_filtrado_full via cols_extra).

Trampa metodologica a evitar: PUNT_GLOBAL es esencialmente una combinacion de
los 5 modulos de puntaje (MOD_*_PUNT), que SON parte de las variables usadas
para construir los clusteres publicados (X_full, 32 columnas). Si probamos
"covariables + cluster publicado" prediciendo PUNT_GLOBAL, el cluster ya
"sabe" el puntaje del estudiante porque se construyo usando esos mismos 5
modulos -> mejora artificial/circular, no una prueba justa.

Disenio (2 pruebas, para ser transparentes con ambas):
  1) PRUEBA LIMPIA (no circular): se construye un clustering alternativo
     SOLO con las covariables socioeconomicas/de acceso (excluyendo los 5
     modulos de puntaje), K=8, mismo pipeline UMAP+KMeans. Se compara, sobre
     el mismo conjunto de evaluacion de 50,000 filas:
        Modelo A: Ridge(PUNT_GLOBAL ~ covariables SES/acceso)
        Modelo B: Ridge(PUNT_GLOBAL ~ covariables SES/acceso + dummies del
                  cluster SES-only)
     Si el cluster aporta senal mas alla de una combinacion LINEAL de esas
     mismas covariables, R2(B) > R2(A) de forma no trivial.
     Tambien se repite con HistGradientBoostingRegressor (combinacion NO
     lineal de las covariables) como prueba mas exigente.
  2) PRUEBA ILUSTRATIVA (circular, con caveat explicito): mismo ejercicio
     pero usando el cluster PUBLICADO (construido con los 5 modulos de
     puntaje incluidos) en vez del cluster SES-only. Se espera una mejora
     mucho mayor, pero se reporta explicitamente como no valida por
     circularidad/fuga de informacion.

Todo se evalua con validacion cruzada de 5 folds (R2) sobre el mismo
conjunto de evaluacion fijo de 50,000 filas usado en el analisis de
estabilidad (E2/R2-2), reutilizando ese idx_eval.npy para consistencia.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
import umap.umap_ as umap_cpu

import sys
sys.path.insert(0, "/home/claude")
from preprocess import preprocesar_saber_pro  # noqa: E402

OUT_DIR = "/home/claude/predictive_value_out"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("Cargando X_full.npy, df_filtrado_full.pkl, idx_eval de estabilidad...")
    X_full = np.load("/home/claude/X_full.npy")
    df_filtrado_full = pd.read_pickle("/home/claude/df_filtrado_full.pkl")
    idx_eval = np.load("/home/claude/stability_out/idx_eval.npy")
    labels_publicado = np.load("/home/claude/stability_out/labels_all.npy")[0]  # rep 0, seed=100
    n_total = X_full.shape[0]

    # y = PUNT_GLOBAL en el conjunto de evaluacion
    y_eval = df_filtrado_full["PUNT_GLOBAL"].values[idx_eval].astype(float)
    mask_y = ~np.isnan(y_eval)
    log(f"PUNT_GLOBAL disponible en {mask_y.sum()}/{len(y_eval)} filas del conjunto de evaluacion")

    # columnas SES/acceso = todo excepto los 5 modulos de puntaje (posiciones 5:10)
    cols_score = slice(5, 10)
    X_ses_full = np.hstack([X_full[:, :5], X_full[:, 10:]])  # 5 ordinales + 22 OHE = 27 cols
    log(f"X_ses_full shape: {X_ses_full.shape}")

    # ---- Clustering SES-only (excluye puntajes), mismo pipeline (K=8) ----
    seed = 600
    rng_fit = np.random.default_rng(seed=seed)
    idx_fit = rng_fit.choice(n_total, size=80_000, replace=False)
    log("Ajustando UMAP sobre covariables SES-only (sin puntajes)...")
    reducer = umap_cpu.UMAP(n_components=2, random_state=seed, n_neighbors=10,
                             low_memory=True, n_jobs=-1)
    reducer.fit(X_ses_full[idx_fit])
    emb_eval_ses = reducer.transform(X_ses_full[idx_eval])
    km = MiniBatchKMeans(n_clusters=8, random_state=seed, n_init="auto", batch_size=10_000)
    labels_ses = km.fit_predict(emb_eval_ses)
    log(f"Clustering SES-only listo. Tamaños de clúster: {np.bincount(labels_ses)}")

    # ---- Matrices de diseño para los modelos ----
    X_cov = X_ses_full[idx_eval][mask_y]
    y = y_eval[mask_y]
    labels_ses_m = labels_ses[mask_y]
    labels_pub_m = labels_publicado[mask_y]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    dummies_ses = ohe.fit_transform(labels_ses_m.reshape(-1, 1))
    dummies_pub = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(
        labels_pub_m.reshape(-1, 1))

    X_A = X_cov
    X_B_ses = np.hstack([X_cov, dummies_ses])
    X_C_pub = np.hstack([X_cov, dummies_pub])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    log("Ridge: Modelo A (solo covariables SES/acceso)...")
    r2_A = cross_val_score(Ridge(alpha=1.0), X_A, y, cv=cv, scoring="r2")
    log(f"  R2 = {r2_A.mean():.4f} +- {r2_A.std():.4f}")

    log("Ridge: Modelo B (covariables + clúster SES-only, no circular)...")
    r2_B = cross_val_score(Ridge(alpha=1.0), X_B_ses, y, cv=cv, scoring="r2")
    log(f"  R2 = {r2_B.mean():.4f} +- {r2_B.std():.4f}")

    log("Ridge: Modelo C (covariables + clúster PUBLICADO, ilustrativo/circular)...")
    r2_C = cross_val_score(Ridge(alpha=1.0), X_C_pub, y, cv=cv, scoring="r2")
    log(f"  R2 = {r2_C.mean():.4f} +- {r2_C.std():.4f}")

    log("HistGBM: baseline (solo covariables SES/acceso)...")
    r2_gbm_A = cross_val_score(HistGradientBoostingRegressor(random_state=42), X_A, y, cv=cv, scoring="r2")
    log(f"  R2 = {r2_gbm_A.mean():.4f} +- {r2_gbm_A.std():.4f}")

    log("HistGBM: covariables + clúster SES-only...")
    r2_gbm_B = cross_val_score(HistGradientBoostingRegressor(random_state=42), X_B_ses, y, cv=cv, scoring="r2")
    log(f"  R2 = {r2_gbm_B.mean():.4f} +- {r2_gbm_B.std():.4f}")

    resultados = {
        "n_eval_con_y": int(mask_y.sum()),
        "ridge_A_solo_covariables": {"mean": float(r2_A.mean()), "sd": float(r2_A.std())},
        "ridge_B_covariables_mas_cluster_ses": {"mean": float(r2_B.mean()), "sd": float(r2_B.std())},
        "ridge_C_covariables_mas_cluster_publicado_CIRCULAR": {"mean": float(r2_C.mean()), "sd": float(r2_C.std())},
        "gbm_A_solo_covariables": {"mean": float(r2_gbm_A.mean()), "sd": float(r2_gbm_A.std())},
        "gbm_B_covariables_mas_cluster_ses": {"mean": float(r2_gbm_B.mean()), "sd": float(r2_gbm_B.std())},
        "delta_r2_ridge_no_circular": float(r2_B.mean() - r2_A.mean()),
        "delta_r2_ridge_circular_ilustrativo": float(r2_C.mean() - r2_A.mean()),
        "delta_r2_gbm_no_circular": float(r2_gbm_B.mean() - r2_gbm_A.mean()),
        "tiempo_total_seg": time.time() - t0,
    }
    json.dump(resultados, open(f"{OUT_DIR}/resultados.json", "w"), indent=2)
    np.save(f"{OUT_DIR}/labels_ses.npy", labels_ses)
    log("LISTO.")
    log(json.dumps(resultados, indent=2))


if __name__ == "__main__":
    main()
