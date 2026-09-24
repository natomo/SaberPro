# -*- coding: utf-8 -*-
"""
R2-5: effect sizes (eta-squared, Cohen's d) matched to the manuscript's real
K=8 clustering (Table 7/8), not to an independently-labeled rerun.
R2-9: sample coverage / repeat-presenter check.
"""
import json, time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import umap.umap_ as umap_cpu

OUT = "/home/claude/effect_size_out"
import os
os.makedirs(OUT, exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

TABLE7 = {
    0: (54623, 135.92), 1: (66104, 150.21), 2: (69891, 145.69), 3: (68047, 140.92),
    4: (62257, 161.98), 5: (63929, 148.48), 6: (60240, 136.30), 7: (6929, 205.28),
}

log("Loading df_filtrado_full.pkl ...")
df = pd.read_pickle("/home/claude/df_filtrado_full.pkl")
n_total = len(df)
log(f"n_total={n_total}")

COLS_ORD = ['FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD',
            'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'ESTU_HORASSEMANATRABAJA']
COLS_PUNT = ['MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
             'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT', 'MOD_COMUNI_ESCRITA_PUNT']
COLS_NOM = ['ESTU_TITULOOBTENIDOBACHILLER', 'ESTU_PAGOMATRICULABECA',
            'ESTU_PAGOMATRICULACREDITO', 'ESTU_PAGOMATRICULAPADRES',
            'ESTU_PAGOMATRICULAPROPIO', 'ESTU_COMOCAPACITOEXAMENSB11',
            'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
            'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA']

X = np.load("/home/claude/X_full.npy")
assert X.shape[0] == n_total

log("Fitting UMAP on 80,000-subsample (seed=42), transforming full data (matches Methodology)...")
rng = np.random.default_rng(42)
idx_fit = rng.choice(n_total, size=80_000, replace=False)
reducer = umap_cpu.UMAP(n_components=2, random_state=42, n_neighbors=10, low_memory=True, n_jobs=1)
reducer.fit(X[idx_fit])
emb = reducer.transform(X)
log("UMAP done.")

km = MiniBatchKMeans(n_clusters=8, random_state=42, n_init="auto", batch_size=10_000)
labels = km.fit_predict(emb)
sizes = pd.Series(labels).value_counts().sort_index()
log(f"Cluster sizes from this run: {dict(sizes)}")

# global mean score per run-cluster to match against Table 7
df['_label_run'] = labels
df['PUNT_GLOBAL'] = df[COLS_PUNT].mean(axis=1)
means_run = df.groupby('_label_run')['PUNT_GLOBAL'].mean().sort_values()
sizes_run = df['_label_run'].value_counts()

log("Run cluster sizes (sorted by mean score):")
for lbl in means_run.index:
    log(f"  run-label {lbl}: n={sizes_run[lbl]}, mean={means_run[lbl]:.2f}")

# Compare to Table 7 sorted by mean score
table7_sorted = sorted(TABLE7.items(), key=lambda kv: kv[1][1])
log("Table 7 (sorted by mean score): " + str(table7_sorted))

# Save raw comparison; matching + effect sizes computed in a second step after eyeballing
np.save(os.path.join(OUT, "labels_run.npy"), labels)
df[['_label_run', 'PUNT_GLOBAL']].to_csv(os.path.join(OUT, "labels_and_scores.csv"), index=False)
json.dump({"sizes_run": {int(k): int(v) for k, v in sizes_run.items()},
           "means_run": {int(k): float(v) for k, v in means_run.items()},
           "table7": TABLE7}, open(os.path.join(OUT, "comparison.json"), "w"), indent=2)
log("Saved comparison data. DONE STEP 1.")
