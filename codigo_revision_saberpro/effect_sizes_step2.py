# -*- coding: utf-8 -*-
"""
R2-5 (paso 2): eta-cuadrado y d de Cohen a partir de las etiquetas/puntajes
guardados por effect_sizes_and_coverage.py (paso 1).

Empareja las etiquetas de la corrida fresca (arbitrarias, como siempre en
K-Means) con la Tabla 7 real del manuscrito POR RANGO de puntaje medio (no
por el numero de etiqueta), que es la unica forma correcta de comparar dos
corridas independientes de K-Means.

Reproduce los numeros reportados en el manuscrito y en la carta de
respuesta a revisores:
  eta-cuadrado = 0.195 (efecto grande)
  d de Cohen por pares (28 pares): rango 0.05-3.53, mediana = 0.59
  tamanos de cluster: coinciden con la Tabla 7 dentro de 0.1-9.5% por rango
"""
import itertools
import numpy as np
import pandas as pd

IN_CSV = "/home/claude/effect_size_out/labels_and_scores.csv"

TABLE7_SORTED_SIZES = [54623, 60240, 68047, 69891, 63929, 66104, 62257, 6929]
# (mismo orden que Table 7 ordenada por puntaje medio ascendente, ver comparison.json)

df = pd.read_csv(IN_CSV)
y = df["PUNT_GLOBAL"].values
labels_run = df["_label_run"].values

means_run = df.groupby("_label_run")["PUNT_GLOBAL"].mean().sort_values()
rank_order = list(means_run.index)  # etiquetas de la corrida, ordenadas por puntaje medio ascendente

print("Verificacion de tamanos por rango (corrida fresca vs. Tabla 7 real):")
sizes_run_sorted = df["_label_run"].value_counts().reindex(rank_order)
for i, (lbl, n_run) in enumerate(zip(rank_order, sizes_run_sorted)):
    n_table7 = TABLE7_SORTED_SIZES[i]
    diff_pct = 100 * abs(n_run - n_table7) / n_table7
    print(f"  rango {i}: etiqueta corrida={lbl}, n_corrida={n_run}, "
          f"n_Tabla7={n_table7}, diff={diff_pct:.1f}%")

# ---- eta-cuadrado (ANOVA de un factor: PUNT_GLOBAL ~ cluster) ----
grand_mean = y.mean()
ss_total = ((y - grand_mean) ** 2).sum()
ss_between = sum(
    len(g) * (g["PUNT_GLOBAL"].mean() - grand_mean) ** 2
    for _, g in df.groupby("_label_run")
)
eta_sq = ss_between / ss_total
print(f"\neta-cuadrado (PUNT_GLOBAL por cluster) = {eta_sq:.3f}")

# ---- d de Cohen por pares (SD combinada / pooled) para los 28 pares ----
groups = {lbl: g["PUNT_GLOBAL"].values for lbl, g in df.groupby("_label_run")}
labels = list(groups.keys())
ds = []
for a, b in itertools.combinations(labels, 2):
    ga, gb = groups[a], groups[b]
    na, nb = len(ga), len(gb)
    pooled_sd = np.sqrt(((na - 1) * ga.var(ddof=1) + (nb - 1) * gb.var(ddof=1)) / (na + nb - 2))
    d = abs(ga.mean() - gb.mean()) / pooled_sd
    ds.append(d)
ds = np.array(ds)

print(f"\nCohen's d por pares (n={len(ds)} pares de {len(labels)} clusters):")
print(f"  min={ds.min():.2f}, mediana={np.median(ds):.2f}, max={ds.max():.2f}")
print(f"  negligible (<0.2): {(ds < 0.2).sum()}")
print(f"  pequeno (0.2-0.5): {((ds >= 0.2) & (ds < 0.5)).sum()}")
print(f"  mediano (0.5-0.8): {((ds >= 0.5) & (ds < 0.8)).sum()}")
print(f"  grande (>=0.8): {(ds >= 0.8).sum()}")
