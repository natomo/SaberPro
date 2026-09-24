# -*- coding: utf-8 -*-
"""Figura de estabilidad del pipeline completo (UMAP+K-Means): distribución
de ARI por pares + matriz de consenso/co-clustering."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = "/home/claude/stability_out"
aris = np.load(f"{OUT}/ari_pairs.npy")
co = np.load(f"{OUT}/co_matrix.npy")
ref_labels_co = np.load(f"{OUT}/ref_labels_co.npy")
summary = json.load(open(f"{OUT}/summary.json"))

# ordenar la matriz de consenso por cluster de referencia para que se vean bloques
order = np.argsort(ref_labels_co, kind="stable")
co_ordered = co[order][:, order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=170,
                                gridspec_kw={"width_ratios": [1, 1.15]})

# --- Panel 1: distribución de ARI por pares ---
ax1.hist(aris, bins=18, color="#3B6FA0", edgecolor="white", linewidth=0.6)
ax1.axvline(aris.mean(), color="#B33F3F", linewidth=1.6, linestyle="--",
            label=f"Media = {aris.mean():.3f}")
ax1.axvline(0.69, color="#555555", linewidth=1.4, linestyle=":",
            label="ARI reportado en el manuscrito = 0.69\n(solo variando init. K-Means)")
ax1.set_xlabel("ARI (Adjusted Rand Index) por par de repeticiones")
ax1.set_ylabel("N.º de pares")
ax1.set_title(f"Estabilidad del pipeline completo\n(n={summary['N_REPS']} repeticiones, "
              f"{len(aris)} pares)", fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=7.5, frameon=False, loc="upper left")

# --- Panel 2: matriz de consenso/co-clustering ---
im = ax2.imshow(co_ordered, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                interpolation="nearest")
ax2.set_title("Matriz de consenso (submuestra n=1,500)\nordenada por clúster de referencia",
              fontsize=10)
ax2.set_xlabel("Puntos (ordenados por clúster)")
ax2.set_ylabel("Puntos (ordenados por clúster)")
ax2.set_xticks([]); ax2.set_yticks([])
cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
cbar.set_label("Prob. de co-asignación\nal mismo clúster", fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/figura_estabilidad_pipeline.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Guardado: figura_estabilidad_pipeline.png")
