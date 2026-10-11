"""
R02b_plot_MM_vs_GS.py — ME2 MM-vs-GS scatter with clearly separated labels for the locked 7 hub genes.
Panel A: all ME2 genes; Panel B: zoom on the hub-gene cluster with one label per point.
Input : analysis/02_WGCNA_signed/A_key_module_ME2_all_genes.csv  (from R02)
Output: analysis/02_WGCNA_signed/Fig_MM_vs_GS_ME2_labelled.png / .tif (300 dpi)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.stats import pearsonr
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "02_WGCNA_signed"
HUBS = pd.read_csv(OUT.parent / "hub_genes_final.csv").gene.tolist()   # locked 7-gene set (R03b)
DARK, LIGHT = "#8e1b10", "#e6a19a"
km = pd.read_csv(OUT / "A_key_module_ME2_all_genes.csv")
hub = km[km.gene_name.isin(HUBS)].set_index("gene_name")
rest = km[~km.gene_name.isin(HUBS)]
r, p = pearsonr(km.MM_own, km.GS_tumour)
x0, x1, y0, y1 = 0.900, 0.940, 0.505, 0.595

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})

# A — full module
a.scatter(rest.MM_own, rest.GS_tumour, s=10, c=LIGHT, alpha=.6, lw=0, label="ME2 genes")
a.scatter(hub.MM_own, hub.GS_tumour, s=30, c=DARK, edgecolor="white", lw=.6, zorder=3, label=f"{len(HUBS)} hub genes")
a.axvline(.8, ls="--", c="grey", lw=.8); a.axhline(.5, ls="--", c="grey", lw=.8)
a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=DARK, lw=1))
a.set(xlabel="Module membership in ME2 (signed)", ylabel="Gene significance for tumour status (signed)",
      title=f"(A) ME2 module: MM vs GS\n(r = {r:.2f}, p = {p:.1e}, n = {len(km)})")
a.legend(loc="upper left", frameon=False, fontsize=9)

# B — zoom
z = km[km.MM_own.between(x0, x1) & km.GS_tumour.between(y0, y1)]
b.scatter(z.MM_own, z.GS_tumour, s=22, c=LIGHT, alpha=.8, lw=0)
b.scatter(hub.MM_own, hub.GS_tumour, s=70, c=DARK, edgecolor="white", lw=1, zorder=3)
offs = {g: (8, 0) for g in HUBS}
offs.update({"CCNB1": (-8, 0), "AURKB": (-8, 0)})
for g, row in hub.iterrows():
    dx, dy = offs[g]
    b.annotate(g, (row.MM_own, row.GS_tumour), xytext=(dx, dy), textcoords="offset points",
               fontsize=11, fontstyle="italic", fontweight="bold", color=DARK,
               ha="left" if dx > 0 else "right", va="center")
b.set_xlim(x0, x1); b.set_ylim(y0, y1)
b.set(xlabel="Module membership in ME2 (signed)", ylabel="",
      title="(B) Zoom: hub-gene cluster")
for s in b.spines.values(): s.set_color(DARK)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(ax is b)
# connector lines from A box to B
for ya, yb in [(y1, 1), (y0, 0)]:
    fig.add_artist(ConnectionPatch((x1, ya), (0, yb), "data", "axes fraction",
                                   axesA=a, axesB=b, color=DARK, lw=.7, alpha=.6))
plt.tight_layout(); plt.subplots_adjust(wspace=0.35)
plt.savefig(OUT / "Fig_MM_vs_GS_ME2_labelled.png", dpi=300)
plt.savefig(OUT / "Fig_MM_vs_GS_ME2_labelled.tif", dpi=300)
print("saved")
