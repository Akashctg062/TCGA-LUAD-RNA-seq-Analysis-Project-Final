"""
R05c_forest_plot.py — publication forest plot for R05 (replaces the draft forest plot drawn by R05).
Input : analysis/05_survival/cox_uni_multi_per_SD_OS_PFI_DSS.csv
Output: analysis/05_survival/Fig_forest_uni_multi_OS_PFI_DSS.png / .tif
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
OUT = NEW / "05_survival"
HUBS = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
res = pd.read_csv(OUT / "cox_uni_multi_per_SD_OS_PFI_DSS.csv")
pfmt = lambda q: "< 0.001" if q < 0.001 else f"= {q:.3f}"

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
ypos = np.arange(len(HUBS))[::-1]
for ax, ep in zip(axes, ["OS", "PFI", "DSS"]):
    r = res[res.endpoint == ep].set_index("gene").loc[HUBS]
    ax.errorbar(r.uni_HR, ypos + .17, xerr=[r.uni_HR - r.uni_lo, r.uni_hi - r.uni_HR], fmt="o",
                color="#7f8c8d", ms=5, capsize=2)
    ax.errorbar(r.multi_HR, ypos - .17, xerr=[r.multi_HR - r.multi_lo, r.multi_hi - r.multi_HR], fmt="s",
                color="#c0392b", ms=5, capsize=2)
    for y, (_, row) in zip(ypos, r.iterrows()):
        ax.text(1.03, y - .17, f"{row.multi_HR:.2f} ({row.multi_lo:.2f}–{row.multi_hi:.2f}); FDR {pfmt(row.multi_FDR)}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="#c0392b")
    ax.axvline(1, ls="--", c="k", lw=.7)
    ax.set_xlim(0.9, max(1.9, r[["uni_hi", "multi_hi"]].max().max() + .05))
    ax.set_title(f"{ep}\nUnivariate: n = {int(r.uni_n.iloc[0])}, {int(r.uni_events.iloc[0])} events\n"
                 f"Multivariable: n = {int(r.multi_n.iloc[0])}, {int(r.multi_events.iloc[0])} events", fontsize=10)
    ax.set_xlabel("HR per 1 SD increase (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(HUBS, fontstyle="italic", fontsize=11)
fig.legend([Line2D([], [], color="#7f8c8d", marker="o", ls=""), Line2D([], [], color="#c0392b", marker="s", ls="")],
           ["Univariate", "Multivariable (age, sex, stage II/III/IV vs I, smoking; each gene modelled separately)"],
           loc="lower center", ncol=2, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.subplots_adjust(wspace=0.75)
plt.savefig(OUT / "Fig_forest_uni_multi_OS_PFI_DSS.png", dpi=300)
plt.savefig(OUT / "Fig_forest_uni_multi_OS_PFI_DSS.tif", dpi=300)
print("saved")
