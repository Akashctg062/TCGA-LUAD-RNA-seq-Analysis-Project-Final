"""R12_fig2_discovery.py — Main Figure 2: discovery of the hub genes (600 dpi).
A volcano (shrunken log2FC if R12a output exists), B signed-WGCNA module-trait, C ME2 MM vs GS,
D STRING consensus-rank robustness across 9 networks."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from adjustText import adjust_text
from R12_figstyle import NEW, FIG_MAIN, W2, RED, BLUE, GREY, panel, save

ROOT = NEW.parent
HUB = pd.read_csv(NEW / "hub_genes_final.csv")
HUB7 = HUB.gene.tolist()

fig = plt.figure(figsize=(W2, 6.3))
G = gs.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.42, width_ratios=[1, 1])

# ── A volcano ──
ax = fig.add_subplot(G[0, 0])
shr = NEW / "12_DE_revision" / "DE_full_517T_59N.csv"
if shr.exists():
    d = pd.read_csv(shr, index_col=0); lfc = d.log2FC_shrunk; lab = "Shrunken log$_2$ fold change (tumor vs normal)"
    cls = d.DEG
else:
    d = pd.read_csv(ROOT / "results" / "deg_all_results.csv", index_col=0); lfc = d.log2FoldChange
    lab = "log$_2$ fold change (MLE; tumor vs normal)"
    cls = np.where((d.padj < .05) & (lfc > 1), "Up", np.where((d.padj < .05) & (lfc < -1), "Down", "NS"))
y = -np.log10(d.padj.clip(lower=1e-300))
col = {"Up": RED, "Down": BLUE, "NS": "#d0d3d4"}
cls = pd.Series(cls, index=d.index)
for k in ["NS", "Down", "Up"]:
    m = cls == k
    ax.scatter(lfc[m], y[m], s=1.2, c=col[k], lw=0, rasterized=True,
               label=f"{'Not significant' if k == 'NS' else k} ({m.sum():,})")
hm = d.gene_name.isin(HUB7) if "gene_name" in d else d.index.isin(HUB.gene_id)
ax.scatter(lfc[hm], y[hm], s=10, facecolor="none", edgecolor="k", lw=0.6, zorder=3)
hub_idx = list(d.index[hm]); hub_idx.sort(key=lambda i: -y[i])
xs = lfc.max() * 0.80
ys = np.linspace(y.max() * 0.97, y.max() * 0.55, len(hub_idx))
for i, yy in zip(hub_idx, ys):
    ax.annotate(d.loc[i, "gene_name"], xy=(lfc[i], y[i]), xytext=(xs, yy), fontsize=5.8, fontstyle="italic",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.35, color="k", shrinkA=0, shrinkB=2))
ax.axvline(1, ls="--", lw=.5, c=GREY); ax.axvline(-1, ls="--", lw=.5, c=GREY)
ax.axhline(-np.log10(.05), ls="--", lw=.5, c=GREY)
ax.set(xlabel=lab, ylabel="$-$log$_{10}$ (adjusted p)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, markerscale=4, handletextpad=0.1,
          columnspacing=0.8, borderaxespad=0.1, fontsize=5.8)
panel(ax, "A")

# ── B module-trait (signed WGCNA, all samples) ──
ax = fig.add_subplot(G[0, 1])
mt = pd.read_csv(NEW / "02_WGCNA_signed" / "A_module_trait_tumour_status.csv").sort_values("r_tumour")
cols = [RED if (r > 0 and q < .05) else BLUE if (r < 0 and q < .05) else GREY for r, q in zip(mt.r_tumour, mt.FDR)]
ax.barh(range(len(mt)), mt.r_tumour, color=cols, height=0.7)
ax.set_yticks(range(len(mt))); ax.set_yticklabels([f"{m} ({s})" for m, s in zip(mt.module, mt["size"])])
for i, (r, q) in enumerate(zip(mt.r_tumour, mt.FDR)):
    ax.text(r + (0.02 if r >= 0 else -0.02), i, f"{r:+.2f}", va="center", ha="left" if r >= 0 else "right", fontsize=6)
ax.axvline(0, c="k", lw=0.6); ax.set_xlim(-0.95, 0.95)
ax.set(xlabel="Correlation of module eigengene with tumor status (r)",
       title="Signed WGCNA (power 15), 576 samples")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
panel(ax, "B", x=-0.30)

# ── C MM vs GS in ME2 (left) + zoom on hub-gene cluster (right) ──
sub = G[1, 0].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.08)
ax = fig.add_subplot(sub[0]); az = fig.add_subplot(sub[1])
km = pd.read_csv(NEW / "02_WGCNA_signed" / "A_key_module_ME2_all_genes.csv")
rest = km[~km.gene_name.isin(HUB7)]; hub = km[km.gene_name.isin(HUB7)].set_index("gene_name").loc[HUB7]
ax.scatter(rest.MM_own, rest.GS_tumour, s=3, c="#e6a19a", lw=0, alpha=.8, rasterized=True)
ax.scatter(hub.MM_own, hub.GS_tumour, s=12, c=RED, edgecolor="k", lw=0.4, zorder=3)
ax.axvline(.8, ls="--", lw=.5, c=GREY); ax.axhline(.5, ls="--", lw=.5, c=GREY)
r = np.corrcoef(km.MM_own, km.GS_tumour)[0, 1]
X0, X1, Y0, Y1 = .898, .962, .505, .595
ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec="k", lw=.6))
ax.set(xlabel="Module membership (ME2)", ylabel="Gene significance (tumor status)",
       xlim=(-0.25, 1.0), ylim=(-0.25, 0.8))
ax.set_title(f"ME2 (834 genes), r = {r:.2f}", loc="left", fontsize=7)
z = km[km.MM_own.between(X0, X1) & km.GS_tumour.between(Y0, Y1)]
az.scatter(z.MM_own, z.GS_tumour, s=6, c="#e6a19a", lw=0)
az.scatter(hub.MM_own, hub.GS_tumour, s=18, c=RED, edgecolor="k", lw=0.4, zorder=3)
offs = {g: (5, 0, "left") for g in HUB7}
offs.update({"AURKB": (5, 0, "left"), "CCNA2": (5, 0, "left")})
for g, rr in hub.iterrows():
    dx, dy, ha = offs[g]
    az.annotate(g, (rr.MM_own, rr.GS_tumour), xytext=(dx, dy), textcoords="offset points", fontsize=5.6,
                fontstyle="italic", ha=ha, va="center")
az.set_xlim(X0, X1); az.set_ylim(Y0, Y1); az.set_title("zoom (boxed area)", loc="right", fontsize=6)
az.set_xticks([.91, .93, .95]); az.tick_params(labelsize=5.5); az.yaxis.tick_right()
for s_ in az.spines.values(): s_.set_visible(True); s_.set_linewidth(.6)
panel(ax, "C", x=-0.30)

# ── D consensus rank heatmap ──
ax = fig.add_subplot(G[1, 1])
R = pd.read_csv(NEW / "03_PPI_hub_selection" / "consensus_rank_across_9_networks.csv", index_col=0)
top = R.head(12)
M = top.drop(columns=["n_top10", "n_networks_present", "median_rank"])
im = ax.imshow(M.clip(upper=40).values, cmap="viridis_r", vmin=1, vmax=40, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                color="white" if (not np.isnan(v) and v > 20) else "black")
ax.set_xticks(range(M.shape[1]))
ax.set_xticklabels([c.replace("_", "\n≥") for c in M.columns], fontsize=5.6)
ax.set_yticks(range(M.shape[0]))
ax.set_yticklabels([f"{g}  ({int(n)}/9)" for g, n in zip(top.index, top.n_top10)], fontsize=6)
for t in ax.get_yticklabels():
    if t.get_text().split()[0] in HUB7: t.set_color(RED); t.set_fontweight("bold")
ax.set_title("STRING v12 consensus centrality rank\n(E1 all channels, E2 without co-expression, E3 physical)")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("Consensus rank", fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
panel(ax, "D", x=-0.36)

save(fig, FIG_MAIN / "Figure2_discovery")
print("saved; volcano source:", "shrunken" if shr.exists() else "MLE (re-run after R12a)")
