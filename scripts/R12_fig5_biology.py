"""R12_fig5_biology.py — Main Figure 5: biological context, pan-cancer and druggability (600 dpi).
A  TF activity (CollecTRI regulons without the 7 hub genes) vs 7-gene score: TF activity rho and mRNA rho
B  immune context: partial Spearman rho (adjusted for ABSOLUTE purity) of 7-gene score with CIBERSORT /
   MCP-counter populations and checkpoint genes (FDR < 0.05 filled)
C  pan-cancer overall survival: HR per SD of 7-gene score in 33 TCGA cancer types
D  DepMap Public 26Q1 CRISPR gene effect in 53 LUAD cell lines + ChEMBL highest clinical phase"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from R12_figstyle import NEW, FIG_MAIN, W2, RED, BLUE, GREY, DARK, panel, save

HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
fig = plt.figure(figsize=(W2, 8.4))
G = gs.GridSpec(2, 2, figure=fig, height_ratios=[1.55, 0.85], width_ratios=[1.05, 0.95], hspace=0.38, wspace=0.62)
left = G[0, 0].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.55)

# ── A TF activity ──
ax = fig.add_subplot(left[0])
ta = pd.read_csv(NEW / "11_extended" / "M9_TF_activity_vs_score.csv").set_index("TF")
FOC = ["E2F1", "E2F2", "E2F4", "E2F3", "MYC", "MYBL2", "FOXM1", "STAT3", "RELA", "RB1", "TP53"]
t = ta.loc[FOC].iloc[::-1]
y = np.arange(len(t))
ax.barh(y + 0.19, t.rho_activity_vs_score, 0.36, color=[RED if (q < .05 and r > 0) else BLUE if q < .05 else GREY
                                                        for r, q in zip(t.rho_activity_vs_score, t.FDR)], label="TF activity")
ax.barh(y - 0.19, t.rho_mRNA_vs_score, 0.36, color="#d5d8dc", edgecolor="#7f8c8d", lw=.4, label="TF mRNA")
ax.set_yticks(y); ax.set_yticklabels(t.index, fontstyle="italic")
ax.axvline(0, c="k", lw=.6); ax.set_xlim(-0.6, 1.0)
ax.set_xlabel("Spearman ρ with 7-gene score")
ax.set_title("TF activity (regulons without hub genes)", loc="left", fontsize=7)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, fc="#d5d8dc", ec="#7f8c8d", lw=.4)],
          ["Inferred activity", "mRNA level"], loc="lower right", fontsize=5.6)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
panel(ax, "A", x=-0.26)

# ── B immune (purity-adjusted) ──
ax = fig.add_subplot(left[1])
im = pd.read_csv(NEW / "10_immune" / "immune_correlations_raw_and_purity_adjusted.csv")
im = im[im.variable == "HUB7 score"]
keep = {("CIBERSORT", "T cells CD4 memory activated"): "Activated CD4 memory T (CS)",
        ("CIBERSORT", "Macrophages M1"): "M1 macrophages (CS)", ("CIBERSORT", "T cells CD8"): "CD8 T cells (CS)",
        ("CIBERSORT", "Macrophages M0"): "M0 macrophages (CS)", ("CIBERSORT", "Mast cells activated"): "Activated mast cells (CS)",
        ("CIBERSORT", "T cells follicular helper"): "T follicular helper (CS)",
        ("MCP-counter", "Cytotoxic lymphocytes"): "Cytotoxic lymphocytes (MCP)",
        ("CIBERSORT", "Monocytes"): "Monocytes (CS)", ("CIBERSORT", "Dendritic cells resting"): "Resting dendritic cells (CS)",
        ("CIBERSORT", "T cells CD4 memory resting"): "Resting CD4 memory T (CS)",
        ("CIBERSORT", "Mast cells resting"): "Resting mast cells (CS)",
        ("MCP-counter", "Endothelial cells"): "Endothelial cells (MCP)",
        ("MCP-counter", "Myeloid dendritic cells"): "Myeloid dendritic cells (MCP)",
        ("Checkpoint gene", "LAG3"): "LAG3", ("Checkpoint gene", "PDCD1"): "PDCD1 (PD-1)",
        ("Checkpoint gene", "CD274"): "CD274 (PD-L1)", ("Checkpoint gene", "CTLA4"): "CTLA4",
        ("Checkpoint gene", "TIGIT"): "TIGIT", ("Checkpoint gene", "HAVCR2"): "HAVCR2 (TIM-3)"}
rows = [dict(lab=v, fam=k[0], **im[(im.family == k[0]) & (im.feature == k[1])].iloc[0][["rho", "rho_purity_adj", "FDR_purity_adj"]].to_dict())
        for k, v in keep.items()]
b = pd.DataFrame(rows)
b["grp"] = np.where(b.fam == "Checkpoint gene", 0, 1)
b = b.sort_values(["grp", "rho_purity_adj"]).reset_index(drop=True)
yy = np.arange(len(b)).astype(float); yy[b.grp.values == 1] += 0.8
for i, r in b.iterrows():
    sig = r.FDR_purity_adj < .05
    c = RED if r.rho_purity_adj > 0 else BLUE
    ax.plot([0, r.rho_purity_adj], [yy[i], yy[i]], c=c if sig else GREY, lw=.8)
    ax.scatter(r.rho_purity_adj, yy[i], s=14, c=c if sig else "white", edgecolor=c if sig else GREY, lw=.7, zorder=3)
    ax.scatter(r.rho, yy[i], s=8, marker="|", c="k", lw=.6, zorder=4)
ax.set_yticks(yy); ax.set_yticklabels(b.lab, fontsize=5.6)
for tk, f in zip(ax.get_yticklabels(), b.fam):
    if f == "Checkpoint gene": tk.set_fontstyle("italic")
ax.axvline(0, c="k", lw=.6); ax.set_xlim(-0.55, 0.55)
ax.axhline(yy[b.grp.values == 1].min() - 0.9, c="#bbbbbb", lw=.5)
ax.set_xlabel("Partial Spearman ρ with 7-gene score\n(adjusted for ABSOLUTE tumor purity)")
ax.set_title("Immune context (CS: CIBERSORT; MCP: MCP-counter)", loc="left", fontsize=7)
ax.legend([Line2D([], [], marker="o", color=RED, ls="", ms=3.5), Line2D([], [], marker="o", mfc="white", mec=GREY, ls="", ms=3.5),
           Line2D([], [], marker="|", color="k", ls="", ms=5)], ["FDR < 0.05", "FDR ≥ 0.05", "unadjusted ρ"],
          loc="lower left", fontsize=5.4, handletextpad=0.2)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
panel(ax, "B", x=-0.62)

# ── C pan-cancer OS forest ──
ax = fig.add_subplot(G[0, 1])
sv = pd.read_csv(NEW / "11_extended" / "pancancer" / "pancancer_survival_score_per_SD.csv")
sv = sv[(sv.endpoint == "OS") & sv.HR.notna()].sort_values("HR").reset_index(drop=True)
for i, r in sv.iterrows():
    sig = r.FDR < .05
    c = RED if (sig and r.HR > 1) else BLUE if (sig and r.HR < 1) else GREY
    ax.errorbar(r.HR, i, xerr=[[r.HR - r.lo], [min(r.hi, 9) - r.HR]], fmt="s" if r.cancer != "LUAD" else "D",
                ms=2.8 if r.cancer != "LUAD" else 4, color=c, capsize=1, lw=.7,
                mfc=c if sig else "white")
ax.set_yticks(range(len(sv)))
ax.set_yticklabels([f"{c} ({int(e)})" for c, e in zip(sv.cancer, sv.events)], fontsize=5.4)
for tk, c in zip(ax.get_yticklabels(), sv.cancer):
    if c in ("LUAD", "LUSC"): tk.set_fontweight("bold")
ax.axvline(1, ls="--", c="k", lw=.5)
ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
ax.minorticks_off(); ax.set_xlim(0.2, 9)
ax.set_xlabel("OS HR per 1 SD of 7-gene score (95% CI)")
n_sig = int(((sv.FDR < .05) & (sv.HR > 1)).sum())
ax.set_title(f"33 TCGA cancer types (deaths): adverse in {n_sig}", loc="left", fontsize=7)
ax.legend([Line2D([], [], marker="s", color=RED, ls="", ms=3), Line2D([], [], marker="s", mfc="white", mec=GREY, ls="", ms=3)],
          ["FDR < 0.05", "not significant"], loc="lower right", fontsize=5.6)
panel(ax, "C", x=-0.30)

# ── D DepMap + ChEMBL ──
ax = fig.add_subplot(G[1, :])
eff = pd.read_csv(NEW / "11_extended" / "raw" / "DepMap_CRISPRGeneEffect.csv", index_col=0)
eff.columns = [c.split(" (")[0] for c in eff.columns]
mod = pd.read_csv(NEW / "11_extended" / "raw" / "DepMap_Model.csv").set_index("ModelID")
luad = mod.index[mod.OncotreeSubtype.astype(str).str.contains("Lung Adenocarcinoma", case=False)].intersection(eff.index)
dm = pd.read_csv(NEW / "11_extended" / "drugs" / "DepMap_dependency_LUAD.csv").set_index("gene")
dt = pd.read_csv(NEW / "11_extended" / "drugs" / "Table_hub_gene_drug_mechanisms_ChEMBL.csv").set_index("gene")
rng = np.random.default_rng(1)
data = [eff.loc[luad, g].dropna().values for g in HUB7]
bp = ax.boxplot(data, positions=range(len(HUB7)), widths=0.5, showfliers=False, patch_artist=True,
                medianprops=dict(color="k", lw=.8), boxprops=dict(lw=.6), whiskerprops=dict(lw=.6), capprops=dict(lw=.6))
for p_ in bp["boxes"]: p_.set_facecolor("#f2c4be")
for i, d_ in enumerate(data):
    ax.scatter(rng.normal(i, .07, len(d_)), d_, s=3, c=RED, alpha=.6, lw=0, zorder=3)
ax.axhline(-1, ls="--", c=GREY, lw=.6); ax.axhline(0, c="k", lw=.5)
ax.set_xticks(range(len(HUB7))); ax.set_xticklabels(HUB7, fontstyle="italic")
ax.set_ylabel("CRISPR gene effect (Chronos)"); ax.set_ylim(-4.6, 1.35)
for i, g in enumerate(HUB7):
    ph = dt.loc[g, "highest_phase"]
    drug = ("phase " + str(int(ph)) + " agents" if pd.notna(ph) else "no direct agent")
    ax.text(i, 1.18, f"{dm.loc[g, 'pct_LUAD_dependent']:.0f}% dependent", ha="center", fontsize=5.5)
    ax.text(i, 0.88, ("common essential" if dm.loc[g, "common_essential"] else "not essential"), ha="center", fontsize=5.3,
            color="#555555")
    ax.text(i, 0.58, drug, ha="center", fontsize=5.3, color=RED if pd.notna(ph) else "#555555", fontweight="bold" if pd.notna(ph) else None)
ax.set_title(f"DepMap Public 26Q1: {len(luad)} lung adenocarcinoma cell lines (dashed: effect = −1); "
             "ChEMBL highest clinical phase", loc="left", fontsize=7)
panel(ax, "D", x=-0.06)

save(fig, FIG_MAIN / "Figure5_biology_pancancer_drugs")
print("saved")
