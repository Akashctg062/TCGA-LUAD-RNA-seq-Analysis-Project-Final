"""
R13_supp_figures.py — Supplementary Figures S13-S17 for the limitation analyses (600 dpi PNG + TIFF).
S13 normal-reference (GTEx) analysis   S14 single-cell source   S15 CPTAC protein
S16 published signatures              S17 follow-up, adjuvant treatment, ancestry
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from R12_figstyle import NEW, FIG_SUPP, W2, RED, BLUE, GREY, DARK, panel, save, pfmt

L = NEW / "13_limitations"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
ORANGE = "#e67e22"

def fdr_txt(q):
    return "FDR < 0.001" if q < 0.001 else f"FDR = {q:.2f}"

# ───────────────────────── S13 ─────────────────────────
def s13():
    D = pd.read_csv(L / "field_effect" / "toil_LUAD_lung_normal_hub_expression.csv", index_col=0)
    R = pd.read_csv(L / "field_effect" / "R13a_hub_tumour_NAT_GTEx.csv").set_index("gene")
    J = pd.read_csv(L / "field_effect" / "R13b_concordance_gene_level.csv", index_col=0)
    cand = set(pd.read_csv(NEW / "02_WGCNA_signed" / "A_key_module_ME2_hub_candidates_MM0.8_GS0.5_DEGup.csv").gene_id)
    S = __import__("json").load(open(L / "field_effect" / "R13b_summary.json"))
    fig = plt.figure(figsize=(W2, 2.75))
    G = gs.GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1.15, 1.1], wspace=0.45)
    # A
    ax = fig.add_subplot(G[0])
    order = [("Tumour", "Tumour", RED), ("NAT", "NAT", GREY), ("GTEx", "GTEx", BLUE)]
    data = [D.loc[D.group == k, "score"].values for k, _, _ in order]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=0.8))
    for b, (_, _, c) in zip(bp["boxes"], order): b.set(facecolor=c, alpha=0.55, lw=0.6)
    rng = np.random.default_rng(1)
    for i, x in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(x)), x, s=1.2, color=order[i][2], alpha=0.35, lw=0)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"{l}\n({len(x)})" for (_, l, _), x in zip(order, data)], fontsize=6)
    ax.set_ylabel("7-gene score (mean z)")
    ymax = max(x.max() for x in data)
    for (a, b, txt, h) in [(2, 3, f"δ = {R.loc['score', 'cliff_NAT_vs_GTEx']:.2f}", 0.3), (1, 3, f"δ = {R.loc['score', 'cliff_T_vs_GTEx']:.2f}", 1.0)]:
        y = ymax + h
        ax.plot([a, a, b, b], [y - 0.12, y, y, y - 0.12], c="k", lw=0.5); ax.text((a + b) / 2, y + 0.06, txt, ha="center", fontsize=5.6)
    nlow = int(sum((x < -3.3).sum() for x in data))
    ax.set_ylim(-3.3, ymax + 1.7)
    if nlow: ax.text(0.98, 0.01, f"{nlow} value < −3.3 not shown", transform=ax.transAxes, ha="right", fontsize=5.0, color="#555555")
    panel(ax, "A", x=-0.42)
    # B
    ax = fig.add_subplot(G[1])
    genes = HUB7 + ["MKI67"]
    y = np.arange(len(genes))[::-1]
    ax.barh(y + 0.18, R.loc[genes, "diff_T_vs_GTEx"], height=0.34, color=RED, label="Tumour − GTEx lung")
    ax.barh(y - 0.18, R.loc[genes, "diff_NAT_vs_GTEx"], height=0.34, color=GREY, label="Adjacent normal − GTEx lung")
    ax.axvline(0, c="k", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xlabel("Difference in median log$_2$(TPM)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=5.8)
    panel(ax, "B", x=-0.3)
    # C
    ax = fig.add_subplot(G[2])
    ax.scatter(J.log2FC_MLE, J.log2FoldChange_TvsGTEx, s=0.6, color="#bbbbbb", lw=0, rasterized=True)
    cj = J.loc[J.index.intersection(list(cand))]
    ax.scatter(cj.log2FC_MLE, cj.log2FoldChange_TvsGTEx, s=5, color=ORANGE, lw=0, label=f"86 candidate genes")
    hj = J[J.gene_name.isin(HUB7)]
    ax.scatter(hj.log2FC_MLE, hj.log2FoldChange_TvsGTEx, s=12, color=RED, edgecolor="k", lw=0.3, label="7 hub genes", zorder=3)
    lim = (-8, 10); ax.plot(lim, lim, ls="--", c="k", lw=0.5); ax.axhline(0, c="#999999", lw=0.4); ax.axvline(0, c="#999999", lw=0.4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("log$_2$FC tumour vs adjacent normal\n(original TCGA analysis)")
    ax.set_ylabel("log$_2$FC tumour vs GTEx lung (Toil)")
    ax.text(0.03, 0.97, f"Spearman ρ = {S['spearman_log2FC_TvsNAT_vs_TvsGTEx']:.2f}\n"
            f"{100 * S['orig_Up_same_direction_vs_GTEx']:.0f}% of up- and {100 * S['orig_Down_same_direction_vs_GTEx']:.0f}% of\n"
            f"down-DEGs same direction", transform=ax.transAxes, va="top", fontsize=5.6)
    ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.2, markerscale=1.2)
    panel(ax, "C", x=-0.3)
    save(fig, FIG_SUPP / "FigureS13_normal_reference_GTEx")

# ───────────────────────── S14 ─────────────────────────
def s14():
    G_ = pd.read_csv(L / "single_cell" / "R13f_gene_by_celltype_tumour.csv")
    C_ = pd.read_csv(L / "single_cell" / "R13f_cycling_by_group.csv").set_index("group")
    groups = ["Malignant epithelial", "T/NK cells", "Myeloid cells", "B cells", "Mast cells", "Fibroblasts",
              "Endothelial cells", "Other epithelial", "Other"]
    lab = {g: g for g in groups}; lab["Other"] = "Unassigned"
    cols = [RED, "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#795548", "#16a085", "#e8a39b", "#bdc3c7"]
    fig = plt.figure(figsize=(W2, 2.9))
    Gs = gs.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.1, 0.75], wspace=1.05)
    # A stacked share
    ax = fig.add_subplot(Gs[0])
    sh = G_[G_.gene.isin(HUB7)].pivot(index="gene", columns="group", values="share_of_tumour_transcripts").loc[HUB7[::-1], groups]
    left = np.zeros(len(sh))
    for g, c in zip(groups, cols):
        ax.barh(range(len(sh)), sh[g], left=left, color=c, height=0.7, label=lab[g]); left += sh[g].values
    ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh.index, fontstyle="italic")
    ax.set_xlim(0, 100); ax.set_xlabel("Share of hub-gene transcripts\nin tumour samples (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=2, fontsize=5.4, handlelength=0.9, columnspacing=0.6)
    panel(ax, "A", x=-0.3)
    # B dot plot
    ax = fig.add_subplot(Gs[1])
    genes = HUB7 + ["MKI67"]
    sub = G_[G_.group.isin(groups)]
    vmax = float(np.ceil(sub[sub.gene.isin(genes)].mean_log2TPM.max() * 20) / 20)
    for i, g in enumerate(groups):
        for j, gene in enumerate(genes):
            r = sub[(sub.group == g) & (sub.gene == gene)].iloc[0]
            ax.scatter(j, len(groups) - 1 - i, s=3 + 6 * r.pct_expressing, c=[r.mean_log2TPM], cmap="Reds",
                       vmin=0, vmax=vmax, edgecolor="#555555", lw=0.2)
    sc = ax.scatter([], [], c=[], cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=60, ha="right", fontstyle="italic", fontsize=6)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([lab[g] for g in groups[::-1]], fontsize=6)
    ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(-0.6, len(groups) - 0.4)
    cax = ax.inset_axes([0.05, -0.47, 0.9, 0.04])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("Mean log$_2$(TPM+1)", fontsize=5.8); cb.ax.tick_params(labelsize=5.4)
    for k, p_ in enumerate([2, 10, 20]):
        ax.scatter([], [], s=3 + 6 * p_, c="white", edgecolor="#555555", lw=0.3, label=f"{p_}%")
    ax.legend(title="% cells expressing", title_fontsize=5.6, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=3)
    panel(ax, "B", x=-0.75)
    # C cycling
    ax = fig.add_subplot(Gs[2])
    order = ["Malignant epithelial", "Myeloid cells", "T/NK cells", "B cells", "Fibroblasts", "Endothelial cells"]
    nrm = pd.read_csv(L / "single_cell" / "R13f_summary.csv", index_col=0)["value"]
    vals = [C_.loc[g, "pct_cycling"] for g in order] + [float(nrm["pct_cycling_normal"])]
    labs = [lab[g] for g in order] + ["Normal epithelial\n(normal lung)"]
    cc = [RED, "#27ae60", "#2980b9", "#8e44ad", "#795548", "#16a085", GREY]
    ax.barh(range(len(vals))[::-1], vals, color=cc, height=0.65)
    for i, v in enumerate(vals): ax.text(v + 0.3, len(vals) - 1 - i, f"{v:.1f}", va="center", fontsize=5.6)
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labs, fontsize=6)
    ax.set_xlabel("Cycling cells (%)"); ax.set_xlim(0, max(vals) * 1.3)
    panel(ax, "C", x=-1.15)
    save(fig, FIG_SUPP / "FigureS14_single_cell_source")

# ───────────────────────── S15 ─────────────────────────
def s15():
    R = pd.read_csv(L / "protein" / "R13h_protein_tumour_vs_NAT.csv").set_index("gene")
    P = pd.read_csv(L / "protein" / "R13h_paired_scores.csv", index_col=0)
    M = pd.read_csv(L / "protein" / "R13h_mRNA_vs_protein_score.csv", index_col=0)
    raw = NEW / "13_limitations" / "raw" / "CPTAC"
    PT = pd.read_csv(raw / "HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct", sep="\t", index_col=0)
    PN = pd.read_csv(raw / "HS_CPTAC_LUAD_proteome_ratio_NArm_NORMAL.cct", sep="\t", index_col=0)
    genes = [g for g in HUB7 if R.loc[g, "protein_quantified"]] + ["MKI67"]
    fig = plt.figure(figsize=(W2, 2.6))
    Gs = gs.GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.8, 0.9], wspace=0.5)
    ax = fig.add_subplot(Gs[0])
    pairs = PT.columns.intersection(PN.columns)
    for j, g in enumerate(genes):
        d = (PT.loc[g, pairs] - PN.loc[g, pairs]).dropna()
        ax.boxplot([d.values], positions=[j], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=RED if g != "MKI67" else GREY, alpha=0.5, lw=0.6), medianprops=dict(color="k", lw=0.8))
        ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(d)), d, s=1.5, color="k", alpha=0.4, lw=0)
        q = R.loc[g, "FDR_paired"]
        ax.text(j, 1.0, f"n = {int(R.loc[g, 'n_pairs'])}\n{R.loc[g, 'pct_tumour_higher']:.0f}%\n{'***' if q < 0.001 else 'n.s.' if q >= 0.05 else '*'}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.3)
    ax.axhline(0, c="k", lw=0.5, ls="--")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, fontstyle="italic", rotation=40, ha="right")
    ax.set_ylabel("Protein log$_2$ ratio, tumour − paired NAT")
    panel(ax, "A", x=-0.2, y=1.2)
    ax = fig.add_subplot(Gs[1])
    gg = genes
    ax.barh(range(len(gg))[::-1], R.loc[gg, "rho_mRNA_protein"], color=[RED if g != "MKI67" else GREY for g in gg], height=0.6)
    for i, g in enumerate(gg): ax.text(R.loc[g, "rho_mRNA_protein"] + 0.02, len(gg) - 1 - i, f"{R.loc[g, 'rho_mRNA_protein']:.2f}", va="center", fontsize=5.6)
    ax.set_yticks(range(len(gg))[::-1]); ax.set_yticklabels(gg, fontstyle="italic"); ax.set_xlim(0, 1.1)
    ax.set_xlabel("mRNA–protein Spearman ρ\n(tumours)")
    panel(ax, "B", x=-0.42, y=1.2)
    ax = fig.add_subplot(Gs[2])
    ax.scatter(M.mRNA_score, M.protein_score, s=5, color=RED, alpha=0.7, lw=0)
    from scipy.stats import spearmanr
    rho = spearmanr(M.mRNA_score, M.protein_score).correlation
    ax.text(0.04, 0.96, f"ρ = {rho:.2f}\nn = {len(M)} tumours", transform=ax.transAxes, va="top", fontsize=5.8)
    ax.set_xlabel("mRNA 7-gene score"); ax.set_ylabel("Protein score (5 hub proteins)")
    panel(ax, "C", x=-0.32, y=1.2)
    save(fig, FIG_SUPP / "FigureS15_protein_CPTAC")

# ───────────────────────── S16 ─────────────────────────
NAMES = {"HUB7_mean": "7-gene score (this study)", "PAM50_prolif": "PAM50 proliferation (11)", "CCP31": "CCP (31; Wistuba 2013)",
         "KRATZ11": "Kratz (11; 2012)", "CHEN5": "Chen (5; 2007)", "SHUKLA4": "Shukla (4; 2017)",
         "ALDHERASI7": "Al-Dherasi (7; 2021)", "LI8": "Li (8; 2020)"}
def s16():
    M = pd.read_csv(L / "published_signatures" / "R13c_meta_analysis.csv").set_index("signature")
    R = pd.read_csv(L / "published_signatures" / "R13c_signature_performance.csv")
    H = pd.read_csv(L / "published_signatures" / "R13c_head_to_head_vs_HUB7.csv")
    fig = plt.figure(figsize=(W2, 5.4))
    G0 = gs.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.62], hspace=0.95)
    Gs = G0[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.75)
    ax = fig.add_subplot(Gs[0])
    sigs = ["HUB7_mean", "PAM50_prolif", "CCP31", "KRATZ11", "SHUKLA4", "CHEN5", "ALDHERASI7", "LI8"]
    y = 0; yt, yl = [], []
    for s in sigs:
        for var, mk, col, off in [("", "s", RED if s == "HUB7_mean" else DARK, 0.17), ("_refit", "o", GREY, -0.17)]:
            k = s + var
            if k not in M.index: continue
            r = M.loc[k]
            ax.errorbar(r.HR, -y - off, xerr=[[r.HR - r.lo], [r.hi - r.HR]], fmt=mk, ms=3, color=col, capsize=1.2, lw=0.8)
        yt.append(-y); yl.append(NAMES[s]); y += 1
    ax.axvline(1, c="k", ls="--", lw=0.5); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6)
    ax.set_xscale("log"); ax.set_xticks([0.8, 1, 1.25, 1.5]); ax.set_xticklabels(["0.8", "1", "1.25", "1.5"]); ax.minorticks_off()
    ax.set_xlabel("Pooled HR per SD (95% CI), OS\n(random effects; cohorts not used by the authors)")
    ax.legend([Line2D([], [], marker="s", color=DARK, ls="", ms=3), Line2D([], [], marker="o", color=GREY, ls="", ms=3)],
              ["As published (locked)", "Re-weighted in TCGA, locked (GEO only)"], loc="upper center", bbox_to_anchor=(0.3, -0.3), fontsize=5.6)
    panel(ax, "A", x=-0.75)
    ax = fig.add_subplot(Gs[1])
    coh = ["TCGA-LUAD", "GSE31210", "GSE68465", "GSE72094"]
    Cm = R[R.signature.isin(sigs)].pivot(index="signature", columns="cohort", values="C_uni").loc[sigs, coh]
    tr = R[R.signature.isin(sigs)].pivot(index="signature", columns="cohort", values="trained_in_this_cohort").loc[sigs, coh]
    im = ax.imshow(Cm.values, cmap="viridis", vmin=0.45, vmax=0.72, aspect="auto")
    for i in range(Cm.shape[0]):
        for j in range(Cm.shape[1]):
            v = Cm.values[i, j]
            if np.isnan(v): ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.2, color="#666666"); continue
            ax.text(j, i, f"{v:.2f}{'†' if tr.values[i, j] is True or tr.values[i, j] == 1 else ''}", ha="center", va="center",
                    fontsize=5.4, color="white" if v < 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(["TCGA", "GSE\n31210", "GSE\n68465", "GSE\n72094"], fontsize=5.8)
    ax.set_yticks(range(len(sigs))); ax.set_yticklabels([NAMES[s].split(" (")[0] for s in sigs], fontsize=6)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cax = ax.inset_axes([0.1, -0.3, 0.8, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Harrell C (as published)", fontsize=5.8); cb.ax.tick_params(labelsize=5.4)
    ax.text(0, -0.52, "† cohort used for training/validation by the authors", transform=ax.transAxes, fontsize=5.2)
    panel(ax, "B", x=-0.5)
    ax = fig.add_subplot(G0[1])
    sg = ["PAM50_prolif", "CCP31", "KRATZ11", "SHUKLA4", "CHEN5", "ALDHERASI7", "LI8"]
    for j, s in enumerate(sg):
        h = H[H.signature == s]
        ax.scatter(np.full(len(h), j), h.r_spearman, s=10, color=DARK, alpha=0.8, lw=0)
    ax.axhline(0, c="k", lw=0.4)
    ax.set_xticks(range(len(sg))); ax.set_xticklabels([NAMES[s].replace(" (", "\n(") for s in sg], fontsize=5.8)
    ax.set_ylabel("Spearman ρ with\nthe 7-gene score"); ax.set_ylim(-0.45, 1.05); ax.set_xlim(-0.5, len(sg) - 0.5)
    ax.text(1.0, 1.02, "each point = one cohort", transform=ax.transAxes, ha="right", fontsize=5.6)
    panel(ax, "C", x=-0.09)
    save(fig, FIG_SUPP / "FigureS16_published_signatures")

# ───────────────────────── S17 ─────────────────────────
def s17():
    R = pd.read_csv(L / "extra_cohorts" / "R13g_cohort_results.csv").set_index("cohort")
    M = pd.read_csv(L / "extra_cohorts" / "R13g_meta_analysis.csv").set_index("analysis")
    order = list(R.index)
    lab = {c: c.replace("TCGA-LUAD", "TCGA") for c in order}
    fig = plt.figure(figsize=(W2, 3.4))
    Gs = gs.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.95)
    def forest(ax, rows, title):
        yy = np.arange(len(rows))[::-1]
        for y, (name, hr, lo, hi, col, mk) in zip(yy, rows):
            ax.errorbar(hr, y, xerr=[[hr - lo], [hi - hr]], fmt=mk, ms=3.2 if mk == "D" else 2.6, color=col, capsize=1.2, lw=0.8)
        ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
        ax.axvline(1, c="k", ls="--", lw=0.5); ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.set_title(title, fontsize=6.6, loc="left")
        ax.set_xlabel("HR per SD (95% CI)")
    ci = lambda hr, se: (hr, np.exp(np.log(hr) - 1.96 * se), np.exp(np.log(hr) + 1.96 * se))
    # A follow-up
    rows = []
    for c in order:
        h, l, u = ci(R.loc[c, "HR_0_5y"], R.loc[c, "se_0_5y"]); rows.append((f"{lab[c]} (0–5 y)", h, l, u, BLUE, "s"))
    m = M.loc["deaths within 5 years"]; rows.append(("Pooled, deaths ≤ 5 y", m.HR, m.lo, m.hi, DARK, "D"))
    for c in R.dropna(subset=["HR_after_5y"]).index:
        rows.append((f"{lab[c]} (> 5 y; {int(R.loc[c, 'events_landmark5'])} ev.)", R.loc[c, "HR_after_5y"], R.loc[c, "lo_after_5y"], R.loc[c, "hi_after_5y"], ORANGE, "s"))
    m = M.loc["5-year landmark (after 5 years)"]; rows.append(("Pooled, 5-year survivors", m.HR, m.lo, m.hi, DARK, "D"))
    ax = fig.add_subplot(Gs[0]); forest(ax, rows, "Follow-up period"); panel(ax, "A", x=-1.1)
    # B treatment
    rows = []
    for c in R.dropna(subset=["HR_no_adjuvant"]).index:
        txt = f"{lab[c]} (n = {int(R.loc[c, 'n_no_adjuvant'])}){'*' if c == 'GSE31210' else ''}"
        rows.append((txt, R.loc[c, "HR_no_adjuvant"], R.loc[c, "lo_no_adjuvant"], R.loc[c, "hi_no_adjuvant"], BLUE, "s"))
    m = M.loc["no adjuvant therapy"]; rows.append(("Pooled", m.HR, m.lo, m.hi, DARK, "D"))
    ax = fig.add_subplot(Gs[1]); forest(ax, rows, "No adjuvant therapy")
    ip = R.dropna(subset=["interaction_p"])
    ax.text(0.0, -0.17, "* complete resection, no adjuvant therapy\nScore × adjuvant chemotherapy:\n" + "\n".join(f"{lab[c]} p = {ip.loc[c, 'interaction_p']:.2f}" for c in ip.index),
            transform=ax.transAxes, fontsize=5.4, va="top")
    panel(ax, "B", x=-0.95)
    # C region + leave-one-out
    rows = []
    for k, t in [("all 9 cohorts, multivariable", "All 9 cohorts"), ("5 new cohorts, multivariable", "5 additional cohorts"),
                 ("East Asian, multivariable", "East Asian (2)"), ("Western, multivariable", "Western (7)")]:
        m = M.loc[k]; rows.append((t, m.HR, m.lo, m.hi, DARK, "D"))
    for c in order:
        m = M.loc[f"leave-one-out: without {c}"]; rows.append((f"without {lab[c]}", m.HR, m.lo, m.hi, GREY, "o"))
    ax = fig.add_subplot(Gs[2]); forest(ax, rows, "Subgroups and leave-one-out"); ax.set_xlim(0.9, 2.0)
    ax.set_xticks([1, 1.25, 1.5, 2]); ax.set_xticklabels(["1", "1.25", "1.5", "2"])
    panel(ax, "C", x=-0.9)
    save(fig, FIG_SUPP / "FigureS17_followup_treatment_ancestry")

if __name__ == "__main__":
    for f in (s13, s14, s15, s16, s17):
        f(); print(f.__name__, "saved", flush=True)
