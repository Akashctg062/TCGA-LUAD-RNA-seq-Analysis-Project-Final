"""R12_supp_figures_v2.py — Supplementary Figures S1-S12 (figures only, no table panels; 600 dpi).
Tables formerly embedded in figures are provided in the Supplementary Tables Word file (R12_supp_tables_docx.py).
Reuses plotting code from R12_supp_figures.py (old numbering in brackets).
S1 differential expression [S2] | S2 signed WGCNA [S3] | S3 PPI network [S4A] | S4 PFI/DSS forest [S5A-B]
S5 single-gene KM [S6] | S6 benchmark null distributions [S7A-B] | S7 external validation details [S8A-C]
S8 diagnostic ML [S9A-C] | S9 mutation & stage [S10A-C] | S10 TF/CNV/methylation [S11A-C] | S11 immune [S12]
S12 pan-cancer: tumor vs normal + PFI forest [S13A + new]"""
import sys, types, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
import R12_supp_figures as old
from R12_figstyle import NEW, FIG_SUPP, W2, RED, BLUE, GREY, DARK, panel, save, pfmt
from R12_supp_helpers import HUB7, fp, hrci, km_with_risk

NAMES = {1: "differential_expression", 2: "WGCNA_signed", 3: "PPI_network", 4: "survival_PFI_DSS",
         5: "KM_single_genes", 6: "benchmark_random_sets", 7: "external_validation_details", 8: "diagnostic_ML",
         9: "mutation_stage", 10: "regulation_TF_CNV_methylation", 11: "immune_purity", 12: "pancancer"}
def out(fig, n):
    save(fig, FIG_SUPP / f"FigureS{n}_{NAMES[n]}"); print(f"S{n} saved")

def with_number(fun, n):
    old.out = lambda fig, _n, _name: out(fig, n)
    fun()

def s3():
    import networkx as nx
    from adjustText import adjust_text
    P = NEW / "03_PPI_hub_selection"
    func = pd.read_csv(P / "string_cache" / "string_v12_functional_score150.tsv", sep="\t")
    ids = pd.read_csv(P / "string_cache" / "string_v12_id_mapping.tsv", sep="\t")
    back = dict(zip(ids.preferredName, ids.queryItem)); PRIOR = .041
    p = 1.0
    for c in ["nscore", "fscore", "pscore", "escore", "dscore", "tscore"]:
        p = p * (1 - ((func[c] - PRIOR) / (1 - PRIOR)).clip(lower=0))
    func["E2"] = (1 - p) * (1 - PRIOR) + PRIOR
    d = func[func.E2 >= .7]
    Gx = nx.Graph(); Gx.add_edges_from((back.get(a, a), back.get(b, b)) for a, b in zip(d.preferredName_A, d.preferredName_B))
    top = list(pd.read_csv(P / "centrality_E2_0.7.csv", index_col=0).index[:15])
    fig, ax = plt.subplots(figsize=(W2, 5.6))
    pos = nx.spring_layout(Gx, seed=7, k=0.55)
    nx.draw_networkx_edges(Gx, pos, ax=ax, alpha=.15, width=.4)
    nx.draw_networkx_nodes(Gx, pos, ax=ax, node_color=[RED if n in HUB7 else ("#f0b27a" if n in top else "#aab7b8") for n in Gx],
                           node_size=[10 + 3 * Gx.degree(n) for n in Gx], linewidths=.3, edgecolors="white")
    tx = [ax.text(pos[n][0], pos[n][1], n, fontsize=5.6, fontweight="bold" if n in HUB7 else None, ha="center", va="center")
          for n in Gx if n in HUB7 or n in top]
    adjust_text(tx, ax=ax, expand=(1.4, 1.8), force_text=(0.6, 0.9), arrowprops=dict(arrowstyle="-", lw=0.35, color="#555555"))
    ax.axis("off")
    ax.set_title(f"STRING v12 network of the 86 ME2 candidates, combined score without co-expression ≥ 0.7 "
                 f"({Gx.number_of_nodes()} connected nodes, {Gx.number_of_edges()} edges)", fontsize=7, loc="left")
    ax.legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for c in [RED, "#f0b27a", "#aab7b8"]],
              ["7 hub genes", "other consensus top-15", "other candidates"], loc="lower left")
    out(fig, 3)

def s4():
    res = pd.read_csv(NEW / "05_survival" / "cox_uni_multi_per_SD_OS_PFI_DSS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.3), gridspec_kw=dict(wspace=0.95))
    for j, (ax, ep) in enumerate(zip(axes, ["PFI", "DSS"])):
        r = res[res.endpoint == ep].set_index("gene").loc[HUB7]
        yy = np.arange(len(HUB7))[::-1]
        ax.errorbar(r.uni_HR, yy + .15, xerr=[r.uni_HR - r.uni_lo, r.uni_hi - r.uni_HR], fmt="o", ms=2.5, color=GREY, capsize=1.3, lw=.8)
        ax.errorbar(r.multi_HR, yy - .15, xerr=[r.multi_HR - r.multi_lo, r.multi_hi - r.multi_HR], fmt="s", ms=2.5, color=RED, capsize=1.3, lw=.8)
        for y, (_, row) in zip(yy, r.iterrows()):
            ax.text(1.03, y - .15, f"{hrci(row.multi_HR, row.multi_lo, row.multi_hi)}; FDR {fp(row.multi_FDR)}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=5.3)
        ax.axvline(1, ls="--", c="k", lw=.5); ax.set_yticks(yy); ax.set_yticklabels(HUB7, fontstyle="italic")
        ax.set_xlabel("HR per 1 SD (95% CI)")
        ax.set_title(f"{'Progression-free interval' if ep == 'PFI' else 'Disease-specific survival'}\n"
                     f"(multivariable n = {int(r.multi_n.iloc[0])}, {int(r.multi_events.iloc[0])} events)", fontsize=6.6, loc="left")
        panel(ax, "AB"[j], x=-0.3)
    fig.legend([Line2D([], [], marker="o", color=GREY, ls="", ms=3), Line2D([], [], marker="s", color=RED, ls="", ms=3)],
               ["Univariate", "Multivariable (age, sex, stage, smoking)"], loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.06))
    out(fig, 4)

def s6():
    B = NEW / "06_benchmark"
    null = pd.read_csv(B / "null_distributions_1000x2.csv"); obs = pd.read_csv(B / "signature_performance.csv", index_col=0)
    pct = pd.read_csv(B / "HUB7_vs_null_empirical.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(wspace=0.35))
    for j, (ax, (m, lab)) in enumerate(zip(axes, [("C_multi", "OS C-index, clinical model + score"), ("AUC", "Tumor vs normal AUC")])):
        for (nm, g), c in zip(null.groupby("null"), ["#bdc3c7", "#f5b041"]):
            ax.hist(g[m], bins=35, alpha=.75, color=c, label="expression-matched up-DEGs" if nm.startswith("N1") else "G2M/E2F genes")
        ax.axvline(obs.loc["HUB7", m], color=RED, lw=1.1, label="7-gene score")
        if m == "C_multi": ax.axvline(0.6723, color="k", ls=":", lw=1, label="clinical model alone")
        pp = pct[pct.metric == m]
        ax.set_title(f"{lab}\n7-gene score percentile: {pp.HUB7_percentile.iloc[0]:.0f} / {pp.HUB7_percentile.iloc[1]:.0f}", fontsize=6.5)
        ax.set(xlabel=lab, ylabel="Random 7-gene sets"); ax.set_ylim(0, ax.get_ylim()[1] * 1.65)
        ax.legend(fontsize=5.2, loc="upper left", ncol=2, columnspacing=0.8)
        panel(ax, "AB"[j], x=-0.2)
    out(fig, 6)

def s7():
    EV = NEW / "07_external_validation"; DAT = EV / "data"
    ta = pd.read_csv(EV / "time_dependent_AUC_HUB7_mean.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.0), gridspec_kw=dict(wspace=0.45, bottom=0.3))
    for j, (gse, ep, lab) in enumerate([("GSE31210", "RFS", "Relapse-free survival"), ("GSE68465", "PFS", "Progression-free survival")]):
        ax = axes[j]
        E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
        C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
        Z = (E.loc[C.index, HUB7] - E.loc[C.index, HUB7].mean()) / E.loc[C.index, HUB7].std()
        k = pd.DataFrame({"T": C[f"{ep}_days"] / 365.25, "E": C[f"{ep}_event"], "s": Z.mean(1)}).dropna()
        hi = (k.s > k.s.median()).values
        p = km_with_risk(ax, k["T"], k["E"], [hi, ~hi], [RED, BLUE], ["High", "Low"], risk_y0=-0.36, dy=0.1, show_row_labels=j == 0)
        ax.set_title(f"{gse}: {lab}\n(n = {len(k)}; {pfmt(p)})", fontsize=6.3); ax.set_xlabel("Years")
        ax.get_legend().remove()
        if j == 0: ax.legend(loc="lower left", fontsize=5.4)
        panel(ax, "AB"[j], x=-0.3)
    ax = axes[2]
    t = ta.copy(); t["lab"] = t.cohort.str.replace("TCGA-LUAD", "TCGA") + " " + t.endpoint
    piv = t.pivot(index="lab", columns="year", values="AUC"); x = np.arange(len(piv))
    ax.bar(x - .18, piv[3], .36, color="#85c1e9", label="3 years"); ax.bar(x + .18, piv[5], .36, color=BLUE, label="5 years")
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=55, ha="right", fontsize=5.4)
    ax.axhline(.5, c="k", lw=.5); ax.set_ylim(.4, .8); ax.set(ylabel="Time-dependent AUC", title="7-gene score (Uno, IPCW)")
    ax.legend(loc="upper left", fontsize=5.4); panel(ax, "C", x=-0.32)
    out(fig, 7)

def s8():
    M = NEW / "09_ML"
    summ = pd.read_csv(M / "A_diagnostic_nested_cv_summary.csv", index_col=[0, 1])
    ext = pd.read_csv(M / "A_external_GSE31210_diagnostic.csv").set_index("model")
    order = ["Logistic regression", "SVM (RBF)", "Random forest", "XGBoost", "MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.4), gridspec_kw=dict(wspace=0.35))
    for j, (ax, (met, lab, lo)) in enumerate(zip(axes, [("AUC", "AUC", .75), ("balanced_accuracy", "Balanced accuracy", .5), ("MCC", "MCC", 0)])):
        for i, fs in enumerate(["HUB7", "FOLD7"]):
            s = summ.loc[fs].loc[order]; x = np.arange(len(order)) + (i - .5) * .36
            ax.bar(x, s[f"{met}_mean"], .34, yerr=[s[f"{met}_mean"] - s[f"{met}_lo"], s[f"{met}_hi"] - s[f"{met}_mean"]],
                   color=[RED, "#aab7b8"][i], error_kw=dict(lw=.6, capsize=1.5),
                   label=["7 hub genes (nested CV)", "7 DEGs re-selected per fold"][i])
        ax.scatter(np.arange(len(order)), ext.loc[order, met], marker="D", s=12, color=BLUE, zorder=3, label="External GSE31210 (locked)")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=5.6)
        ax.set_ylim(lo, 1.0); ax.set_title(lab); panel(ax, "ABC"[j], x=-0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    out(fig, 8)

def strip_last_axes_and_save(fun, n, keep):
    """Run an old function but intercept the figure, remove table axes beyond the first `keep` panels."""
    captured = {}
    old.out = lambda fig, _n, _name: captured.setdefault("fig", fig)
    fun()
    fig = captured["fig"]
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    for a in fig.axes[keep:]:
        a.remove()
    return fig

def s9():
    fig = strip_last_axes_and_save(old.s10, 9, keep=3)
    fig.set_size_inches(W2, 5.2)
    gsx = gs.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.62, wspace=0.42)
    a, b, c = fig.axes[:3]
    a.set_position(gsx[0, 0].get_position(fig)); b.set_position(gsx[0, 1].get_position(fig)); c.set_position(gsx[1, :].get_position(fig))
    out(fig, 9)

def s10():
    fig = strip_last_axes_and_save(old.s11, 10, keep=3)
    fig.set_size_inches(W2, 3.4)
    gsx = gs.GridSpec(1, 3, figure=fig, wspace=0.55, bottom=0.2, top=0.86)
    for a, g_ in zip(fig.axes[:3], [gsx[0], gsx[1], gsx[2]]):
        a.set_position(g_.get_position(fig))
    out(fig, 10)

def s12():
    P = NEW / "11_extended" / "pancancer"
    tn = pd.read_csv(P / "pancancer_tumour_vs_TCGA_plus_GTEx_normal.csv")
    sv = pd.read_csv(P / "pancancer_survival_score_per_SD.csv")
    fig = plt.figure(figsize=(W2, 7.6))
    G = gs.GridSpec(2, 1, height_ratios=[1, 1.15], hspace=0.62)
    ax = fig.add_subplot(G[0])
    t = tn.dropna(subset=["diff"]).sort_values("diff"); x = np.arange(len(t))
    ax.vlines(x, t.median_normal, t.median_tumour, color="#aab7b8", lw=1)
    ax.scatter(x, t.median_normal, s=12, color=BLUE, label="Normal (TCGA adjacent + GTEx)", zorder=3)
    ax.scatter(x, t.median_tumour, s=12, color=RED, label="Primary tumor", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{c} ({int(a)}/{int(b + g)})" for c, a, b, g in
                                          zip(t.cancer, t.n_tumour, t.n_TCGA_normal, t.n_GTEx_normal)], rotation=90, fontsize=5.2)
    for tk, c in zip(ax.get_xticklabels(), t.cancer):
        if c == "LUAD": tk.set_fontweight("bold")
    ax.set_ylabel("Median 7-gene score\n(UCSC Toil TCGA+GTEx, z)")
    ax.set_title(f"Tumor vs same-site normal tissue in {len(t)} cancer types (tumors/normals); all FDR < 0.01 (Mann–Whitney)",
                 loc="left", fontsize=6.8)
    ax.legend(loc="upper left", fontsize=5.6); ax.set_ylim(-1.9, 1.9); panel(ax, "A", x=-0.07)
    ax = fig.add_subplot(G[1])
    s = sv[(sv.endpoint == "PFI") & sv.HR.notna()].sort_values("HR", ascending=False).reset_index(drop=True)
    x = np.arange(len(s))
    for i, r in s.iterrows():
        sig = r.FDR < .05; c = RED if (sig and r.HR > 1) else BLUE if (sig and r.HR < 1) else GREY
        ax.errorbar(i, r.HR, yerr=[[r.HR - r.lo], [min(r.hi, 9) - r.HR]], fmt="D" if r.cancer == "LUAD" else "s",
                    ms=4 if r.cancer == "LUAD" else 3, color=c, mfc=c if sig else "white", capsize=1, lw=.7)
    ax.axhline(1, ls="--", c="k", lw=.5); ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_xticks(x); ax.set_xticklabels([f"{c} ({int(e)})" for c, e in zip(s.cancer, s.events)], rotation=90, fontsize=5.2)
    for tk, c in zip(ax.get_xticklabels(), s.cancer):
        if c in ("LUAD", "LUSC"): tk.set_fontweight("bold")
    ax.set_ylabel("PFI HR per 1 SD (95% CI)")
    ax.set_title(f"Progression-free interval in 32 TCGA cancer types (events): adverse in {int(((s.FDR < .05) & (s.HR > 1)).sum())} (FDR < 0.05)",
                 loc="left", fontsize=6.8)
    ax.legend([Line2D([], [], marker="s", color=RED, ls="", ms=3), Line2D([], [], marker="s", mfc="white", mec=GREY, ls="", ms=3)],
              ["FDR < 0.05", "not significant"], loc="upper right", fontsize=5.6)
    panel(ax, "B", x=-0.07)
    out(fig, 12)

if __name__ == "__main__":
    sel = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
    F = {1: lambda: with_number(old.s2, 1), 2: lambda: with_number(old.s3, 2), 3: s3, 4: s4,
         5: lambda: with_number(old.s6, 5), 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: lambda: with_number(old.s12, 11), 12: s12}
    for n in sel: F[n]()
