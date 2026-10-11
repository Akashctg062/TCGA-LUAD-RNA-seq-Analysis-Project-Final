"""R12_supp_figures.py — Supplementary Figures S1-S14 (600 dpi; images and tables combined).
Usage: python R12_supp_figures.py            (all)   |   python R12_supp_figures.py 3 5   (selected)"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from R12_figstyle import NEW, FIG_SUPP, W2, RED, BLUE, GREY, DARK, panel, save, pfmt
from R12_supp_helpers import ROOT, HUB7, table, f2, fp, hrci, tcga_expr, km_with_risk

def out(fig, n, name):
    save(fig, FIG_SUPP / f"FigureS{n}_{name}")
    print(f"S{n} saved")

# ═════════ S1 cohorts and clinical data ═════════
def s1():
    clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv")
    DAT = NEW / "07_external_validation" / "data"
    rows = []
    def add(name, C, plat, ep2):
        st = C.stage.value_counts()
        rows.append({"Cohort": name, "Platform": plat, "Tumors": len(C),
                     "Age": f2(C.age.median(), 0), "Male %": f2(100 * C.male.mean(), 0),
                     "Smoker %": f2(100 * C.ever_smoker.mean(), 0),
                     "Stage I/II/III/IV": "/".join(str(int(st.get(s, 0))) for s in ["I", "II", "III", "IV"]),
                     "OS n (deaths)": f"{int(C.OS_days.notna().sum())} ({int(C.OS_event.sum())})",
                     "Other endpoint": ep2})
    c = clin.assign(male=(clin.sex == "Male").astype(float))
    add("TCGA-LUAD", c, "RNA-seq", f"PFI {int(c.PFI_days.notna().sum())} ({int(c.PFI_event.sum())})")
    for gse, plat in [("GSE31210", "HG-U133 Plus 2.0"), ("GSE68465", "HG-U133A"), ("GSE72094", "Merck custom array")]:
        C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
        ep2 = (f"RFS {int(C.RFS_days.notna().sum())} ({int(C.RFS_event.sum())})" if "RFS_days" in C else
               f"PFS {int(C.PFS_days.notna().sum())} ({int(C.PFS_event.sum())})" if "PFS_days" in C else "–")
        add(gse, C, plat, ep2)
    T1 = pd.DataFrame(rows)
    comp = pd.read_csv(NEW / "04_clinical" / "completeness_old_vs_new.csv", index_col=0)
    T2 = pd.DataFrame({"Variable": comp.index, "Missing, original metadata": comp.old_metadata_missing.map(lambda v: "" if pd.isna(v) else int(v)).replace("", "not available"),
                       "Missing, revised (TCGA-CDR + GDC)": comp.new_missing.astype(int), "Available, revised": comp.new_available.astype(int)})
    flow = ["576 TCGA-LUAD STAR-Counts samples after QC: 517 primary tumors, 59 adjacent normal lung",
            "517 tumors matched to the TCGA Clinical Data Resource (TCGA-CDR)",
            "Usable follow-up (time > 0): OS 504 (186 deaths), PFI 504 (210 events), DSS 504 (115 events)",
            "Pathological stage available for 509: I 276, II 122, III 85, IV 26",
            "Sex: 278 female, 239 male; smoking: 75 never, 308 former, 120 current, 14 unknown",
            "Complete cases for multivariable OS models (age, sex, stage, smoking): 473 (168 deaths)",
            "566 TCGA-LUAD tumors with MC3 somatic mutation calls; 503 with ABSOLUTE purity estimates"]
    fig = plt.figure(figsize=(W2, 6.4))
    G = gs.GridSpec(3, 1, height_ratios=[1.05, 1.2, 0.85], hspace=0.30)
    ax = fig.add_subplot(G[0]); table(ax, T1, "Cohort characteristics (tumors only)", fontsize=5.4,
                                      col_widths=[.12, .15, .08, .06, .07, .08, .15, .13, .14])
    panel(ax, "A", x=-0.04, y=1.02)
    ax = fig.add_subplot(G[1]); table(ax, T2, "TCGA-LUAD clinical data completeness (517 tumors)", fontsize=5.6,
                                      col_widths=[.16, .26, .3, .2], row_height=1.25)
    panel(ax, "B", x=-0.04, y=1.02)
    ax = fig.add_subplot(G[2]); ax.axis("off")
    ax.set_title("TCGA-LUAD sample flow", loc="left", fontsize=7, pad=4)
    for i, l in enumerate(flow):
        ax.text(0.01, 0.95 - i * 0.135, "• " + l, fontsize=5.9, va="top", transform=ax.transAxes)
    panel(ax, "C", x=-0.04, y=1.02)
    out(fig, 1, "cohorts_clinical")

# ═════════ S2 differential expression ═════════
def s2():
    d = pd.read_csv(NEW / "12_DE_revision" / "DE_full_517T_59N.csv", index_col=0)
    pr = pd.read_csv(NEW / "12_DE_revision" / "DE_paired_58pairs.csv", index_col=0)
    gsea = pd.read_csv(ROOT / "results" / "gsea_hallmark.csv")
    go = pd.read_csv(ROOT / "results" / "go_bp_down.csv")
    fig = plt.figure(figsize=(W2, 6.4))
    G = gs.GridSpec(2, 2, hspace=0.5, wspace=1.05)
    ax = fig.add_subplot(G[0, 0])
    col = d.DEG.map({"Up": RED, "Down": BLUE, "NS": "#d0d3d4"})
    ax.scatter(np.log10(d.baseMean + 1), d.log2FC_shrunk, s=1, c=col, lw=0, rasterized=True)
    ax.axhline(0, c="k", lw=.5)
    ax.set(xlabel="log$_{10}$ (mean normalized count + 1)", ylabel="Shrunken log$_2$ FC", title="MA plot (apeGLM-type shrinkage)")
    ax.legend([Line2D([], [], marker="o", color=c, ls="", ms=3) for c in [RED, BLUE, "#d0d3d4"]],
              [f"Up ({(d.DEG == 'Up').sum():,})", f"Down ({(d.DEG == 'Down').sum():,})", "NS"], loc="upper right")
    panel(ax, "A", x=-0.25)
    ax = fig.add_subplot(G[0, 1])
    j = d[["log2FC_MLE", "gene_name"]].join(pr[["log2FC_MLE"]], rsuffix="_p")
    ax.scatter(j.log2FC_MLE, j.log2FC_MLE_p, s=1, c="#aab7b8", lw=0, rasterized=True)
    h = j[j.gene_name.isin(HUB7)]
    ax.scatter(h.log2FC_MLE, h.log2FC_MLE_p, s=10, c=RED, edgecolor="k", lw=.3, zorder=3)
    lim = [-8, 12]; ax.plot(lim, lim, ls="--", c="k", lw=.5); ax.set_xlim(lim); ax.set_ylim(lim)
    r = j[["log2FC_MLE", "log2FC_MLE_p"]].corr("spearman").iloc[0, 1]
    ax.set(xlabel="log$_2$ FC, all samples (517 T / 59 N)", ylabel="log$_2$ FC, paired design (58 pairs)",
           title=f"Paired-design sensitivity (ρ = {r:.2f})")
    ax.text(0.97, 0.05, "red: 7 hub genes", transform=ax.transAxes, ha="right", fontsize=5.6, color=RED)
    panel(ax, "B", x=-0.25)
    ax = fig.add_subplot(G[1, 0])
    g = gsea[gsea["FDR q-val"] < 0.05].sort_values("NES")
    g = pd.concat([g.head(8), g.tail(8)])
    ax.barh(range(len(g)), g.NES, color=[RED if v > 0 else BLUE for v in g.NES])
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.Term, fontsize=5.4); ax.axvline(0, c="k", lw=.5)
    ax.set(xlabel="Normalized enrichment score", title="GSEA, MSigDB Hallmark (FDR < 0.05)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    panel(ax, "C", x=-0.95, y=1.10)
    ax = fig.add_subplot(G[1, 1])
    g = go.sort_values("Adjusted P-value").head(10).iloc[::-1]
    lab = g.Term.str.replace(r" \(GO:\d+\)", "", regex=True).map(lambda t: t if len(t) <= 34 else t[:32] + "…")
    ax.barh(range(len(g)), -np.log10(g["Adjusted P-value"]), color=BLUE)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(lab, fontsize=5.4)
    ax.set(xlabel="$-$log$_{10}$ adjusted p"); ax.set_title("GO BP, down-regulated DEGs (Enrichr)", loc="right")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    panel(ax, "D", x=-0.95, y=1.10)
    out(fig, 2, "differential_expression")

# ═════════ S3 signed WGCNA ═════════
def s3():
    W = NEW / "02_WGCNA_signed"
    sA = pd.read_csv(W / "soft_threshold_A_all_samples.csv"); sB = pd.read_csv(W / "soft_threshold_B_tumour_only.csv")
    trB = pd.read_csv(W / "B_tumour_only_module_clinical_traits.csv")
    mods = pd.read_csv(W / "gene_module_assignments_A_B.csv")
    fig = plt.figure(figsize=(W2, 6.0))
    G = gs.GridSpec(2, 2, hspace=0.55, wspace=0.45)
    ax = fig.add_subplot(G[0, 0])
    ax.plot(sA.power, sA.R2, "o-", ms=2.5, color=RED, label="All samples (β = 15)")
    ax.plot(sB.power, sB.R2, "s-", ms=2.5, color=BLUE, label="Tumors only (β = 10)")
    ax.axhline(.85, ls="--", c=GREY, lw=.6); ax.axvline(15, ls=":", c=RED, lw=.6); ax.axvline(10, ls=":", c=BLUE, lw=.6)
    ax.set(xlabel="Soft-threshold power β", ylabel="Scale-free fit (signed R²)", title="Scale independence (signed network)")
    ax.legend(loc="lower right"); panel(ax, "A")
    ax = fig.add_subplot(G[0, 1])
    ax.plot(sA.power, sA.mean_k, "o-", ms=2.5, color=RED); ax.plot(sB.power, sB.mean_k, "s-", ms=2.5, color=BLUE)
    ax.set_yscale("log"); ax.set(xlabel="Soft-threshold power β", ylabel="Mean connectivity", title="Mean connectivity")
    panel(ax, "B")
    ax = fig.add_subplot(G[1, 0])
    vc = mods.module_A.value_counts().sort_index()
    ax.bar([f"M{i}" if i else "grey" for i in vc.index], vc.values,
           color=["#bdc3c7" if i == 0 else (RED if i == 2 else "#5d6d7e") for i in vc.index])
    for i, v in enumerate(vc.values): ax.text(i, v + 15, str(v), ha="center", fontsize=5.4)
    ax.set(ylabel="Genes", title="Module sizes, all-sample network (ME2 in red)")
    panel(ax, "C")
    ax = fig.add_subplot(G[1, 1])
    M = trB.set_index("module")[["rho_stage", "rho_age"]].assign(OS=np.log2(trB.set_index("module").HR_OS_perSD))
    Q = trB.set_index("module")[["FDR_stage", "FDR_age", "FDR_OS"]]
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-.45, vmax=.45, aspect="auto")
    for i in range(M.shape[0]):
        for k, (v, q) in enumerate(zip(M.values[i], Q.values[i])):
            lab = (f"{v:.2f}" if k < 2 else f"HR {2 ** v:.2f}") + ("*" if q < .05 else "")
            ax.text(k, i, lab, ha="center", va="center", fontsize=5.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Stage (ρ)", "Age (ρ)", "OS (HR/SD)"])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"{m} ({s})" for m, s in zip(trB.module, trB["size"])])
    ax.set_title("Tumor-only network: module vs clinical traits\n(* FDR < 0.05; hub genes in M2)", fontsize=6.8)
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.ax.tick_params(labelsize=5.5); cb.set_label("ρ / log$_2$ HR", fontsize=6)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    panel(ax, "D", x=-0.35)
    out(fig, 3, "WGCNA_signed")

# ═════════ S4 PPI ═════════
def s4():
    import networkx as nx
    P = NEW / "03_PPI_hub_selection"
    func = pd.read_csv(P / "string_cache" / "string_v12_functional_score150.tsv", sep="\t")
    ids = pd.read_csv(P / "string_cache" / "string_v12_id_mapping.tsv", sep="\t")
    back = dict(zip(ids.preferredName, ids.queryItem))
    PRIOR = .041
    def comb(df, ch):
        p = 1.0
        for c in ch: p = p * (1 - ((df[c] - PRIOR) / (1 - PRIOR)).clip(lower=0))
        return (1 - p) * (1 - PRIOR) + PRIOR
    func["E2"] = comb(func, ["nscore", "fscore", "pscore", "escore", "dscore", "tscore"])
    d = func[func.E2 >= .7]
    Gx = nx.Graph(); Gx.add_edges_from((back.get(a, a), back.get(b, b)) for a, b in zip(d.preferredName_A, d.preferredName_B))
    prim = pd.read_csv(P / "centrality_E2_0.7.csv", index_col=0)
    top = list(prim.index[:15])
    summ = pd.read_csv(P / "network_summary_9_settings.csv")
    fig = plt.figure(figsize=(W2, 7.6))
    G = gs.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.12)
    ax = fig.add_subplot(G[0])
    pos = nx.spring_layout(Gx, seed=7, k=0.55)
    nx.draw_networkx_edges(Gx, pos, ax=ax, alpha=.15, width=.4)
    cols = [RED if n in HUB7 else ("#f0b27a" if n in top else "#aab7b8") for n in Gx]
    nx.draw_networkx_nodes(Gx, pos, ax=ax, node_color=cols, node_size=[10 + 3 * Gx.degree(n) for n in Gx],
                           linewidths=.3, edgecolors="white")
    from adjustText import adjust_text
    tx = [ax.text(pos[n][0], pos[n][1], n, fontsize=5.4, fontweight="bold" if n in HUB7 else None,
                  color="k", ha="center", va="center") for n in Gx if n in HUB7 or n in top]
    adjust_text(tx, ax=ax, expand=(1.4, 1.8), force_text=(0.6, 0.9),
                arrowprops=dict(arrowstyle="-", lw=0.35, color="#555555"))
    ax.axis("off")
    ax.set_title(f"STRING v12, ME2 candidates, combined score without co-expression ≥ 0.7 "
                 f"({Gx.number_of_nodes()} nodes, {Gx.number_of_edges()} edges)", fontsize=7, loc="left")
    ax.legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for c in [RED, "#f0b27a", "#aab7b8"]],
              ["7 hub genes", "other consensus top-15", "other candidates"], loc="lower left")
    panel(ax, "A", x=-0.02, y=1.0)
    ax = fig.add_subplot(G[1])
    ev_ = {"E1": "all channels", "E2": "no co-expression", "E3": "physical"}
    t10 = summ.top10.map(lambda t: ", ".join(t.split(", ")[:5]) + ",\n" + ", ".join(t.split(", ")[5:]))
    T = pd.DataFrame({"Network": summ.network, "Evidence": summ.network.str[:2].map(ev_),
                      "Cut-off": summ.cutoff, "Nodes": summ.nodes_with_edges, "Edges": summ.edges,
                      "Density": summ.density.map(lambda v: f"{v:.3f}"), "Consensus top 10 (rank order)": t10})
    table(ax, T, "Nine STRING networks used for the robustness analysis", fontsize=5.2,
          col_widths=[.08, .14, .07, .07, .07, .08, .49], row_height=2.3)
    panel(ax, "B", x=-0.02, y=1.0)
    out(fig, 4, "PPI_network")

# ═════════ S5 survival details ═════════
def s5():
    S = NEW / "05_survival"
    res = pd.read_csv(S / "cox_uni_multi_per_SD_OS_PFI_DSS.csv")
    add = pd.read_csv(S / "added_value_over_clinical_model.csv")
    ph = pd.read_csv(S / "proportional_hazards_tests.csv")
    nl = pd.read_csv(S / "nonlinearity_spline_tests.csv")
    ss = pd.read_csv(S / "sensitivity_stage_stratified_cox.csv")
    fig = plt.figure(figsize=(W2, 6.6))
    G = gs.GridSpec(2, 1, height_ratios=[1, 1.05], hspace=0.38)
    top = G[0].subgridspec(1, 2, wspace=0.85)
    for j, ep in enumerate(["PFI", "DSS"]):
        ax = fig.add_subplot(top[j])
        r = res[res.endpoint == ep].set_index("gene").loc[HUB7]
        yy = np.arange(len(HUB7))[::-1]
        ax.errorbar(r.uni_HR, yy + .15, xerr=[r.uni_HR - r.uni_lo, r.uni_hi - r.uni_HR], fmt="o", ms=2.5, color=GREY, capsize=1.3, lw=.8)
        ax.errorbar(r.multi_HR, yy - .15, xerr=[r.multi_HR - r.multi_lo, r.multi_hi - r.multi_HR], fmt="s", ms=2.5, color=RED, capsize=1.3, lw=.8)
        for y, (_, row) in zip(yy, r.iterrows()):
            ax.text(1.03, y - .15, f"{hrci(row.multi_HR, row.multi_lo, row.multi_hi)}; FDR {fp(row.multi_FDR)}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=5.3)
        ax.axvline(1, ls="--", c="k", lw=.5); ax.set_yticks(yy); ax.set_yticklabels(HUB7, fontstyle="italic")
        ax.set_xlabel("HR per 1 SD (95% CI)")
        ax.set_title(f"{'Progression-free interval' if ep == 'PFI' else 'Disease-specific survival'} "
                     f"(multivariable n = {int(r.multi_n.iloc[0])}, {int(r.multi_events.iloc[0])} events)", fontsize=6.5, loc="left")
        panel(ax, "AB"[j], x=-0.25)
    ax = fig.add_subplot(top[0]).figure.axes[0]
    ax.legend([Line2D([], [], marker="o", color=GREY, ls="", ms=3), Line2D([], [], marker="s", color=RED, ls="", ms=3)],
              ["Univariate", "Multivariable"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.axes[-1].remove() if len(fig.axes) > 2 else None
    o = add[add.endpoint == "OS"].set_index("gene").loc[HUB7]
    p_ = ph[ph.endpoint == "OS"].set_index("gene").loc[HUB7]
    n_ = nl[nl.endpoint == "OS"].set_index("gene").loc[HUB7]
    s_ = ss[ss.endpoint == "OS"].set_index("gene").loc[HUB7]
    T = pd.DataFrame({"Gene": HUB7,
                      "C clinical": o.C_clinical.map(lambda v: f"{v:.3f}"),
                      "C + gene": o.C_clinical_plus_gene.map(lambda v: f"{v:.3f}"),
                      "ΔC (95% CI)": [f"{a:+.3f} ({b:+.3f}, {c:+.3f})" for a, b, c in zip(o.delta_C, o.delta_C_lo, o.delta_C_hi)],
                      "LRT FDR": o.LRT_FDR.map(fp),
                      "PH p": p_.ph_p_gene.map(fp),
                      "Non-linear p": n_.p_nonlinear.map(fp),
                      "Stage-stratified HR": [hrci(a, b, c) for a, b, c in zip(s_.HR, s_.lo, s_.hi)],
                      "Strat. FDR": s_.FDR.map(fp)})
    T["ΔC (95% CI)"] = T["ΔC (95% CI)"].str.replace("-0.000", "0.000").str.replace("+0.000", "0.000")
    ax = fig.add_subplot(G[1])
    table(ax, T, "Overall survival, TCGA-LUAD: added value over age, sex, stage and smoking; model checks; "
                 "stage-stratified sensitivity analysis", fontsize=5.3,
          col_widths=[.08, .09, .08, .19, .08, .07, .09, .17, .08], bold_first_col=True, row_height=1.55)
    panel(ax, "C", x=-0.02, y=1.0)
    out(fig, 5, "survival_details")

# ═════════ S6 single-gene KM curves ═════════
def s6():
    X = pd.read_csv(NEW / "05_survival" / "hub_expression_log2CPM_tumours.csv", index_col=0)
    clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
    d = clin.join(X)
    fig, axes = plt.subplots(2, 4, figsize=(W2, 4.6), gridspec_kw=dict(hspace=0.95, wspace=0.42))
    Z = (X - X.mean()) / X.std(); d["score"] = Z[HUB7].mean(1)
    items = [(g, "OS") for g in HUB7] + [("score", "PFI")]
    for ax, (g, ep) in zip(axes.flat, items):
        k = d[[f"{ep}_days", f"{ep}_event", g]].dropna()
        hi = k[g] > k[g].median()
        p = km_with_risk(ax, k[f"{ep}_days"] / 365.25, k[f"{ep}_event"], [hi.values, (~hi).values], [RED, BLUE],
                         ["High", "Low"], risk_y0=-0.36, dy=0.09, show_row_labels=(ax in (axes[0, 0], axes[1, 0])))
        ax.set_title(("7-gene score, PFI" if g == "score" else f"{g}, OS") + f"  ({pfmt(p)})", fontsize=6.3,
                     fontstyle="normal" if g == "score" else "italic")
        ax.set_xlabel("Years", fontsize=6); ax.set_ylabel("Survival" if ax in (axes[0, 0], axes[1, 0]) else "", fontsize=6)
        ax.get_legend().remove()
    axes[0, 0].legend(loc="lower left", fontsize=5.4)
    out(fig, 6, "KM_single_genes")

# ═════════ S7 benchmark extras ═════════
def s7():
    B = NEW / "06_benchmark"
    null = pd.read_csv(B / "null_distributions_1000x2.csv"); obs = pd.read_csv(B / "signature_performance.csv", index_col=0)
    pct = pd.read_csv(B / "HUB7_vs_null_empirical.csv"); nest = pd.read_csv(B / "nested_LRT_HUB7_vs_proliferation.csv")
    sd = pd.read_csv(B / "signature_definitions.csv").set_index("signature")
    fig = plt.figure(figsize=(W2, 6.8))
    G = gs.GridSpec(3, 2, height_ratios=[1, .9, .8], hspace=0.62, wspace=0.35)
    for j, (m, lab) in enumerate([("C_multi", "OS C-index, clinical model + score"), ("AUC", "Tumor vs normal AUC")]):
        ax = fig.add_subplot(G[0, j])
        for (nm, g), c in zip(null.groupby("null"), ["#bdc3c7", "#f5b041"]):
            ax.hist(g[m], bins=35, alpha=.75, color=c, label="expression-matched up-DEGs" if nm.startswith("N1") else "G2M/E2F genes")
        ax.axvline(obs.loc["HUB7", m], color=RED, lw=1.1, label="7-gene score")
        if m == "C_multi":
            ax.axvline(0.6723, color="k", ls=":", lw=1, label="clinical model alone")
        pp = pct[pct.metric == m]
        ax.set_title(f"{lab}\npercentile {pp.HUB7_percentile.iloc[0]:.0f} / {pp.HUB7_percentile.iloc[1]:.0f}", fontsize=6.5)
        ax.set(xlabel=lab, ylabel="Random 7-gene sets")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.65)
        ax.legend(fontsize=5.2, loc="upper left", ncol=2, columnspacing=0.8)
        panel(ax, "AB"[j], x=-0.2)
    names = {"HUB7": "7-gene score (this study)", "OLD5": "Original 5 genes", "MKI67": "MKI67", "PAM50_prolif": "PAM50 proliferation (11 genes)",
             "PCNA_meta": "PCNA metagene (Venet rule)", "HALLMARK_G2M": "Hallmark G2M checkpoint", "HALLMARK_E2F": "Hallmark E2F targets"}
    T = pd.DataFrame({"Signature": [names[k] for k in names], "Genes": [int(sd.loc[k, "n_present"]) for k in names],
                      "C score": [f"{obs.loc[k, 'C_uni']:.3f}" for k in names],
                      "C clin.+score": [f"{obs.loc[k, 'C_multi']:.3f}" for k in names],
                      "Adj. HR/SD": [f"{obs.loc[k, 'HR_multi']:.2f}" for k in names],
                      "p": [fp(obs.loc[k, "p_multi"]) for k in names],
                      "AUC T vs N": [f"{obs.loc[k, 'AUC']:.3f}" for k in names],
                      "r vs 7-gene": [f"{obs.loc[k, 'r_with_HUB7']:.2f}" for k in names]})
    ax = fig.add_subplot(G[1, :]); table(ax, T, "Signature performance, TCGA-LUAD (OS; adjusted for age, sex, stage, smoking)",
                                         fontsize=5.4, col_widths=[.25, .06, .09, .13, .12, .07, .13, .13], highlight_rows=[0])
    panel(ax, "C", x=-0.02, y=1.0)
    N = pd.DataFrame({"Comparator": [names[c] for c in nest.comparator], "r": nest.r_HUB7_comparator.map(lambda v: f"{v:.2f}"),
                      "7-gene adds: LR χ² (p)": [f"{a:.2f} ({fp(b)})" for a, b in zip(nest.HUB7_adds_to_clin_plus_comparator_LR, nest.p_HUB7_adds)],
                      "Comparator adds: LR χ² (p)": [f"{a:.2f} ({fp(b)})" for a, b in zip(nest.comparator_adds_to_clin_plus_HUB7_LR, nest.p_comparator_adds)]})
    ax = fig.add_subplot(G[2, :]); table(ax, N, "Nested likelihood-ratio tests (both models include age, sex, stage, smoking)",
                                         fontsize=5.4, col_widths=[.32, .08, .28, .28])
    panel(ax, "D", x=-0.02, y=1.0)
    out(fig, 7, "benchmark_details")

# ═════════ S8 external validation extras ═════════
def s8():
    EV = NEW / "07_external_validation"
    DAT = EV / "data"
    ta = pd.read_csv(EV / "time_dependent_AUC_HUB7_mean.csv")
    sens = pd.read_csv(EV / "sensitivity_HUB7_mean.csv")
    meta = pd.read_csv(EV / "meta_analysis_OS_random_effects.csv")
    fig = plt.figure(figsize=(W2, 8.6))
    G = gs.GridSpec(3, 1, height_ratios=[1, 1.25, 1.1], hspace=0.55)
    top = G[0].subgridspec(1, 3, wspace=0.45)
    for j, (gse, ep, lab) in enumerate([("GSE31210", "RFS", "Relapse-free survival"), ("GSE68465", "PFS", "Progression-free survival")]):
        ax = fig.add_subplot(top[j])
        E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
        C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
        Z = (E.loc[C.index, HUB7] - E.loc[C.index, HUB7].mean()) / E.loc[C.index, HUB7].std()
        k = pd.DataFrame({"T": C[f"{ep}_days"] / 365.25, "E": C[f"{ep}_event"], "s": Z.mean(1)}).dropna()
        hi = (k.s > k.s.median()).values
        p = km_with_risk(ax, k["T"], k["E"], [hi, ~hi], [RED, BLUE], ["High", "Low"], risk_y0=-0.36, dy=0.09, show_row_labels=j == 0)
        ax.set_title(f"{gse}: {lab}\n(n = {len(k)}; {pfmt(p)})", fontsize=6.3); ax.set_xlabel("Years")
        ax.get_legend().remove()
        if j == 0: ax.legend(loc="lower left", fontsize=5.4)
        panel(ax, "AB"[j], x=-0.3)
    ax = fig.add_subplot(top[2])
    t = ta.copy(); t["lab"] = t.cohort.str.replace("TCGA-LUAD", "TCGA") + " " + t.endpoint
    piv = t.pivot(index="lab", columns="year", values="AUC")
    x = np.arange(len(piv))
    ax.bar(x - .18, piv[3], .36, color="#85c1e9", label="3 years"); ax.bar(x + .18, piv[5], .36, color=BLUE, label="5 years")
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=55, ha="right", fontsize=5.4)
    ax.axhline(.5, c="k", lw=.5); ax.set_ylim(.4, .8); ax.set(ylabel="Time-dependent AUC", title="7-gene score (Uno, IPCW)")
    ax.legend(loc="upper left", fontsize=5.4)
    panel(ax, "C", x=-0.32)
    S = pd.DataFrame({"Cohort": sens.cohort, "Endpoint": sens.endpoint, "Analysis": sens.analysis.str.replace(" (GSE31210)", ""),
                      "n (events)": [f"{a} ({b})" for a, b in zip(sens.n, sens.events)],
                      "HR per SD (95% CI)": [hrci(a, b, c) for a, b, c in zip(sens.HR, sens.lo, sens.hi)], "p": sens.p.map(fp)})
    ax = fig.add_subplot(G[1]); table(ax, S, "Sensitivity analyses of the locked 7-gene score (adjusted for age, sex, stage; "
                                             "GSE68465 stratified by site)", fontsize=5.4, col_widths=[.13, .08, .34, .12, .19, .08], row_height=1.15)
    panel(ax, "D", x=-0.02, y=1.0)
    names = {"HUB7_mean": "7-gene score (mean z)", "HUB7_risk": "7-gene ridge-Cox score (TCGA weights)", "OLD5": "Original 5 genes",
             "PAM50_prolif": "PAM50 proliferation", "PCNA_meta": "PCNA metagene", "MKI67": "MKI67",
             "HALLMARK_G2M": "Hallmark G2M", "HALLMARK_E2F": "Hallmark E2F"}
    m = meta[meta.item.isin(names)]
    rows = []
    for k_ in names:
        u = m[(m.item == k_) & (m.model == "uni")].iloc[0]; v = m[(m.item == k_) & (m.model == "multi")].iloc[0]
        rows.append({"Signature": names[k_], "Univariate HR (95% CI)": hrci(u.HR, u.lo, u.hi), "I² (%)": f"{u.I2:.0f}",
                     "Adjusted HR (95% CI)": hrci(v.HR, v.lo, v.hi), "I² (%) ": f"{v.I2:.0f}", "p (adjusted)": fp(v.p)})
    ax = fig.add_subplot(G[2]); table(ax, pd.DataFrame(rows), "Random-effects meta-analysis of OS across TCGA and three GEO cohorts "
                                                             "(HR per SD)", fontsize=5.4,
                                      col_widths=[.3, .17, .07, .17, .07, .1], highlight_rows=[0], row_height=1.15)
    panel(ax, "E", x=-0.02, y=1.0)
    out(fig, 8, "external_validation_details")

# ═════════ S9 diagnostic / ML ═════════
def s9():
    M = NEW / "09_ML"
    summ = pd.read_csv(M / "A_diagnostic_nested_cv_summary.csv", index_col=[0, 1])
    ext = pd.read_csv(M / "A_external_GSE31210_diagnostic.csv").set_index("model")
    p0 = pd.read_csv(M / "part0_reproduction_of_original_errors.csv")
    order = ["Logistic regression", "SVM (RBF)", "Random forest", "XGBoost", "MLP"]
    fig = plt.figure(figsize=(W2, 5.6))
    G = gs.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=1.0)
    top = G[0].subgridspec(1, 3, wspace=0.35)
    for j, (met, lab, lo) in enumerate([("AUC", "AUC", .75), ("balanced_accuracy", "Balanced accuracy", .5), ("MCC", "MCC", 0)]):
        ax = fig.add_subplot(top[j])
        for i, fs in enumerate(["HUB7", "FOLD7"]):
            s = summ.loc[fs].loc[order]
            x = np.arange(len(order)) + (i - .5) * .36
            ax.bar(x, s[f"{met}_mean"], .34, yerr=[s[f"{met}_mean"] - s[f"{met}_lo"], s[f"{met}_hi"] - s[f"{met}_mean"]],
                   color=[RED, "#aab7b8"][i], error_kw=dict(lw=.6, capsize=1.5),
                   label=["7 hub genes (nested CV)", "7 DEGs re-selected per fold"][i])
        ax.scatter(np.arange(len(order)), ext.loc[order, met], marker="D", s=12, color=BLUE, zorder=3, label="External GSE31210")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=5.6)
        ax.set_ylim(lo, 1.0); ax.set_title(lab)
        if j == 0: ax.legend(loc="upper center", bbox_to_anchor=(1.75, -0.42), ncol=3, fontsize=5.6)
        panel(ax, "ABC"[j], x=-0.3)
    T = pd.DataFrame({
        "Analysis": ["Original MLP (early stopping)", "Same MLP, no early stopping",
                     "LR, 5-fold CV (original text)", "LR, apparent fit (original Fig. S14C)"],
        "AUC": [f2(p0.AUC[0], 3), f2(p0.AUC[1], 3), "", ""],
        "TN / FP / FN / TP": ["", "", f"{int(p0.TN[2])} / {int(p0.FP[2])} / {int(p0.FN[2])} / {int(p0.TP[2])}",
                              f"{int(p0.TN[3])} / {int(p0.FP[3])} / {int(p0.FN[3])} / {int(p0.TP[3])}"],
        "Specificity": ["", "", f2(p0.specificity[2], 3), f2(p0.specificity[3], 3)],
        "Note": ["stopped after 12–30 iterations (under-fitted)", "under-fitting resolved",
                 "cross-validated predictions", "model evaluated on its training data"]})
    ax = fig.add_subplot(G[1]); table(ax, T, "Reproduction of two errors in the original diagnostic analysis", fontsize=5.4,
                                      col_widths=[.3, .07, .16, .1, .37])
    panel(ax, "D", x=-0.02, y=1.0)
    out(fig, 9, "diagnostic_ML")

# ═════════ S10 mutation context and stage ═════════
def s10():
    MC = NEW / "08_mutation_context"
    fr = pd.read_csv(MC / "mutation_frequencies_patient_level.csv")
    inter = pd.read_csv(MC / "score_by_mutation_interaction_cox.csv")
    drv = pd.read_csv(MC / "TCGA_driver_status_patient_level.csv", index_col=0)
    X = pd.read_csv(NEW / "05_survival" / "hub_expression_log2CPM_tumours.csv", index_col=0)
    clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
    Z = (X - X.mean()) / X.std(); sc = Z[HUB7].mean(1)
    d = clin[["patient_id", "stage"]].join(sc.rename("score")).join(drv, on="patient_id")
    tr = pd.read_csv(MC / "stage_trend_jonckheere.csv")
    fig = plt.figure(figsize=(W2, 7.6))
    G = gs.GridSpec(3, 2, height_ratios=[0.9, 0.9, 1.2], hspace=0.62, wspace=0.42)
    ax = fig.add_subplot(G[0, 0])
    f = fr[fr.gene.isin(["TP53", "KRAS", "KEAP1", "STK11", "EGFR", "BRAF"] + HUB7)].copy()
    f["pct"] = np.where(f.role == "driver", f.pct_driver_definition, f.pct_any)
    f = f.set_index("gene").loc[["TP53", "KRAS", "KEAP1", "STK11", "EGFR", "BRAF"] + HUB7]
    ax.bar(range(len(f)), f.pct, color=[DARK if r == "driver" else RED for r in f.role])
    for i, v in enumerate(f.pct): ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=5)
    ax.set_xticks(range(len(f))); ax.set_xticklabels(f.index, rotation=60, ha="right", fontstyle="italic", fontsize=5.6)
    ax.set_ylabel("% of tumors (n = 566)")
    ax.set_title("Protein-altering mutations (MC3)", loc="right")
    ax.set_ylim(0, 60); panel(ax, "A", x=-0.17, y=1.04)
    ax = fig.add_subplot(G[0, 1])
    muts = ["TP53", "KRAS", "EGFR", "STK11", "KEAP1"]
    pos = 0; xt = []
    for m_ in muts:
        a = d.score[d[m_] == 1].dropna(); b = d.score[d[m_] == 0].dropna()
        bp = ax.boxplot([b, a], positions=[pos, pos + .8], widths=.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k", lw=.7), boxprops=dict(lw=.5), whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
        bp["boxes"][0].set_facecolor("#d6eaf8"); bp["boxes"][1].set_facecolor("#f5b7b1")
        xt.append(pos + .4); pos += 2.2
    sa = pd.read_csv(MC / "score_by_mutation_status.csv").set_index("mutation")
    for x_, m_ in zip(xt, muts):
        ax.text(x_, 2.75, f"FDR {fp(sa.loc[m_, 'FDR'])}", ha="center", fontsize=5.2)
    ax.set_ylim(-3.5, 3.3)
    ax.set_xticks(xt); ax.set_xticklabels(muts, fontstyle="italic"); ax.set_ylabel("7-gene score (z)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#d6eaf8"), plt.Rectangle((0, 0), 1, 1, fc="#f5b7b1")], ["wild-type", "mutant"],
              loc="lower center", ncol=2, fontsize=5.4)
    ax.set_title("7-gene score by driver status (TCGA-LUAD)"); panel(ax, "B", x=-0.2)
    ax = fig.add_subplot(G[1, :])
    cohorts = ["TCGA-LUAD", "GSE68465", "GSE72094", "GSE31210"]
    DAT = NEW / "07_external_validation" / "data"
    pos = 0; xt = []
    for coh in cohorts:
        if coh == "TCGA-LUAD":
            dd = d[["stage", "score"]].dropna()
        else:
            E = pd.read_csv(DAT / f"{coh}_expression_genes.csv", index_col=0)
            C = pd.read_csv(DAT / f"{coh}_clinical.csv", index_col=0); C = C[C.tumour == 1]
            Zg = (E.loc[C.index, HUB7] - E.loc[C.index, HUB7].mean()) / E.loc[C.index, HUB7].std()
            dd = pd.DataFrame({"stage": C.stage, "score": Zg.mean(1)}).dropna()
        st = [s for s in ["I", "II", "III", "IV"] if (dd.stage == s).sum() >= 5]
        bp = ax.boxplot([dd.score[dd.stage == s] for s in st], positions=[pos + i for i in range(len(st))], widths=.6,
                        showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=.7))
        for b_, c_ in zip(bp["boxes"], ["#fadbd8", "#f1948a", "#cb4335", "#7b241c"]): b_.set_facecolor(c_)
        t = tr[(tr.cohort == coh) & (tr.variable == "HUB7_score")].iloc[0]
        ax.text(pos + (len(st) - 1) / 2, 3.1, f"{coh}\nJT p {'< 0.001' if t.JT_p < .001 else '= ' + fp(t.JT_p)}, ρ = {t.spearman_rho:.2f}", ha="center", fontsize=5.6)
        for i, s in enumerate(st): xt.append((pos + i, f"{s}\n({(dd.stage == s).sum()})"))
        pos += len(st) + 1
    ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([l for _, l in xt], fontsize=5.4)
    ax.set_ylim(-3, 3.9); ax.set_ylabel("7-gene score (z)"); ax.set_xlim(-0.7, pos - 0.3)
    ax.set_title("7-gene score by pathological stage (Jonckheere–Terpstra trend test)", loc="left", fontsize=7)
    panel(ax, "C", x=-0.07)
    T = pd.DataFrame({"Cohort": inter.cohort, "Endpoint": inter.endpoint, "Mutation": inter.mutation,
                      "Mutant n (events)": [f"{a} ({b})" for a, b in zip(inter.n_mut, inter.events_mut)],
                      "HR mutant (95% CI)": [("too few events" if e < 10 else hrci(a, b, c)) for a, b, c, e in
                                             zip(inter.HR_Mut, inter.lo_Mut, inter.hi_Mut, inter.events_mut)],
                      "HR wild-type (95% CI)": [hrci(a, b, c) for a, b, c in zip(inter.HR_WT, inter.lo_WT, inter.hi_WT)],
                      "Interaction p": inter.p_interaction.map(fp), "FDR": inter.FDR_interaction.map(fp)})
    ax = fig.add_subplot(G[2, :]); table(ax, T, "7-gene score × driver-mutation interaction (Cox, adjusted for age, sex, stage; HR per SD)",
                                         fontsize=5.0, col_widths=[.12, .08, .08, .13, .19, .19, .1, .08], row_height=1.22)
    panel(ax, "D", x=-0.02, y=1.0)
    out(fig, 10, "mutation_stage")

# ═════════ S11 regulation: TF, CNV, methylation ═════════
def s11():
    import textwrap
    E11 = NEW / "11_extended"
    ta = pd.read_csv(E11 / "M9_TF_activity_vs_score.csv").head(15).iloc[::-1]
    cn = pd.read_csv(E11 / "M10_copy_number_vs_expression.csv").set_index("gene").loc[HUB7]
    gl = pd.read_csv(E11 / "M10_methylation_promoter_gene_level.csv").set_index("gene").loc[HUB7]
    ev = pd.read_csv(E11 / "M9_curated_TF_target_evidence.csv")
    fig = plt.figure(figsize=(W2, 6.6))
    G = gs.GridSpec(2, 3, height_ratios=[1, 1.0], hspace=0.42, wspace=0.55)
    ax = fig.add_subplot(G[0, 0])
    ax.barh(range(len(ta)), ta.rho_activity_vs_score, color=RED)
    ax.set_yticks(range(len(ta))); ax.set_yticklabels(ta.TF, fontstyle="italic", fontsize=5.6)
    ax.set(xlabel="ρ, TF activity vs score", xlim=(0, 1)); ax.set_title("Top 15 TF activities\n(CollecTRI, ULM)", fontsize=6.8)
    panel(ax, "A", x=-0.42)
    ax = fig.add_subplot(G[0, 1])
    c = cn[["pct_deep_del", "pct_shallow_del", "pct_gain", "pct_amp"]]
    bottom = np.zeros(len(c))
    for col_, cc, lab in zip(c.columns, ["#2166ac", "#92c5de", "#f4a582", "#b2182b"], ["Deep del.", "Shallow del.", "Gain", "Amp."]):
        ax.bar(range(len(c)), c[col_], bottom=bottom, color=cc, label=lab, width=.7); bottom += c[col_].values
    for i, (g, r) in enumerate(cn.iterrows()):
        ax.text(i, bottom[i] + 1.5, f"{r.rho_CN_vs_expr:.2f}", ha="center", fontsize=5)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(HUB7, fontstyle="italic", rotation=55, ha="right", fontsize=5.6)
    ax.set_ylabel("% of tumors (GISTIC2)"); ax.set_ylim(0, 90)
    ax.set_title("Copy number\n(numbers: ρ with expression)", fontsize=6.8)
    ax.legend(fontsize=5, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1); panel(ax, "B", x=-0.35)
    ax = fig.add_subplot(G[0, 2])
    x = np.arange(len(gl))
    ax.bar(x - .2, gl.median_promoter_beta_normal, .4, color=BLUE, label=f"Normal ({int(gl.n_normal.iloc[0])})")
    ax.bar(x + .2, gl.median_promoter_beta_tumour, .4, color=RED, label=f"Tumor ({int(gl.n_tumour.iloc[0])})")
    for i, r in enumerate(gl.itertuples()):
        ax.text(i, max(r.median_promoter_beta_normal, r.median_promoter_beta_tumour) + .01, f"{r.rho_promoter_beta_vs_expr:.2f}",
                ha="center", fontsize=5)
    ax.set_xticks(x); ax.set_xticklabels(HUB7, fontstyle="italic", rotation=55, ha="right", fontsize=5.6); ax.set_ylim(0, .3)
    ax.set_ylabel("Median promoter β (450K)")
    ax.set_title("Promoter methylation\n(numbers: ρ with expression)", fontsize=6.8); ax.legend(fontsize=5.2, loc="upper right")
    panel(ax, "C", x=-0.38)
    ax = fig.add_subplot(G[1, :])
    rows = []
    for g in HUB7:
        e = ev[ev.hub_gene == g]
        tr_ = ", ".join(sorted(e[e.in_TRRUST].TF.unique())) or "–"
        key = [t for t in ["E2F1", "E2F2", "E2F3", "E2F4", "FOXM1", "MYC", "MYBL2", "TP53", "RB1"] if t in set(e.TF)]
        rows.append({"Gene": g, "TRRUST v2 regulators": "\n".join(textwrap.wrap(tr_, 62)),
                     "Cell-cycle TFs (TRRUST/CollecTRI)": ", ".join(key) or "–",
                     "TFs (n)": e.TF.nunique()})
    t = table(ax, pd.DataFrame(rows), "Curated TF–target evidence for the hub genes", fontsize=5.4,
              col_widths=[.08, .46, .36, .1], bold_first_col=True, row_height=2.4)
    panel(ax, "D", x=-0.04, y=1.0)
    out(fig, 11, "regulation_TF_CNV_methylation")

# ═════════ S12 immune ═════════
def s12():
    im = pd.read_csv(NEW / "10_immune" / "immune_correlations_raw_and_purity_adjusted.csv")
    D = pd.read_csv(NEW / "10_immune" / "tumour_immune_purity_matrix.csv", index_col=0)
    pv = pd.read_csv(NEW / "10_immune" / "purity_vs_hub_genes.csv").set_index("variable")
    cols = ["HUB7 score"] + HUB7
    fig = plt.figure(figsize=(W2, 8.0))
    G = gs.GridSpec(2, 2, height_ratios=[1, 1.45], width_ratios=[1, 1.25], hspace=0.42, wspace=0.65)
    def hm(ax, fam, title):
        r = im[im.family == fam]
        M = r.pivot(index="feature", columns="variable", values="rho_purity_adj")[cols]
        Q = r.pivot(index="feature", columns="variable", values="FDR_purity_adj")[cols]
        M = M.loc[M["HUB7 score"].sort_values().index]; Q = Q.loc[M.index]
        h = ax.imshow(M.values, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                q = Q.values[i, j]
                ax.text(j, i, f"{M.values[i, j]:.2f}" + ("*" if q < .05 else ""), ha="center", va="center", fontsize=4.6,
                        color="white" if abs(M.values[i, j]) > .33 else "black")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(["Score"] + HUB7, rotation=55, ha="right", fontsize=5.4)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=5.4)
        ax.set_title(title, loc="left", fontsize=6.8); ax.spines[:].set_visible(False); ax.tick_params(length=0)
        return h
    ax = fig.add_subplot(G[0, 0])
    s = D[["HUB7 score", "ABSOLUTE_purity"]].dropna()
    ax.scatter(s.ABSOLUTE_purity, s["HUB7 score"], s=3, c=RED, alpha=.5, lw=0)
    ax.set(xlabel="ABSOLUTE tumor purity", ylabel="7-gene score")
    ax.set_title(f"Score vs purity: ρ = {pv.loc['HUB7 score', 'rho_with_purity']:.3f}\n(p = {pv.loc['HUB7 score', 'p']:.2f}, n = {int(pv.loc['HUB7 score', 'n'])})", fontsize=6.8)
    panel(ax, "A", x=-0.2)
    ax = fig.add_subplot(G[0, 1]); h = hm(ax, "MCP-counter", "MCP-counter (partial ρ, purity-adjusted; * FDR < 0.05)")
    panel(ax, "B", x=-0.45)
    ax = fig.add_subplot(G[1, :]); h = hm(ax, "CIBERSORT", "CIBERSORT, PanCanAtlas (partial ρ, purity-adjusted; * FDR < 0.05)")
    cb = fig.colorbar(h, ax=ax, fraction=.03, pad=.02); cb.set_label("partial Spearman ρ", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    panel(ax, "C", x=-0.25)
    out(fig, 12, "immune_purity")

# ═════════ S13 pan-cancer ═════════
def s13():
    P = NEW / "11_extended" / "pancancer"
    tn = pd.read_csv(P / "pancancer_tumour_vs_TCGA_plus_GTEx_normal.csv")
    sv = pd.read_csv(P / "pancancer_survival_score_per_SD.csv")
    fig = plt.figure(figsize=(W2, 9.4))
    G = gs.GridSpec(2, 1, height_ratios=[0.75, 1.25], hspace=0.38)
    ax = fig.add_subplot(G[0])
    t = tn.dropna(subset=["diff"]).sort_values("diff")
    x = np.arange(len(t))
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
    ax.legend(loc="lower right", fontsize=5.6); panel(ax, "A", x=-0.07)
    rows = []
    for c in sorted(sv.cancer.unique()):
        o = sv[(sv.cancer == c) & (sv.endpoint == "OS")].iloc[0]; p_ = sv[(sv.cancer == c) & (sv.endpoint == "PFI")].iloc[0]
        rows.append({"Cancer": c, "OS n (deaths)": f"{int(o.n)} ({int(o.events)})",
                     "OS HR (95% CI)": "–" if pd.isna(o.HR) else hrci(o.HR, o.lo, o.hi), "OS FDR": fp(o.FDR) or "–",
                     "Adjusted OS HR": f2(o.HR_adj) or "–", "Adjusted FDR": fp(o.FDR_adj) or "–",
                     "PFI events": int(p_.events), "PFI HR": f2(p_.HR) or "–", "PFI FDR": fp(p_.FDR) or "–"})
    T = pd.DataFrame(rows)
    ax = fig.add_subplot(G[1])
    hl = [i for i, c in enumerate(T.Cancer) if c == "LUAD"]
    table(ax, T, "7-gene score and survival per cancer type (HR per SD; adjusted: age, sex and stage where ≥ 70% staged)",
          fontsize=5.0, col_widths=[.08, .13, .19, .09, .12, .1, .09, .09, .09], row_height=1.06, highlight_rows=hl)
    panel(ax, "B", x=-0.04, y=1.0)
    out(fig, 13, "pancancer")

# ═════════ S14 druggability ═════════
def s14():
    import textwrap
    dt = pd.read_csv(NEW / "11_extended" / "drugs" / "Table_hub_gene_drug_mechanisms_ChEMBL.csv").set_index("gene").loc[HUB7]
    dm = pd.read_csv(NEW / "11_extended" / "drugs" / "DepMap_dependency_LUAD.csv").set_index("gene").loc[HUB7]
    fig = plt.figure(figsize=(W2, 5.8))
    G = gs.GridSpec(2, 1, height_ratios=[1.15, 0.85], hspace=0.22)
    rows = []
    for g, r in dt.iterrows():
        parts = []
        if isinstance(r.agents_phase3, str) and r.agents_phase3: parts.append(f"Phase 3: {r.agents_phase3}")
        if isinstance(r.agents_phase2, str) and r.agents_phase2: parts.append(f"Phase 2: {r.agents_phase2}")
        txt = "; ".join(parts) if parts else (r.note if isinstance(r.note, str) else "–")
        txt = txt.replace("No curated drug mechanism; druggable only indirectly via partner kinase", "No direct agent; indirect via")
        rows.append({"Gene": g, "Agents": int(r.n_clinical_stage_agents),
                     "Max phase": "–" if pd.isna(r.highest_phase) else int(r.highest_phase),
                     "Clinical-stage agents (ChEMBL drug_mechanism)": "\n".join(textwrap.wrap(txt, 95))})
    ax = fig.add_subplot(G[0])
    t = table(ax, pd.DataFrame(rows), "Curated drug mechanisms (ChEMBL); phase 1 agents in Supplementary Table S10", fontsize=5.3,
              col_widths=[.08, .07, .08, .77], bold_first_col=True, row_height=1.9)
    for (r, c), cl in t.get_celld().items():
        if c == 3 and r > 0: cl._loc = "left"; cl.set_text_props(ha="left")
    panel(ax, "A", x=-0.04, y=1.0)
    D = pd.DataFrame({"Gene": HUB7, "LUAD lines": dm.n_LUAD_lines.astype(int),
                      "Median, LUAD": dm.median_effect_LUAD.map(lambda v: f"{v:.2f}"),
                      "% dep., LUAD": dm.pct_LUAD_dependent.map(lambda v: f"{v:.0f}"),
                      "Median, other": dm.median_effect_other.map(lambda v: f"{v:.2f}"),
                      "% dep., other": dm.pct_other_dependent.map(lambda v: f"{v:.0f}"),
                      "p (LUAD vs other)": dm.p_LUAD_vs_other.map(fp),
                      "Common essential": dm.common_essential.map({True: "yes", False: "no"})})
    ax = fig.add_subplot(G[1])
    table(ax, D, "DepMap Public 26Q1 CRISPR (Chronos) gene effect: 53 LUAD vs 1,155 other cancer cell lines "
                 "(dependent: effect < −1)", fontsize=5.3,
          col_widths=[.1, .1, .12, .14, .12, .14, .14, .14], bold_first_col=True, row_height=1.35)
    panel(ax, "B", x=-0.04, y=1.0)
    out(fig, 14, "druggability")

FUN = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11, 12: s12, 13: s13, 14: s14}
if __name__ == "__main__":
    sel = [int(a) for a in sys.argv[1:]] or list(FUN)
    for n in sel:
        FUN[n]()
