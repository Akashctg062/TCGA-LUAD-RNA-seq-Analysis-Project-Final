"""R13_fig4_external.py — Main Figure 4 (revision 2): external validation in eight GEO cohorts (600 dpi).
A  locked 7-gene score: multivariable OS HR per SD in TCGA (training) and eight GEO cohorts + random-effects meta
B  each hub gene x cohort (OS, multivariable HR per SD, FDR) for all nine cohorts
C  Kaplan-Meier OS by locked score (median split) in GSE31210, GSE68465, GSE72094, with numbers at risk
D  prognostic models locked in TCGA: Harrell C in TCGA (repeated CV) and each GEO cohort"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from R12_figstyle import NEW, FIG_MAIN, W2, RED, BLUE, GREY, DARK, panel, save, pfmt

EV = NEW / "07_external_validation"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
perf = pd.read_csv(EV / "signature_performance_all_cohorts.csv")
meta = pd.read_csv(EV / "meta_analysis_OS_random_effects.csv")
genes = pd.read_csv(EV / "individual_genes_all_cohorts.csv")

fig = plt.figure(figsize=(W2, 9.0))
G = gs.GridSpec(3, 1, figure=fig, height_ratios=[1.35, 1.0, 0.85], hspace=0.7)
top = G[0].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.62)

# ── A forest ──
X = NEW / "13_limitations" / "extra_cohorts"
R9 = pd.read_csv(X / "R13g_cohort_results.csv")
M9 = pd.read_csv(X / "R13g_meta_analysis.csv").set_index("analysis")
G9 = pd.read_csv(X / "R13g_single_genes_all_cohorts.csv")
ax = fig.add_subplot(top[0])
m = M9.loc["all 9 cohorts, multivariable"]
labs = [c.replace("TCGA-LUAD", "TCGA") for c in R9.cohort] + ["Random-effects, 9 cohorts"]
ns = [f"{n} / {e}" for n, e in zip(R9.n, R9.events)] + [f"I² = {m.I2:.0f}%"]
hr = list(R9.HR_multi) + [m.HR]; lo = list(R9.lo_multi) + [m.lo]; hi = list(R9.hi_multi) + [m.hi]
newc = list(R9.new) + [False]
yy = np.arange(len(hr))[::-1].astype(float); yy[-1] -= 0.5
for i in range(len(hr)):
    isM = i == len(hr) - 1; isT = labs[i].startswith("TCGA")
    col = DARK if isM else (RED if isT else ("#5dade2" if newc[i] else BLUE))
    ax.errorbar(hr[i], yy[i], xerr=[[hr[i] - lo[i]], [hi[i] - hr[i]]], fmt="D" if isM else "s", ms=4 if isM else 3,
                color=col, capsize=1.5, lw=.9)
    ax.text(1.03, yy[i], f"{hr[i]:.2f} ({lo[i]:.2f}–{hi[i]:.2f})", transform=ax.get_yaxis_transform(), va="center", fontsize=5.6)
ax.text(1.03, 1.0, "HR (95% CI)", transform=ax.transAxes, fontsize=5.6, fontweight="bold", va="bottom")
ax.axvline(1, ls="--", c="k", lw=.5); ax.axhline(yy[-1] + 0.75, c="#bbbbbb", lw=.5)
ax.set_yticks(yy); ax.set_yticklabels([f"{l}  ({n})" for l, n in zip(labs, ns)], fontsize=5.8)
ax.set_xscale("log"); ax.set_xticks([0.8, 1, 1.5, 2, 2.5]); ax.set_xticklabels(["0.8", "1", "1.5", "2", "2.5"]); ax.minorticks_off()
ax.set_xlim(0.85, 2.7)
ax.set_xlabel("HR per 1 SD of 7-gene score (95% CI), OS")
ax.set_title("Locked 7-gene score (n / deaths)", loc="left", fontsize=7)
ax.legend([Line2D([], [], marker="s", color=RED, ls="", ms=3), Line2D([], [], marker="s", color=BLUE, ls="", ms=3),
           Line2D([], [], marker="s", color="#5dade2", ls="", ms=3), Line2D([], [], marker="D", color=DARK, ls="", ms=3.5)],
          ["Training", "Validation (original)", "Validation (additional)", "Pooled"],
          loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, columnspacing=0.8, handletextpad=0.2)
panel(ax, "A", x=-0.85)

# ── B gene x cohort heatmap ──
ax = fig.add_subplot(top[1])
order = list(R9.cohort)
H = G9.pivot(index="gene", columns="cohort", values="HR_multi").loc[HUB7, order]
Q = G9.pivot(index="gene", columns="cohort", values="FDR_multi").loc[HUB7, order]
H["Meta"] = [M9.loc[f"gene {g}, all 9 cohorts, multivariable", "HR"] for g in HUB7]
Q["Meta"] = [M9.loc[f"gene {g}, all 9 cohorts, multivariable", "p"] for g in HUB7]
im = ax.imshow(np.log2(H.values), cmap="RdBu_r", vmin=-1.1, vmax=1.1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        q = Q.values[i, j]; st = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
        ax.text(j, i - 0.12, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                color="white" if abs(np.log2(H.values[i, j])) > .75 else "black")
        ax.text(j, i + 0.27, st, ha="center", va="center", fontsize=4.6,
                color="white" if abs(np.log2(H.values[i, j])) > .75 else "black")
ax.set_xticks(range(H.shape[1]))
ax.set_xticklabels([c.replace("TCGA-LUAD", "TCGA").replace("GSE", "") for c in order] + ["Meta"], fontsize=5.2, rotation=60, ha="right")
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(HUB7, fontstyle="italic")
ax.axvline(len(order) - 0.5, c="k", lw=.8)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("log$_2$ HR per SD", fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.set_title("Single genes, OS (multivariable)", loc="left", fontsize=7)
ax.text(0, -0.3, "Columns: GEO series (GSE prefix omitted). * FDR < 0.05, ** < 0.01, *** < 0.001", transform=ax.transAxes, fontsize=5.0)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
panel(ax, "B", x=-0.25)

# ── C KMs in GEO ──
mid = G[1].subgridspec(1, 3, wspace=0.35)
DAT = EV / "data"
for j, gse in enumerate(["GSE31210", "GSE68465", "GSE72094"]):
    ax = fig.add_subplot(mid[j])
    E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
    Z = (E.loc[C.index, HUB7] - E.loc[C.index, HUB7].mean()) / E.loc[C.index, HUB7].std()
    d = pd.DataFrame({"T": C.OS_days / 365.25, "E": C.OS_event, "s": Z.mean(1)}).dropna()
    d.loc[d["T"] > 10, "E"] = 0; d["T"] = d["T"].clip(upper=10)
    hi = d.s > d.s.median()
    tmax = 6 if gse == "GSE72094" else 10
    k1, k0 = KaplanMeierFitter(), KaplanMeierFitter()
    k1.fit(d["T"][hi], d.E[hi], label="High").plot_survival_function(ax=ax, ci_show=False, color=RED, lw=1)
    k0.fit(d["T"][~hi], d.E[~hi], label="Low").plot_survival_function(ax=ax, ci_show=False, color=BLUE, lw=1)
    p = logrank_test(d["T"][hi], d["T"][~hi], d.E[hi], d.E[~hi]).p_value
    ax.set(xlim=(0, tmax), ylim=(0, 1.03), xlabel="Years", ylabel="Overall survival" if j == 0 else "")
    ax.set_title(f"{gse} (n = {len(d)}; {int(d.E.sum())} deaths ≤ 10 y)", fontsize=6.4)
    ax.text(0.04, 0.05, f"log-rank {pfmt(p)}", transform=ax.transAxes, fontsize=5.8)
    ax.legend(loc="lower right", title="7-gene score", title_fontsize=5.6, fontsize=5.6)
    ticks = list(range(0, tmax + 1, 2)); ax.set_xticks(ticks)
    for lab_, msk, c_, yy_ in [("High", hi, RED, -0.33), ("Low", ~hi, BLUE, -0.41)]:
        if j == 0: ax.text(-0.13, yy_, lab_, transform=ax.transAxes, ha="right", fontsize=5.4, color=c_)
        for t_ in ticks:
            ax.text(t_, yy_, str(int((d["T"][msk] >= t_).sum())), transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.4, color=c_)
    if j == 0:
        ax.text(-0.13, -0.25, "At risk", transform=ax.transAxes, ha="right", fontsize=5.4, fontweight="bold")
        panel(ax, "C", x=-0.42)

# ── D prognostic models ──
ax = fig.add_subplot(G[2])
cv = pd.read_csv(NEW / "09_ML" / "B_prognostic_cv_summary.csv", index_col=0)
ex = pd.read_csv(NEW / "09_ML" / "B_prognostic_external_locked_models.csv")
kinds = ["Cox clinical", "Cox clinical + HUB7 score", "Elastic-net Cox clinical + 7 genes",
         "Random survival forest clinical + 7 genes"]
lab = {"Cox clinical": "Clinical Cox (age, sex, stage)", "Cox clinical + HUB7 score": "Clinical + 7-gene score",
       "Elastic-net Cox clinical + 7 genes": "Elastic-net Cox, clinical + 7 genes",
       "Random survival forest clinical + 7 genes": "Random survival forest, clinical + 7 genes"}
cols = [GREY, RED, "#e67e22", "#8e44ad"]
cohorts = ["TCGA\n(repeated 5-fold CV)", "GSE31210", "GSE68465", "GSE72094"]
for i, k_ in enumerate(kinds):
    v = [cv.loc[k_, "C_mean"]]; l = [cv.loc[k_, "C_lo"]]; h = [cv.loc[k_, "C_hi"]]
    for g_ in ["GSE31210", "GSE68465", "GSE72094"]:
        rr = ex[(ex.cohort == g_) & (ex.model == k_)].iloc[0]; v.append(rr.C); l.append(rr.C_lo); h.append(rr.C_hi)
    x = np.arange(4) + (i - 1.5) * 0.17
    ax.errorbar(x, v, yerr=[np.array(v) - l, np.array(h) - v], fmt="o", ms=3.2, color=cols[i], capsize=1.5, lw=.9, label=lab[k_])
ax.set_xticks(range(4)); ax.set_xticklabels(cohorts)
ax.set_ylabel("Harrell C-index (OS)"); ax.set_ylim(0.5, 0.85); ax.axhline(0.5, c="#bbbbbb", lw=.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, columnspacing=1.0, handletextpad=0.2, fontsize=5.8)
ax.set_title("Prediction models fitted in TCGA and applied unchanged to validation cohorts", loc="left", fontsize=7)
panel(ax, "D", x=-0.07)

save(fig, FIG_MAIN / "Figure4_external_validation")
print("saved")
