"""R12_fig3_prognosis_benchmark.py — Main Figure 3 (600 dpi).
A  OS hazard ratios per SD (univariate / multivariable) for the 7 genes and the 7-gene score, TCGA-LUAD
B  Kaplan-Meier OS by 7-gene score (median split; illustration), numbers at risk
C  7-gene score vs 2 x 1,000 random 7-gene sets (OS Harrell C), with comparator signatures
D  Comparator signatures: C-index and nested likelihood-ratio test (does the 7-gene score add?)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from R12_figstyle import NEW, FIG_MAIN, W2, RED, BLUE, GREY, DARK, panel, save, pfmt

HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
res = pd.read_csv(NEW / "05_survival" / "cox_uni_multi_per_SD_OS_PFI_DSS.csv")
X = pd.read_csv(NEW / "05_survival" / "hub_expression_log2CPM_tumours.csv", index_col=0)
clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
Z = (X - X.mean()) / X.std(); s = Z[HUB7].mean(1); s = (s - s.mean()) / s.std()
d = clin.join(s.rename("score"))
d["age10"] = d.age / 10; d["male"] = (d.sex == "Male").astype(float)
for st in ["II", "III", "IV"]: d[f"st_{st}"] = (d.stage == st).astype(float).where(d.stage.notna())
COV = ["age10", "male", "st_II", "st_III", "st_IV", "ever_smoker"]
fu = d[["OS_days", "OS_event", "score"]].dropna(); mu = CoxPHFitter().fit(fu, "OS_days", "OS_event")
fm = d[["OS_days", "OS_event", "score"] + COV].dropna(); mm = CoxPHFitter().fit(fm, "OS_days", "OS_event")
srow = dict(gene="7-gene score", uni_HR=np.exp(mu.params_.score), uni_lo=np.exp(mu.confidence_intervals_.loc["score"].iloc[0]),
            uni_hi=np.exp(mu.confidence_intervals_.loc["score"].iloc[1]),
            multi_HR=np.exp(mm.params_.score), multi_lo=np.exp(mm.confidence_intervals_.loc["score"].iloc[0]),
            multi_hi=np.exp(mm.confidence_intervals_.loc["score"].iloc[1]), multi_p=mm.summary.loc["score", "p"],
            multi_FDR=mm.summary.loc["score", "p"])

fig = plt.figure(figsize=(W2, 6.9))
G = gs.GridSpec(2, 2, figure=fig, wspace=0.78, hspace=0.70, width_ratios=[1.0, 1.0])

# ── A forest ──
ax = fig.add_subplot(G[0, 0])
r = res[res.endpoint == "OS"].set_index("gene").loc[HUB7].reset_index()
r = pd.concat([r, pd.DataFrame([srow])], ignore_index=True)
yy = np.arange(len(r))[::-1].astype(float); yy[-1] -= 0.4
for i, row in r.iterrows():
    is_s = row.gene == "7-gene score"
    ax.errorbar(row.uni_HR, yy[i] + .17, xerr=[[row.uni_HR - row.uni_lo], [row.uni_hi - row.uni_HR]], fmt="o",
                ms=3, color=GREY, capsize=1.5, lw=.8)
    ax.errorbar(row.multi_HR, yy[i] - .17, xerr=[[row.multi_HR - row.multi_lo], [row.multi_hi - row.multi_HR]],
                fmt="D" if is_s else "s", ms=3.5 if is_s else 3, color=DARK if is_s else RED, capsize=1.5, lw=.9)
    q = row.multi_FDR
    ax.text(1.02, yy[i] - .17, f"{row.multi_HR:.2f} ({row.multi_lo:.2f}–{row.multi_hi:.2f})", transform=ax.get_yaxis_transform(),
            va="center", fontsize=5.8)
ax.axvline(1, ls="--", c="k", lw=.5)
ax.axhline(yy[-1] + 0.7, c="#bbbbbb", lw=.5)
ax.set_yticks(yy); ax.set_yticklabels(r.gene)
for t in ax.get_yticklabels():
    t.set_fontstyle("normal" if t.get_text() == "7-gene score" else "italic")
    if t.get_text() == "7-gene score": t.set_fontweight("bold")
ax.set_xlim(0.95, 1.8)
ax.set_xlabel("Overall survival, HR per 1 SD (95% CI)")
ax.set_title("TCGA-LUAD, overall survival", loc="left", fontsize=7)
ax.legend([Line2D([], [], marker="o", color=GREY, ls="", ms=3), Line2D([], [], marker="s", color=RED, ls="", ms=3)],
          ["Univariate", f"Multivariable* (n = {len(fm)}, {int(fm.OS_event.sum())} deaths)"],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.0, handletextpad=0.3)
ax.text(1.02, 1.01, "Multivariable HR", transform=ax.transAxes, fontsize=5.8, fontweight="bold", va="bottom")
panel(ax, "A", x=-0.32)

# ── B KM ──
ax = fig.add_subplot(G[0, 1])
k = d[["OS_days", "OS_event", "score"]].dropna().copy()
lim = 3652.5; k.loc[k.OS_days > lim, "OS_event"] = 0; k["T"] = k.OS_days.clip(upper=lim) / 365.25
hi = k.score > k.score.median()
k1, k0 = KaplanMeierFitter(), KaplanMeierFitter()
k1.fit(k["T"][hi], k.OS_event[hi], label="High score").plot_survival_function(ax=ax, ci_show=True, color=RED, lw=1, ci_alpha=.12)
k0.fit(k["T"][~hi], k.OS_event[~hi], label="Low score").plot_survival_function(ax=ax, ci_show=True, color=BLUE, lw=1, ci_alpha=.12)
p = logrank_test(k["T"][hi], k["T"][~hi], k.OS_event[hi], k.OS_event[~hi]).p_value
ax.text(0.97, 0.95, f"log-rank {pfmt(p)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3)
ax.set(xlabel="Years", ylabel="Overall survival probability", ylim=(0, 1.02), xlim=(0, 10),
       title="7-gene score, median split (TCGA-LUAD)")
ax.legend(loc="lower left")
ticks = [0, 2, 4, 6, 8, 10]
ax.set_xticks(ticks)
ax.text(-0.09, -0.25, "At risk", transform=ax.transAxes, ha="right", fontsize=5.6, fontweight="bold")
for lab_, m_, c_, yy_ in [("High", hi, RED, -0.31), ("Low", ~hi, BLUE, -0.37)]:
    ax.text(-0.09, yy_, lab_, transform=ax.transAxes, ha="right", fontsize=5.6, color=c_)
    for t_ in ticks:
        n_ = int((k["T"][m_] >= t_).sum()) if t_ > 0 else int(m_.sum())
        ax.text(t_, yy_, str(n_), transform=ax.get_xaxis_transform(), ha="center", fontsize=5.6, color=c_)
panel(ax, "B", x=-0.22)

# ── C null distributions ──
ax = fig.add_subplot(G[1, 0])
null = pd.read_csv(NEW / "06_benchmark" / "null_distributions_1000x2.csv")
obs = pd.read_csv(NEW / "06_benchmark" / "signature_performance.csv", index_col=0)
pct = pd.read_csv(NEW / "06_benchmark" / "HUB7_vs_null_empirical.csv")
for (name, g), c, lab in zip(null.groupby("null"), ["#bdc3c7", "#f5b041"],
                             ["Random up-regulated DEGs\n(expression-matched)", "Random G2M/E2F\nproliferation genes"]):
    ax.hist(g.C_uni, bins=35, alpha=.75, color=c, label=lab)
for kk, c, ls in [("HUB7", RED, "-"), ("PAM50_prolif", "#8e44ad", "--"), ("MKI67", "k", ":"), ("HALLMARK_G2M", DARK, "-.")]:
    ax.axvline(obs.loc[kk, "C_uni"], color=c, ls=ls, lw=1.1)
ax.set(xlabel="OS Harrell C-index of score (TCGA-LUAD)", ylabel="Random 7-gene sets (n = 1,000 each)")
p1 = pct[(pct.null.str.startswith("N1")) & (pct.metric == "C_uni")].iloc[0]
p2 = pct[(pct.null.str.startswith("N2")) & (pct.metric == "C_uni")].iloc[0]
ax.set_title(f"Percentile vs random sets: {p1.HUB7_percentile:.0f}th (p = {p1.empirical_p:.3f}) / "
             f"{p2.HUB7_percentile:.0f}th (p = {p2.empirical_p:.3f})", fontsize=6.0, loc="center")
h1, l1 = ax.get_legend_handles_labels()
h2 = [Line2D([], [], color=c, ls=ls, lw=1.1) for c, ls in [(RED, "-"), ("#8e44ad", "--"), ("k", ":"), (DARK, "-.")]]
ax.legend(h1 + h2, l1 + ["7-gene score", "PAM50 proliferation", "MKI67", "Hallmark G2M"],
          loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.4, handlelength=1.6)
ax.set_xlim(0.42, 0.66); ax.set_ylim(0, 150)
panel(ax, "C", x=-0.26, y=1.10)

# ── D: paired-bootstrap difference in C-index vs each comparator ──
ax = fig.add_subplot(G[1, 1])
bt = pd.read_csv(NEW / "06_benchmark" / "paired_bootstrap_deltaC_HUB7_vs_comparators.csv", index_col=0)
nest = pd.read_csv(NEW / "06_benchmark" / "nested_LRT_HUB7_vs_proliferation.csv").set_index("comparator")
names = {"OLD5": "Original 5 genes", "PAM50_prolif": "PAM50 proliferation", "PCNA_meta": "PCNA metagene",
         "MKI67": "MKI67", "HALLMARK_G2M": "Hallmark G2M", "HALLMARK_E2F": "Hallmark E2F"}
o = bt.loc[list(names)].sort_values("delta_C")
for i, (k_, b) in enumerate(o.iterrows()):
    sig = b.lo > 0
    ax.errorbar(b.delta_C, i, xerr=[[b.delta_C - b.lo], [b.hi - b.delta_C]], fmt="o", ms=3.5,
                color=RED if sig else GREY, capsize=1.8, lw=.9)
    lr = nest.loc[k_, "p_HUB7_adds"]
    ax.text(1.02, i, ("<0.001" if lr < .001 else f"{lr:.3f}"), transform=ax.get_yaxis_transform(), va="center", fontsize=5.8)
ax.text(1.02, len(o) - 0.35, "LRT p", transform=ax.get_yaxis_transform(), fontsize=5.8, fontweight="bold", va="bottom")
ax.set_yticks(range(len(o)))
ax.set_yticklabels([f"{names[k_]} (C = {obs.loc[k_, 'C_uni']:.3f})" for k_ in o.index])
ax.axvline(0, ls="--", c="k", lw=.5)
ax.set_xlabel("ΔC-index: 7-gene score minus comparator\n(95% paired-bootstrap CI)")
ax.set_title(f"7-gene score (C = {obs.loc['HUB7', 'C_uni']:.3f}) vs proliferation signatures", loc="right", fontsize=6.8)
ax.set_xlim(-0.02, 0.055)
panel(ax, "D", x=-0.72, y=1.10)

fig.text(0.01, 0.0, "*Adjusted for age, sex, pathological stage (II, III, IV vs I) and smoking; each gene modelled separately. "
         "In D, red = 95% CI excludes zero; LRT p = likelihood-ratio test for adding the 7-gene score to a Cox model containing age, sex, stage, smoking and the comparator.",
         fontsize=5.5, ha="left", va="top", wrap=True)
save(fig, FIG_MAIN / "Figure3_prognosis_benchmark")
print("saved")
