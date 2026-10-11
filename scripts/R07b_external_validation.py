"""
R07b_external_validation.py — Revision step 7b (PI comments M5, M6, M4)
=========================================================================
External validation of the locked 7-gene signature in three independent GEO cohorts
(tumours only) + random-effects meta-analysis with TCGA-LUAD.

Locked scores (nothing re-estimated in the validation cohorts)
  HUB7_mean   pre-specified: mean z-score of the 7 genes (z within cohort; zero fitted parameters)
  HUB7_risk   TCGA-trained ridge Cox (OS, penalizer 0.1) on the 7 z-scored genes; coefficients locked
              and applied unchanged to each GEO cohort
Comparators: OLD5, MKI67, PAM50_prolif, PCNA_meta, HALLMARK_G2M, HALLMARK_E2F (definitions from R06)

Per cohort and endpoint (OS everywhere; RFS GSE31210; PFS GSE68465)
  * HR per 1 SD (univariate; multivariable age + sex + stage; GSE68465 stratified by site)
  * sensitivity: + smoking; GSE31210 excluding incomplete resection/adjuvant therapy
  * Harrell C; delta C over clinical model (bootstrap 95% CI) and LRT
  * time-dependent AUC (cumulative/dynamic, IPCW; Uno) at 3 and 5 years
  * HUB7_mean percentile among 1,000 random 7-gene sets (N1 expression-matched up-DEGs, N2 G2M ∪ E2F)
  * each hub gene: HR per SD (BH within cohort)
Meta-analysis: DerSimonian–Laird random effects over TCGA + 3 GEO (OS), with I².
Diagnosis: GSE31210 tumour (226) vs adjacent normal lung (20), AUC with bootstrap CI and null percentile.
Outputs: analysis/07_external_validation/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2, norm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "07_external_validation"; DAT = OUT / "data"
SEED, N_NULL, N_BOOT = 42, 1000, 300
rng = np.random.default_rng(SEED)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
sigdef = pd.read_csv(NEW / "06_benchmark" / "signature_definitions.csv").set_index("signature")
SIGS = {k: v.split(";") for k, v in sigdef.genes.items()}
fmt = lambda v: f"{v:.3g}"

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

# ═════════ TCGA training cohort ═════════
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
tc = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
tcga_expr = lc[tc.index].T
tcga_clin = pd.DataFrame({"tumour": 1, "age": tc.age, "male": (tc.sex == "Male").astype(float),
                          "ever_smoker": tc.ever_smoker, "stage": tc.stage, "site": "TCGA",
                          "OS_event": tc.OS_event, "OS_days": tc.OS_days,
                          "PFI_event": tc.PFI_event, "PFI_days": tc.PFI_days}, index=tc.index)

def zscore(E):
    return (E - E.mean()) / E.std().replace(0, np.nan)

# locked ridge-Cox weights
Zt = zscore(tcga_expr[HUB7])
f = pd.concat([tcga_clin.OS_days.rename("T"), tcga_clin.OS_event.rename("E"), Zt], axis=1).dropna()
ridge = CoxPHFitter(penalizer=0.1, l1_ratio=0.0).fit(f, "T", "E")
W = ridge.params_[HUB7]
W.rename("locked_coef").to_csv(OUT / "locked_ridge_cox_coefficients_TCGA.csv")

def scores(E):
    Z = zscore(E)
    s = {"HUB7_mean": Z[HUB7].mean(1), "HUB7_risk": Z[HUB7] @ W}
    for k, genes in SIGS.items():
        if k == "HUB7": continue
        g = [x for x in genes if x in Z.columns]
        s[k] = Z[g].mean(1)
        s[f"{k}__n"] = len(g)
    return pd.DataFrame({k: v for k, v in s.items() if not k.endswith("__n")}), \
           {k: v for k, v in s.items() if k.endswith("__n")}

# ═════════ cohorts ═════════
cohorts = {"TCGA-LUAD": (tcga_expr, tcga_clin, {"OS": "OS", "PFI": "PFI"})}
for gse, eps in [("GSE31210", {"OS": "OS", "RFS": "RFS"}), ("GSE68465", {"OS": "OS", "PFS": "PFS"}),
                 ("GSE72094", {"OS": "OS"})]:
    E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0)
    cohorts[gse] = (E, C, eps)

# DEG pool for N1 null (TCGA up-DEGs, decile of TCGA mean expression)
deg = pd.read_csv(ROOT / "results" / "deg_all_results.csv")
up = [g for g in deg[(deg.padj < 0.05) & (deg.log2FoldChange > 1)].gene_name.dropna().unique()
      if g in tcga_expr.columns and g not in HUB7]
dec = pd.qcut(tcga_expr[up + HUB7].mean(), 10, labels=False)
dec_d = dec.to_dict()
up_by_dec = {k: [g for g in up if dec_d[g] == k] for k in range(10)}
prolif = set(SIGS["HALLMARK_G2M"]) | set(SIGS["HALLMARK_E2F"])

def cox(df, cols, strata=None, pen=0.0):
    m = CoxPHFitter(penalizer=pen).fit(df, "T", "E", strata=strata)
    return m

def design(C, ep, cols, extra=()):
    d = pd.DataFrame({"T": C[f"{ep}_days"], "E": C[f"{ep}_event"]})
    for c in cols: d[c] = C[c]
    st = C.stage
    for s in ["II", "III", "IV"]:
        if (st == s).sum() >= 5: d[f"stage_{s}"] = (st == s).astype(float).where(st.notna())
    for e in extra: d[e] = C[e]
    return d

perf, genes_out, ta_rows, null_rows, sens_rows, km_data = [], [], [], [], [], {}
for name, (E, C, eps) in cohorts.items():
    T = C[C.tumour == 1].index.intersection(E.index)
    Et, Ct = E.loc[T], C.loc[T]
    S, nmap = scores(Et)
    Zg = zscore(Et[HUB7])
    strata = ["site"] if Ct.site.nunique() > 1 else None
    for ep_label, ep in eps.items():
        base_cols = ["age", "male"]
        for sig in S.columns:
            d = design(Ct, ep, base_cols).join((S[sig] - S[sig].mean()) / S[sig].std()).rename(columns={sig: "s"})
            if strata: d["site"] = Ct.site
            d = d.dropna()
            mu = cox(d[["T", "E", "s"] + (["site"] if strata else [])], None, strata)
            mm = cox(d, None, strata)
            mc = cox(d.drop(columns="s"), None, strata)
            c_uni = concordance_index(d["T"], -d["s"], d["E"])
            c_cl = concordance_index(d["T"], -mc.predict_partial_hazard(d), d["E"])
            c_mm = concordance_index(d["T"], -mm.predict_partial_hazard(d), d["E"])
            row = dict(cohort=name, endpoint=ep_label, signature=sig, n=len(d), events=int(d.E.sum()),
                       HR_uni=np.exp(mu.params_["s"]), lo_uni=np.exp(mu.confidence_intervals_.loc["s"].iloc[0]),
                       hi_uni=np.exp(mu.confidence_intervals_.loc["s"].iloc[1]), p_uni=mu.summary.loc["s", "p"],
                       se_logHR_uni=mu.standard_errors_["s"],
                       HR_multi=np.exp(mm.params_["s"]), lo_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[0]),
                       hi_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[1]), p_multi=mm.summary.loc["s", "p"],
                       se_logHR_multi=mm.standard_errors_["s"],
                       C_uni=c_uni, C_clinical=c_cl, C_clinical_plus_sig=c_mm, delta_C=c_mm - c_cl,
                       LRT_p=chi2.sf(2 * (mm.log_likelihood_ - mc.log_likelihood_), 1),
                       n_genes_present=nmap.get(f"{sig}__n", 7))
            if sig == "HUB7_mean":
                bs = []
                for _ in range(N_BOOT):
                    b = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
                    try:
                        m1, m0 = cox(b, None, strata), cox(b.drop(columns="s"), None, strata)
                        bs.append(concordance_index(b["T"], -m1.predict_partial_hazard(b), b["E"]) -
                                  concordance_index(b["T"], -m0.predict_partial_hazard(b), b["E"]))
                    except Exception: pass
                row.update(delta_C_lo=np.percentile(bs, 2.5), delta_C_hi=np.percentile(bs, 97.5))
                # time-dependent AUC
                try:
                    from sksurv.metrics import cumulative_dynamic_auc
                    from sksurv.util import Surv
                    y = Surv.from_arrays(d.E.astype(bool), d["T"])
                    tmax = d["T"][d.E == 1].max()
                    for yr in [3, 5]:
                        t = yr * 365.25
                        if t < tmax and (d["T"] > t).sum() > 10:
                            auc, _ = cumulative_dynamic_auc(y, y, d["s"].values, [t])
                            ta_rows.append(dict(cohort=name, endpoint=ep_label, year=yr, AUC=float(auc[0])))
                except ImportError:
                    pass
                # nulls
                Za = zscore(Et)
                pools = {h: [g for g in up_by_dec[dec_d[h]] if g in Za.columns] for h in HUB7}
                ppool = [g for g in prolif if g in Za.columns and g not in HUB7]
                for nul in ["N1_expr_matched_upDEG", "N2_G2M_E2F"]:
                    cs = []
                    for _ in range(N_NULL):
                        if nul.startswith("N1"):
                            gs = [rng.choice(pools[h]) for h in HUB7]
                        else:
                            gs = list(rng.choice(ppool, 7, replace=False))
                        sc = Za.loc[d.index, list(dict.fromkeys(gs))].mean(1)
                        cs.append(concordance_index(d["T"], -sc, d["E"]))
                    cs = np.array(cs)
                    null_rows.append(dict(cohort=name, endpoint=ep_label, null=nul, HUB7_C=c_uni,
                                          null_median=np.median(cs), null_95th=np.percentile(cs, 95),
                                          percentile=100 * (cs < c_uni).mean(),
                                          empirical_p=(1 + (cs >= c_uni).sum()) / (1 + len(cs))))
                km_data[(name, ep_label)] = d[["T", "E", "s"]]
            perf.append(row)
        # sensitivity analyses for HUB7_mean
        for label, extra, mask in [("+ smoking", ["ever_smoker"], None),
                                   ("complete resection, no adjuvant (GSE31210)", [], "exclude_prognosis")]:
            if mask and mask not in Ct: continue
            Cs = Ct[Ct[mask] == 0] if mask else Ct
            s = S.loc[Cs.index, "HUB7_mean"]
            d = design(Cs, ep, ["age", "male"], extra).join(((s - s.mean()) / s.std()).rename("s"))
            if strata: d["site"] = Cs.site
            d = d.dropna()
            if d.E.sum() < 10: continue
            m = cox(d, None, strata)
            sens_rows.append(dict(cohort=name, endpoint=ep_label, analysis=label, n=len(d), events=int(d.E.sum()),
                                  HR=np.exp(m.params_["s"]), lo=np.exp(m.confidence_intervals_.loc["s"].iloc[0]),
                                  hi=np.exp(m.confidence_intervals_.loc["s"].iloc[1]), p=m.summary.loc["s", "p"]))
        # individual genes
        for g in HUB7:
            d = design(Ct, ep, ["age", "male"]).join(Zg[g].rename("s"))
            if strata: d["site"] = Ct.site
            d = d.dropna()
            mu = cox(d[["T", "E", "s"] + (["site"] if strata else [])], None, strata); mm = cox(d, None, strata)
            genes_out.append(dict(cohort=name, endpoint=ep_label, gene=g, n=len(d), events=int(d.E.sum()),
                                  HR_uni=np.exp(mu.params_["s"]), p_uni=mu.summary.loc["s", "p"],
                                  se_uni=mu.standard_errors_["s"],
                                  HR_multi=np.exp(mm.params_["s"]), lo_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[0]),
                                  hi_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[1]), p_multi=mm.summary.loc["s", "p"],
                                  se_multi=mm.standard_errors_["s"]))
    print(name, "done", flush=True)

perf = pd.DataFrame(perf); perf.to_csv(OUT / "signature_performance_all_cohorts.csv", index=False)
gdf = pd.DataFrame(genes_out)
for (c, e), idx in gdf.groupby(["cohort", "endpoint"]).groups.items():
    gdf.loc[idx, "FDR_uni"] = bh(gdf.loc[idx, "p_uni"]); gdf.loc[idx, "FDR_multi"] = bh(gdf.loc[idx, "p_multi"])
gdf.to_csv(OUT / "individual_genes_all_cohorts.csv", index=False)
ta = pd.DataFrame(ta_rows); ta.to_csv(OUT / "time_dependent_AUC_HUB7_mean.csv", index=False)
nl = pd.DataFrame(null_rows); nl.to_csv(OUT / "HUB7_vs_random_sets_all_cohorts.csv", index=False)
sens = pd.DataFrame(sens_rows); sens.to_csv(OUT / "sensitivity_HUB7_mean.csv", index=False)

# ═════════ meta-analysis (OS) ═════════
def dl_meta(y, se):
    w = 1 / se ** 2; yf = (w * y).sum() / w.sum()
    Q = (w * (y - yf) ** 2).sum(); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se ** 2 + tau2); yr = (wr * y).sum() / wr.sum(); ser = np.sqrt(1 / wr.sum())
    I2 = max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0
    return dict(HR=np.exp(yr), lo=np.exp(yr - 1.96 * ser), hi=np.exp(yr + 1.96 * ser),
                p=2 * norm.sf(abs(yr / ser)), I2=I2, Q_p=chi2.sf(Q, k - 1), tau2=tau2, k=k)
meta = []
os_ = perf[perf.endpoint == "OS"]
for sig in os_.signature.unique():
    for typ in ["uni", "multi"]:
        r = os_[os_.signature == sig]
        meta.append(dict(item=sig, model=typ, **dl_meta(np.log(r[f"HR_{typ}"].values), r[f"se_logHR_{typ}"].values)))
gos = gdf[gdf.endpoint == "OS"]
for g in HUB7:
    for typ in ["uni", "multi"]:
        r = gos[gos.gene == g]
        meta.append(dict(item=g, model=typ, **dl_meta(np.log(r[f"HR_{typ}"].values), r[f"se_{typ}"].values)))
meta = pd.DataFrame(meta); meta.to_csv(OUT / "meta_analysis_OS_random_effects.csv", index=False)

# ═════════ diagnostic: GSE31210 tumour vs normal ═════════
E, C, _ = cohorts["GSE31210"]
S_all, _ = scores(E.loc[C.index])
y = C.tumour.values
auc = roc_auc_score(y, S_all.HUB7_mean)
bt = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if len(set(y[i])) == 2: bt.append(roc_auc_score(y[i], S_all.HUB7_mean.values[i]))
Za = zscore(E.loc[C.index])
pools = {h: [g for g in up_by_dec[dec_d[h]] if g in Za.columns] for h in HUB7}
null_auc = [roc_auc_score(y, Za[list(dict.fromkeys(rng.choice(pools[h]) for h in HUB7))].mean(1))
            for _ in range(N_NULL)]
diag = pd.DataFrame([dict(cohort="GSE31210", n_tumour=int(y.sum()), n_normal=int((1 - y).sum()), AUC=auc,
                          lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5),
                          null_median=np.median(null_auc), percentile_vs_N1=100 * np.mean(np.array(null_auc) < auc),
                          **{f"AUC_{k}": roc_auc_score(y, S_all[k]) for k in S_all.columns if k != "HUB7_mean"})])
diag.to_csv(OUT / "diagnostic_GSE31210_tumour_vs_normal.csv", index=False)

# ═════════ figures ═════════
# 1. forest: HUB7_mean multivariable per cohort/endpoint + meta
rows = perf[perf.signature == "HUB7_mean"].reset_index(drop=True)
m = meta[(meta.item == "HUB7_mean") & (meta.model == "multi")].iloc[0]
labels = [f"{r.cohort} — {r.endpoint}  (n={r.n}, {r.events} ev)" for r in rows.itertuples()] + \
         [f"Random-effects meta, OS (k={int(m.k)}; I²={m.I2:.0f}%)"]
hr = list(rows.HR_multi) + [m.HR]; lo = list(rows.lo_multi) + [m.lo]; hi = list(rows.hi_multi) + [m.hi]
pv = list(rows.p_multi) + [m.p]
fig, ax = plt.subplots(figsize=(10, 0.55 * len(hr) + 1.6))
yy = np.arange(len(hr))[::-1]
for i, (h, l, u, p) in enumerate(zip(hr, lo, hi, pv)):
    meta_row = i == len(hr) - 1
    ax.errorbar(h, yy[i], xerr=[[h - l], [u - h]], fmt="D" if meta_row else "s", ms=8 if meta_row else 6,
                color="#2c3e50" if meta_row else ("#c0392b" if "TCGA" in labels[i] else "#2980b9"), capsize=3)
    ax.text(1.02, yy[i], f"{h:.2f} ({l:.2f}–{u:.2f})  p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
            transform=ax.get_yaxis_transform(), va="center", fontsize=9)
ax.axvline(1, ls="--", c="k", lw=.7); ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("HR per 1 SD of 7-gene score (95% CI), adjusted for age, sex, stage\n(GSE68465 stratified by site)")
ax.set_title("Locked 7-gene score: training (TCGA, red) and external validation (GEO, blue)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(rect=(0, 0, 0.78, 1)); plt.savefig(OUT / "Fig_forest_HUB7_external_validation.png", dpi=300); plt.close()

# 2. gene x cohort heatmap (OS, multivariable HR)
hm = gos.pivot(index="gene", columns="cohort", values="HR_multi").loc[HUB7]
fd = gos.pivot(index="gene", columns="cohort", values="FDR_multi").loc[HUB7]
ann = hm.round(2).astype(str) + np.where(fd < 0.001, "***", np.where(fd < 0.01, "**", np.where(fd < 0.05, "*", "")))
import seaborn as sns
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(np.log2(hm), annot=ann, fmt="", cmap="RdBu_r", center=0, vmin=-.6, vmax=.6, ax=ax,
            cbar_kws={"label": "log2 HR per SD (OS, multivariable)"})
ax.set_yticklabels(ax.get_yticklabels(), fontstyle="italic", rotation=0); ax.set_xlabel(""); ax.set_ylabel("")
ax.set_title("Hub genes and OS across cohorts (* FDR<0.05, ** <0.01, *** <0.001)")
plt.tight_layout(); plt.savefig(OUT / "Fig_genes_by_cohort_OS.png", dpi=300); plt.close()

# 3. KM per cohort/endpoint (median split, 10 y)
keys = list(km_data)
nc = 3; nr = int(np.ceil(len(keys) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(6.5 * nc, 5.6 * nr))
for ax, k in zip(axes.flat, keys):
    d = km_data[k].copy(); lim = 3652.5
    d.loc[d["T"] > lim, "E"] = 0; d["T"] = d["T"].clip(upper=lim) / 365.25
    hi_ = d.s > d.s.median()
    k1, k0 = KaplanMeierFitter(), KaplanMeierFitter()
    k1.fit(d["T"][hi_], d.E[hi_], label="High score").plot_survival_function(ax=ax, ci_show=False, color="#c0392b")
    k0.fit(d["T"][~hi_], d.E[~hi_], label="Low score").plot_survival_function(ax=ax, ci_show=False, color="#2980b9")
    p = logrank_test(d["T"][hi_], d["T"][~hi_], d.E[hi_], d.E[~hi_]).p_value
    ax.set(title=f"{k[0]} — {k[1]}   log-rank p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
           xlabel="Years", ylabel="Survival probability", ylim=(0, 1.02), xlim=(0, 10))
    add_at_risk_counts(k1, k0, ax=ax, rows_to_show=["At risk"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in list(axes.flat)[len(keys):]: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "Fig_KM_HUB7_all_cohorts.png", dpi=300); plt.close()

# ═════════ report ═════════
cols = ["cohort", "endpoint", "signature", "n", "events", "HR_uni", "p_uni", "HR_multi", "lo_multi", "hi_multi",
        "p_multi", "C_uni", "C_clinical", "C_clinical_plus_sig", "delta_C", "delta_C_lo", "delta_C_hi", "LRT_p"]
rep = ["R07 EXTERNAL VALIDATION", "=" * 70,
       "Locked ridge-Cox weights (TCGA OS): " + ", ".join(f"{g} {w:+.3f}" for g, w in W.items()),
       "\nHUB7 scores in every cohort:",
       perf[perf.signature.isin(["HUB7_mean", "HUB7_risk"])][cols].to_string(index=False, float_format=fmt),
       "\nAll signatures, OS, Harrell C of score alone:",
       perf[perf.endpoint == "OS"].pivot(index="signature", columns="cohort", values="C_uni").to_string(float_format=fmt),
       "\nAll signatures, OS, multivariable HR per SD:",
       perf[perf.endpoint == "OS"].pivot(index="signature", columns="cohort", values="HR_multi").to_string(float_format=fmt),
       "\nHUB7_mean vs 1,000 random 7-gene sets (C of score alone):", nl.to_string(index=False, float_format=fmt),
       "\nTime-dependent AUC (HUB7_mean):", ta.to_string(index=False, float_format=fmt),
       "\nSensitivity (HUB7_mean, multivariable):", sens.to_string(index=False, float_format=fmt),
       "\nIndividual genes, OS multivariable HR per SD (FDR):",
       gos.assign(v=gos.apply(lambda r: f"{r.HR_multi:.2f} ({r.FDR_multi:.3g})", axis=1))
          .pivot(index="gene", columns="cohort", values="v").loc[HUB7].to_string(),
       "\nRandom-effects meta-analysis, OS (TCGA + 3 GEO):", meta.to_string(index=False, float_format=fmt),
       "\nDiagnostic, GSE31210 tumour vs normal:", diag.T.to_string(float_format=fmt)]
(OUT / "R07_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
