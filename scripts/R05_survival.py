"""
R05_survival.py — Revision step 5 (PI comment M5)
==================================================
Survival analysis of the 7 locked hub genes in TCGA-LUAD tumours.

Expression : log2(CPM + 1) from filtered STAR counts, tumour samples only,
             standardised (z-score) within tumours  ->  every HR is "per 1 SD increase"
Clinical   : analysis/04_clinical/clinical_tumour_harmonised.csv (TCGA-CDR endpoints)
Endpoints  : OS (primary), PFI, DSS
Models     : (1) univariate Cox, continuous expression
             (2) multivariable Cox: gene + age (per 10 y) + sex + stage (II, III, IV vs I) + ever-smoker
                 each gene modelled SEPARATELY (genes are collinear, r ~ 0.9)
             (3) added value over clinical model: likelihood-ratio test and delta Harrell C (bootstrap 95% CI)
             (4) non-linearity: restricted cubic spline (4 knots) vs linear, LRT
             (5) proportional hazards: Schoenfeld-residual test (rank time) for every multivariable model
             (6) sensitivity: OS administratively censored at 10 years
Multiple testing: Benjamini-Hochberg across the 7 genes within each endpoint/model.
KM curves  : median split, illustration only, follow-up truncated at 10 years, numbers at risk shown.

Outputs: analysis/05_survival/
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from scipy.stats import chi2
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "05_survival"; OUT.mkdir(exist_ok=True)
HUBS = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
ENDPOINTS = ["OS", "PFI", "DSS"]
N_BOOT, SEED = 300, 42
rng = np.random.default_rng(SEED)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

# ── data ──
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
logcpm = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
ids = {g: gmap[gmap == g].index[0] for g in HUBS}
X = logcpm.loc[list(ids.values()), clin.index].T
X.columns = HUBS
X.to_csv(OUT / "hub_expression_log2CPM_tumours.csv")
Z = (X - X.mean()) / X.std()

d = clin.join(Z)
d["age10"] = d.age / 10
d["male"] = (d.sex == "Male").astype(float)
for s in ["II", "III", "IV"]:
    d[f"stage_{s}"] = (d.stage == s).astype(float).where(d.stage.notna())
COV = ["age10", "male", "stage_II", "stage_III", "stage_IV", "ever_smoker"]

def frame(ep, cols, censor_years=None):
    t, e = d[f"{ep}_days"].copy(), d[f"{ep}_event"].copy()
    if censor_years:
        lim = censor_years * 365.25
        e = e.where(t <= lim, 0); t = t.clip(upper=lim)
    f = pd.concat([t.rename("T"), e.rename("E"), d[cols]], axis=1).dropna()
    return f

def fit(f):
    return CoxPHFitter().fit(f, "T", "E")

def rcs_basis(x, knots):
    """Harrell restricted cubic spline: returns k-2 non-linear terms."""
    k = knots; t_k, t_k1 = k[-1], k[-2]; den = (t_k - k[0]) ** 2
    pos = lambda u: np.maximum(u, 0) ** 3
    return np.column_stack([(pos(x - k[j]) - pos(x - t_k1) * (t_k - k[j]) / (t_k - t_k1)
                             + pos(x - t_k) * (t_k1 - k[j]) / (t_k - t_k1)) / den
                            for j in range(len(k) - 2)])

def harrell_c(model, f):
    return concordance_index(f["T"], -model.predict_partial_hazard(f), f["E"])

rows, addv, ph_rows, nl_rows = [], [], [], []
for ep in ENDPOINTS:
    f_clin = frame(ep, COV)
    m_clin = fit(f_clin)
    for g in HUBS:
        # univariate
        fu = frame(ep, [g]); mu = fit(fu)
        # multivariable (complete cases)
        fm = frame(ep, [g] + COV); mm = fit(fm)
        rows.append(dict(endpoint=ep, gene=g,
            uni_n=len(fu), uni_events=int(fu.E.sum()),
            uni_HR=np.exp(mu.params_[g]), uni_lo=np.exp(mu.confidence_intervals_.loc[g].iloc[0]),
            uni_hi=np.exp(mu.confidence_intervals_.loc[g].iloc[1]), uni_p=mu.summary.loc[g, "p"],
            multi_n=len(fm), multi_events=int(fm.E.sum()),
            multi_HR=np.exp(mm.params_[g]), multi_lo=np.exp(mm.confidence_intervals_.loc[g].iloc[0]),
            multi_hi=np.exp(mm.confidence_intervals_.loc[g].iloc[1]), multi_p=mm.summary.loc[g, "p"]))
        # PH test
        ph = proportional_hazard_test(mm, fm, time_transform="rank").summary
        ph_rows.append(dict(endpoint=ep, gene=g, ph_p_gene=ph.loc[g, "p"], ph_min_p_any_term=ph.p.min(),
                            ph_terms_p_lt_0_05=", ".join(ph.index[ph.p < 0.05])))
        # non-linearity (univariate, OS-type endpoint)
        knots = np.quantile(fu[g], [.05, .35, .65, .95])
        B = pd.DataFrame(rcs_basis(fu[g].values, knots), index=fu.index, columns=["s1", "s2"])
        mnl = fit(pd.concat([fu, B], axis=1))
        lr = 2 * (mnl.log_likelihood_ - mu.log_likelihood_)
        nl_rows.append(dict(endpoint=ep, gene=g, LR_nonlinear=lr, p_nonlinear=chi2.sf(lr, 2)))
        # added value over clinical model (same complete cases)
        mc = fit(fm[["T", "E"] + COV])
        lr_add = 2 * (mm.log_likelihood_ - mc.log_likelihood_)
        c0, c1 = harrell_c(mc, fm), harrell_c(mm, fm)
        boot = []
        if ep == "OS":
            for _ in range(N_BOOT):
                b = fm.sample(len(fm), replace=True, random_state=int(rng.integers(1e9)))
                try:
                    boot.append(harrell_c(fit(b), b) - harrell_c(fit(b[["T", "E"] + COV]), b))
                except Exception:
                    pass
        addv.append(dict(endpoint=ep, gene=g, n=len(fm), events=int(fm.E.sum()),
                         C_clinical=c0, C_clinical_plus_gene=c1, delta_C=c1 - c0,
                         delta_C_lo=np.percentile(boot, 2.5) if boot else np.nan,
                         delta_C_hi=np.percentile(boot, 97.5) if boot else np.nan,
                         LRT_chi2=lr_add, LRT_p=chi2.sf(lr_add, 1)))
    print(f"{ep}: done")

res = pd.DataFrame(rows)
for ep in ENDPOINTS:
    m = res.endpoint == ep
    res.loc[m, "uni_FDR"] = bh(res.loc[m, "uni_p"]); res.loc[m, "multi_FDR"] = bh(res.loc[m, "multi_p"])
res.to_csv(OUT / "cox_uni_multi_per_SD_OS_PFI_DSS.csv", index=False)
add = pd.DataFrame(addv)
for ep in ENDPOINTS:
    m = add.endpoint == ep; add.loc[m, "LRT_FDR"] = bh(add.loc[m, "LRT_p"])
add.to_csv(OUT / "added_value_over_clinical_model.csv", index=False)
ph = pd.DataFrame(ph_rows); ph.to_csv(OUT / "proportional_hazards_tests.csv", index=False)
nl = pd.DataFrame(nl_rows); nl.to_csv(OUT / "nonlinearity_spline_tests.csv", index=False)

# sensitivity: OS censored at 10 years
sens = []
for g in HUBS:
    fm = frame("OS", [g] + COV, censor_years=10); mm = fit(fm)
    sens.append(dict(gene=g, n=len(fm), events=int(fm.E.sum()), HR=np.exp(mm.params_[g]),
                     lo=np.exp(mm.confidence_intervals_.loc[g].iloc[0]),
                     hi=np.exp(mm.confidence_intervals_.loc[g].iloc[1]), p=mm.summary.loc[g, "p"]))
sens = pd.DataFrame(sens); sens["FDR"] = bh(sens.p)
sens.to_csv(OUT / "sensitivity_OS_multivariable_censored_10y.csv", index=False)

# clinical-model coefficients (for the table in the paper)
fc = frame("OS", COV); mclin = fit(fc)
mclin.summary.to_csv(OUT / "clinical_model_OS_coefficients.csv")

# ── forest plot ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
ypos = np.arange(len(HUBS))[::-1]
for ax, ep in zip(axes, ENDPOINTS):
    r = res[res.endpoint == ep].set_index("gene").loc[HUBS]
    ax.errorbar(r.uni_HR, ypos + .15, xerr=[r.uni_HR - r.uni_lo, r.uni_hi - r.uni_HR], fmt="o",
                color="#7f8c8d", ms=5, capsize=2, label="Univariate")
    ax.errorbar(r.multi_HR, ypos - .15, xerr=[r.multi_HR - r.multi_lo, r.multi_hi - r.multi_HR], fmt="s",
                color="#c0392b", ms=5, capsize=2, label="Multivariable*")
    for y, (_, row) in zip(ypos, r.iterrows()):
        ax.text(1.02, y, f"{row.multi_HR:.2f} ({row.multi_lo:.2f}–{row.multi_hi:.2f})  FDR {row.multi_FDR:.3f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    ax.axvline(1, ls="--", c="k", lw=.7)
    n_u, e_u = int(r.uni_n.iloc[0]), int(r.uni_events.iloc[0])
    n_m, e_m = int(r.multi_n.iloc[0]), int(r.multi_events.iloc[0])
    ax.set(title=f"{ep}  (uni n={n_u}/{e_u} events; multi n={n_m}/{e_m})", xlabel="HR per 1 SD (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(HUBS, fontstyle="italic")
axes[0].legend(fontsize=8, loc="lower left", frameon=False)
fig.text(0.01, 0.005, "*Each gene modelled separately, adjusted for age, sex, stage (II/III/IV vs I) and smoking.",
         fontsize=8)
plt.tight_layout(rect=(0, 0.03, 0.86, 1)); plt.savefig(OUT / "Fig_forest_uni_multi_OS_PFI_DSS.png", dpi=300); plt.close()

# ── KM grid, OS, 10-year truncation, numbers at risk ──
fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
for ax, g in zip(axes.flat, HUBS):
    f = frame("OS", [g], censor_years=10)
    f["T"] = f["T"] / 365.25
    hi = f[g] > f[g].median()
    k1, k0 = KaplanMeierFitter(), KaplanMeierFitter()
    k1.fit(f["T"][hi], f["E"][hi], label=f"High (n={hi.sum()})").plot_survival_function(ax=ax, ci_show=False, color="#c0392b")
    k0.fit(f["T"][~hi], f["E"][~hi], label=f"Low (n={(~hi).sum()})").plot_survival_function(ax=ax, ci_show=False, color="#2980b9")
    p = logrank_test(f["T"][hi], f["T"][~hi], f["E"][hi], f["E"][~hi]).p_value
    ax.set(title=f"{g}   log-rank p = {p:.2g}" if p >= 1e-3 else f"{g}   log-rank p < 0.001",
           xlabel="Years", ylabel="Overall survival", ylim=(0, 1.02), xlim=(0, 10))
    ax.title.set_fontstyle("italic")
    add_at_risk_counts(k1, k0, ax=ax, rows_to_show=["At risk"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes.flat[-1].axis("off")
axes.flat[-1].text(0.05, 0.5, "Median split of log2 CPM\nwithin tumours.\nFollow-up truncated at 10 years.\n"
                   "Illustration only; primary\nanalysis = continuous Cox.", fontsize=11, va="center")
plt.tight_layout(); plt.savefig(OUT / "Fig_KM_OS_7genes_10y.png", dpi=300); plt.close()

# ── report ──
pd.set_option("display.width", 200)
fmt = lambda v: f"{v:.3g}"
txt = ["R05 SURVIVAL (per-SD Cox, TCGA-CDR endpoints)", "=" * 60]
for ep in ENDPOINTS:
    r = res[res.endpoint == ep]
    txt += [f"\n{ep}", r[["gene", "uni_HR", "uni_lo", "uni_hi", "uni_FDR", "multi_HR", "multi_lo", "multi_hi",
                          "multi_FDR"]].to_string(index=False, float_format=fmt)]
txt += ["\nAdded value over clinical model (age, sex, stage, smoking):",
        add[["endpoint", "gene", "n", "events", "C_clinical", "C_clinical_plus_gene", "delta_C", "delta_C_lo",
             "delta_C_hi", "LRT_p", "LRT_FDR"]].to_string(index=False, float_format=fmt),
        "\nProportional hazards (multivariable models, rank time):",
        ph.to_string(index=False, float_format=fmt),
        "\nNon-linearity (RCS 4 knots vs linear, univariate):",
        nl.to_string(index=False, float_format=fmt),
        "\nSensitivity: OS multivariable, censored at 10 y:",
        sens.to_string(index=False, float_format=fmt)]
(OUT / "R05_report.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
