"""
R11c_pancancer.py — Revision step 11c (PI comment M11)
=======================================================
Pan-cancer analysis of the 7-gene score with the FULL TCGA PanCanAtlas data
(replaces the original analysis that used <=20 tumours per cancer type).

Data
  * PanCanAtlas EB++ batch-corrected RNA-seq (RSEM, all 33 cancer types), log2(x + 1); R11a (streamed subset)
  * TCGA-CDR curated survival endpoints (Liu et al., Cell 2018); stage, age, sex from TCGA-CDR
  * one primary tumour per patient (sample type 01; 03 for LAML; 01 or 06 for SKCM)
  * optional: GTEx normal tissue via UCSC Toil (TCGA+GTEx recompute), if R11a obtained it
Analyses
  (1) tumour vs adjacent normal (TCGA normals, cancer types with >= 10 normals): difference in 7-gene score
      and each gene (log2 scale), Mann-Whitney, BH across cancer types
  (2) prognosis per cancer type: Cox HR per 1 SD of the 7-gene score (z within cancer type) for OS and PFI,
      univariate and adjusted for age + sex (+ stage where >= 70% staged); BH across cancer types;
      cancer types with < 20 events flagged as under-powered
Outputs: analysis/11_extended/pancancer/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from lifelines import CoxPHFitter

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "11_extended" / "raw"
OUT = NEW / "11_extended" / "pancancer"; OUT.mkdir(exist_ok=True)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
fmt = lambda v: f"{v:.3g}"
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

X = pd.read_csv(RAW / "PanCanAtlas_EBpp_selected_genes.tsv", sep="\t", index_col=0)
X.index = X.index.str.split("|").str[0]
X = np.log2(X.clip(lower=0) + 1).T
X["patient"] = X.index.str[:12]; X["stype"] = X.index.str[13:15]
cdr = pd.read_excel(NEW / "04_clinical" / "raw" / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
cdr = cdr.set_index("bcr_patient_barcode")
X["cancer"] = X.patient.map(cdr.type)
X = X.dropna(subset=["cancer"])

def primary(df):
    ok = (df.stype == "01") | ((df.cancer == "LAML") & (df.stype == "03")) | ((df.cancer == "SKCM") & df.stype.isin(["01", "06"]))
    d = df[ok].sort_values("stype")
    return d[~d.patient.duplicated()]
T = primary(X)
N = X[X.stype == "11"]
N = N[~N.patient.duplicated()]

# score: mean z of the 7 genes, z computed over all primary tumours pan-cancer (comparable across types)
mu, sd = T[HUB7].mean(), T[HUB7].std()
T["score"] = ((T[HUB7] - mu) / sd).mean(1); N["score"] = ((N[HUB7] - mu) / sd).mean(1)

# (1) tumour vs normal
tn = []
for c, g in T.groupby("cancer"):
    n = N[N.cancer == c]
    row = dict(cancer=c, n_tumour=len(g), n_normal=len(n))
    if len(n) >= 10:
        for v in ["score"] + HUB7:
            row[f"diff_{v}"] = g[v].median() - n[v].median()
            row[f"p_{v}"] = mannwhitneyu(g[v], n[v]).pvalue
    tn.append(row)
tn = pd.DataFrame(tn)
m = tn.n_normal >= 10
for v in ["score"] + HUB7:
    tn.loc[m, f"FDR_{v}"] = bh(tn.loc[m, f"p_{v}"])
tn.to_csv(OUT / "pancancer_tumour_vs_TCGA_normal.csv", index=False)

# (2) survival
def stage_num(s):
    s = str(s).upper()
    for k, v in [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]:
        if s.startswith("STAGE " + k): return v
    return np.nan
sv = []
for c, g in T.groupby("cancer"):
    cl = cdr.loc[g.patient]
    z = (g.score - g.score.mean()) / g.score.std()
    for ep in ["OS", "PFI"]:
        d = pd.DataFrame({"T": cl[f"{ep}.time"].values, "E": cl[ep].values, "s": z.values,
                          "age": cl.age_at_initial_pathologic_diagnosis.values / 10,
                          "male": (cl.gender == "MALE").astype(float).values,
                          "stage": cl.ajcc_pathologic_tumor_stage.map(stage_num).values}, index=g.index)
        d = d[(d["T"] > 0)].dropna(subset=["T", "E", "s"])
        if d.E.sum() < 5 or len(d) < 30:
            sv.append(dict(cancer=c, endpoint=ep, n=len(d), events=int(d.E.sum()), note="too few events")); continue
        try:
            mu_ = CoxPHFitter(penalizer=1e-4).fit(d[["T", "E", "s"]], "T", "E")
        except Exception as e:
            sv.append(dict(cancer=c, endpoint=ep, n=len(d), events=int(d.E.sum()), note=f"fit failed: {e}")); continue
        cov = ["age", "male"] + (["stage"] if d.stage.notna().mean() >= .7 else [])
        if d.male.nunique() < 2: cov.remove("male")
        dm = d[["T", "E", "s"] + cov].dropna()
        try:
            mm = CoxPHFitter(penalizer=1e-4).fit(dm, "T", "E")
            mrow = dict(HR_adj=np.exp(mm.params_["s"]), lo_adj=np.exp(mm.confidence_intervals_.loc["s"].iloc[0]),
                        hi_adj=np.exp(mm.confidence_intervals_.loc["s"].iloc[1]), p_adj=mm.summary.loc["s", "p"],
                        n_adj=len(dm), covariates="+".join(cov))
        except Exception:
            mrow = {}
        sv.append(dict(cancer=c, endpoint=ep, n=len(d), events=int(d.E.sum()),
                       HR=np.exp(mu_.params_["s"]), lo=np.exp(mu_.confidence_intervals_.loc["s"].iloc[0]),
                       hi=np.exp(mu_.confidence_intervals_.loc["s"].iloc[1]), p=mu_.summary.loc["s", "p"],
                       note="under-powered (<20 events)" if d.E.sum() < 20 else "", **mrow))
sv = pd.DataFrame(sv)
for ep in ["OS", "PFI"]:
    k = (sv.endpoint == ep) & sv.p.notna()
    sv.loc[k, "FDR"] = bh(sv.loc[k, "p"])
    k2 = (sv.endpoint == ep) & sv.p_adj.notna()
    sv.loc[k2, "FDR_adj"] = bh(sv.loc[k2, "p_adj"])
sv.to_csv(OUT / "pancancer_survival_score_per_SD.csv", index=False)

# ── figures ──
order = T.groupby("cancer").score.median().sort_values().index
fig, ax = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.2, 1]}, sharex=True)
data = pd.concat([T[["cancer", "score"]].assign(tissue="Tumour"), N[["cancer", "score"]].assign(tissue="Normal")])
sns.boxplot(data=data, x="cancer", y="score", hue="tissue", order=order, ax=ax[0], fliersize=1,
            palette={"Tumour": "#c0392b", "Normal": "#2980b9"}, linewidth=.6)
ax[0].set(ylabel="7-gene score (pan-cancer z)", xlabel="", title="7-gene score across 33 TCGA cancer types (PanCanAtlas)")
for i, c in enumerate(order):
    r = tn.set_index("cancer").loc[c]
    if r.n_normal >= 10 and r.FDR_score < 0.05:
        ax[0].text(i, ax[0].get_ylim()[1] * .92, "*", ha="center", fontsize=12)
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
s = sv[sv.endpoint == "OS"].set_index("cancer").reindex(order)
cols = ["#c0392b" if (r.FDR < 0.05 and r.HR > 1) else "#2980b9" if (r.FDR < 0.05 and r.HR < 1) else "#95a5a6"
        for r in s.itertuples()]
x = np.arange(len(order))
ax[1].errorbar(x, s.HR, yerr=[s.HR - s.lo, s.hi - s.HR], fmt="none", ecolor="#7f8c8d", lw=.8)
ax[1].scatter(x, s.HR, c=cols, zorder=3, s=25)
ax[1].axhline(1, ls="--", c="k", lw=.7); ax[1].set_yscale("log")
ax[1].set_yticks([0.5, 1, 2, 4]); ax[1].set_yticklabels(["0.5", "1", "2", "4"]); ax[1].minorticks_off()
ax[1].set(ylabel="OS HR per 1 SD (95% CI)", title="Overall survival (TCGA-CDR); red/blue = FDR < 0.05")
ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=90)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_pancancer_expression_and_OS.png", dpi=300); plt.close()

hm = sv.pivot(index="cancer", columns="endpoint", values="HR").reindex(order)
q = sv.pivot(index="cancer", columns="endpoint", values="FDR").reindex(order)
fig, ax = plt.subplots(figsize=(4, 11))
ann = hm.round(2).astype(str) + np.where(q < 0.05, "*", "")
sns.heatmap(np.log2(hm), annot=ann.where(hm.notna(), ""), fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={"label": "log2 HR per SD"})
ax.set_title("7-gene score: HR per SD\n(* FDR < 0.05)"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(OUT / "Fig_pancancer_HR_heatmap.png", dpi=300); plt.close()

os_ = sv[(sv.endpoint == "OS") & sv.p.notna()]
rep = [f"R11c PAN-CANCER  primary tumours: {len(T)} in {T.cancer.nunique()} types; TCGA normals: {len(N)}",
       "\nTumour vs normal (types with >=10 normals):",
       tn[tn.n_normal >= 10][["cancer", "n_tumour", "n_normal", "diff_score", "FDR_score"]].to_string(index=False, float_format=fmt),
       "\nOS, HR per SD of 7-gene score:",
       sv[sv.endpoint == "OS"][["cancer", "n", "events", "HR", "lo", "hi", "FDR", "HR_adj", "FDR_adj", "covariates", "note"]]
         .to_string(index=False, float_format=fmt),
       "\nPFI, HR per SD:",
       sv[sv.endpoint == "PFI"][["cancer", "n", "events", "HR", "FDR", "HR_adj", "FDR_adj", "note"]].to_string(index=False, float_format=fmt),
       f"\nOS summary: significant adverse (FDR<0.05, HR>1): {((os_.FDR < .05) & (os_.HR > 1)).sum()} types: "
       f"{', '.join(os_[(os_.FDR < .05) & (os_.HR > 1)].cancer)}; protective: "
       f"{', '.join(os_[(os_.FDR < .05) & (os_.HR < 1)].cancer)}"]
(OUT / "R11c_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
