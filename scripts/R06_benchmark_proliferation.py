"""
R06_benchmark_proliferation.py — Revision step 6 (PI comment M1: novelty / proliferation problem)
==================================================================================================
Question: does the locked 7-gene hub signature carry prognostic (or diagnostic) information
BEYOND a generic proliferation signal?

Signature score (identical rule for every signature): mean z-score of the member genes
  * prognostic analyses: z-scores computed within the tumours
  * diagnostic analyses: z-scores computed over all 576 samples
Expression: log2(CPM + 1), filtered STAR counts.

Comparators
  HUB7          locked hub genes (analysis/hub_genes_final.csv)
  OLD5          original hub genes (BIRC5, CCNB1, CCNB2, CDC20, KIF2C)
  MKI67         single gene
  PAM50_prolif  11-gene PAM50 proliferation index (Nielsen et al., Clin Cancer Res 2010)
  PCNA_meta     PCNA metagene, Venet et al. (PLoS Comput Biol 2011) rule applied in-cohort:
                top 1% of expressed genes most correlated with PCNA in TCGA-LUAD tumours
  HALLMARK_G2M, HALLMARK_E2F   MSigDB Hallmark sets (R06a download)
Null distributions (1,000 random 7-gene sets each, seed 42)
  N1  expression-matched up-regulated DEGs (each hub gene replaced by a random up-DEG from the same
      decile of mean expression)
  N2  proliferation-matched: random genes from HALLMARK_G2M ∪ HALLMARK_E2F
Metrics
  OS Harrell C of the score alone; C of clinical model (age, sex, stage, smoking) + score; AUC tumour vs normal
  Paired bootstrap (1,000) of C(HUB7) − C(comparator)
  Nested likelihood-ratio tests: does HUB7 add to clinical + comparator, and vice versa?
Outputs: analysis/06_benchmark/
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2, pearsonr
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "06_benchmark"; OUT.mkdir(exist_ok=True)
N_NULL, N_BOOT, SEED = 1000, 1000, 42
rng = np.random.default_rng(SEED)

HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
OLD5 = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
PAM50_PROLIF = ["BIRC5", "CCNB1", "CDC20", "NUF2", "CEP55", "NDC80", "MKI67", "PTTG1", "RRM2", "TYMS", "UBE2C"]

# ── expression ──
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
meta = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("sample_submitter_id")
logcpm = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
logcpm.index = gmap.reindex(logcpm.index).fillna(pd.Series(logcpm.index, index=logcpm.index)).values
logcpm = logcpm[~logcpm.index.duplicated(keep="first")]
E = logcpm.T                                                   # samples x genes (symbols)
is_tum = (meta.loc[E.index, "condition"] == "Tumor").values
Et = E[is_tum]
Zt = (Et - Et.mean()) / Et.std().replace(0, np.nan)            # tumour z
Za = (E - E.mean()) / E.std().replace(0, np.nan)               # all-sample z
y_diag = is_tum.astype(int)

clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
clin = clin.loc[Et.index]
cov = pd.DataFrame({"age10": clin.age / 10, "male": (clin.sex == "Male").astype(float),
                    "stage_II": (clin.stage == "II").astype(float).where(clin.stage.notna()),
                    "stage_III": (clin.stage == "III").astype(float).where(clin.stage.notna()),
                    "stage_IV": (clin.stage == "IV").astype(float).where(clin.stage.notna()),
                    "ever_smoker": clin.ever_smoker}, index=clin.index)
surv = pd.concat([clin.OS_days.rename("T"), clin.OS_event.rename("E"), cov], axis=1).dropna()
COV = list(cov.columns)
base = CoxPHFitter().fit(surv, "T", "E")
C_CLIN = concordance_index(surv["T"], -base.predict_partial_hazard(surv), surv["E"])

# ── gene sets ──
gmt = sorted(glob.glob(str(OUT / "raw" / "h.all.*.gmt")))
assert gmt, "Run R06a_download_genesets.py first"
hall = {l.split("\t")[0]: l.strip().split("\t")[2:] for l in open(gmt[-1], encoding="utf-8") if l.strip()}
G2M, E2F = hall["HALLMARK_G2M_CHECKPOINT"], hall["HALLMARK_E2F_TARGETS"]

r_pcna = Et.corrwith(Et["PCNA"]).drop("PCNA").dropna()
expressed = Et.columns[Et.mean() > 1]
r_pcna = r_pcna[r_pcna.index.isin(expressed)]
PCNA_META = r_pcna.nlargest(int(round(0.01 * len(r_pcna)))).index.tolist()

SIGS = {"HUB7": HUB7, "OLD5": OLD5, "MKI67": ["MKI67"], "PAM50_prolif": PAM50_PROLIF,
        "PCNA_meta": PCNA_META, "HALLMARK_G2M": G2M, "HALLMARK_E2F": E2F}
present = {k: [g for g in v if g in E.columns] for k, v in SIGS.items()}
pd.DataFrame([dict(signature=k, n_defined=len(SIGS[k]), n_present=len(v), genes=";".join(v))
              for k, v in present.items()]).to_csv(OUT / "signature_definitions.csv", index=False)

def score(genes, Z):
    return Z[genes].mean(1)

def metrics(genes, multi=True):
    st = score(genes, Zt).loc[surv.index]
    c_uni = concordance_index(surv["T"], -st, surv["E"])
    auc = roc_auc_score(y_diag, score(genes, Za))
    out = dict(C_uni=c_uni, AUC=auc)
    if multi:
        f = surv.assign(s=(st - st.mean()) / st.std())
        m = CoxPHFitter().fit(f, "T", "E")
        out.update(C_multi=concordance_index(f["T"], -m.predict_partial_hazard(f), f["E"]),
                   HR_multi=np.exp(m.params_["s"]), p_multi=m.summary.loc["s", "p"],
                   LR_vs_clin=2 * (m.log_likelihood_ - base.log_likelihood_))
    return out

# ── observed signatures ──
obs = pd.DataFrame({k: metrics(v) for k, v in present.items()}).T
obs["delta_C_vs_clinical"] = obs.C_multi - C_CLIN
obs["r_with_HUB7"] = [pearsonr(score(present["HUB7"], Zt), score(v, Zt))[0] for v in present.values()]
obs.to_csv(OUT / "signature_performance.csv")

# ── paired bootstrap of C(HUB7) − C(comparator), univariate ──
S = pd.DataFrame({k: score(v, Zt).loc[surv.index] for k, v in present.items()})
T_, E_ = surv["T"].values, surv["E"].values
boot = {k: [] for k in present if k != "HUB7"}
for _ in range(N_BOOT):
    i = rng.integers(0, len(surv), len(surv))
    c7 = concordance_index(T_[i], -S["HUB7"].values[i], E_[i])
    for k in boot:
        boot[k].append(c7 - concordance_index(T_[i], -S[k].values[i], E_[i]))
bt = pd.DataFrame({k: dict(delta_C=obs.loc["HUB7", "C_uni"] - obs.loc[k, "C_uni"],
                           lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                           p_two_sided=min(1, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))))
                   for k, v in boot.items()}).T
bt.to_csv(OUT / "paired_bootstrap_deltaC_HUB7_vs_comparators.csv")

# ── nested LRTs ──
nest = []
for k in ["MKI67", "PAM50_prolif", "PCNA_meta", "HALLMARK_G2M", "HALLMARK_E2F", "OLD5"]:
    f = surv.assign(h=(S["HUB7"] - S["HUB7"].mean()) / S["HUB7"].std(), c=(S[k] - S[k].mean()) / S[k].std())
    m_c = CoxPHFitter().fit(f.drop(columns="h"), "T", "E")
    m_h = CoxPHFitter().fit(f.drop(columns="c"), "T", "E")
    m_b = CoxPHFitter().fit(f, "T", "E")
    lr_h = 2 * (m_b.log_likelihood_ - m_c.log_likelihood_)
    lr_c = 2 * (m_b.log_likelihood_ - m_h.log_likelihood_)
    nest.append(dict(comparator=k, r_HUB7_comparator=np.corrcoef(f.h, f.c)[0, 1],
                     HUB7_adds_to_clin_plus_comparator_LR=lr_h, p_HUB7_adds=chi2.sf(lr_h, 1),
                     comparator_adds_to_clin_plus_HUB7_LR=lr_c, p_comparator_adds=chi2.sf(lr_c, 1),
                     HR_HUB7_joint=np.exp(m_b.params_["h"]), p_HUB7_joint=m_b.summary.loc["h", "p"],
                     HR_comp_joint=np.exp(m_b.params_["c"]), p_comp_joint=m_b.summary.loc["c", "p"]))
nest = pd.DataFrame(nest); nest.to_csv(OUT / "nested_LRT_HUB7_vs_proliferation.csv", index=False)

# ── null distributions ──
deg = pd.read_csv(ROOT / "results" / "deg_all_results.csv")
up = deg[(deg.padj < 0.05) & (deg.log2FoldChange > 1)].gene_name.dropna()
up = [g for g in up.unique() if g in Et.columns and g not in HUB7]
mean_expr = Et[up + HUB7].mean()
dec = pd.qcut(mean_expr, 10, labels=False)
pool_by_dec = {d: [g for g in up if dec[g] == d] for d in range(10)}
prolif_pool = [g for g in set(G2M) | set(E2F) if g in Et.columns and g not in HUB7]

null = []
for name, draw in [("N1_expression_matched_upDEG", lambda: [rng.choice(pool_by_dec[dec[g]]) for g in HUB7]),
                   ("N2_proliferation_G2M_E2F", lambda: list(rng.choice(prolif_pool, 7, replace=False)))]:
    for i in range(N_NULL):
        gs = draw()
        while len(set(gs)) < 7: gs = draw()
        null.append(dict(null=name, **metrics(gs)))
    print(name, "done")
null = pd.DataFrame(null); null.to_csv(OUT / "null_distributions_1000x2.csv", index=False)

pct = []
for name, g in null.groupby("null"):
    for m in ["C_uni", "C_multi", "AUC"]:
        pct.append(dict(null=name, metric=m, HUB7=obs.loc["HUB7", m], null_median=g[m].median(),
                        null_95th=g[m].quantile(.95), empirical_p=(1 + (g[m] >= obs.loc["HUB7", m]).sum()) / (1 + len(g)),
                        HUB7_percentile=100 * (g[m] < obs.loc["HUB7", m]).mean()))
pct = pd.DataFrame(pct); pct.to_csv(OUT / "HUB7_vs_null_empirical.csv", index=False)

# ── figure ──
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, m, lab in [(ax[0], "C_uni", "OS Harrell C (score alone)"), (ax[1], "C_multi", "OS C (clinical + score)")]:
    for (name, g), col in zip(null.groupby("null"), ["#95a5a6", "#e67e22"]):
        a.hist(g[m], bins=40, alpha=.55, color=col, label=name.replace("_", " "))
    for k, c, ls in [("HUB7", "#c0392b", "-"), ("MKI67", "k", ":"), ("HALLMARK_G2M", "#2c3e50", "--"),
                     ("PAM50_prolif", "#8e44ad", "-.")]:
        a.axvline(obs.loc[k, m], color=c, ls=ls, lw=1.6, label=k)
    if m == "C_multi": a.axvline(C_CLIN, color="#16a085", lw=1.2, label="clinical only")
    a.set(xlabel=lab, ylabel="Random 7-gene sets"); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=7, frameon=False)
o = obs.sort_values("C_uni")
ax[2].barh(o.index, o.C_uni, color=["#c0392b" if i == "HUB7" else "#7f8c8d" for i in o.index])
for i, (k, v) in enumerate(o.C_uni.items()):
    ax[2].text(v + .002, i, f"{v:.3f}", va="center", fontsize=8)
ax[2].set(xlim=(0.5, max(o.C_uni) + .04), xlabel="OS Harrell C (score alone)", title="Signature comparison")
ax[2].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_benchmark_null_and_comparators.png", dpi=300); plt.close()

# ── report ──
fmt = lambda v: f"{v:.3g}"
rep = [f"R06 BENCHMARK  (OS n = {len(surv)}, events = {int(surv.E.sum())}; clinical-only C = {C_CLIN:.3f})",
       f"PCNA metagene size: {len(PCNA_META)}; Hallmark file: {Path(gmt[-1]).name}",
       "\nObserved signatures:", obs.to_string(float_format=fmt),
       "\nPaired bootstrap C(HUB7) - C(comparator):", bt.to_string(float_format=fmt),
       "\nNested LRT (all models include age, sex, stage, smoking):", nest.to_string(index=False, float_format=fmt),
       "\nHUB7 vs 1,000 random 7-gene sets:", pct.to_string(index=False, float_format=fmt)]
(OUT / "R06_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
