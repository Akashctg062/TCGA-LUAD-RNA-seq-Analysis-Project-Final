"""
R13c_published_signatures.py — Limitation 7: head-to-head comparison with published LUAD prognostic signatures.

Published signatures (gene lists, directions and weights checked against the original sources):
  CCP31     Wistuba et al., Clin Cancer Res 2013 (cell-cycle progression score; 31 genes, unweighted mean;
            validated by the authors in GSE31210 and the Director's Consortium = GSE68465)
  KRATZ11   Kratz et al., Lancet 2012 (14-gene qPCR assay; 11 cancer-related genes; direction from US 9476098 B2,
            claim 2: BAG1, BRCA1, CDC6, CDK2AP1, FUT3, IL11, RND3 risk; ERBB3, LCK, SH3BGR, WNT3A protective)
  CHEN5     Chen et al., N Engl J Med 2007 (DUSP6, MMD, ERBB3 risk; STAT1, LCK protective; weights = ln HR)
  SHUKLA4   Shukla et al., J Natl Cancer Inst 2017 (RHOV, CD109, LINC00941, FRRS1; all risk; TCGA-trained)
  ALDHERASI7 Al-Dherasi et al., Cancer Cell Int 2021 (published Cox coefficients; trained in Fudan cohort,
            validated in TCGA)
  LI8       Li et al., Int J Mol Sci 2020 (published LASSO-Cox coefficients; TCGA-trained)
Reference: the locked 7-gene score (HUB7_mean, no fitted weights), and PAM50 proliferation index.

Two versions of every published signature
  (a) "as published": score = sum(w_i * z_i) over the genes measured on the platform (z within cohort),
      w_i = published coefficient, ln HR, or +1/-1 direction (CCP31, KRATZ11, SHUKLA4) — nothing refitted.
  (b) "TCGA-refit": ridge Cox (penalizer 0.1) on the signature genes in TCGA, weights locked and applied to GEO
      (same procedure as the locked HUB7 ridge score). Only reported for GEO cohorts.
A signature is evaluated in a cohort only if >= 50% of its genes are measured.
Per cohort (OS): HR per SD (univariable; multivariable age, sex, stage; GSE68465 stratified by site), Harrell C,
delta C over the clinical model, C difference vs HUB7_mean (paired bootstrap, 500 replicates), and mutual
likelihood-ratio tests (HUB7 added to clinical + signature, and signature added to clinical + HUB7).
Random-effects (DerSimonian-Laird) pooled multivariable HR over the cohorts in which a signature was not trained.
Outputs: analysis/13_limitations/published_signatures/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
DAT = NEW / "07_external_validation" / "data"
OUT = NEW / "13_limitations" / "published_signatures"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42); N_BOOT = 500
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
sigdef = pd.read_csv(NEW / "06_benchmark" / "signature_definitions.csv").set_index("signature")
PAM50P = sigdef.loc["PAM50_prolif", "genes"].split(";")

ALIAS = {"CDK1": ["CDC2"], "PCLAF": ["KIAA0101"], "SKA1": ["C18orf24"], "ORC6": ["ORC6L"], "CAVIN2": ["SDPR"],
         "FRRS1": [], "HMMR": []}
CCP = ("FOXM1 CDC20 CDKN3 CDK1 KIF11 PCLAF NUSAP1 CENPF ASPM BUB1B RRM2 DLGAP5 BIRC5 KIF20A PLK1 TOP2A TK1 PBK "
       "ASF1B SKA1 RAD54L PTTG1 CDCA3 MCM10 PRC1 DTL CEP55 RAD51 CENPM CDCA8 ORC6").split()
SIGS = {
    "CCP31": ({g: 1.0 for g in CCP}, "Wistuba 2013", "prostate-derived; qPCR in stage I LUAD"),
    "KRATZ11": ({"BAG1": 1, "BRCA1": 1, "CDC6": 1, "CDK2AP1": 1, "FUT3": 1, "IL11": 1, "RND3": 1,
                 "ERBB3": -1, "LCK": -1, "SH3BGR": -1, "WNT3A": -1}, "Kratz 2012", "qPCR; non-squamous NSCLC"),
    "CHEN5": ({"DUSP6": np.log(2.12), "MMD": np.log(2.50), "ERBB3": np.log(1.73), "STAT1": np.log(0.56),
               "LCK": np.log(0.43)}, "Chen 2007", "NSCLC; Taiwan"),
    "SHUKLA4": ({"RHOV": 1, "CD109": 1, "LINC00941": 1, "FRRS1": 1}, "Shukla 2017", "TCGA-trained"),
    "ALDHERASI7": ({"GRIA1": -0.3658, "UCN2": 0.5701, "PKHD1L1": -0.601, "RIMS2": 0.2192, "PGM5": -0.3617,
                    "CLIC6": -0.6036, "CAVIN2": 1.1686}, "Al-Dherasi 2021", "trained Fudan; validated TCGA"),
    "LI8": ({"TTK": -0.462, "HMMR": 0.633, "ASPM": -0.550, "CDCA8": -1.309, "KIF2C": 1.188, "CCNB2": -0.474,
             "MKI67": 0.570, "CCNA2": 0.484}, "Li 2020", "TCGA-trained"),
}
TRAINED_IN = {"SHUKLA4": {"TCGA-LUAD"}, "LI8": {"TCGA-LUAD"}, "ALDHERASI7": {"TCGA-LUAD"},
              "CCP31": {"GSE31210", "GSE68465"}}  # cohorts used for training or validation by the original authors

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
def zscore(E): return (E - E.mean()) / E.std().replace(0, np.nan)
def resolve(cols, g):
    for x in [g] + ALIAS.get(g, []):
        if x in cols: return x
    return None

# ── cohorts (identical loading to R07b) ──
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
tc = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
cohorts = {"TCGA-LUAD": (lc[tc.index].T, pd.DataFrame({"age": tc.age, "male": (tc.sex == "Male").astype(float),
           "stage": tc.stage, "site": "TCGA", "OS_event": tc.OS_event, "OS_days": tc.OS_days, "tumour": 1},
           index=tc.index))}
for gse in ["GSE31210", "GSE68465", "GSE72094"]:
    cohorts[gse] = (pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0),
                    pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0))

def design(C):
    d = pd.DataFrame({"T": C.OS_days, "E": C.OS_event, "age": C.age, "male": C.male})
    for s in ["II", "III", "IV"]:
        if (C.stage == s).sum() >= 5: d[f"stage_{s}"] = (C.stage == s).astype(float).where(C.stage.notna())
    return d

def sig_score(Z, weights):
    use = {resolve(Z.columns, g): w for g, w in weights.items() if resolve(Z.columns, g)}
    if len(use) < 0.5 * len(weights): return None, len(use)
    return sum(Z[g] * w for g, w in use.items()) / sum(abs(w) for w in use.values()), len(use)

# TCGA-refit weights (per GEO cohort: restricted to the genes measured there)
Et_tcga = cohorts["TCGA-LUAD"][0]; Ct_tcga = cohorts["TCGA-LUAD"][1]
Zt_tcga = zscore(Et_tcga)
def refit_weights(genes):
    f = pd.concat([Ct_tcga.OS_days.rename("T"), Ct_tcga.OS_event.rename("E"), Zt_tcga[genes]], axis=1).dropna()
    return CoxPHFitter(penalizer=0.1).fit(f, "T", "E").params_

rows, h2h, wrows = [], [], []
for name, (E, C) in cohorts.items():
    T = C[C.tumour == 1].index.intersection(E.index)
    Et, Ct = E.loc[T], C.loc[T]
    Z = zscore(Et)
    strata = ["site"] if Ct.site.nunique() > 1 else None
    S = {"HUB7_mean": Z[HUB7].mean(1), "PAM50_prolif": Z[[g for g in PAM50P if g in Z]].mean(1)}
    ngen = {"HUB7_mean": 7, "PAM50_prolif": sum(g in Z for g in PAM50P)}
    for k, (w, *_rest) in SIGS.items():
        s, n = sig_score(Z, w); ngen[k] = n
        if s is not None: S[k] = s
        if name != "TCGA-LUAD" and s is not None:
            genes_t = [resolve(Zt_tcga.columns, g) for g in w]
            pairs = [(gt, resolve(Z.columns, g)) for gt, g in zip(genes_t, w) if gt and resolve(Z.columns, g)]
            W = refit_weights([a for a, _ in pairs])
            S[k + "_refit"] = sum(Z[b] * W[a] for a, b in pairs); ngen[k + "_refit"] = len(pairs)
            for a, b in pairs: wrows.append(dict(cohort=name, signature=k, gene=a, refit_coef=W[a]))
    S = pd.DataFrame(S)
    base = design(Ct)
    if strata: base["site"] = Ct.site
    for sig in S.columns:
        d = base.join(zscore(S[[sig]]).rename(columns={sig: "s"})).dropna()
        mu = CoxPHFitter().fit(d[["T", "E", "s"] + (["site"] if strata else [])], "T", "E", strata=strata)
        mm = CoxPHFitter().fit(d, "T", "E", strata=strata)
        mc = CoxPHFitter().fit(d.drop(columns="s"), "T", "E", strata=strata)
        c_uni = concordance_index(d["T"], -d["s"], d["E"])
        c_cl = concordance_index(d["T"], -mc.predict_partial_hazard(d), d["E"])
        c_mm = concordance_index(d["T"], -mm.predict_partial_hazard(d), d["E"])
        rows.append(dict(cohort=name, signature=sig, n_genes=ngen[sig], n=len(d), events=int(d.E.sum()),
                         HR_uni=np.exp(mu.params_["s"]), p_uni=mu.summary.loc["s", "p"],
                         HR_multi=np.exp(mm.params_["s"]), lo_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[0]),
                         hi_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[1]), p_multi=mm.summary.loc["s", "p"],
                         se_logHR_multi=mm.standard_errors_["s"], C_uni=c_uni, C_clinical=c_cl,
                         C_clinical_plus_sig=c_mm, delta_C=c_mm - c_cl,
                         trained_in_this_cohort=name in TRAINED_IN.get(sig.replace("_refit", ""), set())
                                                or (sig.endswith("_refit") and False)))
    # head-to-head vs HUB7_mean
    for sig in [c for c in S.columns if c != "HUB7_mean"]:
        d = base.join(zscore(S[["HUB7_mean", sig]]).rename(columns={"HUB7_mean": "h", sig: "s"})).dropna()
        m_both = CoxPHFitter().fit(d, "T", "E", strata=strata)
        m_s = CoxPHFitter().fit(d.drop(columns="h"), "T", "E", strata=strata)
        m_h = CoxPHFitter().fit(d.drop(columns="s"), "T", "E", strata=strata)
        diffs = []
        idx = np.arange(len(d))
        for _ in range(N_BOOT):
            b = d.iloc[rng.choice(idx, len(idx))]
            diffs.append(concordance_index(b["T"], -b["h"], b["E"]) - concordance_index(b["T"], -b["s"], b["E"]))
        dC = concordance_index(d["T"], -d["h"], d["E"]) - concordance_index(d["T"], -d["s"], d["E"])
        h2h.append(dict(cohort=name, signature=sig, n=len(d), r_spearman=d[["h", "s"]].corr("spearman").iloc[0, 1],
                        C_HUB7_minus_C_sig=dC, lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5),
                        LRT_p_HUB7_added_to_sig=chi2.sf(2 * (m_both.log_likelihood_ - m_s.log_likelihood_), 1),
                        LRT_p_sig_added_to_HUB7=chi2.sf(2 * (m_both.log_likelihood_ - m_h.log_likelihood_), 1)))
    print(name, "done", flush=True)

R = pd.DataFrame(rows); H = pd.DataFrame(h2h)
H["FDR_HUB7_added"] = bh(H.LRT_p_HUB7_added_to_sig); H["FDR_sig_added"] = bh(H.LRT_p_sig_added_to_HUB7)
R.to_csv(OUT / "R13c_signature_performance.csv", index=False)
H.to_csv(OUT / "R13c_head_to_head_vs_HUB7.csv", index=False)
pd.DataFrame(wrows).to_csv(OUT / "R13c_refit_weights.csv", index=False)

# random-effects meta-analysis (multivariable HR), excluding cohorts used for training/validation by the authors
meta = []
for sig, g in R.groupby("signature"):
    g = g[~g.trained_in_this_cohort]
    if len(g) < 2: continue
    y, v = np.log(g.HR_multi.values), g.se_logHR_multi.values ** 2
    w = 1 / v; ybar = (w * y).sum() / w.sum(); Q = (w * (y - ybar) ** 2).sum(); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2); mu = (ws * y).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    meta.append(dict(signature=sig, k=k, cohorts=";".join(g.cohort), HR=np.exp(mu), lo=np.exp(mu - 1.96 * se),
                     hi=np.exp(mu + 1.96 * se), p=2 * norm.sf(abs(mu / se)),
                     I2=max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0, mean_C_uni=g.C_uni.mean()))
M = pd.DataFrame(meta).sort_values("HR", ascending=False)
M.to_csv(OUT / "R13c_meta_analysis.csv", index=False)
pd.set_option("display.width", 250)
print(R[["cohort", "signature", "n_genes", "events", "HR_multi", "p_multi", "C_uni", "delta_C", "trained_in_this_cohort"]]
      .round(3).to_string(index=False))
print(H.round(3).to_string(index=False))
print(M.round(3).to_string(index=False))
