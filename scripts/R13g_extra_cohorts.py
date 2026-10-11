"""
R13g_extra_cohorts.py — Limitation 3: longer follow-up, treatment information and ancestry.
(1) Harmonise five additional GEO cohorts (adenocarcinomas only):
      GSE50081 Der 2014 (Toronto, GPL570; OS/DFS in years)        GSE30219 Rousseaux 2013 (France, GPL570; pTNM)
      GSE37745 Botling 2013 (Uppsala, GPL570; adjuvant therapy)   GSE42127 Tang 2013 (MDACC, GPL6884; adjuvant chemo)
      GSE13213 Tomida 2009 (Japan, GPL6480; EGFR/KRAS/TP53)
    probe -> gene from the GPL annotation ('Gene symbol'; multi-gene probes dropped; highest-mean probe per gene).
(2) Locked 7-gene score (mean within-cohort z; nothing refitted): OS HR per SD (univariable; multivariable age, sex,
    stage), Harrell C, delta C over clinical model; median follow-up (reverse Kaplan-Meier).
(3) Follow-up: HR for deaths within 5 years (censored at 5 y) and 5-year landmark (patients alive at 5 y);
    time-varying effect (score x log(time)) in the pooled long-follow-up cohorts.
(4) Treatment: patients without adjuvant therapy only; score x adjuvant-chemotherapy interaction
    (GSE42127, GSE37745, GSE68465).
(5) Updated random-effects meta-analysis over all 9 cohorts, leave-one-out, and East Asian vs Western subgroups.
Outputs: analysis/13_limitations/extra_cohorts/
"""
import gzip, re, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
RAW = NEW / "13_limitations" / "raw"
OUT = NEW / "13_limitations" / "extra_cohorts"; OUT.mkdir(parents=True, exist_ok=True)
DAT = NEW / "07_external_validation" / "data"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
YEAR = 365.25

def read_series(acc):
    ch, expr, intab = {}, [], False
    with gzip.open(RAW / f"{acc}_series_matrix.txt.gz", "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"): intab = True; continue
            if line.startswith("!series_matrix_table_end"): break
            p = [x.strip('"') for x in line.rstrip("\n").split("\t")]
            if intab: expr.append(p); continue
            if p[0] == "!Sample_geo_accession": ids = p[1:]
            if p[0] in ("!Sample_characteristics_ch1", "!Sample_source_name_ch1", "!Sample_title"):
                for i, v in enumerate(p[1:]):
                    k, _, val = v.partition(":") if p[0] == "!Sample_characteristics_ch1" else (p[0][8:], "", v)
                    if v == "": continue
                    ch.setdefault(k.strip(), [None] * len(p[1:]))[i] = val.strip()
    clin = pd.DataFrame(ch, index=ids)
    E = pd.DataFrame(expr[1:], columns=expr[0]).set_index("ID_REF").apply(pd.to_numeric, errors="coerce")
    return clin, E

def annot(gpl):
    with gzip.open(RAW / f"{gpl}.annot.gz", "rt", encoding="latin-1") as fh:
        lines = [l for l in fh if not l.startswith(("!", "#", "^"))]
    from io import StringIO
    a = pd.read_csv(StringIO("".join(lines)), sep="\t", usecols=["ID", "Gene symbol"], dtype=str)
    a = a.dropna(); a = a[~a["Gene symbol"].str.contains("///")]
    return a.set_index("ID")["Gene symbol"]

def collapse(E, sym):
    E = E.loc[E.index.intersection(sym.index)].copy()
    if np.nanmax(E.values) > 100: E = np.log2(E.clip(lower=0) + 1)
    E["__s"] = sym.loc[E.index].values; E["__m"] = E.drop(columns="__s").mean(1)
    return E.sort_values("__m", ascending=False).drop_duplicates("__s").set_index("__s").drop(columns="__m").T

def stage_roman(s):
    s = str(s).upper().strip()
    for k, v in [("IV", "IV"), ("4", "IV"), ("III", "III"), ("3", "III"), ("II", "II"), ("2", "II"), ("I", "I"), ("1", "I")]:
        if s.startswith(k): return v
    return np.nan
def tnm_stage(t, n, m):   # AJCC 7th approximation from pT/pN/pM (T2a/T2b not distinguishable -> T2N0 = I)
    t, n, m = (str(x).upper() for x in (t, n, m))
    if "X" in n or "X" in t or t in ("NAN", "NONE") : return np.nan
    if "M1" in m: return "IV"
    tn, nn = re.findall(r"\d", t), re.findall(r"\d", n)
    if not tn or not nn: return np.nan
    T, N = int(tn[0]), int(nn[0])
    if N == 3 or N == 2 or (T == 4): return "III"
    if N == 1: return "III" if T == 3 else "II"
    return "I" if T <= 2 else "II"

cohorts = {}
num = lambda s: pd.to_numeric(s, errors="coerce")
# GSE50081
c, E = read_series("GSE50081"); g = collapse(E, annot("GPL570"))
c = c[c.histology == "adenocarcinoma"]
cohorts["GSE50081"] = (g, pd.DataFrame({"age": num(c.age), "male": (c.Sex == "M").astype(float),
    "stage": c.Stage.map(stage_roman), "OS_days": num(c["survival time"]) * YEAR,
    "ever_smoker": c.smoking.map({"Current": 1, "Ex-smoker": 1, "Never": 0}),
    "OS_event": c.status.map({"dead": 1, "alive": 0}), "adjuvant": np.nan, "region": "Western"}))
# GSE30219
c, E = read_series("GSE30219"); g = collapse(E, annot("GPL570"))
c = c[(c.histology == "ADC") & (c.tissue == "lung tumour")]
cohorts["GSE30219"] = (g, pd.DataFrame({"age": num(c["age at surgery"]), "male": (c.gender == "M").astype(float),
    "stage": [tnm_stage(a, b, d) for a, b, d in zip(c["pt stage"], c["pn stage"], c["pm stage"])],
    "OS_days": num(c["follow-up time (months)"]) * 30.4375, "OS_event": c.status.map({"DEAD": 1, "ALIVE": 0}),
    "adjuvant": np.nan, "region": "Western"}, index=c.index))
# GSE37745
c, E = read_series("GSE37745"); g = collapse(E, annot("GPL570"))
c = c[c.histology == "adeno"]
cohorts["GSE37745"] = (g, pd.DataFrame({"age": num(c.age), "male": (c.gender == "male").astype(float),
    "stage": c["tumor stage"].map(stage_roman), "OS_days": num(c["days to determined death status"]),
    "OS_event": c.dead.map({"yes": 1, "no": 0}), "adjuvant": c["adjuvant treatment"].map({"yes": 1, "no": 0}),
    "region": "Western"}))
# GSE42127
c, E = read_series("GSE42127"); g = collapse(E, annot("GPL6884"))
c = c[c.histology == "Adenocarcinoma"]
cohorts["GSE42127"] = (g, pd.DataFrame({"age": num(c["age at surgery"]), "male": (c.gender == "M").astype(float),
    "stage": c["final.pat.stage"].map(stage_roman), "OS_days": num(c["overall survival months"]) * 30.4375,
    "OS_event": c["survival status"].map({"D": 1, "A": 0}),
    "adjuvant": c.had_adjuvant_chemo.map({"TRUE": 1, "FALSE": 0}), "region": "Western"}))
# GSE13213
c, E = read_series("GSE13213"); g = collapse(E, annot("GPL6480"))
cohorts["GSE13213"] = (g, pd.DataFrame({"age": num(c.Age), "male": (c.Sex == "M").astype(float),
    "stage": c["Stage (Pathological )"].map(stage_roman), "OS_days": num(c["Survival (days)"]),
    "ever_smoker": (num(c["Smoking (BI)"]) > 0).astype(float).where(num(c["Smoking (BI)"]).notna()),
    "OS_event": c.Status.map({"Dead": 1, "Alive": 0}), "adjuvant": np.nan, "region": "East Asian"}))
# original cohorts (from R07a), for the pooled analyses
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
tc = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
orig = {"TCGA-LUAD": (lc[tc.index].T, pd.DataFrame({"age": tc.age, "male": (tc.sex == "Male").astype(float),
        "stage": tc.stage, "OS_days": tc.OS_days, "OS_event": tc.OS_event, "adjuvant": np.nan,
        "region": "Western"}, index=tc.index))}
for gse, reg in [("GSE31210", "East Asian"), ("GSE68465", "Western"), ("GSE72094", "Western")]:
    E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
    C = C.assign(adjuvant=C.get("adjuvant_chemo", np.nan), region=reg)
    if gse == "GSE31210": C["adjuvant"] = np.where(C.exclude_prognosis == 1, np.nan, 0)  # excluded = incomplete resection/adjuvant
    orig[gse] = (E, C)
allc = {**orig, **cohorts}

def zscore(x): return (x - x.mean()) / x.std()
def design(C, s):
    d = pd.DataFrame({"T": C.OS_days, "E": C.OS_event, "age": C.age, "male": C.male, "s": s})
    for st in ["II", "III", "IV"]:
        if (C.stage == st).sum() >= 5: d[f"stage_{st}"] = (C.stage == st).astype(float).where(C.stage.notna())
    if "site" in C and C.site.nunique() > 1: d["site"] = C.site
    return d
def fit(d):
    strata = ["site"] if "site" in d else None
    return CoxPHFitter(penalizer=1e-4).fit(d, "T", "E", strata=strata)

rows, pooled, grows = [], [], []
for name, (E, C) in allc.items():
    C = C.loc[C.index.intersection(E.index)]
    C = C[(C.OS_days > 0) & C.OS_event.notna()]
    s = zscore(zscore(E.loc[C.index, HUB7]).mean(1))
    d = design(C, s).dropna()
    mu = CoxPHFitter().fit(d[["T", "E", "s"]], "T", "E"); mm = fit(d); mc = fit(d.drop(columns="s"))
    rk = KaplanMeierFitter().fit(C.OS_days / YEAR, 1 - C.OS_event)
    c_cl = concordance_index(d["T"], -mc.predict_partial_hazard(d), d["E"])
    c_mm = concordance_index(d["T"], -mm.predict_partial_hazard(d), d["E"])
    r = dict(cohort=name, new=name in cohorts, region=C.region.iloc[0], n=len(d), events=int(d.E.sum()),
             genes_present=int(E.columns.isin(HUB7).sum()),
             median_followup_y=rk.median_survival_time_, max_followup_y=C.OS_days.max() / YEAR,
             stage_I_pct=100 * (C.stage == "I").mean(),
             HR_uni=np.exp(mu.params_.s), p_uni=mu.summary.p.s,
             HR_multi=np.exp(mm.params_.s), lo_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[0]),
             hi_multi=np.exp(mm.confidence_intervals_.loc["s"].iloc[1]), p_multi=mm.summary.p.s,
             se_multi=mm.standard_errors_.s, se_uni=mu.standard_errors_.s,
             lo_uni=np.exp(mu.confidence_intervals_.loc["s"].iloc[0]), hi_uni=np.exp(mu.confidence_intervals_.loc["s"].iloc[1]),
             C_uni=concordance_index(d["T"], -d["s"], d["E"]), C_clinical=c_cl, C_clinical_plus_score=c_mm, delta_C=c_mm - c_cl,
             LRT_p=chi2.sf(2 * (mm.log_likelihood_ - mc.log_likelihood_), 1))
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    y_ = Surv.from_arrays(d.E.astype(bool), d["T"])
    for yr in [3, 5]:
        t_ = yr * YEAR
        if t_ < d["T"][d.E == 1].max() and (d["T"] > t_).sum() > 10:
            r[f"tdAUC_{yr}y"] = float(cumulative_dynamic_auc(y_, y_, d["s"].values, [t_])[0][0])
    # 0-5 years and 5-year landmark
    d5 = d.copy(); late = d5["T"] > 5 * YEAR; d5.loc[late, "E"] = 0; d5.loc[late, "T"] = 5 * YEAR
    m5 = fit(d5); r.update(HR_0_5y=np.exp(m5.params_.s), p_0_5y=m5.summary.p.s, se_0_5y=m5.standard_errors_.s)
    dl = d[d["T"] > 5 * YEAR].copy()
    if dl.E.sum() >= 10:
        dl["T"] -= 5 * YEAR; ml = fit(dl)
        r.update(n_landmark5=len(dl), events_landmark5=int(dl.E.sum()), HR_after_5y=np.exp(ml.params_.s),
                 lo_after_5y=np.exp(ml.confidence_intervals_.loc["s"].iloc[0]),
                 hi_after_5y=np.exp(ml.confidence_intervals_.loc["s"].iloc[1]), p_after_5y=ml.summary.p.s,
                 se_after_5y=ml.standard_errors_.s)
    # treatment
    if C.adjuvant.notna().sum() > 20 and C.adjuvant.nunique() > 1:
        du = d.join(C.adjuvant).dropna()
        dn = du[du.adjuvant == 0].drop(columns="adjuvant")
        mn = fit(dn)
        di = du.assign(sxa=du.s * du.adjuvant); mi = fit(di)
        r.update(n_no_adjuvant=len(dn), events_no_adjuvant=int(dn.E.sum()), HR_no_adjuvant=np.exp(mn.params_.s),
                 lo_no_adjuvant=np.exp(mn.confidence_intervals_.loc["s"].iloc[0]),
                 hi_no_adjuvant=np.exp(mn.confidence_intervals_.loc["s"].iloc[1]), p_no_adjuvant=mn.summary.p.s,
                 n_adjuvant=int((du.adjuvant == 1).sum()), interaction_HR=np.exp(mi.params_.sxa),
                 interaction_p=mi.summary.p.sxa)
    elif name == "GSE31210":
        dn = d.loc[d.index.intersection(C.index[C.adjuvant == 0])]; mn = fit(dn)
        r.update(n_no_adjuvant=len(dn), events_no_adjuvant=int(dn.E.sum()), HR_no_adjuvant=np.exp(mn.params_.s),
                 lo_no_adjuvant=np.exp(mn.confidence_intervals_.loc["s"].iloc[0]),
                 hi_no_adjuvant=np.exp(mn.confidence_intervals_.loc["s"].iloc[1]), p_no_adjuvant=mn.summary.p.s)
    rows.append(r)
    Zg = zscore(E.loc[d.index, HUB7])
    for gene in HUB7:
        dg = d.drop(columns="s").join(zscore(Zg[gene]).rename("s")); mg = fit(dg)
        grows.append(dict(cohort=name, gene=gene, HR_multi=np.exp(mg.params_.s), se_multi=mg.standard_errors_.s,
                          p_multi=mg.summary.p.s))
    if r["max_followup_y"] > 8:
        pooled.append(d[["T", "E", "s"]].assign(cohort=name))
    print(name, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "R13g_cohort_results.csv", index=False)
GR = pd.DataFrame(grows)
def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
GR["FDR_multi"] = GR.groupby("cohort").p_multi.transform(_bh)
for gse in cohorts:
    E, C = cohorts[gse]; C.to_csv(OUT / f"{gse}_clinical_adeno.csv")

# time-varying effect in pooled long-follow-up cohorts (stratified by cohort)
P = pd.concat(pooled)
from lifelines import CoxTimeVaryingFitter
Pm = CoxPHFitter().fit(P.assign(s_logt=0)[["T", "E", "s", "cohort"]], "T", "E", strata=["cohort"])
from lifelines.statistics import proportional_hazard_test
ph = proportional_hazard_test(Pm, P[["T", "E", "s", "cohort"]], time_transform="log")
tv = dict(n=len(P), events=int(P.E.sum()), cohorts=";".join(P.cohort.unique()), HR=np.exp(Pm.params_.s),
          PH_test_p_logtime=float(ph.summary.p.iloc[0]))
pd.Series(tv).to_csv(OUT / "R13g_time_varying.csv", header=["value"])

def dl_meta(y, se):
    w = 1 / se ** 2; yb = (w * y).sum() / w.sum(); Q = (w * (y - yb) ** 2).sum(); k = len(y)
    t2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); ws = 1 / (se ** 2 + t2)
    mu = (ws * y).sum() / ws.sum(); s = np.sqrt(1 / ws.sum())
    return dict(k=k, HR=np.exp(mu), lo=np.exp(mu - 1.96 * s), hi=np.exp(mu + 1.96 * s), p=2 * norm.sf(abs(mu / s)),
                I2=100 * max(0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, tau2=t2)
M = []
for lab, sub, col, se in [("all 9 cohorts, multivariable", R, "HR_multi", "se_multi"),
                          ("all 9 cohorts, univariable", R, "HR_uni", "se_uni"),
                          ("5 new cohorts, multivariable", R[R.new], "HR_multi", "se_multi"),
                          ("East Asian, multivariable", R[R.region == "East Asian"], "HR_multi", "se_multi"),
                          ("Western, multivariable", R[R.region == "Western"], "HR_multi", "se_multi"),
                          ("deaths within 5 years", R, "HR_0_5y", "se_0_5y"),
                          ("5-year landmark (after 5 years)", R.dropna(subset=["HR_after_5y"]), "HR_after_5y", "se_after_5y")]:
    M.append(dict(analysis=lab, **dl_meta(np.log(sub[col].values), sub[se].values)))
na = R.dropna(subset=["HR_no_adjuvant"])
M.append(dict(analysis="no adjuvant therapy", **dl_meta(np.log(na.HR_no_adjuvant.values),
              ((np.log(na.hi_no_adjuvant) - np.log(na.lo_no_adjuvant)) / 3.92).values)))
for cname in R.cohort:
    sub = R[R.cohort != cname]
    M.append(dict(analysis=f"leave-one-out: without {cname}", **dl_meta(np.log(sub.HR_multi.values), sub.se_multi.values)))
for gene in HUB7:
    sub = GR[GR.gene == gene]
    M.append(dict(analysis=f"gene {gene}, all 9 cohorts, multivariable", **dl_meta(np.log(sub.HR_multi.values), sub.se_multi.values)))
GR.to_csv(OUT / "R13g_single_genes_all_cohorts.csv", index=False)
M = pd.DataFrame(M); M.to_csv(OUT / "R13g_meta_analysis.csv", index=False)
pd.set_option("display.width", 250)
print(R[["cohort", "region", "n", "events", "genes_present", "median_followup_y", "max_followup_y", "HR_multi", "lo_multi",
         "hi_multi", "p_multi", "C_uni", "delta_C", "HR_0_5y", "HR_after_5y", "p_after_5y", "events_landmark5"]].round(3).to_string(index=False))
print(R[["cohort", "n_no_adjuvant", "HR_no_adjuvant", "p_no_adjuvant", "n_adjuvant", "interaction_HR", "interaction_p"]].dropna(subset=["n_no_adjuvant"]).round(3).to_string(index=False))
print(pd.Series(tv).to_string()); print(M.round(3).to_string(index=False))
