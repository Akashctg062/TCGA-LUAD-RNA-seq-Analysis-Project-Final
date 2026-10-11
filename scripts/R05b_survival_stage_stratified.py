"""
R05b_survival_stage_stratified.py — sensitivity analysis for R05.
Stage terms violated proportional hazards in R05 (Schoenfeld test), so the multivariable Cox models
are refitted STRATIFIED by stage (I / II / III / IV; separate baseline hazards), adjusted for
age, sex and smoking. HR per 1 SD of log2 CPM.
Output: analysis/05_survival/sensitivity_stage_stratified_cox.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

NEW = Path(__file__).resolve().parents[1]
OUT = NEW / "05_survival"
HUBS = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
X = pd.read_csv(OUT / "hub_expression_log2CPM_tumours.csv", index_col=0)
d = clin.join((X - X.mean()) / X.std())
d["age10"] = d.age / 10
d["male"] = (d.sex == "Male").astype(float)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

rows = []
for ep in ["OS", "PFI", "DSS"]:
    for g in HUBS:
        f = d[[f"{ep}_days", f"{ep}_event", g, "age10", "male", "ever_smoker", "stage"]].dropna()
        f.columns = ["T", "E", g, "age10", "male", "ever_smoker", "stage"]
        m = CoxPHFitter().fit(f, "T", "E", strata=["stage"])
        ph = proportional_hazard_test(m, f, time_transform="rank").summary
        ci = np.exp(m.confidence_intervals_.loc[g])
        rows.append(dict(endpoint=ep, gene=g, n=len(f), events=int(f.E.sum()),
                         HR=np.exp(m.params_[g]), lo=ci.iloc[0], hi=ci.iloc[1], p=m.summary.loc[g, "p"],
                         ph_p_gene=ph.loc[g, "p"], ph_min_p_any_term=ph.p.min()))
r = pd.DataFrame(rows)
for ep in r.endpoint.unique():
    m = r.endpoint == ep; r.loc[m, "FDR"] = bh(r.loc[m, "p"])
r.to_csv(OUT / "sensitivity_stage_stratified_cox.csv", index=False)
print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
