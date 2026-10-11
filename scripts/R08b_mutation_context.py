"""
R08b_mutation_context.py — Revision step 8b (PI comments M10, M12)
====================================================================
1. Patient-level mutation frequencies (TCGA PanCanAtlas MC3 via cBioPortal; R08a)
   protein-altering = missense, nonsense, frameshift, in-frame indel, splice site (Splice_Region excluded)
   driver definitions
     KRAS   codon 12/13/61 hotspots          EGFR  activating (L858R, L861Q, G719X, S768I, T790M,
     TP53, STK11, KEAP1  any protein-altering         exon-19 in-frame deletions, exon-20 insertions)
     BRAF   V600E                             ALK   NOT assessed: ALK alterations in LUAD are fusions,
                                                    which mutation calls do not capture
2. Interaction tests (replace the four-group log-rank tests)
   Cox: score_z + mutation + score_z x mutation + age + sex + stage, for OS and PFI (TCGA)
   subgroup HR per SD of the 7-gene score within mutant and wild-type tumours (same covariates)
   BH correction across the mutations tested, per endpoint
   replication: GSE72094 (KRAS, EGFR, TP53, STK11 as provided), GSE31210 (EGFR, KRAS)
3. Score distribution by mutation status (Mann-Whitney)
4. Stage trend (PI: Kruskal-Wallis tests heterogeneity, not trend): Jonckheere-Terpstra test and
   Spearman rho across stage I-IV, TCGA and GEO cohorts
Outputs: analysis/08_mutation_context/
"""
import json, re, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, norm
from lifelines import CoxPHFitter

NEW = Path(__file__).resolve().parents[1]
OUT = NEW / "08_mutation_context"; RAW = OUT / "raw"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
OLD5 = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
fmt = lambda v: f"{v:.3g}"

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

# ── 1. mutations ──
m = pd.DataFrame([dict(patient=x["patientId"], sample=x["sampleId"], gene=x["gene"]["hugoGeneSymbol"],
                       type=x["mutationType"], aa=str(x.get("proteinChange", "")),
                       pos=x.get("proteinPosStart")) for x in json.load(open(RAW / "mutations_selected_genes.json"))])
seq = [s for s in json.load(open(RAW / "sequenced_samples.json")) if s.endswith("-01")]
seq_pat = sorted({s[:12] for s in seq})
PA = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del",
      "In_Frame_Ins", "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"}
m = m[m.type.isin(PA) & m.patient.isin(seq_pat)]

def egfr_act(r):
    aa = r.aa
    if re.match(r"^(L858R|L861Q|S768I|T790M|G719[ACDS])$", aa): return True
    if r.type == "In_Frame_Del" and r.pos and 729 <= r.pos <= 761: return True
    if r.type == "In_Frame_Ins" and r.pos and 762 <= r.pos <= 774: return True
    return False

drv = pd.DataFrame(index=seq_pat)
drv["TP53"] = drv.index.isin(m[m.gene == "TP53"].patient)
drv["KRAS"] = drv.index.isin(m[(m.gene == "KRAS") & m.pos.isin([12, 13, 61])].patient)
drv["EGFR"] = drv.index.isin(m[(m.gene == "EGFR")][m[m.gene == "EGFR"].apply(egfr_act, axis=1)].patient) \
    if (m.gene == "EGFR").any() else False
drv["STK11"] = drv.index.isin(m[m.gene == "STK11"].patient)
drv["KEAP1"] = drv.index.isin(m[m.gene == "KEAP1"].patient)
drv["BRAF_V600E"] = drv.index.isin(m[(m.gene == "BRAF") & (m.aa == "V600E")].patient)
drv = drv.astype(int)
drv.to_csv(OUT / "TCGA_driver_status_patient_level.csv")

freq = []
for g in ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "BRAF"] + HUB7 + [g for g in OLD5 if g not in HUB7]:
    sub = m[m.gene == g]
    freq.append(dict(gene=g, role="hub (current)" if g in HUB7 else ("hub (original)" if g in OLD5 else "driver"),
                     patients_any_protein_altering=sub.patient.nunique(),
                     pct_any=100 * sub.patient.nunique() / len(seq_pat),
                     driver_definition_patients=int(drv[g].sum()) if g in drv else (int(drv["BRAF_V600E"].sum()) if g == "BRAF" else np.nan),
                     n_records=len(sub), types=sub.type.value_counts().to_dict()))
freq = pd.DataFrame(freq)
freq["pct_driver_definition"] = 100 * freq.driver_definition_patients / len(seq_pat)
freq.to_csv(OUT / "mutation_frequencies_patient_level.csv", index=False)

# ── TCGA analysis frame ──
X = pd.read_csv(NEW / "05_survival" / "hub_expression_log2CPM_tumours.csv", index_col=0)
clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
Z = (X - X.mean()) / X.std()
score = Z[HUB7].mean(1)
d = clin[["patient_id", "age", "sex", "stage", "OS_event", "OS_days", "PFI_event", "PFI_days"]].copy()
d["score"] = (score - score.mean()) / score.std()
d = d.join(drv, on="patient_id")
d = d[d.patient_id.isin(seq_pat)]
d["male"] = (d.sex == "Male").astype(float); d["age10"] = d.age / 10
for s in ["II", "III", "IV"]: d[f"st_{s}"] = (d.stage == s).astype(float).where(d.stage.notna())
COV = ["age10", "male", "st_II", "st_III", "st_IV"]

def inter_model(df, mut, ep, cov):
    f = pd.DataFrame({"T": df[f"{ep}_days"], "E": df[f"{ep}_event"], "score": df.score, "mut": df[mut]})
    for c in cov: f[c] = df[c]
    f = f.dropna(); f["score_x_mut"] = f.score * f["mut"]
    m_ = CoxPHFitter().fit(f, "T", "E")
    out = dict(n=len(f), events=int(f.E.sum()), n_mut=int(f["mut"].sum()), events_mut=int(f.E[f["mut"] == 1].sum()),
               HR_interaction=np.exp(m_.params_["score_x_mut"]), p_interaction=m_.summary.loc["score_x_mut", "p"],
               HR_mut_main=np.exp(m_.params_["mut"]), p_mut_main=m_.summary.loc["mut", "p"])
    for lab, val in [("WT", 0), ("Mut", 1)]:
        s = f[f["mut"] == val].drop(columns=["mut", "score_x_mut"])
        s = s.loc[:, s.nunique() > 1]
        ms = CoxPHFitter(penalizer=1e-4).fit(s, "T", "E")
        ci = np.exp(ms.confidence_intervals_.loc["score"])
        out.update({f"HR_{lab}": np.exp(ms.params_["score"]), f"lo_{lab}": ci.iloc[0], f"hi_{lab}": ci.iloc[1],
                    f"p_{lab}": ms.summary.loc["score", "p"]})
    return out

rows = []
for ep in ["OS", "PFI"]:
    for mut in ["TP53", "KRAS", "EGFR", "STK11", "KEAP1"]:
        rows.append(dict(cohort="TCGA-LUAD", endpoint=ep, mutation=mut, **inter_model(d, mut, ep, COV)))

# ── GEO replication ──
DAT = NEW / "07_external_validation" / "data"
for gse, muts in [("GSE72094", ["TP53", "KRAS", "EGFR", "STK11"]), ("GSE31210", ["EGFR", "KRAS"])]:
    E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0)
    C = C[C.tumour == 1]; E = E.loc[C.index]
    s = ((E[HUB7] - E[HUB7].mean()) / E[HUB7].std()).mean(1)
    g = C.copy(); g["score"] = (s - s.mean()) / s.std(); g["age10"] = g.age / 10
    for st in ["II", "III", "IV"]:
        if (g.stage == st).sum() >= 5: g[f"st_{st}"] = (g.stage == st).astype(float).where(g.stage.notna())
    if gse == "GSE31210":
        g["EGFR"] = (g.driver == "EGFR mutation +").astype(int); g["KRAS"] = (g.driver == "KRAS mutation +").astype(int)
    else:
        for mu in muts: g[mu] = g[f"{mu}_mut"]
    cov = ["age10", "male"] + [c for c in g.columns if c.startswith("st_")]
    for mu in muts:
        if g[mu].sum() >= 10:
            rows.append(dict(cohort=gse, endpoint="OS", mutation=mu, **inter_model(g, mu, "OS", cov)))
res = pd.DataFrame(rows)
for (c, e), idx in res.groupby(["cohort", "endpoint"]).groups.items():
    res.loc[idx, "FDR_interaction"] = bh(res.loc[idx, "p_interaction"])
res.to_csv(OUT / "score_by_mutation_interaction_cox.csv", index=False)

# ── 3. score by mutation status ──
assoc = []
for mut in ["TP53", "KRAS", "EGFR", "STK11", "KEAP1"]:
    a, b = d.score[d[mut] == 1], d.score[d[mut] == 0]
    assoc.append(dict(mutation=mut, n_mut=len(a), median_mut=a.median(), median_wt=b.median(),
                      p=mannwhitneyu(a, b).pvalue))
assoc = pd.DataFrame(assoc); assoc["FDR"] = bh(assoc.p); assoc.to_csv(OUT / "score_by_mutation_status.csv", index=False)

# ── 4. stage trend ──
def jonckheere(x, groups):
    """Jonckheere-Terpstra test for ordered alternative (normal approximation, ties ignored)."""
    levels = sorted(set(groups)); xs = [x[groups == l] for l in levels]
    J = sum((np.sign(b[None, :] - a[:, None]) + 1).sum() / 2 for i, a in enumerate(xs) for b in xs[i + 1:])
    n = np.array([len(v) for v in xs]); N = n.sum()
    mu = (N ** 2 - (n ** 2).sum()) / 4
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    z = (J - mu) / np.sqrt(var)
    return z, 2 * norm.sf(abs(z))

trend = []
sets = {"TCGA-LUAD": (Z.join(clin.stage), None)}
for gse in ["GSE68465", "GSE72094", "GSE31210"]:
    E = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0)
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
    sets[gse] = (((E.loc[C.index, HUB7] - E.loc[C.index, HUB7].mean()) / E.loc[C.index, HUB7].std()).join(C.stage), None)
for coh, (df, _) in sets.items():
    df = df.dropna(subset=["stage"]); df["HUB7_score"] = df[HUB7].mean(1)
    st = df.stage.map({"I": 1, "II": 2, "III": 3, "IV": 4})
    for v in HUB7 + ["HUB7_score"]:
        z, p = jonckheere(df[v].values, st.values)
        rho, prho = spearmanr(df[v], st)
        trend.append(dict(cohort=coh, variable=v, n=len(df), stages=st.value_counts().sort_index().to_dict(),
                          JT_z=z, JT_p=p, spearman_rho=rho, spearman_p=prho,
                          median_by_stage={k: round(float(x), 2) for k, x in df.groupby(st)[v].median().items()}))
trend = pd.DataFrame(trend)
for c, idx in trend.groupby("cohort").groups.items():
    trend.loc[idx, "JT_FDR"] = bh(trend.loc[idx, "JT_p"])
trend.to_csv(OUT / "stage_trend_jonckheere.csv", index=False)

# ── figure: subgroup forest ──
r = res[res.endpoint == "OS"].reset_index(drop=True)
lab, hr, lo, hi, col = [], [], [], [], []
for _, x in r.iterrows():
    for g_, c_ in [("Mut", "#c0392b"), ("WT", "#2980b9")]:
        n_ = x.n_mut if g_ == "Mut" else x.n - x.n_mut
        ev = x.events_mut if g_ == "Mut" else x.events - x.events_mut
        few = ev < 10
        lab.append(f"{x.cohort}  {x.mutation} {'mutant' if g_ == 'Mut' else 'wild-type'} (n={int(n_)}, {int(ev)} deaths)"
                   + ("  — too few events" if few else ""))
        hr.append(np.nan if few else x[f"HR_{g_}"]); lo.append(x[f"lo_{g_}"]); hi.append(x[f"hi_{g_}"]); col.append(c_)
    lab.append(f"      interaction p = {x.p_interaction:.3f} (FDR {x.FDR_interaction:.2f})"); hr.append(np.nan)
    lo.append(np.nan); hi.append(np.nan); col.append("w")
fig, ax = plt.subplots(figsize=(9.5, 0.33 * len(lab) + 1.2))
yy = np.arange(len(lab))[::-1]
for y, h, l, u, c in zip(yy, hr, lo, hi, col):
    if np.isfinite(h):
        ax.errorbar(h, y, xerr=[[h - l], [min(u, 6) - h]], fmt="s", color=c, ms=4, capsize=2)
ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=7.5)
for t, h in zip(ax.get_yticklabels(), hr):
    if not np.isfinite(h): t.set_fontstyle("italic"); t.set_color("#555555")
    if t.get_text().startswith("GSE") or t.get_text().startswith("TCGA"): t.set_fontstyle("normal"); t.set_color("black")
ax.axvline(1, ls="--", c="k", lw=.7); ax.set_xscale("log"); ax.set_xlim(0.3, 6)
ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel("HR per 1 SD of 7-gene score within subgroup (OS; adjusted for age, sex, stage)")
ax.set_title("Prognostic effect of the 7-gene score by driver-mutation status")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_subgroup_forest_mutation.png", dpi=300); plt.close()

# ── report ──
rep = [f"R08 MUTATION CONTEXT  (TCGA sequenced patients: {len(seq_pat)}; with RNA-seq & survival: {d.OS_days.notna().sum()})",
       "\nMutation frequencies (patient level, protein-altering):",
       freq[["gene", "role", "patients_any_protein_altering", "pct_any", "driver_definition_patients",
             "pct_driver_definition"]].to_string(index=False, float_format=fmt),
       "\nInteraction Cox models:",
       res[["cohort", "endpoint", "mutation", "n", "events", "n_mut", "events_mut", "HR_Mut", "lo_Mut", "hi_Mut",
            "HR_WT", "lo_WT", "hi_WT", "HR_interaction", "p_interaction", "FDR_interaction"]].to_string(index=False, float_format=fmt),
       "\nScore by mutation status (TCGA):", assoc.to_string(index=False, float_format=fmt),
       "\nStage trend (Jonckheere-Terpstra):",
       trend[["cohort", "variable", "n", "stages", "JT_z", "JT_p", "JT_FDR", "spearman_rho"]].to_string(index=False, float_format=fmt)]
(OUT / "R08_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
