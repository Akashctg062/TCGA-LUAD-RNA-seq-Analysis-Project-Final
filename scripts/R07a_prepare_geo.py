"""
R07a_prepare_geo.py — Revision step 7a (PI comment M6): harmonise the three GEO validation cohorts.
Uses the files already downloaded by the original pipeline (data/geo/<GSE>/), read-only.

Cohort        Platform                                   Tumours  Normals            Endpoints
GSE31210      GPL570  Affymetrix HG-U133 Plus 2.0 (MAS5)    226    20 normal lung      OS, RFS
GSE68465      GPL96   Affymetrix HG-U133A (MAS5)            443    19 Stratagene/NCI   OS, PFS (first progression/relapse)
                                                                   control RNAs -> NOT normal lung, excluded
GSE72094      GPL15048 Rosetta/Merck custom Affymetrix      442    none               OS (+ KRAS/EGFR/TP53/STK11 status)

Processing
  * probe -> gene symbol from the platform table inside the GEO family SOFT file
    (GPL15048 provides a GeneSymbol column, so GSE72094 IS annotatable — the original
     "platform annotation limitation" was incorrect)
  * probes mapping to >1 gene ("///") dropped; one probe per gene = highest mean expression
  * log2(x + 1) applied when the series is on the linear MAS5 scale (max > 100)
  * stage collapsed to I / II / III / IV; GSE68465 stage derived from pT/pN (AJCC 6th, all resected M0)
Outputs (analysis/07_external_validation/data/): <GSE>_expression_genes.csv (samples x genes, all genes),
         <GSE>_clinical.csv, cohort_summary.csv
"""
import gzip, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data" / "geo"
OUT = ROOT / "analysis" / "07_external_validation" / "data"
OUT.mkdir(parents=True, exist_ok=True)

def read_soft(gse):
    """Return (platform table, {GSM: source_name})."""
    rows, src, cur, in_tab, header = [], {}, None, False, None
    with gzip.open(GEO / gse / f"{gse}_family.soft.gz", "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_tab = True; header = None; continue
            if line.startswith("!platform_table_end"):
                in_tab = False; continue
            if in_tab:
                parts = line.rstrip("\n").split("\t")
                if header is None: header = parts
                else: rows.append(parts)
                continue
            if line.startswith("^SAMPLE"):
                cur = line.split("=")[1].strip()
            elif line.startswith("!Sample_source_name_ch1") and cur:
                src[cur] = line.split("=", 1)[1].strip()
    plat = pd.DataFrame(rows, columns=header)
    return plat, src

def gene_matrix(gse, sym_col):
    plat, src = read_soft(gse)
    sym = plat.set_index("ID")[sym_col].astype(str).str.strip()
    sym = sym[(sym != "") & (sym != "nan") & ~sym.str.contains("///")]
    e = pd.read_csv(GEO / gse / "expression.csv", index_col=0)
    e.index = e.index.astype(str)
    if np.nanmax(e.values) > 100:
        e = np.log2(e.clip(lower=0) + 1)
    e = e.loc[e.index.intersection(sym.index)]
    e["__sym"] = sym.loc[e.index].values
    e["__mean"] = e.drop(columns="__sym").mean(1)
    e = e.sort_values("__mean", ascending=False).drop_duplicates("__sym")
    g = e.set_index("__sym").drop(columns="__mean").T
    g.index.name = "sample_id"
    return g, src, len(plat)

def stage4(s):
    if pd.isna(s): return np.nan
    s = str(s).upper().replace("STAGE", "").strip()
    m = {"4": "IV", "3": "III", "2": "II", "1": "I"}
    if s and s[0] in m: return m[s[0]]
    for k in ["IV", "III", "II", "I"]:
        if s.startswith(k): return k
    return np.nan

def tnm6(code):
    m = re.match(r"pN(\w)pT(\w)", str(code))
    if not m: return np.nan
    n, t = m.groups()
    if "X" in (n, t): return np.nan
    n, t = int(n), int(t)
    if n == 3 or t == 4: return "III"
    if n == 2: return "III"
    if n == 1: return "III" if t == 3 else "II"
    return {1: "I", 2: "I", 3: "II"}[t]

summary = []
# ── GSE31210 ──
g, _, nprobe = gene_matrix("GSE31210", "Gene Symbol")
p = pd.read_csv(GEO / "GSE31210" / "phenotype.csv").set_index("sample_id")
c = pd.DataFrame(index=p.index)
c["tumour"] = (p.tissue == "primary lung tumor").astype(int)
c["age"] = p["age_(years)"]; c["male"] = (p.gender == "male").astype(float)
c["ever_smoker"] = p.smoking_status.map({"Ever-smoker": 1, "Never-smoker": 0})
c["stage"] = p.pathological_stage.map(stage4)
c["driver"] = p.gene_alteration_status
c["OS_event"] = p.death.map({"dead": 1, "alive": 0}); c["OS_days"] = p["days_before_death/censor"]
c["RFS_event"] = p.relapse.map({"relapsed": 1, "not relapsed": 0}); c["RFS_days"] = p["days_before_relapse/censor"]
c["exclude_prognosis"] = (p.exclude_for_prognosis_analysis_due_to_incomplete_resection_or_adjuvant_therapy == "exclude").astype(int)
c["site"] = "NCCRI"
cohorts = {"GSE31210": (g, c)}
# ── GSE68465 ──
g, src, _ = gene_matrix("GSE68465", "Gene Symbol")
p = pd.read_csv(GEO / "GSE68465" / "phenotype.csv").set_index("sample_id")
num = lambda s: pd.to_numeric(s.replace({"--": np.nan, "Unknown": np.nan}), errors="coerce")
c = pd.DataFrame(index=p.index)
c["tumour"] = (p.disease_state == "Lung Adenocarcinoma").astype(int)
c["age"] = p.age; c["male"] = p.sex.map({"Male": 1.0, "Female": 0.0})
c["ever_smoker"] = p.smoking_history.map({"Smoked in the past": 1, "Currently smoking": 1, "Never smoked": 0})
c["stage"] = p.disease_stage.map(tnm6)
c["adjuvant_chemo"] = p.clinical_treatment_adjuvant_chemo.map({"Yes": 1, "No": 0})
c["OS_event"] = p.vital_status.map({"Dead": 1, "Alive": 0})
c["OS_days"] = num(p.months_to_last_contact_or_death) * 30.4375
prog = p.first_progression_or_relapse
c["PFS_event"] = prog.map({"Yes": 1, "No": 0})
c["PFS_days"] = np.where(prog == "Yes", num(p.months_to_first_progression),
                         np.where(prog == "No", num(p.mths_to_last_clinical_assessment), np.nan)) * 30.4375
c["site"] = pd.Series(src).reindex(c.index)
cohorts["GSE68465"] = (g, c)
# ── GSE72094 ──
g, _, _ = gene_matrix("GSE72094", "GeneSymbol")
p = pd.read_csv(GEO / "GSE72094" / "phenotype.csv").set_index("sample_id")
c = pd.DataFrame(index=p.index)
c["tumour"] = 1
c["age"] = p.age_at_diagnosis; c["male"] = p.gender.map({"M": 1.0, "F": 0.0})
c["ever_smoker"] = p.smoking_status.map({"Ever": 1, "Never": 0})
c["stage"] = p.stage.map(stage4)
c["OS_event"] = p.vital_status.map({"Dead": 1, "Alive": 0}); c["OS_days"] = p.survival_time_in_days
for m in ["kras", "egfr", "tp53", "stk11"]:
    c[f"{m.upper()}_mut"] = p[f"{m}_status"].map({"Mut": 1, "WT": 0})
c["site"] = "Moffitt"
cohorts["GSE72094"] = (g, c)

for gse, (g, c) in cohorts.items():
    c = c.loc[g.index.intersection(c.index)]
    for e in ["OS", "RFS", "PFS"]:
        if f"{e}_days" in c:
            bad = ~(c[f"{e}_days"] > 0)
            c.loc[bad, [f"{e}_days", f"{e}_event"]] = np.nan
    g.loc[c.index].to_csv(OUT / f"{gse}_expression_genes.csv")
    c.to_csv(OUT / f"{gse}_clinical.csv")
    t = c[c.tumour == 1]
    summary.append(dict(cohort=gse, genes=g.shape[1], samples=len(c), tumours=len(t),
                        normals_used=int((c.tumour == 0).sum()) if gse == "GSE31210" else 0,
                        OS_usable=int(t.OS_days.notna().sum()), OS_events=int(t.OS_event.sum()),
                        second_endpoint=("RFS" if "RFS_days" in t else "PFS" if "PFS_days" in t else ""),
                        second_usable=int(t.get("RFS_days", t.get("PFS_days", pd.Series(dtype=float))).notna().sum()),
                        stage=t.stage.value_counts().reindex(["I", "II", "III", "IV"]).fillna(0).astype(int).to_dict(),
                        stage_missing=int(t.stage.isna().sum()),
                        sites=t.site.value_counts().to_dict(),
                        hub7_present=None))
s = pd.DataFrame(summary)
hub7 = pd.read_csv(ROOT / "analysis" / "hub_genes_final.csv").gene.tolist()
s["hub7_present"] = [sum(h in pd.read_csv(OUT / f"{g}_expression_genes.csv", nrows=0).columns for h in hub7)
                     for g in s.cohort]
s.to_csv(OUT / "cohort_summary.csv", index=False)
print(s.T.to_string())
