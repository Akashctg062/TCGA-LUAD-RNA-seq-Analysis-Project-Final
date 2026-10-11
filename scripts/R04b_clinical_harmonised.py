"""
R04b_clinical_harmonised.py — Revision step 4b (PI comment M5, M13)
=====================================================================
Builds one clean clinical table for the 517 TCGA-LUAD tumours with RNA-seq.

Sources (downloaded by R04a, raw files untouched):
  * TCGA-CDR (Liu et al., Cell 2018): sex, age, AJCC pathological stage, residual tumour,
    curated endpoints OS / DSS / PFI (+ times)               -> PRIMARY source
  * GDC case records: smoking status, pack-years, and stage/T/N/M from the PRIMARY diagnosis
    (diagnosis_is_primary_disease = True)                    -> smoking + stage fallback

Diagnosis of the old error: GDC cases often hold several "diagnoses" records (primary tumour,
recurrence, not reported). Only the primary record carries the AJCC stage; reading another
record gives a missing stage. Sex is not returned in the current GDC demographic record.

Outputs: analysis/04_clinical/
  clinical_tumour_harmonised.csv   one row per tumour sample (517)
  completeness_old_vs_new.csv      missing-data comparison
  stage_old_vs_new_crosstab.csv
  sample_flow.txt                  numbers for the sample-flow diagram (M13)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "04_clinical"; RAW = OUT / "raw"
meta = pd.read_csv(ROOT / "data" / "metadata.csv")

# ── sources ──
cdr = pd.read_excel(RAW / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
cdr = cdr[cdr.type == "LUAD"].set_index("bcr_patient_barcode")
NA_TOKENS = {"[Not Available]", "[Not Applicable]", "[Unknown]", "[Discrepancy]", "[Not Evaluated]",
             "[Completed]", "[Not Applicable]", "nan", "Not Reported", "not reported", "Unknown"}
cdr = cdr.replace(list(NA_TOKENS), np.nan)

hits = json.load(open(RAW / "gdc_TCGA-LUAD_cases.json", encoding="utf-8"))
rows = []
for h in hits:
    prim = [d for d in h.get("diagnoses", []) if d.get("diagnosis_is_primary_disease")]
    d = prim[0] if prim else {}
    ex = (h.get("exposures") or [{}])[0]
    rows.append(dict(patient_id=h["submitter_id"],
                     gdc_stage=d.get("ajcc_pathologic_stage"), gdc_T=d.get("ajcc_pathologic_t"),
                     gdc_N=d.get("ajcc_pathologic_n"), gdc_M=d.get("ajcc_pathologic_m"),
                     gdc_ajcc_edition=d.get("ajcc_staging_system_edition"),
                     gdc_residual=d.get("residual_disease"),
                     smoking_raw=ex.get("tobacco_smoking_status"), pack_years=ex.get("pack_years_smoked")))
gdc = pd.DataFrame(rows).set_index("patient_id").replace(list(NA_TOKENS), np.nan)

# ── tumour samples ──
tum = meta[meta.condition == "Tumor"][["sample_submitter_id", "patient_id"]].copy()
assert tum.patient_id.is_unique, "more than one tumour sample per patient"
df = tum.join(cdr, on="patient_id").join(gdc, on="patient_id")

def stage_group(s):
    if pd.isna(s): return np.nan
    s = str(s).upper().replace("STAGE", "").strip()
    for k, v in [("IV", "IV"), ("III", "III"), ("II", "II"), ("I", "I")]:
        if s.startswith(k): return v
    return np.nan

df["stage_cdr"] = df.ajcc_pathologic_tumor_stage.map(stage_group)
df["stage_gdc"] = df.gdc_stage.map(stage_group)
df["stage"] = df.stage_cdr.fillna(df.stage_gdc)
df["stage_source"] = np.where(df.stage_cdr.notna(), "CDR", np.where(df.stage_gdc.notna(), "GDC", "missing"))
df["stage_num"] = df.stage.map({"I": 1, "II": 2, "III": 3, "IV": 4})
df["stage_early"] = df.stage_num.map(lambda x: np.nan if pd.isna(x) else int(x >= 3))   # 0 = I-II, 1 = III-IV

df["sex"] = df.gender.str.capitalize()
df["age"] = df.age_at_initial_pathologic_diagnosis

def smoke(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if "lifelong non" in s or "never" in s: return "Never"
    if "reformed" in s or "former" in s: return "Former"
    if "current" in s: return "Current"
    return np.nan
df["smoking"] = df.smoking_raw.map(smoke)
df["ever_smoker"] = df.smoking.map({"Never": 0, "Former": 1, "Current": 1})

res = df.residual_tumor.fillna(df.gdc_residual)
df["residual_tumour"] = res.where(res.isin(["R0", "R1", "R2", "RX"]))

for e in ["OS", "DSS", "PFI", "DFI"]:
    df[f"{e}_event"] = df[e]
    df[f"{e}_days"] = df[f"{e}.time"]
    bad = df[f"{e}_days"] <= 0
    df.loc[bad, [f"{e}_event", f"{e}_days"]] = np.nan          # zero/negative follow-up not usable
    df[f"{e}_years"] = df[f"{e}_days"] / 365.25

keep = ["sample_submitter_id", "patient_id", "age", "sex", "smoking", "ever_smoker", "pack_years",
        "stage", "stage_num", "stage_early", "stage_source", "gdc_T", "gdc_N", "gdc_M", "gdc_ajcc_edition",
        "residual_tumour"] + [f"{e}_{s}" for e in ["OS", "DSS", "PFI", "DFI"] for s in ["event", "days", "years"]]
clin = df[keep].rename(columns={"gdc_T": "pT", "gdc_N": "pN", "gdc_M": "pM", "gdc_ajcc_edition": "ajcc_edition"})
clin.to_csv(OUT / "clinical_tumour_harmonised.csv", index=False)

# ── comparison with the old metadata ──
old = meta[meta.condition == "Tumor"].set_index("sample_submitter_id")
new = clin.set_index("sample_submitter_id")
comp = pd.DataFrame({
    "old_metadata_missing": [old.tumor_stage.isna().sum(), old.gender.isna().sum(),
                             old.tobacco_smoking_status.isna().sum(), old.os_time_days.isna().sum(), np.nan, np.nan],
    "new_missing": [new.stage.isna().sum(), new.sex.isna().sum(), new.smoking.isna().sum(),
                    new.OS_days.isna().sum(), new.PFI_days.isna().sum(), new.DSS_days.isna().sum()],
}, index=["stage", "sex", "smoking", "OS time", "PFI time", "DSS time"])
comp["new_available"] = len(new) - comp.new_missing
comp.to_csv(OUT / "completeness_old_vs_new.csv")
ct = pd.crosstab(old.tumor_stage.map(stage_group).fillna("missing"), new.stage.fillna("missing"),
                 rownames=["old"], colnames=["new"])
ct.to_csv(OUT / "stage_old_vs_new_crosstab.csv")

cc_cox = new.dropna(subset=["OS_days", "OS_event", "stage_num", "age", "sex", "ever_smoker"])
flow = [
    f"GDC TCGA-LUAD STAR-Counts samples retained after QC: {len(meta)} (tumour {len(tum)}, normal {(meta.condition=='Normal').sum()})",
    f"Tumour samples matched to TCGA-CDR: {df.type.notna().sum()}",
    f"Tumours with usable OS (time > 0 and status known): {new.dropna(subset=['OS_days', 'OS_event']).shape[0]}  (events {int(new.dropna(subset=['OS_days', 'OS_event']).OS_event.sum())})",
    f"Tumours with usable PFI (time > 0 and status known): {new.dropna(subset=['PFI_days', 'PFI_event']).shape[0]}  (events {int(new.dropna(subset=['PFI_days', 'PFI_event']).PFI_event.sum())})",
    f"Tumours with usable DSS (time > 0 and status known): {new.dropna(subset=['DSS_days', 'DSS_event']).shape[0]}  (events {int(new.dropna(subset=['DSS_days', 'DSS_event']).DSS_event.sum())})",
    f"Tumours with stage: {new.stage.notna().sum()}  ({new.stage_source.value_counts().to_dict()})",
    f"Stage distribution: {new.stage.value_counts().reindex(['I','II','III','IV']).to_dict()}",
    f"Sex: {new.sex.value_counts().to_dict()}",
    f"Smoking: {new.smoking.value_counts().to_dict()}  missing {new.smoking.isna().sum()}",
    f"Residual tumour: {new.residual_tumour.value_counts().to_dict()}  missing {new.residual_tumour.isna().sum()}",
    f"Complete cases for multivariable OS Cox (age, sex, stage, smoking): {len(cc_cox)} (events {int(cc_cox.OS_event.sum())})",
]
(OUT / "sample_flow.txt").write_text("\n".join(flow), encoding="utf-8")
print("\n".join(flow)); print(); print(comp); print(); print(ct)
