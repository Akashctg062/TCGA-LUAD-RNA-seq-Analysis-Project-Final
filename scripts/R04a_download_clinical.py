"""
R04a_download_clinical.py — Revision step 4a (PI comment M5): download clean clinical data.
Run ONCE on a computer with internet:
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R04a_download_clinical.py"

Downloads (raw, unmodified) into analysis/04_clinical/raw/:
  1. GDC API case records for TCGA-LUAD: demographic (sex, vital status), all diagnoses
     (AJCC pathologic stage / T / N / M, age, follow-up, residual disease) and exposures (smoking, pack-years)
  2. TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR; Liu et al., Cell 2018) — curated OS, DSS, DFI, PFI
"""
import json, requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "04_clinical" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

fields = ",".join([
    "submitter_id", "case_id",
    "demographic.gender", "demographic.vital_status", "demographic.days_to_death", "demographic.race",
    "diagnoses.diagnosis_is_primary_disease", "diagnoses.classification_of_tumor",
    "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m", "diagnoses.ajcc_staging_system_edition",
    "diagnoses.age_at_diagnosis", "diagnoses.days_to_last_follow_up", "diagnoses.residual_disease",
    "diagnoses.primary_diagnosis",
    "exposures.tobacco_smoking_status", "exposures.pack_years_smoked",
])
payload = {
    "filters": {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-LUAD"]}},
    "fields": fields, "format": "JSON", "size": "2000",
}
print("1/2 GDC case records ...")
r = requests.post("https://api.gdc.cancer.gov/cases", json=payload, timeout=180)
r.raise_for_status()
hits = r.json()["data"]["hits"]
(RAW / "gdc_TCGA-LUAD_cases.json").write_text(json.dumps(hits, indent=1), encoding="utf-8")
print(f"    {len(hits)} cases saved")

print("2/2 TCGA-CDR (Liu et al. 2018) ...")
r = requests.get("https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81", timeout=300)
r.raise_for_status()
(RAW / "TCGA-CDR-SupplementalTableS1.xlsx").write_bytes(r.content)
print(f"    {len(r.content)/1e6:.1f} MB saved")
print(f"\nDone. Files in {RAW}")
