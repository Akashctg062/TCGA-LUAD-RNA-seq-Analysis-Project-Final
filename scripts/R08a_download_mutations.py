"""
R08a_download_mutations.py — Revision step 8a (PI comments M10, M12): patient-level somatic mutations.
Run ONCE on a computer with internet:
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R08a_download_mutations.py"

Source: cBioPortal REST API, study luad_tcga_pan_can_atlas_2018 (TCGA PanCanAtlas MC3 calls).
Saves raw responses in analysis/08_mutation_context/raw/:
  mutations_selected_genes.json   all mutation records for drivers + hub genes
  sequenced_samples.json          samples profiled for mutations (denominator)
  clinical_sample.json            PanCanAtlas sample-level attributes (e.g. TMB, aneuploidy)
  clinical_patient.json           PanCanAtlas patient-level attributes
"""
import json, requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "08_mutation_context" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
API = "https://www.cbioportal.org/api"
STUDY = "luad_tcga_pan_can_atlas_2018"
GENES = {  # Entrez IDs
    "TP53": 7157, "KRAS": 3845, "EGFR": 1956, "STK11": 6794, "KEAP1": 9817, "BRAF": 673, "ALK": 238,
    "BUB1B": 701, "CCNB1": 891, "CDC20": 991, "AURKB": 9212, "PLK1": 5347, "CCNB2": 9133, "CCNA2": 890,
    "BIRC5": 332, "KIF2C": 11004,
}
H = {"Accept": "application/json", "Content-Type": "application/json"}

def save(name, obj):
    (RAW / name).write_text(json.dumps(obj), encoding="utf-8")
    print(f"  {name}: {len(obj)} records")

print("1/4 mutations ...")
r = requests.post(f"{API}/molecular-profiles/{STUDY}_mutations/mutations/fetch?projection=DETAILED",
                  headers=H, timeout=300,
                  data=json.dumps({"sampleListId": f"{STUDY}_sequenced", "entrezGeneIds": list(GENES.values())}))
r.raise_for_status(); save("mutations_selected_genes.json", r.json())

print("2/4 sequenced sample list ...")
r = requests.get(f"{API}/sample-lists/{STUDY}_sequenced/sample-ids", headers=H, timeout=120)
r.raise_for_status(); save("sequenced_samples.json", r.json())

print("3/4 sample-level clinical attributes ...")
r = requests.get(f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY", headers=H, timeout=300)
r.raise_for_status(); save("clinical_sample.json", r.json())

print("4/4 patient-level clinical attributes ...")
r = requests.get(f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY", headers=H, timeout=300)
r.raise_for_status(); save("clinical_patient.json", r.json())
print(f"\nDone. Files in {RAW}")
