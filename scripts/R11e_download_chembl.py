"""
R11e_download_chembl.py — Revision step 11e (PI comment M8): curated drug mechanisms from ChEMBL.
Run ONCE with internet:
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R11e_download_chembl.py"

For each hub protein (UniProt accession) all ChEMBL targets that contain it are retrieved
(single proteins AND complexes, e.g. CDK1/cyclin B1), then every curated drug MECHANISM on those
targets (ChEMBL drug_mechanism table: mechanism of action, action type, max clinical phase).
Saves analysis/11_extended/raw/ChEMBL_hub_targets.json and ChEMBL_hub_mechanisms.json (raw API records)
and a flat table ChEMBL_hub_drug_mechanisms.csv.
"""
import csv, json, time, requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "11_extended" / "raw"
API = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = {"BUB1B": "O60566", "CCNB1": "P14635", "CDC20": "Q12834", "AURKB": "Q96GD4",
           "PLK1": "P53350", "CCNB2": "O95067", "CCNA2": "P20248"}

def get(url, **params):
    params.setdefault("format", "json"); params.setdefault("limit", 1000)
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429: time.sleep(5 * (attempt + 1)); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

targets, mechs, rows = {}, [], []
for gene, acc in UNIPROT.items():
    t = get(f"{API}/target.json", target_components__accession=acc)["targets"]
    t = [x for x in t if x.get("organism") == "Homo sapiens"]
    targets[gene] = t
    print(f"{gene}: {len(t)} ChEMBL targets ({', '.join(x['target_chembl_id'] + ':' + x['target_type'] for x in t)})")
    for x in t:
        m = get(f"{API}/mechanism.json", target_chembl_id=x["target_chembl_id"])["mechanisms"]
        for mm in m:
            mm["_hub_gene"] = gene; mm["_target_pref_name"] = x["pref_name"]; mm["_target_type"] = x["target_type"]
        mechs.extend(m)
        time.sleep(0.3)

mol_ids = sorted({m["molecule_chembl_id"] for m in mechs})
names = {}
for i in range(0, len(mol_ids), 50):
    for mol in get(f"{API}/molecule.json", molecule_chembl_id__in=",".join(mol_ids[i:i + 50]))["molecules"]:
        names[mol["molecule_chembl_id"]] = (mol.get("pref_name"), mol.get("max_phase"))

(RAW / "ChEMBL_hub_targets.json").write_text(json.dumps(targets, indent=1), encoding="utf-8")
(RAW / "ChEMBL_hub_mechanisms.json").write_text(json.dumps(mechs, indent=1), encoding="utf-8")
with open(RAW / "ChEMBL_hub_drug_mechanisms.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["hub_gene", "target_chembl_id", "target_name", "target_type", "molecule_chembl_id", "drug_name",
                "max_phase", "mechanism_of_action", "action_type"])
    for m in mechs:
        nm, ph = names.get(m["molecule_chembl_id"], (None, None))
        w.writerow([m["_hub_gene"], m["target_chembl_id"], m["_target_pref_name"], m["_target_type"],
                    m["molecule_chembl_id"], nm, ph, m.get("mechanism_of_action"), m.get("action_type")])
print(f"\n{len(mechs)} mechanism records for {len(mol_ids)} molecules -> {RAW / 'ChEMBL_hub_drug_mechanisms.csv'}")
