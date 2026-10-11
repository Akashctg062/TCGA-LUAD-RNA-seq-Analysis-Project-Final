"""
R13e_download_cptac.py — Step 1 (protein level): CPTAC-LUAD tables from LinkedOmics (Gillette et al., Cell 2020).
(The 'cptac' pip package needs C++ build tools on Windows, so the same data are taken from LinkedOmics instead.)
Run:  .\\venv\\Scripts\\python.exe "analysis\\scripts\\R13e_download_cptac.py"
Output: analysis/13_limitations/raw/CPTAC/
"""
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parents[1] / "13_limitations" / "raw" / "CPTAC"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://linkedomics.org/data_download/CPTAC-LUAD/"
FILES = ["HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct", "HS_CPTAC_LUAD_proteome_ratio_NArm_NORMAL.cct",
         "HS_CPTAC_LUAD_rnaseq_uq_rpkm_log2_NArm_TUMOR.cct", "HS_CPTAC_LUAD_rnaseq_uq_rpkm_log2_NArm_NORMAL.cct",
         "HS_CPTAC_LUAD_cli.tsi"]
for f in FILES:
    d = OUT / f
    if d.exists() and d.stat().st_size > 0:
        print("SKIP", f); continue
    try:
        r = requests.get(BASE + f, timeout=600); r.raise_for_status()
        d.write_bytes(r.content); print(f"OK   {f} ({len(r.content)/1e6:.1f} MB)")
    except Exception as e:
        print(f"FAIL {f}: {e}")
print("finished")
