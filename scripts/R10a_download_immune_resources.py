"""
R10a_download_immune_resources.py — Revision step 10a (PI comment M7): published immune / purity resources.
Run ONCE on a computer with internet:
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R10a_download_immune_resources.py"

Saves (raw, unmodified) into analysis/10_immune/raw/:
  MCPcounter_genes.txt              MCP-counter marker genes (Becht et al., Genome Biol 2016; GitHub ebecht/MCPcounter)
  TCGA_ABSOLUTE_purity.txt          PanCanAtlas ABSOLUTE purity/ploidy (TCGA_mastercalls.abs_tables_JSedit.fixed.txt)
  TCGA_CIBERSORT_relative.tsv       PanCanAtlas CIBERSORT relative fractions (Thorsson et al., Immunity 2018)
Each item is attempted independently; failures are reported, not fatal.
"""
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "10_immune" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ITEMS = [
    ("MCPcounter_genes.txt",
     ["https://raw.githubusercontent.com/ebecht/MCPcounter/master/Signatures/genes.txt"]),
    ("TCGA_ABSOLUTE_purity.txt",
     ["https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5"]),
    ("TCGA_CIBERSORT_relative.tsv",
     ["https://api.gdc.cancer.gov/data/b3df502e-3594-46ef-9f94-d041a20a0b9a"]),
]
for fname, urls in ITEMS:
    for u in urls:
        try:
            r = requests.get(u, timeout=600)
            r.raise_for_status()
            (RAW / fname).write_bytes(r.content)
            head = r.content[:120].decode("utf-8", "replace").replace("\n", " | ")
            print(f"OK   {fname}  ({len(r.content)/1e6:.2f} MB)  starts: {head}")
            break
        except Exception as e:
            print(f"FAIL {fname} from {u}: {e}")
print(f"\nFiles in {RAW}")
