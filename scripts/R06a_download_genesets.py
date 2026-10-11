"""
R06a_download_genesets.py — Revision step 6a: download MSigDB Hallmark gene sets (human symbols).
Run ONCE on a computer with internet:
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R06a_download_genesets.py"
Saves: analysis/06_benchmark/raw/h.all.<version>.Hs.symbols.gmt
"""
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "06_benchmark" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{v}/h.all.v{v}.symbols.gmt"
for v in ["2024.1.Hs", "2023.2.Hs", "2023.1.Hs"]:
    try:
        r = requests.get(BASE.format(v=v), timeout=120)
        r.raise_for_status()
        f = RAW / f"h.all.v{v}.symbols.gmt"
        f.write_text(r.text, encoding="utf-8")
        n = sum(1 for line in r.text.splitlines() if line.strip())
        print(f"Saved {f.name}  ({n} Hallmark gene sets)")
        break
    except Exception as e:
        print(f"MSigDB {v} failed: {e}")
else:
    raise SystemExit("Could not download Hallmark gene sets.")
