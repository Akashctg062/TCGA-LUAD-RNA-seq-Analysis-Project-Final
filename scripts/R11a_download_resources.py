"""
R11a_download_resources.py — Revision step 11a (PI comments M8, M9, M10, M11): download public resources.
Run on a computer with internet (large files are STREAMED and filtered to the genes of interest,
so disk use stays small; total time ~10-30 min depending on connection):
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R11a_download_resources.py"
Every item is independent and skipped if already present; re-run safely after a failure.

Saves into analysis/11_extended/raw/:
  M9  TRRUST_v2_human.tsv                     curated TF-target pairs (Han et al., NAR 2018)
      CollecTRI_omnipath.tsv                  CollecTRI TF-target network via OmniPath (Müller-Dott et al., NAR 2023)
  M10 LUAD_GISTIC2_thresholded_by_genes.gz    TCGA-LUAD GISTIC2 copy number (UCSC Xena, TCGA hub)
      LUAD_methylation450_selected.tsv        TCGA-LUAD 450K beta values, probes of the hub genes only (streamed)
      illuminaMethyl450_probeMap.tsv          probe annotation (UCSC Xena)
  M11 PanCanAtlas_EBpp_selected_genes.tsv     PanCanAtlas batch-corrected RNA-seq, selected genes, all samples (streamed)
      ToilTcgaTargetGtex_tpm_selected.tsv     UCSC Toil recompute TCGA+GTEx log2(TPM+0.001), selected genes (Xena API query)
      ToilTcgaTargetGtex_phenotype.txt.gz     sample annotation for the Toil matrix
  M8  DepMap: CRISPRGeneEffect.csv (selected genes), Model.csv, CRISPRInferredCommonEssentials.csv
"""
import csv, gzip, io, json, sys, time
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "11_extended" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
HUB7 = ["BUB1B", "CCNB1", "CDC20", "AURKB", "PLK1", "CCNB2", "CCNA2"]
GENES = HUB7 + ["MKI67"]
ENSG = {"BUB1B": "ENSG00000156970", "CCNB1": "ENSG00000134057", "CDC20": "ENSG00000117399",
        "AURKB": "ENSG00000178999", "PLK1": "ENSG00000166851", "CCNB2": "ENSG00000157456",
        "CCNA2": "ENSG00000145386", "MKI67": "ENSG00000148773"}
XENA_TCGA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
TOIL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"

def done(name):
    f = RAW / name
    if f.exists() and f.stat().st_size > 0:
        print(f"SKIP {name} (exists)"); return True
    return False

def get(name, url, binary=True):
    if done(name): return
    try:
        r = requests.get(url, timeout=900); r.raise_for_status()
        (RAW / name).write_bytes(r.content)
        print(f"OK   {name} ({len(r.content)/1e6:.1f} MB)")
    except Exception as e:
        print(f"FAIL {name}: {e}")

def stream_filter(name, url, keep_row, gz=True, sep="\t"):
    """Stream a (gzipped) matrix, keep header + rows where keep_row(first_field) is True."""
    if done(name): return
    t0 = time.time(); n = kept = 0
    try:
        with requests.get(url, stream=True, timeout=900) as r:
            r.raise_for_status()
            raw = r.raw; raw.decode_content = True
            fh = gzip.GzipFile(fileobj=raw) if gz else raw
            tmp = RAW / (name + ".part")                      # renamed only when complete
            out = open(tmp, "w", encoding="utf-8")
            try:
                for i, line in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
                    if i == 0:
                        out.write(line); continue
                    n += 1
                    key = line.split(sep, 1)[0].strip().strip('"')
                    if keep_row(key):
                        out.write(line); kept += 1
                    if n % 2000 == 0:
                        print(f"     {name}: scanned {n:,} rows, kept {kept} ({time.time()-t0:.0f}s)", flush=True)
            except ValueError as e:              # "I/O operation on closed file" = end of HTTP stream
                if "closed file" not in str(e): raise
            finally:
                out.close()
        tmp.replace(RAW / name)
        print(f"OK   {name}: kept {kept} of {n:,} rows ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"FAIL {name}: {e}")
        (RAW / (name + ".part")).unlink(missing_ok=True)

# ── M9 transcription factors ──
get("TRRUST_v2_human.tsv", "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv")
get("CollecTRI_omnipath.tsv",
    "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=sources,references&format=tsv")

# ── M10 copy number + methylation ──
get("LUAD_GISTIC2_thresholded_by_genes.gz",
    XENA_TCGA + "TCGA.LUAD.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz")
get("illuminaMethyl450_probeMap.tsv", XENA_TCGA + "probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy")
probes = set()
pm = RAW / "illuminaMethyl450_probeMap.tsv"
if pm.exists():
    for row in csv.reader(open(pm, encoding="utf-8", errors="replace"), delimiter="\t"):
        if len(row) > 1 and any(g in row[1].split(",") for g in GENES):
            probes.add(row[0])
    print(f"     {len(probes)} 450K probes map to the selected genes")
if probes:
    stream_filter("LUAD_methylation450_selected.tsv", XENA_TCGA + "TCGA.LUAD.sampleMap%2FHumanMethylation450.gz",
                  lambda k: k in probes)

# ── M11 pan-cancer ──
stream_filter("PanCanAtlas_EBpp_selected_genes.tsv",
              "https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611",
              lambda k: k.split("|")[0] in GENES, gz=False)
# Toil TCGA+GTEx: query only the selected genes from the UCSC Xena server (avoids a ~7 GB download)
if not done("ToilTcgaTargetGtex_tpm_selected.tsv"):
    try:
        import xenaPython as xena
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xenaPython"])
        import xenaPython as xena
    try:
        host, ds = "https://toil.xenahubs.net", "TcgaTargetGtex_rsem_gene_tpm"
        samples = xena.dataset_samples(host, ds, None)
        probes = [p for p in xena.dataset_field(host, ds) if p.split(".")[0] in set(ENSG.values())]
        vals = xena.dataset_probe_values(host, ds, samples, probes)[1]
        with open(RAW / "ToilTcgaTargetGtex_tpm_selected.tsv", "w", encoding="utf-8") as o:
            o.write("sample\t" + "\t".join(samples) + "\n")
            for p_, v in zip(probes, vals):
                o.write(p_ + "\t" + "\t".join("" if x is None or x == "NaN" else str(x) for x in v) + "\n")
        print(f"OK   ToilTcgaTargetGtex_tpm_selected.tsv ({len(probes)} genes x {len(samples)} samples)")
    except Exception as e:
        print(f"FAIL Toil TCGA+GTEx query: {e}")
get("ToilTcgaTargetGtex_phenotype.txt.gz", TOIL + "TcgaTargetGTEX_phenotype.txt.gz")

# ── M8 DepMap (latest public release, via the portal's file-listing API) ──
want = {"CRISPRGeneEffect.csv", "Model.csv", "CRISPRInferredCommonEssentials.csv"}
if not all((RAW / f"DepMap_{w}").exists() for w in want):
    try:
        lst = requests.get("https://depmap.org/portal/api/download/files", timeout=300)
        lst.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(lst.text)))
        pub = [r for r in rows if "Public" in r.get("release", "") and r.get("filename") in want]
        latest = sorted({r["release"] for r in pub}, key=lambda s: s)[-1]
        print(f"     DepMap release: {latest}")
        for r in pub:
            if r["release"] != latest: continue
            fn = f"DepMap_{r['filename']}"
            if r["filename"] == "CRISPRGeneEffect.csv":
                # gene effect: columns are genes -> download whole file (~400 MB) then keep selected columns
                if not done(fn):
                    big = requests.get(r["url"], timeout=1800); big.raise_for_status()
                    lines = big.text.splitlines()
                    head = next(csv.reader([lines[0]]))
                    idx = [0] + [i for i, h in enumerate(head) if h.split(" (")[0] in GENES]
                    with open(RAW / fn, "w", encoding="utf-8", newline="") as o:
                        w = csv.writer(o)
                        for row in csv.reader(lines):
                            w.writerow([row[i] for i in idx])
                    print(f"OK   {fn} (subset to {len(idx)-1} genes)")
            else:
                get(fn, r["url"])
        (RAW / "DepMap_release.txt").write_text(latest)
    except Exception as e:
        print(f"FAIL DepMap: {e}\n     Manual alternative: https://depmap.org/portal/data_page/?tab=allData -> "
              f"download CRISPRGeneEffect.csv, Model.csv, CRISPRInferredCommonEssentials.csv into {RAW} "
              f"with the prefix 'DepMap_'.")
print(f"\nFinished. Files in {RAW}")
