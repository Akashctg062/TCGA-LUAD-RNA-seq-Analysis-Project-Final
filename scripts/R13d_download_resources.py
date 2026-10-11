"""
R13d_download_resources.py — downloads for limitation steps 5 (single-cell), 1 (protein) and 3 (extra cohorts).
Run on your PC (internet needed; ~30-60 min, mostly the single-cell stream). Safe to re-run: finished files are skipped.
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R13d_download_resources.py"

Step 5  GSE131907 (Kim et al., Nat Commun 2020; 208,506 cells, 58 samples from 44 LUAD patients)
        - cell annotation (1.8 MB)
        - normalized log2TPM matrix (2.9 GB) is STREAMED; only the rows of the genes below are kept (~50 MB)
Step 1  CPTAC-LUAD proteogenomics (Gillette et al., Cell 2020; 110 tumours, 101 normal adjacent tissues)
        via the 'cptac' Python package (data hosted on Zenodo/PDC); proteomics, transcriptomics, clinical
        Human Protein Atlas: pathology (cancer IHC) and normal tissue IHC tables
Step 3  extra GEO cohorts with long follow-up / treatment data (series matrix + platform annotation):
        GSE50081 (Der 2014, Toronto, GPL570)  GSE30219 (Rousseaux 2013, France, GPL570)
        GSE37745 (Botling 2013, Sweden, GPL570)  GSE42127 (Tang 2013, MDACC, GPL6884, adjuvant chemo)
        GSE13213 (Tomida 2009, Japan, GPL6480)
Outputs: analysis/13_limitations/raw/
"""
import gzip, io, re, sys, time, subprocess
from pathlib import Path
import requests

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "13_limitations" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
GEOFTP = "https://ftp.ncbi.nlm.nih.gov/geo"

def done(p):
    if p.exists() and p.stat().st_size > 0:
        log(f"SKIP {p.name} (exists)"); return True
    return False

def get(url, dest, timeout=1800):
    if done(dest): return True
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20): fh.write(chunk)
        tmp.replace(dest); log(f"OK   {dest.name} ({dest.stat().st_size/1e6:.1f} MB)"); return True
    except Exception as e:
        log(f"FAIL {dest.name}: {e}"); return False

def series_dir(acc):
    return f"{acc[:-3]}nnn/{acc}"

# ───────────── Step 5: single-cell GSE131907 ─────────────
SC_GENES = {"BUB1B", "CCNB1", "CDC20", "AURKB", "PLK1", "CCNB2", "CCNA2", "MKI67", "TOP2A",
            "EPCAM", "KRT18", "KRT19", "NKX2-1", "SFTPC", "AGER", "PTPRC", "CD3E", "CD8A", "NKG7", "MS4A1",
            "CD79A", "CD68", "LYZ", "MARCO", "TPSAB1", "COL1A1", "DCN", "PECAM1", "VWF", "ACTA2"}
sc_base = f"{GEOFTP}/series/{series_dir('GSE131907')}/suppl/"
get(sc_base + "GSE131907_Lung_Cancer_cell_annotation.txt.gz", RAW / "GSE131907_cell_annotation.txt.gz")
get(sc_base + "GSE131907_Lung_Cancer_Feature_Summary.xlsx", RAW / "GSE131907_Feature_Summary.xlsx")
sc_out = RAW / "GSE131907_log2TPM_selected_genes.tsv.gz"
if not done(sc_out):
    log("Streaming GSE131907 normalized log2TPM matrix (2.9 GB) - keeping selected genes only ...")
    try:
        tmp = sc_out.with_suffix(".part")
        kept = n = 0
        with requests.get(sc_base + "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz", stream=True,
                          timeout=3600) as r, gzip.open(tmp, "wt", encoding="utf-8") as out:
            r.raise_for_status()
            r.raw.decode_content = True
            fh = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8", errors="replace")
            for i, line in enumerate(fh):
                if i == 0:
                    out.write(line); continue
                n += 1
                key = line[:line.find("\t")].strip().strip('"')
                if key in SC_GENES:
                    out.write(line); kept += 1
                if n % 2000 == 0: log(f"  {n} genes scanned, {kept} kept")
        tmp.replace(sc_out); log(f"OK   {sc_out.name}: {kept} genes kept of {n}")
    except Exception as e:
        log(f"FAIL single-cell stream: {e}  (re-run the script to retry)")

# ───────────── Step 1: CPTAC-LUAD + Human Protein Atlas ─────────────
cp_prot = RAW / "CPTAC_LUAD_proteomics.csv.gz"
if not done(cp_prot):
    try:
        try:
            import cptac
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cptac"]); import cptac
        log(f"cptac version {getattr(cptac, '__version__', '?')}")
        luad = cptac.Luad()
        def first_ok(fn, sources):
            for s in sources:
                try:
                    df = fn(s) if s else fn(); log(f"  {fn.__name__}({s}) -> {df.shape}"); return df, s
                except Exception as e:
                    log(f"  {fn.__name__}({s}) failed: {e}")
            return None, None
        prot, s1 = first_ok(luad.get_proteomics, ["umich", "bcm", None])
        rna, s2 = first_ok(luad.get_transcriptomics, ["washu", "bcm", "broad", None])
        clin, s3 = first_ok(luad.get_clinical, ["mssm", None])
        for df, name in [(rna, "CPTAC_LUAD_transcriptomics.csv.gz"), (clin, "CPTAC_LUAD_clinical.csv.gz")]:
            if df is not None:
                if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                    df.columns = ["|".join(map(str, c)) for c in df.columns]
                df.to_csv(RAW / name, compression="gzip"); log(f"OK   {name}")
        try:
            fu, _ = first_ok(luad.get_followup, ["mssm", None])
            if fu is not None: fu.to_csv(RAW / "CPTAC_LUAD_followup.csv.gz", compression="gzip")
        except AttributeError:
            pass
        if prot is not None:
            if hasattr(prot.columns, "levels") and prot.columns.nlevels > 1:
                prot.columns = ["|".join(map(str, c)) for c in prot.columns]
            prot.to_csv(cp_prot, compression="gzip"); log(f"OK   {cp_prot.name} (source {s1})")
    except Exception as e:
        log(f"FAIL CPTAC: {e}")
        log("     Alternative: download the CPTAC LUAD tables manually from https://www.linkedomics.org (CPTAC-LUAD)")
for name, urls in {"HPA_pathology.tsv.zip": ["https://www.proteinatlas.org/download/tsv/pathology.tsv.zip",
                                             "https://www.proteinatlas.org/download/pathology.tsv.zip"],
                   "HPA_normal_tissue.tsv.zip": ["https://www.proteinatlas.org/download/tsv/normal_tissue.tsv.zip",
                                                 "https://www.proteinatlas.org/download/normal_tissue.tsv.zip"]}.items():
    for u in urls:
        if get(u, RAW / name): break

# ───────────── Step 3: additional GEO cohorts ─────────────
GSES = ["GSE50081", "GSE30219", "GSE37745", "GSE42127", "GSE13213"]
for acc in GSES:
    dest = RAW / f"{acc}_series_matrix.txt.gz"
    if done(dest): continue
    base = f"{GEOFTP}/series/{series_dir(acc)}/matrix/"
    try:
        idx = requests.get(base, timeout=120).text
        files = sorted(set(re.findall(rf'href="({acc}[^"]*series_matrix\.txt\.gz)"', idx)))
        if not files: files = [f"{acc}_series_matrix.txt.gz"]
        log(f"{acc}: {files}")
        if len(files) == 1:
            get(base + files[0], dest)
        else:
            for f in files: get(base + f, RAW / f)
    except Exception as e:
        log(f"FAIL {acc}: {e}")
for gpl in ["GPL570", "GPL6884", "GPL6480"]:
    get(f"{GEOFTP}/platforms/{gpl[:-3]}nnn/{gpl}/annot/{gpl}.annot.gz", RAW / f"{gpl}.annot.gz") or \
        get(f"{GEOFTP}/platforms/{gpl[:-3]}nnn/{gpl}/soft/{gpl}_family.soft.gz", RAW / f"{gpl}_family.soft.gz")
log("finished - paste this output back to Claude")
