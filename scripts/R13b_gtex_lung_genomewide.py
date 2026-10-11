"""
R13b_gtex_lung_genomewide.py — Limitation 4 (field cancerization), part B: genome-wide check against GTEx lung.
Run on your PC (internet needed for the first run; ~10-30 min):
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R13b_gtex_lung_genomewide.py"

(1) Download (once, checkpointed) RSEM expected counts from the UCSC Toil recompute (TCGA and GTEx processed by
    one pipeline; dataset TcgaTargetGtex_gene_expected_count, log2(count+1)) for TCGA-LUAD primary tumours,
    TCGA-LUAD adjacent normal lung (NAT) and GTEx lung, restricted to the 26,370 genes of the original analysis.
(2) PyDESeq2, design ~group (Tumour / NAT / GTEx): contrasts Tumour vs GTEx and NAT vs GTEx.
(3) Concordance with the original Tumour vs NAT analysis (12_DE_revision/DE_full_517T_59N.csv):
    log2FC correlation, recovery of original up/down DEGs, the 86 ME2 candidates and the 7 hub genes.
Outputs: analysis/13_limitations/field_effect/
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
ROOT = NEW.parent
RAW = NEW / "13_limitations" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
OUT = NEW / "13_limitations" / "field_effect"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
COUNTS = RAW / "Toil_LUAD_NAT_GTExLung_expected_count.csv.gz"

# ── (1) download ──
ph = pd.read_csv(NEW / "11_extended" / "raw" / "ToilTcgaTargetGtex_phenotype.txt.gz", sep="\t",
                 encoding="latin-1").set_index("sample")
ph = ph[ph._primary_site == "Lung"]
grp = pd.Series(np.nan, index=ph.index, dtype=object)
luad = (ph._study == "TCGA") & (ph["primary disease or tissue"] == "Lung Adenocarcinoma")
grp[luad & (ph._sample_type == "Primary Tumor")] = "Tumour"
grp[luad & (ph._sample_type == "Solid Tissue Normal")] = "NAT"
grp[ph._study == "GTEX"] = "GTEx"
grp = grp.dropna()

if not COUNTS.exists():
    try:
        import xenaPython as xena
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xenaPython"])
        import xenaPython as xena
    host, ds = "https://toil.xenahubs.net", "TcgaTargetGtex_gene_expected_count"
    universe = set(pd.read_csv(ROOT / "data" / "filtered_counts.csv", usecols=[0]).iloc[:, 0])
    all_samples = set(xena.dataset_samples(host, ds, None))
    samples = [s for s in grp.index if s in all_samples]
    probes = [p for p in xena.dataset_field(host, ds) if p.split(".")[0] in universe]
    log(f"Querying {len(probes)} genes x {len(samples)} samples from {ds}")
    parts, step = [], 400
    for i in range(0, len(probes), step):
        chunk = probes[i:i + step]
        for attempt in range(5):
            try:
                vals = xena.dataset_probe_values(host, ds, samples, chunk)[1]; break
            except Exception as e:
                log(f"  retry {attempt+1} chunk {i}: {e}"); time.sleep(10 * (attempt + 1))
        else:
            sys.exit("Download failed repeatedly - please rerun (completed chunks are not kept).")
        parts.append(pd.DataFrame([[np.nan if (x is None or x == "NaN") else float(x) for x in v] for v in vals],
                                  index=[p.split(".")[0] for p in chunk], columns=samples))
        if (i // step) % 10 == 0: log(f"  {i + len(chunk)}/{len(probes)} genes")
    M = pd.concat(parts)
    M.to_csv(COUNTS, compression="gzip")
    log(f"Saved {COUNTS.name}: {M.shape}")

M = pd.read_csv(COUNTS, index_col=0)
C = np.round(np.clip(2 ** M - 1, 0, None)).fillna(0).astype(int).T            # samples x genes
md = pd.DataFrame({"group": grp.reindex(C.index)})
md = md[md.group.notna()]
tum = md.group == "Tumour"
md = md[~(tum & pd.Series(md.index.str[:12], index=md.index).where(tum).duplicated())]  # one tumour per patient (duplicates counted among tumours only)
C = C.loc[md.index]
C = C.loc[:, (C >= 10).sum() >= 10]                                          # same spirit as the original filter
log(f"Samples: {md.group.value_counts().to_dict()}; genes after filter: {C.shape[1]}")

# ── (2) DESeq2 ──
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
inf = DefaultInference(n_cpus=max(1, (os.cpu_count() or 2) - 1))
dds = DeseqDataSet(counts=C, metadata=md, design="~group", refit_cooks=True, inference=inf)
dds.deseq2()
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)
res = {}
for a in ["Tumour", "NAT"]:
    st = DeseqStats(dds, contrast=["group", a, "GTEx"], inference=inf); st.summary()
    r = st.results_df.join(gmap, how="left")
    r["DEG"] = np.where((r.padj < 0.05) & (r.log2FoldChange > 1), "Up",
                np.where((r.padj < 0.05) & (r.log2FoldChange < -1), "Down", "NS"))
    r.to_csv(OUT / f"DE_{a}_vs_GTExLung.csv"); res[a] = r
    log(f"{a} vs GTEx: {r.DEG.value_counts().to_dict()}")

# ── (3) concordance with the original Tumour vs NAT analysis ──
orig = pd.read_csv(NEW / "12_DE_revision" / "DE_full_517T_59N.csv", index_col=0)
cand = pd.read_csv(NEW / "02_WGCNA_signed" / "A_key_module_ME2_hub_candidates_MM0.8_GS0.5_DEGup.csv").gene_id
hubs = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
tg, ng = res["Tumour"], res["NAT"]
J = orig[["gene_name", "log2FC_MLE", "padj", "DEG"]].join(
    tg[["log2FoldChange", "padj", "DEG"]].add_suffix("_TvsGTEx"), how="inner").join(
    ng[["log2FoldChange", "padj", "DEG"]].add_suffix("_NATvsGTEx"), how="inner")
J.to_csv(OUT / "R13b_concordance_gene_level.csv")
from scipy.stats import spearmanr, pearsonr
S = {"n_genes_compared": len(J),
     "n_TvsGTEx": md.group.value_counts().to_dict(),
     "spearman_log2FC_TvsNAT_vs_TvsGTEx": spearmanr(J.log2FC_MLE, J.log2FoldChange_TvsGTEx).correlation,
     "pearson_log2FC_TvsNAT_vs_TvsGTEx": pearsonr(J.log2FC_MLE, J.log2FoldChange_TvsGTEx)[0]}
for d in ["Up", "Down"]:
    o = J[J.DEG == d]
    S[f"orig_{d}_n"] = len(o)
    S[f"orig_{d}_same_direction_vs_GTEx"] = float((np.sign(o.log2FoldChange_TvsGTEx) == (1 if d == "Up" else -1)).mean())
    S[f"orig_{d}_also_DEG_{d}_vs_GTEx"] = float((o.DEG_TvsGTEx == d).mean())
    # genes whose NAT already differs from GTEx in the tumour direction (possible field effect, incl. batch)
    S[f"orig_{d}_NAT_shifted_towards_tumour"] = float((o.DEG_NATvsGTEx == d).mean())
cj = J.loc[J.index.intersection(cand)]
S["candidates_n"] = len(cj)
S["candidates_up_vs_GTEx_FDR05"] = int(((cj.log2FoldChange_TvsGTEx > 0) & (cj.padj_TvsGTEx < 0.05)).sum())
S["candidates_DEG_up_vs_GTEx"] = int((cj.DEG_TvsGTEx == "Up").sum())
S["candidates_DEG_up_NAT_vs_GTEx"] = int((cj.DEG_NATvsGTEx == "Up").sum())
hj = J[J.gene_name.isin(hubs)][["gene_name", "log2FC_MLE", "log2FoldChange_TvsGTEx", "padj_TvsGTEx",
                                 "log2FoldChange_NATvsGTEx", "padj_NATvsGTEx"]]
hj.to_csv(OUT / "R13b_hub_genes_DE_vs_GTEx.csv")
json.dump(S, open(OUT / "R13b_summary.json", "w"), indent=2, default=float)
print(json.dumps(S, indent=2, default=float)); print(hj.round(3).to_string())
log("done")
