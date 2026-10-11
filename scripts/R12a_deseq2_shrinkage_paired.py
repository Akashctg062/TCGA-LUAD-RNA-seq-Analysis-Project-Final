"""
R12a_deseq2_shrinkage_paired.py — Revision (PI comment M13): LFC shrinkage + paired-design sensitivity.
Run on your PC (uses all CPU cores; ~10-30 min):
    .\\venv\\Scripts\\python.exe "analysis\\scripts\\R12a_deseq2_shrinkage_paired.py"

The original run (scripts/03_differential_expression.py) called lfc_shrink(coeff="condition_Tumor_vs_Normal"),
a coefficient name that does not exist in PyDESeq2 0.5 -> shrinkage was silently skipped and the volcano plot
showed unshrunken MLE log2FC (up to >10 for low-count genes).

(1) Full design ~condition (517 T vs 59 N), identical settings to the original run (DEG calls unchanged:
    Wald padj and MLE |log2FC| > 1), PLUS apeGLM-type shrunken log2FC (Zhu et al., Bioinformatics 2019)
    for visualisation and ranking.
(2) Paired sensitivity analysis: patients with both tumour and adjacent normal, design ~patient + condition.
Outputs: analysis/12_DE_revision/
"""
import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "12_DE_revision"; OUT.mkdir(parents=True, exist_ok=True)
NCPU = max(1, (os.cpu_count() or 2) - 1)
inf = DefaultInference(n_cpus=NCPU)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
meta = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("sample_submitter_id").loc[counts.columns]
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)
C = counts.T.round().astype(int)

def run(C, md, design, contrast, tag):
    dds = DeseqDataSet(counts=C, metadata=md, design=design, refit_cooks=True, inference=inf)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, inference=inf); st.summary()
    res = st.results_df.rename(columns={"log2FoldChange": "log2FC_MLE", "lfcSE": "lfcSE_MLE"}).copy()
    coef = [c for c in dds.varm["LFC"].columns if "Tumor" in c][0]
    log(f"{tag}: shrinking coefficient {coef}")
    st.lfc_shrink(coeff=coef)
    res["log2FC_shrunk"] = st.results_df["log2FoldChange"]; res["lfcSE_shrunk"] = st.results_df["lfcSE"]
    res = res.join(gmap, how="left")
    res["DEG"] = np.where((res.padj < 0.05) & (res.log2FC_MLE > 1), "Up",
                  np.where((res.padj < 0.05) & (res.log2FC_MLE < -1), "Down", "NS"))
    res.to_csv(OUT / f"DE_{tag}.csv")
    log(f"{tag}: done; DEGs {res.DEG.value_counts().to_dict()}")
    return res

log(f"Using {NCPU} CPU cores")
full = run(C, meta[["condition"]], "~condition", ["condition", "Tumor", "Normal"], "full_517T_59N")

pairs = meta.groupby("patient_id").condition.agg(set)
pp = pairs[pairs.apply(lambda s: {"Tumor", "Normal"} <= s)].index
mp = meta[meta.patient_id.isin(pp)]
mp = mp.sort_values("condition").groupby(["patient_id", "condition"]).head(1)
log(f"Paired patients: {len(pp)}  (samples {len(mp)})")
paired = run(C.loc[mp.index], mp[["patient_id", "condition"]], "~patient_id + condition",
             ["condition", "Tumor", "Normal"], f"paired_{len(pp)}pairs")

j = full[["log2FC_MLE", "padj", "DEG"]].join(paired[["log2FC_MLE", "padj", "DEG"]], rsuffix="_paired")
summ = pd.DataFrame([dict(
    n_genes=len(j), DEGs_full=(j.DEG != "NS").sum(), DEGs_paired=(j.DEG_paired != "NS").sum(),
    concordant_direction_among_full_DEGs=((j.DEG != "NS") & (j.DEG == j.DEG_paired)).sum(),
    spearman_log2FC_full_vs_paired=j[["log2FC_MLE", "log2FC_MLE_paired"]].corr("spearman").iloc[0, 1])])
summ.to_csv(OUT / "paired_vs_full_summary.csv", index=False)
hub = pd.read_csv(ROOT / "analysis" / "hub_genes_final.csv").set_index("gene_id")
cmp = full.loc[hub.index, ["gene_name", "log2FC_MLE", "log2FC_shrunk", "padj"]].join(
      paired.loc[hub.index, ["log2FC_MLE", "padj"]], rsuffix="_paired")
cmp.to_csv(OUT / "hub_genes_full_vs_paired.csv")
print(summ.T); print(cmp)
log("finished")
