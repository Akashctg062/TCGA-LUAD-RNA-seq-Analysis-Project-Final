"""
R13f_singlecell_source.py — Limitation 5: cellular source of hub-gene expression (single-cell RNA-seq).
Data: GSE131907 (Kim et al., Nat Commun 2020) — 208,506 cells, normalized log2(TPM+1), author cell annotation.
Analysed: primary-tumour lung samples (tLung, early stage; tL/B, advanced) and normal lung (nLung).
Epithelial cells: malignant = tS1/tS2/tS3 (tLung) or 'Malignant cells' (tL/B), as annotated by the authors (inferCNV);
normal epithelial = AT1, AT2, ciliated, club cells from nLung.
Per cell group and gene: % cells with detectable expression, mean log2TPM, and share of the total hub-gene transcript
mass (sum of TPM) in tumour samples; cycling cells = MKI67 or TOP2A detected.
Single-cell 7-gene score = mean log2TPM of the 7 genes per cell.
Outputs: analysis/13_limitations/single_cell/
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "13_limitations" / "raw"
OUT = NEW / "13_limitations" / "single_cell"; OUT.mkdir(parents=True, exist_ok=True)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()

a = pd.read_csv(RAW / "GSE131907_cell_annotation.txt.gz", sep="\t").set_index("Index")
X = pd.read_csv(RAW / "GSE131907_log2TPM_selected_genes.tsv.gz", sep="\t", index_col=0).T
X = X.loc[a.index.intersection(X.index)]
a = a.loc[X.index]
print("cells", len(a))

def group(r):
    if r.Cell_type == "Epithelial cells":
        if r.Cell_subtype in ("tS1", "tS2", "tS3", "Malignant cells"): return "Malignant epithelial"
        if r.Sample_Origin == "nLung" and r.Cell_subtype in ("AT1", "AT2", "Ciliated", "Club"): return "Normal epithelial"
        return "Other epithelial"
    return {"T/NK cells": "T/NK cells", "Myeloid cells": "Myeloid cells", "B lymphocytes": "B cells",
            "Fibroblasts": "Fibroblasts", "Endothelial cells": "Endothelial cells",
            "MAST cells": "Mast cells"}.get(r["Cell_type.refined"], "Other")
a["group"] = a.apply(group, axis=1)
a["score"] = X[HUB7].mean(1)
a["cycling"] = (X.MKI67 > 0) | (X.TOP2A > 0)
lung_t = a.Sample_Origin.isin(["tLung", "tL/B"])
lung_n = a.Sample_Origin == "nLung"

# per group x gene in tumour lung samples
rows = []
T = a[lung_t]; XT = X.loc[T.index]; TPM = 2 ** XT[HUB7] - 1
tot = TPM.sum()
for gname, g in T.groupby("group"):
    for gene in HUB7 + ["MKI67"]:
        x = XT.loc[g.index, gene]
        rows.append(dict(group=gname, gene=gene, n_cells=len(g), pct_expressing=100 * (x > 0).mean(),
                         mean_log2TPM=x.mean(),
                         share_of_tumour_transcripts=100 * (2 ** x - 1).sum() / tot[gene] if gene in HUB7 else np.nan))
G = pd.DataFrame(rows)
G.to_csv(OUT / "R13f_gene_by_celltype_tumour.csv", index=False)
share = G[G.gene.isin(HUB7)].pivot(index="group", columns="gene", values="share_of_tumour_transcripts")
share["mean_7"] = share.mean(1)
print("Share of hub-gene transcripts in tumour samples by cell group (%):")
print(share.round(1).sort_values("mean_7", ascending=False).to_string())

# cycling vs non-cycling
cyc = []
for gname, g in T.groupby("group"):
    cyc.append(dict(group=gname, n=len(g), pct_cycling=100 * g.cycling.mean(),
                    score_cycling=g.score[g.cycling].mean(), score_noncycling=g.score[~g.cycling].mean(),
                    pct_hub_transcripts_from_cycling=100 * TPM.loc[g.index[g.cycling.values]].sum().sum() /
                                                     max(TPM.loc[g.index].sum().sum(), 1e-9)))
C = pd.DataFrame(cyc); C.to_csv(OUT / "R13f_cycling_by_group.csv", index=False)
print(C.round(2).to_string(index=False))
r_all = spearmanr(a.score, X.MKI67 + X.TOP2A).correlation
print("Spearman score vs MKI67+TOP2A, all cells:", round(r_all, 3))

# malignant vs normal epithelium
mal = a[(a.group == "Malignant epithelial") & lung_t]; nor = a[a.group == "Normal epithelial"]
comp = dict(n_malignant=len(mal), n_normal_epi=len(nor), mean_score_malignant=mal.score.mean(),
            mean_score_normal=nor.score.mean(), pct_cycling_malignant=100 * mal.cycling.mean(),
            pct_cycling_normal=100 * nor.cycling.mean(), p=mannwhitneyu(mal.score, nor.score).pvalue,
            spearman_score_vs_MKI67_TOP2A=r_all)
# per-sample pseudo-bulk score (mean over all cells) tumour vs normal lung
ps = a[lung_t | lung_n].groupby(["Sample", "Sample_Origin"]).score.mean().reset_index()
ps.to_csv(OUT / "R13f_pseudobulk_score_per_sample.csv", index=False)
pt, pn = ps[ps.Sample_Origin != "nLung"].score, ps[ps.Sample_Origin == "nLung"].score
comp.update(pseudobulk_n_tumour=len(pt), pseudobulk_n_normal=len(pn), pseudobulk_p=mannwhitneyu(pt, pn).pvalue,
            pseudobulk_median_tumour=pt.median(), pseudobulk_median_normal=pn.median())
pd.Series(comp).to_csv(OUT / "R13f_summary.csv", header=["value"])
print(pd.Series(comp).round(4).to_string())
a[["Sample", "Sample_Origin", "group", "score", "cycling"]].join(X[HUB7 + ["MKI67", "TOP2A"]]).to_csv(
    OUT / "R13f_cell_level.csv.gz", compression="gzip")
