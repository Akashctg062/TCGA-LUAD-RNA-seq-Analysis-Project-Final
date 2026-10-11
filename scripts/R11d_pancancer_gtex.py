"""
R11d_pancancer_gtex.py — Revision step 11d (PI comment M11): tumour vs normal including GTEx.
Data: UCSC Toil recompute (Vivian et al., Nat Biotechnol 2017) — TCGA and GTEx re-processed with one
pipeline (RSEM, log2(TPM + 0.001)), queried for the 7 hub genes (R11a).
For each TCGA cancer type (primary tumours, one per patient) the 7-gene score is compared with normal tissue
of the same primary site: TCGA adjacent normal + GTEx normal (Mann-Whitney, BH; Cliff's delta as effect size).
Score = mean z of the 7 genes, z over all samples in the Toil matrix.
Outputs: analysis/11_extended/pancancer/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "11_extended" / "raw"
OUT = NEW / "11_extended" / "pancancer"
ENSG = {"ENSG00000156970": "BUB1B", "ENSG00000134057": "CCNB1", "ENSG00000117399": "CDC20", "ENSG00000178999": "AURKB",
        "ENSG00000166851": "PLK1", "ENSG00000157456": "CCNB2", "ENSG00000145386": "CCNA2", "ENSG00000148773": "MKI67"}
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

X = pd.read_csv(RAW / "ToilTcgaTargetGtex_tpm_selected.tsv", sep="\t", index_col=0)
X.index = [ENSG[i.split(".")[0]] for i in X.index]
X = X.T.astype(float)
ph = pd.read_csv(RAW / "ToilTcgaTargetGtex_phenotype.txt.gz", sep="\t", encoding="latin-1").set_index("sample")
X = X.join(ph[["_study", "_sample_type", "_primary_site"]], how="inner")
cdr = pd.read_excel(NEW / "04_clinical" / "raw" / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
ctype = cdr.set_index("bcr_patient_barcode").type
X["cancer"] = np.where(X._study == "TCGA", X.index.str[:12].map(ctype), np.nan)
Z = (X[HUB7] - X[HUB7].mean()) / X[HUB7].std()
X["score"] = Z.mean(1)

tum = X[(X._study == "TCGA") & (X._sample_type.isin(["Primary Tumor", "Primary Blood Derived Cancer - Peripheral Blood"]))]
tum = tum[~tum.index.str[:12].duplicated()].dropna(subset=["cancer"])
site = tum.groupby("cancer")._primary_site.agg(lambda s: s.mode().iloc[0])
nT = X[(X._study == "TCGA") & (X._sample_type == "Solid Tissue Normal")]
nG = X[X._study == "GTEX"]

# TCGA site label -> GTEx site label where the two projects name the same organ differently
GTEX_SITE = {"Thyroid Gland": "Thyroid", "Adrenal gland": "Adrenal Gland", "Cervix": "Cervix Uteri",
             "Endometrium": "Uterus"}
def gtex_normals(s):
    return nG[nG._primary_site == GTEX_SITE.get(s, s)]

def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sign(a[:, None] - b[None, :])).mean()

rows = []
for c, g in tum.groupby("cancer"):
    s = site[c]
    normals = pd.concat([nT[(nT._primary_site == s) & (nT.index.str[:12].map(ctype) == c)], gtex_normals(s)])
    n_t = int(((nT._primary_site == s) & (nT.index.str[:12].map(ctype) == c)).sum()); n_g = len(gtex_normals(s))
    row = dict(cancer=c, primary_site=s, n_tumour=len(g), n_TCGA_normal=n_t, n_GTEx_normal=n_g)
    if len(normals) >= 10:
        row.update(median_tumour=g.score.median(), median_normal=normals.score.median(),
                   diff=g.score.median() - normals.score.median(), cliffs_delta=cliff(g.score, normals.score),
                   p=mannwhitneyu(g.score, normals.score).pvalue)
    rows.append(row)
r = pd.DataFrame(rows)
k = r.p.notna(); r.loc[k, "FDR"] = bh(r.loc[k, "p"])
r = r.sort_values("diff", ascending=False)
r.to_csv(OUT / "pancancer_tumour_vs_TCGA_plus_GTEx_normal.csv", index=False)

# figure
rr = r[k.loc[r.index]].sort_values("diff")
fig, ax = plt.subplots(figsize=(14, 5))
order = list(rr.cancer)
pos = np.arange(len(order))
bt = [tum[tum.cancer == c].score.values for c in order]
bn = [pd.concat([nT[(nT._primary_site == site[c]) & (nT.index.str[:12].map(ctype) == c)],
                 gtex_normals(site[c])]).score.values for c in order]
b1 = ax.boxplot(bt, positions=pos + .2, widths=.35, patch_artist=True, showfliers=False)
b2 = ax.boxplot(bn, positions=pos - .2, widths=.35, patch_artist=True, showfliers=False)
for b, col in [(b1, "#c0392b"), (b2, "#2980b9")]:
    for p_ in b["boxes"]: p_.set_facecolor(col); p_.set_alpha(.8)
    for m_ in b["medians"]: m_.set_color("k")
for i, c in enumerate(order):
    q = rr.set_index("cancer").loc[c]
    ax.text(i, ax.get_ylim()[1], "*" if q.FDR < 0.05 else "", ha="center", va="top", fontsize=12)
ax.set_xticks(pos); ax.set_xticklabels([f"{c}\n({site[c]})" for c in order], rotation=90, fontsize=8)
ax.set_ylabel("7-gene score (z, Toil TCGA+GTEx)")
ax.set_title("Tumour (red) vs normal tissue of the same site (blue: TCGA adjacent normal + GTEx); * FDR < 0.05")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_pancancer_tumour_vs_GTEx_normal.png", dpi=300); plt.close()
print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
