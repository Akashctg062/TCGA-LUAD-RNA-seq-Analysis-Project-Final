"""
R13a_field_effect_hubs.py — Limitation 4 (field cancerization of adjacent normal tissue), part A.
Hub genes and 7-gene score in LUAD primary tumours vs TCGA adjacent normal lung (NAT) vs GTEx healthy lung,
all re-processed with one pipeline (UCSC Toil recompute, RSEM log2(TPM+0.001); Vivian et al. 2017).
  * tumour vs NAT, tumour vs GTEx, NAT vs GTEx: median difference, Cliff's delta, Mann-Whitney p (BH over 3 x 8 tests)
  * fraction of the tumour-GTEx difference already present in NAT (field-effect fraction)
  * AUC of the score: tumour vs NAT and tumour vs GTEx
Score = mean z of the 7 genes, z computed within the LUAD + lung-normal samples analysed here.
Input : analysis/11_extended/raw/ToilTcgaTargetGtex_tpm_selected.tsv, ToilTcgaTargetGtex_phenotype.txt.gz
Output: analysis/13_limitations/field_effect/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "11_extended" / "raw"
OUT = NEW / "13_limitations" / "field_effect"; OUT.mkdir(parents=True, exist_ok=True)
ENSG = {"ENSG00000156970": "BUB1B", "ENSG00000134057": "CCNB1", "ENSG00000117399": "CDC20", "ENSG00000178999": "AURKB",
        "ENSG00000166851": "PLK1", "ENSG00000157456": "CCNB2", "ENSG00000145386": "CCNA2", "ENSG00000148773": "MKI67"}
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sign(a[:, None] - b[None, :]).mean())

X = pd.read_csv(RAW / "ToilTcgaTargetGtex_tpm_selected.tsv", sep="\t", index_col=0)
X.index = [ENSG[i.split(".")[0]] for i in X.index]
X = X.T.astype(float)
ph = pd.read_csv(RAW / "ToilTcgaTargetGtex_phenotype.txt.gz", sep="\t", encoding="latin-1").set_index("sample")
ph = ph[ph._primary_site == "Lung"]
grp = pd.Series(np.nan, index=ph.index, dtype=object)
grp[(ph._study == "TCGA") & (ph["primary disease or tissue"] == "Lung Adenocarcinoma") & (ph._sample_type == "Primary Tumor")] = "Tumour"
grp[(ph._study == "TCGA") & (ph["primary disease or tissue"] == "Lung Adenocarcinoma") & (ph._sample_type == "Solid Tissue Normal")] = "NAT"
grp[(ph._study == "GTEX")] = "GTEx"
grp = grp.dropna()
D = X.loc[X.index.intersection(grp.index)].copy()
D["group"] = grp
tum = D.group == "Tumour"
D = D[~(tum & pd.Series(D.index.str[:12], index=D.index).where(tum).duplicated())]  # one tumour per patient (duplicates counted among tumours only)
Z = (D[HUB7] - D[HUB7].mean()) / D[HUB7].std()
D["score"] = Z.mean(1)
D.to_csv(OUT / "toil_LUAD_lung_normal_hub_expression.csv")
n = D.group.value_counts(); print(n.to_dict())

rows = []
for g in HUB7 + ["MKI67", "score"]:
    t, a, h = (D.loc[D.group == k, g] for k in ("Tumour", "NAT", "GTEx"))
    row = dict(gene=g, n_tumour=len(t), n_NAT=len(a), n_GTEx=len(h),
               median_tumour=t.median(), median_NAT=a.median(), median_GTEx=h.median())
    for lab, x, y in [("T_vs_NAT", t, a), ("T_vs_GTEx", t, h), ("NAT_vs_GTEx", a, h)]:
        row[f"diff_{lab}"] = x.median() - y.median()
        row[f"cliff_{lab}"] = cliff(x, y)
        row[f"p_{lab}"] = mannwhitneyu(x, y).pvalue
    row["field_fraction"] = row["diff_NAT_vs_GTEx"] / row["diff_T_vs_GTEx"]
    row["AUC_T_vs_NAT"] = roc_auc_score((D.group == "Tumour")[D.group.isin(["Tumour", "NAT"])],
                                        D.loc[D.group.isin(["Tumour", "NAT"]), g])
    row["AUC_T_vs_GTEx"] = roc_auc_score((D.group == "Tumour")[D.group.isin(["Tumour", "GTEx"])],
                                         D.loc[D.group.isin(["Tumour", "GTEx"]), g])
    rows.append(row)
R = pd.DataFrame(rows)
pc = [c for c in R if c.startswith("p_")]
q = bh(R[pc].values.ravel()).reshape(R[pc].shape)
for i, c in enumerate(pc): R["FDR_" + c[2:]] = q[:, i]
R.to_csv(OUT / "R13a_hub_tumour_NAT_GTEx.csv", index=False)
pd.set_option("display.width", 250)
print(R[["gene", "diff_T_vs_NAT", "diff_T_vs_GTEx", "diff_NAT_vs_GTEx", "cliff_T_vs_GTEx", "cliff_NAT_vs_GTEx",
         "FDR_NAT_vs_GTEx", "field_fraction", "AUC_T_vs_NAT", "AUC_T_vs_GTEx"]].round(3).to_string(index=False))
