"""
R13h_cptac_protein.py — Limitation 1: protein-level evidence (CPTAC-LUAD; Gillette et al., Cell 2020).
Data (LinkedOmics): TMT proteome (gene level, log2 ratio; proteins with missing values removed by the provider)
and RNA-seq (log2 UQ-FPKM) for 110 tumours and 101 normal adjacent tissues (NAT); clinical annotation.
  * paired tumour vs NAT (Wilcoxon signed-rank; median paired log2 difference; % pairs tumour > NAT), BH over genes
  * mRNA-protein Spearman correlation in tumours (per gene)
  * protein score = mean z of the quantified hub proteins; correlation with the mRNA 7-gene score and with
    MKI67/TOP2A protein; association with stage (Spearman) and with TP53 mutation (Mann-Whitney)
Outputs: analysis/13_limitations/protein/
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu

NEW = Path(__file__).resolve().parents[1]
RAW = NEW / "13_limitations" / "raw" / "CPTAC"
OUT = NEW / "13_limitations" / "protein"; OUT.mkdir(parents=True, exist_ok=True)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
rd = lambda f: pd.read_csv(RAW / f, sep="\t", index_col=0)
PT, PN = rd("HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct"), rd("HS_CPTAC_LUAD_proteome_ratio_NArm_NORMAL.cct")
RT, RN = rd("HS_CPTAC_LUAD_rnaseq_uq_rpkm_log2_NArm_TUMOR.cct"), rd("HS_CPTAC_LUAD_rnaseq_uq_rpkm_log2_NArm_NORMAL.cct")
cli = pd.read_csv(RAW / "HS_CPTAC_LUAD_cli.tsi", sep="\t", index_col=0).drop(index="data_type")
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

prot_genes = [g for g in HUB7 if g in PT.index and g in PN.index]
print("hub proteins quantified:", prot_genes, "| not quantified:", [g for g in HUB7 if g not in prot_genes])
pairs = PT.columns.intersection(PN.columns)
rows = []
for g in HUB7 + ["MKI67", "TOP2A"]:
    r = dict(gene=g, protein_quantified=g in PT.index and g in PN.index)
    if r["protein_quantified"]:
        t, n = PT.loc[g, pairs].astype(float), PN.loc[g, pairs].astype(float)
        ok = t.notna() & n.notna(); d = t[ok] - n[ok]
        r.update(n_tumours_detected=int(PT.loc[g].notna().sum()), n_NAT_detected=int(PN.loc[g].notna().sum()),
                 n_pairs=int(ok.sum()), median_paired_log2_diff=d.median(), pct_tumour_higher=100 * (d > 0).mean(),
                 p_paired=wilcoxon(t[ok], n[ok]).pvalue)
        if g in RT.index:
            c = PT.columns.intersection(RT.columns)
            x, y = RT.loc[g, c].astype(float), PT.loc[g, c].astype(float); ok2 = x.notna() & y.notna()
            r.update(n_tumours_mRNA_protein=int(ok2.sum()), rho_mRNA_protein=spearmanr(x[ok2], y[ok2]).correlation)
    if g in RT.index and g in RN.index:
        c = RT.columns.intersection(RN.columns)
        r.update(mRNA_median_paired_log2_diff=(RT.loc[g, c] - RN.loc[g, c]).median(),
                 mRNA_p_paired=wilcoxon(RT.loc[g, c], RN.loc[g, c]).pvalue)
    rows.append(r)
R = pd.DataFrame(rows)
k = R.p_paired.notna(); R.loc[k, "FDR_paired"] = bh(R.loc[k, "p_paired"])
R.to_csv(OUT / "R13h_protein_tumour_vs_NAT.csv", index=False)

# scores
allP = pd.concat([PT, PN.add_suffix("_N")], axis=1).loc[prot_genes].astype(float)
Zp = allP.sub(allP.mean(1), axis=0).div(allP.std(1), axis=0)
pscore = Zp.mean(0)
nobs = Zp.notna().sum(0); pscore = pscore[nobs >= 3]                       # >= 3 of the 5 proteins detected
ps_t = pscore.reindex(PT.columns).dropna(); ps_n = pscore.reindex([c + "_N" for c in pairs]); ps_n.index = pairs
pairs = ps_t.index.intersection(ps_n.dropna().index); ps_n = ps_n[pairs]
Zr = RT.loc[HUB7].astype(float); Zr = Zr.sub(Zr.mean(1), axis=0).div(Zr.std(1), axis=0); rscore = Zr.mean(0)
c = ps_t.index.intersection(rscore.index)
S = dict(n_tumours_scored=len(ps_t), n_hub_proteins=len(prot_genes), proteins=";".join(prot_genes),
         n_pairs=len(pairs), protein_score_median_paired_diff=(ps_t[pairs] - ps_n).median(),
         protein_score_pct_tumour_higher=100 * ((ps_t[pairs] - ps_n) > 0).mean(),
         protein_score_p_paired=wilcoxon(ps_t[pairs], ps_n).pvalue,
         rho_protein_score_vs_mRNA_score=spearmanr(ps_t[c], rscore[c]).correlation, n_mRNA_protein=len(c))
for m in ["MKI67", "TOP2A"]:
    if m in PT.index:
        y = PT.loc[m, ps_t.index].astype(float); ok3 = y.notna()
        S[f"rho_protein_score_vs_{m}_protein"] = spearmanr(ps_t[ok3], y[ok3]).correlation
st = cli.Stage.astype(str).str.extract(r"^(\d)")[0].astype(float).reindex(ps_t.index)
ok = st.notna()
S.update(stage_n=int(ok.sum()), rho_protein_score_vs_stage=spearmanr(ps_t[ok], st[ok]).correlation,
         p_protein_score_vs_stage=spearmanr(ps_t[ok], st[ok]).pvalue)
tp = pd.to_numeric(cli["TP53.mutation.status"], errors="coerce").reindex(ps_t.index)
S.update(n_TP53_mut=int((tp == 1).sum()), protein_score_TP53_mut_minus_wt=ps_t[tp == 1].median() - ps_t[tp == 0].median(),
         p_TP53=mannwhitneyu(ps_t[tp == 1], ps_t[tp == 0]).pvalue)
pd.Series(S).to_csv(OUT / "R13h_summary.csv", header=["value"])
pd.DataFrame({"protein_score_tumour": ps_t[pairs], "protein_score_NAT": ps_n}).to_csv(OUT / "R13h_paired_scores.csv")
pd.DataFrame({"protein_score": ps_t[c], "mRNA_score": rscore[c]}).to_csv(OUT / "R13h_mRNA_vs_protein_score.csv")
pd.set_option("display.width", 220)
print(R.round(4).to_string(index=False)); print(pd.Series(S).to_string())
