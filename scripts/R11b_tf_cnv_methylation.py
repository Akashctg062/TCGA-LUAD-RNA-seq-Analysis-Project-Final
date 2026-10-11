"""
R11b_tf_cnv_methylation.py — Revision step 11b (PI comments M9, M10)
=====================================================================
M9  Transcription factors — evidence beyond co-expression
    (a) curated TF -> target support for the 7 hub genes: TRRUST v2 and CollecTRI (via OmniPath)
    (b) TF ACTIVITY inferred with a univariate linear model (decoupler "ulm" logic) on CollecTRI regulons,
        with the 7 hub genes REMOVED from every regulon (avoids circularity), TCGA-LUAD tumours;
        activity of each TF correlated (Spearman) with the 7-gene score; BH-FDR over all TFs tested
M10 Real multi-omics (replaces expression-inferred "methylation")
    (a) copy number: GISTIC2 thresholded calls (-2..+2), TCGA-LUAD (UCSC Xena): frequency of gain/loss,
        expression by copy-number state (Spearman, Kruskal-Wallis)
    (b) DNA methylation: Illumina 450K beta values (UCSC Xena) for probes annotated to the hub genes;
        promoter probes = TSS -1500/+500 bp (hg19 RefSeq TSS, strand-aware);
        tumour vs normal delta-beta (Mann-Whitney) and beta vs expression in tumours (Spearman); BH-FDR
Outputs: analysis/11_extended/
"""
import gzip, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, kruskal

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "11_extended"; RAW = OUT / "raw"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
fmt = lambda v: f"{v:.3g}"
# hg19 RefSeq canonical TSS and strand (UCSC hg19)
TSS = {"BUB1B": ("chr15", 40453224, "+"), "CCNB1": ("chr5", 68462837, "+"), "CDC20": ("chr1", 43824626, "+"),
       "AURKB": ("chr17", 8113882, "-"), "PLK1": ("chr16", 23690367, "+"), "CCNB2": ("chr15", 59397302, "+"),
       "CCNA2": ("chr4", 122745087, "-")}

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
meta = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("sample_submitter_id")
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
tum = meta.index[meta.condition == "Tumor"].intersection(lc.columns)
nor = meta.index[meta.condition == "Normal"].intersection(lc.columns)
Et = lc[tum].T
Zt = (Et[HUB7] - Et[HUB7].mean()) / Et[HUB7].std()
score = Zt.mean(1)
s15 = lambda s: s[:15]

# ════════ M9 (a) curated TF -> hub-gene evidence ════════
tr = pd.read_csv(RAW / "TRRUST_v2_human.tsv", sep="\t", header=None, names=["TF", "target", "mode", "pmid"])
ct = pd.read_csv(RAW / "CollecTRI_omnipath.tsv", sep="\t")
ct = ct.drop(columns=["source", "target"]).rename(columns={"source_genesymbol": "TF", "target_genesymbol": "target"})
ct["TF"] = ct.TF.str.split("_")    # complexes (e.g. FOS_JUND) -> members
ct = ct.explode("TF")
ct["sign"] = np.where(ct.is_inhibition & ~ct.is_stimulation, -1, 1)
ev = []
for g in HUB7:
    t1 = tr[tr.target == g]; t2 = ct[ct.target == g]
    for tf in sorted(set(t1.TF) | set(t2.TF)):
        r, p = spearmanr(Et[tf], Et[g]) if tf in Et.columns else (np.nan, np.nan)
        ev.append(dict(hub_gene=g, TF=tf, in_TRRUST=tf in set(t1.TF), in_CollecTRI=tf in set(t2.TF),
                       TRRUST_mode=";".join(t1[t1.TF == tf]["mode"].unique()),
                       TRRUST_PMIDs=";".join(t1[t1.TF == tf].pmid.astype(str).unique()),
                       CollecTRI_refs=";".join(t2[t2.TF == tf].references.dropna().astype(str).unique())[:300],
                       rho_TF_mRNA_vs_target=r, p=p))
ev = pd.DataFrame(ev); ev["FDR"] = bh(ev.p.fillna(1))
ev.to_csv(OUT / "M9_curated_TF_target_evidence.csv", index=False)

# ════════ M9 (b) TF activity (ULM), hub genes excluded from regulons ════════
net = ct[["TF", "target", "sign"]].drop_duplicates(["TF", "target"])
net = net[net.target.isin(Et.columns) & ~net.target.isin(HUB7)]
sizes = net.groupby("TF").size(); net = net[net.TF.isin(sizes[sizes >= 10].index)]
expr = Et[sorted(net.target.unique())]
Xc = (expr - expr.mean()).T                                       # genes x samples, centred per gene
genes = Xc.index
acts = {}
for tf, g in net.groupby("TF"):
    w = pd.Series(0.0, index=genes); w[g.target.values] = g.sign.values
    x = w.values; xm = x - x.mean()
    Y = Xc.values - Xc.values.mean(0)
    beta = xm @ Y / (xm @ xm)
    resid = Y - np.outer(xm, beta)
    se = np.sqrt((resid ** 2).sum(0) / (len(x) - 2) / (xm @ xm))
    acts[tf] = beta / se                                           # t-value = activity
A = pd.DataFrame(acts, index=Xc.columns)
A.to_csv(OUT / "M9_TF_activity_ULM_CollecTRI_tumours.csv")
ta = []
for tf in A.columns:
    r, p = spearmanr(A[tf], score.loc[A.index])
    ta.append(dict(TF=tf, regulon_size=int(sizes[tf]), rho_activity_vs_score=r, p=p,
                   rho_mRNA_vs_score=spearmanr(Et[tf], score)[0] if tf in Et.columns else np.nan))
ta = pd.DataFrame(ta); ta["FDR"] = bh(ta.p)
ta = ta.sort_values("rho_activity_vs_score", ascending=False)
ta.to_csv(OUT / "M9_TF_activity_vs_score.csv", index=False)
FOCUS = ["FOXM1", "E2F1", "E2F2", "E2F3", "E2F4", "E2F7", "E2F8", "MYBL2", "MYC", "TP53", "RB1", "RELA", "STAT3"]

# ════════ M10 (a) copy number ════════
cn = pd.read_csv(RAW / "LUAD_GISTIC2_thresholded_by_genes.gz", sep="\t", index_col=0)
cn = cn.loc[[g for g in HUB7 if g in cn.index]].T
cn.index = cn.index.str[:15]
ex15 = Et.copy(); ex15.index = ex15.index.str[:15]; ex15 = ex15[~ex15.index.duplicated()]
common = cn.index.intersection(ex15.index)
cnr = []
for g in cn.columns:
    c = cn.loc[common, g]; e = ex15.loc[common, g]
    groups = [e[c == k] for k in sorted(c.unique()) if (c == k).sum() >= 5]
    r, p = spearmanr(c, e)
    cnr.append(dict(gene=g, n=len(common), pct_deep_del=100 * (cn[g] == -2).mean(), pct_shallow_del=100 * (cn[g] == -1).mean(),
                    pct_neutral=100 * (cn[g] == 0).mean(), pct_gain=100 * (cn[g] == 1).mean(), pct_amp=100 * (cn[g] == 2).mean(),
                    rho_CN_vs_expr=r, p=p, R2_expr_explained_by_CN=r ** 2,
                    kruskal_p=kruskal(*groups).pvalue if len(groups) > 1 else np.nan))
cnr = pd.DataFrame(cnr); cnr["FDR"] = bh(cnr.p); cnr.to_csv(OUT / "M10_copy_number_vs_expression.csv", index=False)

# ════════ M10 (b) methylation ════════
pm = pd.read_csv(RAW / "illuminaMethyl450_probeMap.tsv", sep="\t").rename(columns={"#id": "probe"})
me = pd.read_csv(RAW / "LUAD_methylation450_selected.tsv", sep="\t", index_col=0)
rows = []
for g, (chrom, tss, strand) in TSS.items():
    pr = pm[pm.gene.fillna("").str.split(",").apply(lambda l: g in l) & (pm.chrom == chrom)].copy()
    pr["dist_to_TSS"] = (pr.chromStart - tss) * (1 if strand == "+" else -1)
    pr["promoter"] = pr.dist_to_TSS.between(-1500, 500)
    for _, p_ in pr.iterrows():
        if p_.probe not in me.index: continue
        b = me.loc[p_.probe]
        bt = b[[c for c in b.index if c[13:15] == "01"]].dropna(); bn = b[[c for c in b.index if c[13:15] == "11"]].dropna()
        e = ex15[g].reindex(bt.index).dropna(); bt2 = bt.loc[e.index]
        r, pr_ = spearmanr(bt2, e) if len(e) > 10 else (np.nan, np.nan)
        rows.append(dict(gene=g, probe=p_.probe, dist_to_TSS=int(p_.dist_to_TSS), promoter=bool(p_.promoter),
                         n_tumour=len(bt), n_normal=len(bn), median_beta_tumour=bt.median(), median_beta_normal=bn.median(),
                         delta_beta=bt.median() - bn.median(),
                         p_tumour_vs_normal=mannwhitneyu(bt, bn).pvalue if len(bn) >= 5 else np.nan,
                         rho_beta_vs_expr=r, p_beta_vs_expr=pr_))
mt = pd.DataFrame(rows)
mt["FDR_tumour_vs_normal"] = bh(mt.p_tumour_vs_normal.fillna(1)); mt["FDR_beta_vs_expr"] = bh(mt.p_beta_vs_expr.fillna(1))
mt.to_csv(OUT / "M10_methylation_probe_level.csv", index=False)
# gene-level promoter summary
gl = []
for g in TSS:
    pp = mt[(mt.gene == g) & mt.promoter].probe.tolist()
    if not pp: gl.append(dict(gene=g, n_promoter_probes=0)); continue
    B = me.loc[pp]
    bt = B[[c for c in B.columns if c[13:15] == "01"]].mean(); bn = B[[c for c in B.columns if c[13:15] == "11"]].mean()
    e = ex15[g].reindex(bt.index).dropna()
    r, p = spearmanr(bt.loc[e.index], e)
    gl.append(dict(gene=g, n_promoter_probes=len(pp), median_promoter_beta_tumour=bt.median(),
                   median_promoter_beta_normal=bn.median(), delta_beta=bt.median() - bn.median(),
                   p_tumour_vs_normal=mannwhitneyu(bt, bn).pvalue, rho_promoter_beta_vs_expr=r, p_beta_vs_expr=p,
                   n_tumour=len(bt), n_normal=len(bn)))
gl = pd.DataFrame(gl)
gl["FDR_tumour_vs_normal"] = bh(gl.p_tumour_vs_normal.fillna(1)); gl["FDR_beta_vs_expr"] = bh(gl.p_beta_vs_expr.fillna(1))
gl.to_csv(OUT / "M10_methylation_promoter_gene_level.csv", index=False)

# ════════ figures ════════
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
f = ta[ta.TF.isin(FOCUS)].sort_values("rho_activity_vs_score")
ax[0].barh(f.TF, f.rho_activity_vs_score, color=np.where(f.FDR < 0.05, "#c0392b", "#bdc3c7"))
ax[0].set(xlabel="Spearman ρ: TF activity (CollecTRI, hub genes excluded) vs 7-gene score",
          title="(A) Inferred TF activity (red: FDR < 0.05)")
c = cnr.set_index("gene")[["pct_deep_del", "pct_shallow_del", "pct_gain", "pct_amp"]]
c.plot.bar(stacked=True, ax=ax[1], color=["#2166ac", "#92c5de", "#f4a582", "#b2182b"], width=.7)
ax[1].set(ylabel="% of TCGA-LUAD tumours", title="(B) GISTIC2 copy number (thresholded)", xlabel="")
ax[1].legend(["Deep deletion", "Shallow deletion", "Gain", "Amplification"], fontsize=8, frameon=False)
for i, (g, r) in enumerate(cnr.set_index("gene").iterrows()):
    ax[1].text(i, c.loc[g].sum() + 1, f"ρ={r.rho_CN_vs_expr:.2f}", ha="center", fontsize=8)
g2 = gl.dropna(subset=["median_promoter_beta_tumour"])
x = np.arange(len(g2))
ax[2].bar(x - .2, g2.median_promoter_beta_normal, .4, color="#2980b9", label="Normal")
ax[2].bar(x + .2, g2.median_promoter_beta_tumour, .4, color="#c0392b", label="Tumour")
for i, r in enumerate(g2.itertuples()):
    ax[2].text(i, max(r.median_promoter_beta_normal, r.median_promoter_beta_tumour) + .02,
               f"ρ={r.rho_promoter_beta_vs_expr:.2f}", ha="center", fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels(g2.gene, fontstyle="italic")
ax[2].set(ylabel="Median promoter β (450K)", title="(C) Promoter methylation (ρ = β vs expression)")
ax[2].legend(frameon=False, fontsize=8)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_M9_M10_TF_CNV_methylation.png", dpi=300); plt.close()

rep = ["R11b TF / CNV / METHYLATION",
       "\nM9a curated support (TRRUST or CollecTRI) for hub genes:",
       ev.groupby("hub_gene").apply(lambda d: ", ".join(f"{r.TF}({'T' if r.in_TRRUST else ''}{'C' if r.in_CollecTRI else ''})"
                                                        for r in d.itertuples())).to_string(),
       "\nM9b TF activity vs 7-gene score (focus TFs):",
       ta[ta.TF.isin(FOCUS)].to_string(index=False, float_format=fmt),
       "\nM9b top 15 TFs by activity correlation:", ta.head(15).to_string(index=False, float_format=fmt),
       "\nM10a copy number:", cnr.to_string(index=False, float_format=fmt),
       "\nM10b promoter methylation (gene level):", gl.to_string(index=False, float_format=fmt),
       f"\nM10b probes: {len(mt)} total, {mt.promoter.sum()} promoter; "
       f"significant tumour-vs-normal (FDR<0.05): {(mt.FDR_tumour_vs_normal < 0.05).sum()}; "
       f"significant beta-vs-expression: {(mt.FDR_beta_vs_expr < 0.05).sum()}"]
(OUT / "R11b_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
