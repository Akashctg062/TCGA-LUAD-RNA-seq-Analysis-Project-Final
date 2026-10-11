"""
R10b_immune_purity.py — Revision step 10b (PI comment M7)
==========================================================
Immune context of the 7-gene proliferation signature, corrected for tumour purity.

Replaces the original analysis, whose "ssGSEA" was in fact a mean z-score over custom marker lists
attributed to Bindea et al. (2013) although several sets (M1/M2, monocytes, CD4) are not in that paper.

Methods (all TCGA-LUAD primary tumours with RNA-seq)
  * MCP-counter (Becht et al., Genome Biol 2016): published marker list (R10a); score = mean log2(CPM+1) of
    the population's markers (MCP-counter definition); 10 populations incl. endothelial cells and fibroblasts
  * CIBERSORT relative fractions, TCGA PanCanAtlas (Thorsson et al., Immunity 2018), 22 leukocyte subsets
  * Tumour purity: ABSOLUTE, TCGA PanCanAtlas (Carter et al., Nat Biotechnol 2012)
  * Immune-checkpoint GENES reported separately: PDCD1, CD274, CTLA4, LAG3, HAVCR2 (TIM-3), TIGIT
  * Statistics: Spearman rho (raw) and partial Spearman rho adjusted for ABSOLUTE purity
                (rank-based residual method); BH-FDR within each method family
  * Tumour vs normal comparison of MCP-counter scores (Mann-Whitney, BH)
Outputs: analysis/10_immune/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, rankdata, t as tdist, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "10_immune"; RAW = OUT / "raw"
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
CHECKPOINTS = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT"]
fmt = lambda v: f"{v:.3g}"

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; n = len(x); df = n - 3
    p = 2 * tdist.sf(abs(r * np.sqrt(df / (1 - r ** 2))), df)
    return r, p

# ── expression ──
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
meta = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("sample_submitter_id")
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
E = lc.T
tum = meta.index[meta.condition == "Tumor"].intersection(E.index)

# ── MCP-counter ──
mcp = pd.read_csv(RAW / "MCPcounter_genes.txt", sep="\t")
MCP = {}
for pop, g in mcp.groupby("Cell population"):
    genes = [x for x in g["HUGO symbols"] if x in E.columns]
    MCP[pop] = E[genes].mean(1)
MCP = pd.DataFrame(MCP)
pd.DataFrame([dict(population=k, n_markers_defined=int((mcp["Cell population"] == k).sum()),
                   n_present=int(sum(x in E.columns for x in mcp.loc[mcp["Cell population"] == k, "HUGO symbols"])))
              for k in MCP.columns]).to_csv(OUT / "MCPcounter_marker_coverage.csv", index=False)

# ── CIBERSORT (PanCanAtlas) ──
cb = pd.read_csv(RAW / "TCGA_CIBERSORT_relative.tsv", sep="\t")
cb = cb[cb.CancerType == "LUAD"].copy()
cb["sample"] = cb.SampleID.str.replace(".", "-", regex=False).str[:16]
CB_COLS = [c for c in cb.columns if c not in ["SampleID", "CancerType", "P.value", "Correlation", "RMSE", "sample"]]
cb_pval = cb.groupby("sample")["P.value"].mean()
CB = cb.groupby("sample")[CB_COLS].mean()
CB.columns = [c.replace("..", " ").replace(".", " ").strip() for c in CB.columns]

# ── purity ──
ab = pd.read_csv(RAW / "TCGA_ABSOLUTE_purity.txt", sep="\t")
pur = ab.drop_duplicates("array").set_index("array")["purity"]
purity = pd.Series({s: pur.get(s[:15], np.nan) for s in tum}, name="ABSOLUTE_purity")

# ── analysis frame (tumours) ──
Zt = (E.loc[tum, HUB7] - E.loc[tum, HUB7].mean()) / E.loc[tum, HUB7].std()
X = pd.DataFrame({"HUB7 score": Zt.mean(1)}).join(E.loc[tum, HUB7])
D = X.join(purity).join(MCP.loc[tum].add_prefix("MCP|")).join(CB.add_prefix("CIBERSORT|"), how="left") \
     .join(E.loc[tum, CHECKPOINTS].add_prefix("ICP|"))
D.to_csv(OUT / "tumour_immune_purity_matrix.csv")

rows = []
for fam, cols in [("MCP-counter", [c for c in D if c.startswith("MCP|")]),
                  ("CIBERSORT", [c for c in D if c.startswith("CIBERSORT|")]),
                  ("Checkpoint gene", [c for c in D if c.startswith("ICP|")])]:
    for xv in X.columns:
        for c in cols:
            s = D[[xv, c, "ABSOLUTE_purity"]].dropna()
            r, p = spearmanr(s[xv], s[c])
            rp, pp = partial_spearman(s[xv].values, s[c].values, s.ABSOLUTE_purity.values)
            rows.append(dict(family=fam, variable=xv, feature=c.split("|")[1], n=len(s), rho=r, p=p,
                             rho_purity_adj=rp, p_purity_adj=pp))
res = pd.DataFrame(rows)
for fam, idx in res.groupby("family").groups.items():
    res.loc[idx, "FDR"] = bh(res.loc[idx, "p"]); res.loc[idx, "FDR_purity_adj"] = bh(res.loc[idx, "p_purity_adj"])
res["attenuation_pct"] = 100 * (1 - res.rho_purity_adj.abs() / res.rho.abs().replace(0, np.nan))
res.to_csv(OUT / "immune_correlations_raw_and_purity_adjusted.csv", index=False)

# sensitivity: CIBERSORT restricted to samples with significant deconvolution (CIBERSORT P < 0.05)
good = cb_pval[cb_pval < 0.05].index
sens = []
for c in [c for c in D if c.startswith("CIBERSORT|")]:
    s = D.loc[D.index.intersection(good), ["HUB7 score", c, "ABSOLUTE_purity"]].dropna()
    rp, pp = partial_spearman(s["HUB7 score"].values, s[c].values, s.ABSOLUTE_purity.values)
    sens.append(dict(feature=c.split("|")[1], n=len(s), rho_purity_adj=rp, p=pp))
sens = pd.DataFrame(sens); sens["FDR"] = bh(sens.p)
sens.to_csv(OUT / "sensitivity_CIBERSORT_P_lt_0.05.csv", index=False)

# purity vs score / genes
pv = []
for xv in X.columns:
    s = D[[xv, "ABSOLUTE_purity"]].dropna(); r, p = spearmanr(s[xv], s.ABSOLUTE_purity)
    pv.append(dict(variable=xv, n=len(s), rho_with_purity=r, p=p))
pv = pd.DataFrame(pv); pv.to_csv(OUT / "purity_vs_hub_genes.csv", index=False)

# tumour vs normal MCP-counter
nor = meta.index[meta.condition == "Normal"].intersection(E.index)
tn = []
for c in MCP.columns:
    a, b = MCP.loc[tum, c], MCP.loc[nor, c]
    tn.append(dict(population=c, median_tumour=a.median(), median_normal=b.median(), p=mannwhitneyu(a, b).pvalue))
tn = pd.DataFrame(tn); tn["FDR"] = bh(tn.p); tn.to_csv(OUT / "MCPcounter_tumour_vs_normal.csv", index=False)

# ── figures ──
def heat(fam, title, fname):
    r = res[res.family == fam]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.38 * r.feature.nunique() + 2.6), sharey=True)
    for a, (col, q, lab) in zip(ax, [("rho", "FDR", "Raw Spearman ρ"), ("rho_purity_adj", "FDR_purity_adj",
                                                                       "Partial Spearman ρ (adjusted for ABSOLUTE purity)")]):
        M = r.pivot(index="feature", columns="variable", values=col)[list(X.columns)]
        Q = r.pivot(index="feature", columns="variable", values=q)[list(X.columns)]
        ann = M.round(2).astype(str) + np.where(Q < 0.001, "***", np.where(Q < 0.01, "**", np.where(Q < 0.05, "*", "")))
        sns.heatmap(M, annot=ann, fmt="", cmap="RdBu_r", center=0, vmin=-.6, vmax=.6, ax=a, cbar=a is ax[1],
                    annot_kws={"fontsize": 7}, cbar_kws={"label": "ρ"})
        a.set_title(lab, fontsize=11); a.set_xlabel(""); a.set_ylabel("")
        a.set_xticklabels(a.get_xticklabels(), rotation=40, ha="right")
    fig.suptitle(f"{title}  (TCGA-LUAD tumours; * FDR<0.05, ** <0.01, *** <0.001)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=300); plt.close()
heat("MCP-counter", "MCP-counter populations", "Fig_immune_MCPcounter_raw_vs_purity.png")
heat("CIBERSORT", "CIBERSORT fractions (PanCanAtlas)", "Fig_immune_CIBERSORT_raw_vs_purity.png")
heat("Checkpoint gene", "Immune-checkpoint genes", "Fig_immune_checkpoints_raw_vs_purity.png")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
s = D[["HUB7 score", "ABSOLUTE_purity"]].dropna()
ax.scatter(s.ABSOLUTE_purity, s["HUB7 score"], s=10, alpha=.5, color="#c0392b")
rr = pv.set_index("variable").loc["HUB7 score"]
ax.set(xlabel="ABSOLUTE tumour purity", ylabel="7-gene score (mean z)",
       title=f"Score vs purity: ρ = {rr.rho_with_purity:.2f}, p = {rr.p:.1e}, n = {int(rr.n)}")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_score_vs_purity.png", dpi=300); plt.close()

# ── report ──
key = res[res.variable == "HUB7 score"].sort_values(["family", "rho"])
rep = ["R10 IMMUNE CONTEXT WITH PURITY ADJUSTMENT",
       f"Tumours: {len(tum)}; with ABSOLUTE purity: {purity.notna().sum()}; with CIBERSORT: {len(CB.index.intersection(tum))}",
       "\nPurity vs hub genes:", pv.to_string(index=False, float_format=fmt),
       "\nHUB7 score vs immune features (raw and purity-adjusted):",
       key[["family", "feature", "n", "rho", "FDR", "rho_purity_adj", "FDR_purity_adj", "attenuation_pct"]]
          .to_string(index=False, float_format=fmt),
       f"\nAll tests: significant raw FDR<0.05: {(res.FDR < 0.05).sum()}/{len(res)}; after purity adjustment: "
       f"{(res.FDR_purity_adj < 0.05).sum()}/{len(res)}",
       f"Median |rho| raw {res.rho.abs().median():.3f} -> adjusted {res.rho_purity_adj.abs().median():.3f}",
       f"\nSensitivity, CIBERSORT P<0.05 samples only (n={sens.n.max()}):", sens.to_string(index=False, float_format=fmt),
       "\nMCP-counter tumour vs normal:", tn.to_string(index=False, float_format=fmt)]
(OUT / "R10_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
