"""
R01_wgcna_audit.py  —  Revision step 1 (PI comment M2)
=======================================================
Audits the ORIGINAL WGCNA run (scripts/05_wgcna.py) without changing it.

Questions answered:
  1. Was the network signed or unsigned?
  2. What soft-threshold power was actually used, and was the scale-free fit computed correctly?
  3. Why are the five hub genes "in ME7" (r = -0.71 with tumour) yet up-regulated?
     -> recompute SIGNED module membership (MM) and gene significance (GS)
  4. What is the composition of ME7 (up vs down DEGs)?

Inputs (read-only, original project):  data/filtered_counts.csv, data/metadata.csv,
          results/wgcna_modules.csv, results/wgcna_eigengenes.csv, results/deg_all_results.csv
Outputs:  analysis/01_WGCNA_audit/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
DATA, RES = ROOT / "data", ROOT / "results"
OUT = ROOT / "analysis" / "01_WGCNA_audit"
OUT.mkdir(parents=True, exist_ok=True)
HUBS = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
CANDS = HUBS + ["SCN7A", "SCN4B", "LAIR1", "IGLL5"]

report = []
def log(s=""):
    print(s); report.append(s)

# ── Reconstruct the exact expression matrix used by 05_wgcna.py ──
counts = pd.read_csv(DATA / "filtered_counts.csv", index_col=0)
meta = pd.read_csv(DATA / "metadata.csv").set_index("sample_submitter_id")
cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
logcpm = np.log2(cpm + 1)
mad = logcpm.sub(logcpm.median(axis=1), axis=0).abs().median(axis=1)
top = mad.nlargest(5000).index
expr = logcpm.loc[top].T                                  # samples x genes

modules = pd.read_csv(RES / "wgcna_modules.csv")
me = pd.read_csv(RES / "wgcna_eigengenes.csv", index_col=0)
deg = pd.read_csv(RES / "deg_all_results.csv", index_col=0)

assert list(modules.gene_id) == list(expr.columns), "gene order mismatch with original run"
expr = expr.loc[me.index]
tumor = (meta.loc[expr.index, "condition"] == "Tumor").astype(float).values

log("=" * 70)
log("R01  WGCNA AUDIT (PI comment M2)")
log("=" * 70)
log(f"Samples: {expr.shape[0]} (tumour {int(tumor.sum())}, normal {int((1-tumor).sum())}); genes: {expr.shape[1]}")

# ── 1. Signed vs unsigned (read from code) ──
log("\n[1] Network type in 05_wgcna.py")
log("    adjacency = np.abs(cor) ** power   -> UNSIGNED network")
log("    MM and GS were stored as abs(r)      -> sign information discarded")
log("    Manuscript Methods 2.5 says 'signed' -> Methods text is WRONG")

# ── 2. Soft-threshold: correct WGCNA scale-free fit ──
log("\n[2] Soft-threshold power")
X = expr.values.astype(np.float32)
X = (X - X.mean(0)) / X.std(0)
cor = (X.T @ X) / X.shape[0]
np.fill_diagonal(cor, 0)

def sft_fit(k, nbreaks=10):
    """WGCNA scaleFreeFitIndex: R^2 of log10 p(k) ~ log10 k (signed by slope)."""
    breaks = np.linspace(k.min(), k.max(), nbreaks + 1)
    h, e = np.histogram(k, bins=breaks)
    mids = (e[:-1] + e[1:]) / 2
    pk = h / h.sum()
    ok = pk > 0
    x, y = np.log10(mids[ok]), np.log10(pk[ok])
    slope, inter = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (slope * x + inter)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return -np.sign(slope) * r2, slope

rows = []
for p in range(1, 21):
    k_u = (np.abs(cor) ** p).sum(0)
    k_s = (((1 + cor) / 2) ** p).sum(0) - 0.5 ** p   # signed adjacency, remove diagonal
    r2u, su = sft_fit(k_u); r2s, ss = sft_fit(k_s)
    rows.append(dict(power=p, unsigned_R2=r2u, unsigned_mean_k=k_u.mean(),
                     signed_R2=r2s, signed_mean_k=k_s.mean()))
sft = pd.DataFrame(rows)
sft.to_csv(OUT / "soft_threshold_corrected.csv", index=False)
orig_power = 1   # Supplementary Fig. S9 title: "power = 1"
log(f"    Original run selected power = {orig_power} (S9 shows R^2 ~ -0.75 for ALL powers: fit was mis-computed)")
for col in ["unsigned", "signed"]:
    ok = sft[sft[f"{col}_R2"] >= 0.85]
    pick = int(ok.power.iloc[0]) if len(ok) else int(sft.loc[sft[f"{col}_R2"].idxmax(), "power"])
    log(f"    Corrected fit, {col:8s}: first power with R^2>=0.85 -> {pick} "
        f"(R^2={sft.loc[sft.power==pick, f'{col}_R2'].item():.2f})")
del cor

# ── 3. Signed MM / GS for ME7 ──
log("\n[3] Signed module membership and gene significance")
mod7 = modules[modules.module == 7].copy()
E7 = expr[mod7.gene_id]
mod7["MM_signed"] = E7.apply(lambda g: np.corrcoef(g, me["ME7"])[0, 1]).values
mod7["GS_signed"] = E7.apply(lambda g: np.corrcoef(g, tumor)[0, 1]).values
mod7 = mod7.join(deg[["log2FoldChange", "padj"]], on="gene_id")
mod7["DEG_direction"] = np.where(mod7.padj < 0.05,
                          np.where(mod7.log2FoldChange > 1, "Up",
                          np.where(mod7.log2FoldChange < -1, "Down", "NS")), "NS")
mod7.sort_values("MM_signed").to_csv(OUT / "ME7_gene_table_signedMM_GS_log2FC.csv", index=False)

log(f"    ME7 size: {len(mod7)} genes")
log(f"    ME7 eigengene vs tumour status: r = {np.corrcoef(me['ME7'], tumor)[0,1]:+.3f}")
log(f"    Genes with MM_signed > 0: {(mod7.MM_signed>0).sum()},  < 0: {(mod7.MM_signed<0).sum()}")
log(f"    DEG direction in ME7: {mod7.DEG_direction.value_counts().to_dict()}")
log(f"    corr(MM_signed, log2FC) = {np.corrcoef(mod7.MM_signed, mod7.log2FoldChange.fillna(0))[0,1]:+.3f}")

cand = pd.DataFrame()
for g in CANDS:
    gid = modules.loc[modules.gene_name == g, "gene_id"]
    if gid.empty: continue
    gid = gid.item()
    m = int(modules.loc[modules.gene_id == gid, "module"].item())
    ge = expr[gid]
    cand = pd.concat([cand, pd.DataFrame([dict(
        gene=g, module=m,
        MM_signed=np.corrcoef(ge, me[f"ME{m}"])[0, 1],
        GS_signed=np.corrcoef(ge, tumor)[0, 1],
        log2FC=deg.loc[gid, "log2FoldChange"])])])
cand.to_csv(OUT / "PPI_candidates_signed_MM_GS.csv", index=False)
log("\n    Nine PPI candidates (signed):")
log(cand.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ── Figures ──
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].hist(mod7.log2FoldChange.dropna(), bins=60, color="#7f8c8d")
ax[0].axvline(0, color="k", lw=.8)
ax[0].set(title=f"log2FC of ME7 genes (n={len(mod7)})", xlabel="log2FC (tumour vs normal)", ylabel="Genes")
c = np.where(mod7.log2FoldChange > 0, "#e74c3c", "#3498db")
ax[1].scatter(mod7.MM_signed, mod7.GS_signed, s=6, c=c, alpha=.5)
for _, r in cand[cand.module == 7].iterrows():
    ax[1].annotate(r.gene, (r.MM_signed, r.GS_signed), fontsize=8)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
ax[1].set(title="ME7: signed MM vs signed GS", xlabel="Module membership (signed r with ME7)",
          ylabel="Gene significance (signed r with tumour)")
ax[2].plot(sft.power, sft.unsigned_R2, "o-", label="unsigned")
ax[2].plot(sft.power, sft.signed_R2, "s-", label="signed")
ax[2].axhline(.85, ls="--", c="grey"); ax[2].legend()
ax[2].set(title="Scale-free fit (corrected calculation)", xlabel="Soft-threshold power", ylabel="Signed R²")
plt.tight_layout(); plt.savefig(OUT / "WGCNA_audit_figure.png", dpi=200); plt.close()

(OUT / "WGCNA_audit_report.txt").write_text("\n".join(report), encoding="utf-8")
print(f"\nSaved to {OUT}")
