"""
R02_wgcna_signed.py  —  Revision step 2 (PI comment M2)
========================================================
Corrected WGCNA, replacing scripts/05_wgcna.py.

Fixes relative to the original run
  * SIGNED network: a_ij = ((1 + r_ij) / 2) ** beta     (original: |r| ** 1, unsigned)
  * Soft-threshold chosen with the standard WGCNA scale-free fit index
  * Signed TOM, average-linkage clustering, Dynamic Tree Cut (hybrid, deepSplit = 2)
  * Modules merged when eigengene correlation > 0.75
  * Signed module membership (MM) and gene significance (GS)
  * Two analyses kept separate (PI request):
      A. All 576 samples  -> modules vs TUMOUR STATUS (discovery of key module)
      B. Tumours only     -> modules vs STAGE, AGE (Spearman) and OVERALL SURVIVAL (Cox)
  * Every parameter written to a parameter file

GPU: correlation and TOM use PyTorch on CUDA automatically if available
     (e.g. RTX 5060: pip install torch --index-url https://download.pytorch.org/whl/cu128).
     Otherwise NumPy on CPU (5,000 genes takes a few minutes).

Outputs: analysis/02_WGCNA_signed/
"""
import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr
from dynamicTreeCut import cutreeHybrid
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / "data", ROOT / "results"
OUT = ROOT / "analysis" / "02_WGCNA_signed"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = dict(
    expression="log2(CPM + 1) of filtered STAR counts",
    gene_selection="top 5000 genes by median absolute deviation (all 576 samples)",
    correlation="Pearson",
    network_type="signed",
    adjacency="((1 + r) / 2) ** beta",
    tom="signed TOM (Zhang & Horvath 2005)",
    soft_threshold_rule="lowest power with scale-free R^2 >= 0.85 (WGCNA scaleFreeFitIndex, 10 bins)",
    clustering="average linkage on 1 - TOM",
    tree_cut="Dynamic Tree Cut hybrid (dynamicTreeCut python port), deepSplit=2, minClusterSize=30, pamStage=True",
    merge_cut_height=0.25,
    trait_A="tumour status (1 tumour / 0 normal), Pearson r, all samples",
    trait_B="tumour-only: Spearman r with stage (I-IV) and age; univariate Cox HR per SD of eigengene for OS",
    multiple_testing="Benjamini-Hochberg across modules within each trait",
    random_seed=42,
)
N_TOP, MIN_SIZE, DEEP_SPLIT, MERGE_H = 5000, 30, 2, 0.25
HUBS = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]

report = []
def log(s=""):
    print(s, flush=True); report.append(str(s))

# ── backend ──
try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else None
except ImportError:
    torch, DEV = None, None
log(f"Compute backend: {'PyTorch CUDA (' + torch.cuda.get_device_name(0) + ')' if DEV else 'NumPy CPU'}")

def corr(X):
    """Gene-gene Pearson correlation; X = samples x genes."""
    X = (X - X.mean(0)) / X.std(0)
    if DEV:
        t = torch.tensor(X, dtype=torch.float32, device=DEV)
        return ((t.T @ t) / t.shape[0]).cpu().numpy()
    X = X.astype(np.float32)
    return (X.T @ X) / X.shape[0]

def signed_tom(C, beta):
    if DEV:
        A = ((1 + torch.tensor(C, device=DEV)) / 2) ** beta
        A.fill_diagonal_(0)
        k = A.sum(0)
        num = A @ A + A
        den = torch.minimum(k[:, None], k[None, :]) + 1 - A
        T = num / den
        T.fill_diagonal_(1)
        return T.cpu().numpy()
    A = ((1 + C) / 2) ** beta
    np.fill_diagonal(A, 0)
    k = A.sum(0)
    T = (A @ A + A)
    T /= (np.minimum(k[:, None], k[None, :]) + 1 - A)
    np.fill_diagonal(T, 1)
    return T

def sft_index(k, nb=10):
    h, e = np.histogram(k, bins=np.linspace(k.min(), k.max(), nb + 1))
    mids = (e[:-1] + e[1:]) / 2
    pk = h / h.sum(); ok = pk > 0
    x, y = np.log10(mids[ok]), np.log10(pk[ok])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (b * x + a)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return -np.sign(b) * r2, b

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def eigengenes(E, labels):
    MEs = {}
    for m in sorted(set(labels) - {0}):
        Z = E.loc[:, labels == m]
        Z = (Z - Z.mean()) / Z.std()
        u, s, vt = np.linalg.svd(Z.values, full_matrices=False)
        me = u[:, 0] * s[0]
        if np.corrcoef(me, Z.mean(1))[0, 1] < 0:
            me = -me
        MEs[f"ME{m}"] = me
    return pd.DataFrame(MEs, index=E.index)

def run_wgcna(E, tag):
    """Full signed WGCNA on samples x genes matrix E. Returns labels, MEs, power, sft table."""
    t0 = time.time()
    log(f"\n--- Network {tag}: {E.shape[0]} samples x {E.shape[1]} genes ---")
    C = corr(E.values)
    np.fill_diagonal(C, 0)
    rows = []
    for p in range(1, 31):
        A = ((1 + C) / 2) ** p
        k = A.sum(0) - 0.5 ** p           # remove self term (diagonal r set to 0 -> a_ii = 0.5**p)
        r2, b = sft_index(k)
        rows.append(dict(power=p, R2=r2, slope=b, mean_k=k.mean(), median_k=np.median(k)))
    sft = pd.DataFrame(rows)
    ok = sft[sft.R2 >= 0.85]
    beta = int(ok.power.iloc[0]) if len(ok) else int(sft.loc[sft.R2.idxmax(), "power"])
    log(f"  soft-threshold power = {beta} (R2 = {sft.loc[sft.power == beta, 'R2'].item():.3f}, "
        f"mean k = {sft.loc[sft.power == beta, 'mean_k'].item():.1f})")
    np.fill_diagonal(C, 1)
    T = signed_tom(C, beta); del C
    D = 1 - T; del T
    np.fill_diagonal(D, 0); D = np.clip(D, 0, 1).astype(np.float64)
    Z = linkage(squareform(D, checks=False), method="average")
    cut = cutreeHybrid(Z, D, minClusterSize=MIN_SIZE, deepSplit=DEEP_SPLIT, pamStage=True, verbose=0)
    labels = np.asarray(cut["labels"]).astype(int)
    del D
    # merge close modules
    while True:
        MEs = eigengenes(E, labels)
        if MEs.shape[1] < 2: break
        R = MEs.corr().values; np.fill_diagonal(R, -1)
        i, j = np.unravel_index(R.argmax(), R.shape)
        if R[i, j] <= 1 - MERGE_H: break
        a, b_ = int(MEs.columns[i][2:]), int(MEs.columns[j][2:])
        labels[labels == b_] = a
    # renumber by size (1 = largest)
    sizes = pd.Series(labels[labels > 0]).value_counts()
    remap = {old: new for new, old in enumerate(sizes.index, 1)}; remap[0] = 0
    labels = np.array([remap[l] for l in labels])
    MEs = eigengenes(E, labels)
    log(f"  modules: {len(sizes)} (+ grey/unassigned {int((labels == 0).sum())} genes); "
        f"sizes: {pd.Series(labels[labels>0]).value_counts().sort_index().to_dict()}")
    log(f"  runtime {time.time() - t0:.0f} s")
    sft.to_csv(OUT / f"soft_threshold_{tag}.csv", index=False)
    return labels, MEs, beta, sft, Z

# ════════════════════ data ════════════════════
counts = pd.read_csv(DATA / "filtered_counts.csv", index_col=0)
meta = pd.read_csv(DATA / "metadata.csv").set_index("sample_submitter_id")
gmap = pd.read_csv(DATA / "gene_id_mapping.csv", index_col=0)["gene_name"]
deg = pd.read_csv(RES / "deg_all_results.csv", index_col=0)

logcpm = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
mad = logcpm.sub(logcpm.median(1), axis=0).abs().median(1)
genes = mad.nlargest(N_TOP).index
expr = logcpm.loc[genes].T
meta = meta.loc[expr.index]
tumour = (meta.condition == "Tumor").astype(int)
gnames = gmap.reindex(genes).fillna(pd.Series(genes, index=genes)).values
log("=" * 70); log("R02  SIGNED WGCNA (corrected)"); log("=" * 70)
log(f"Hub genes present in the 5,000-gene input: "
    f"{[g for g in HUBS if g in set(gnames)]}")

# ════════════════════ A. all samples, tumour status ════════════════════
labA, MEA, betaA, sftA, ZA = run_wgcna(expr, "A_all_samples")
trA = []
for c in MEA:
    r, p = pearsonr(MEA[c], tumour)
    trA.append(dict(module=c, size=int((labA == int(c[2:])).sum()), r_tumour=r, p=p))
trA = pd.DataFrame(trA); trA["FDR"] = bh(trA.p)
trA = trA.sort_values("r_tumour", ascending=False)
trA.to_csv(OUT / "A_module_trait_tumour_status.csv", index=False)
log("\n  Module vs tumour status (all samples):")
log(trA.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# signed MM / GS for every gene
GS = expr.apply(lambda g: np.corrcoef(g, tumour)[0, 1])
genetab = pd.DataFrame({"gene_id": genes, "gene_name": gnames, "module": labA,
                        "GS_tumour": GS.values})
for c in MEA:
    genetab[f"MM_{c}"] = [np.corrcoef(expr[g], MEA[c])[0, 1] for g in genes]
genetab["MM_own"] = [row[f"MM_ME{int(row.module)}"] if row.module > 0 else np.nan
                     for _, row in genetab.iterrows()]
genetab = genetab.join(deg[["log2FoldChange", "padj"]], on="gene_id")
genetab.to_csv(OUT / "A_gene_module_MM_GS.csv", index=False)

key = trA.iloc[0]["module"]; key_n = int(key[2:])
log(f"\n  KEY MODULE (most positively tumour-associated): {key} "
    f"(n = {int(trA.iloc[0]['size'])}, r = {trA.iloc[0]['r_tumour']:+.3f}, FDR = {trA.iloc[0]['FDR']:.2e})")
hubs5 = genetab[genetab.gene_name.isin(HUBS)][["gene_name", "module", "MM_own", f"MM_{key}", "GS_tumour", "log2FoldChange"]]
log("\n  Where are the five original hub genes now?")
log(hubs5.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

km = genetab[genetab.module == key_n].copy()
km["is_DEG_up"] = (km.padj < 0.05) & (km.log2FoldChange > 1)
cand = km[(km.MM_own > 0.8) & (km.GS_tumour > 0.5) & km.is_DEG_up].sort_values("MM_own", ascending=False)
cand.to_csv(OUT / f"A_key_module_{key}_hub_candidates_MM0.8_GS0.5_DEGup.csv", index=False)
km.sort_values("MM_own", ascending=False).to_csv(OUT / f"A_key_module_{key}_all_genes.csv", index=False)
log(f"\n  Key-module hub candidates (MM > 0.8, GS > 0.5, up-regulated DEG): {len(cand)} genes")
log("  top 30: " + ", ".join(cand.gene_name.head(30)))
log(f"  Original five hub genes among candidates: {[g for g in HUBS if g in set(cand.gene_name)]}")

# ════════════════════ B. tumours only, clinical traits ════════════════════
tum = expr.index[tumour.values == 1]
labB, MEB, betaB, sftB, ZB = run_wgcna(expr.loc[tum], "B_tumour_only")
mt = meta.loc[tum]
stage = mt.tumor_stage.astype(str).str.upper().map(
    lambda s: 4 if "IV" in s else 3 if "III" in s else 2 if "II" in s else 1 if "STAGE I" in s else np.nan)
trB = []
for c in MEB:
    rs, ps = spearmanr(MEB[c], stage, nan_policy="omit")
    ra, pa = spearmanr(MEB[c], mt.age_years, nan_policy="omit")
    d = pd.DataFrame({"me": (MEB[c] - MEB[c].mean()) / MEB[c].std(),
                      "T": mt.os_time_days, "E": mt.os_event}).dropna()
    d = d[d["T"] > 0]
    cph = CoxPHFitter().fit(d, "T", "E")
    trB.append(dict(module=c, size=int((labB == int(c[2:])).sum()),
                    rho_stage=rs, p_stage=ps, rho_age=ra, p_age=pa,
                    HR_OS_perSD=float(np.exp(cph.params_["me"])), p_OS=float(cph.summary.loc["me", "p"]),
                    n_stage=int(stage.notna().sum()), n_OS=len(d)))
trB = pd.DataFrame(trB)
for t in ["stage", "age", "OS"]:
    trB[f"FDR_{t}"] = bh(trB[f"p_{t}"])
trB.to_csv(OUT / "B_tumour_only_module_clinical_traits.csv", index=False)
log("\n  Tumour-only modules vs clinical traits:")
log(trB[["module", "size", "rho_stage", "FDR_stage", "HR_OS_perSD", "FDR_OS", "rho_age", "FDR_age"]]
    .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
gB = pd.DataFrame({"gene_id": genes, "gene_name": gnames, "module_tumour_only": labB})
gB.to_csv(OUT / "B_tumour_only_gene_modules.csv", index=False)
hubB = gB[gB.gene_name.isin(HUBS)]
log("\n  Five hub genes in tumour-only network: " +
    ", ".join(f"{r.gene_name}->M{r.module_tumour_only}" for r in hubB.itertuples()))

# overlap between key module (A) and tumour-only modules
ov = pd.crosstab(labA == key_n, labB).loc[True]
log(f"  {key} genes distribute over tumour-only modules: {ov[ov > 0].to_dict()}")

# ════════════════════ figures ════════════════════
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for s, lab, col in [(sftA, "A: all samples", "#c0392b"), (sftB, "B: tumour only", "#2980b9")]:
    ax[0].plot(s.power, s.R2, "o-", ms=4, label=lab, color=col)
    ax[1].plot(s.power, s.mean_k, "o-", ms=4, label=lab, color=col)
ax[0].axhline(.85, ls="--", c="grey")
for a, b_, c in [(ax[0], betaA, "#c0392b"), (ax[0], betaB, "#2980b9")]:
    a.axvline(b_, ls=":", c=c)
ax[0].set(xlabel="Soft-threshold power (β)", ylabel="Scale-free topology fit (signed R²)",
          title=f"Scale independence (β_A = {betaA}, β_B = {betaB})")
ax[1].set(xlabel="Soft-threshold power (β)", ylabel="Mean connectivity", title="Mean connectivity", yscale="log")
for a in ax: a.legend(); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_soft_threshold.png", dpi=300); plt.close()

# module-trait heatmap A + B
fig, ax = plt.subplots(1, 2, figsize=(11, max(4, 0.45 * max(len(trA), len(trB)) + 1.5)),
                       gridspec_kw={"width_ratios": [1, 3]})
hA = trA.set_index("module")[["r_tumour"]]
annA = trA.set_index("module").apply(lambda r: f"{r.r_tumour:.2f}\n({r.FDR:.1e})", axis=1).to_frame()
sns.heatmap(hA, annot=annA, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax[0], cbar=False,
            yticklabels=[f"{m} ({s})" for m, s in zip(trA.module, trA["size"])], xticklabels=["Tumour status"])
ax[0].set_title("A. All samples (Pearson r, FDR)"); ax[0].set_ylabel("")
hB = trB.set_index("module")[["rho_stage", "rho_age"]].assign(logHR=np.log2(trB.set_index("module").HR_OS_perSD))
annB = trB.set_index("module").apply(lambda r: pd.Series([f"{r.rho_stage:.2f}\n({r.FDR_stage:.1e})",
                                                         f"{r.rho_age:.2f}\n({r.FDR_age:.1e})",
                                                         f"HR {r.HR_OS_perSD:.2f}\n({r.FDR_OS:.1e})"]), axis=1)
sns.heatmap(hB, annot=annB.values, fmt="", cmap="RdBu_r", center=0, vmin=-.5, vmax=.5, ax=ax[1],
            yticklabels=[f"{m} ({s})" for m, s in zip(trB.module, trB["size"])],
            xticklabels=["Stage (ρ)", "Age (ρ)", "OS (log2 HR/SD)"], cbar_kws={"label": "ρ or log2 HR"})
ax[1].set_title("B. Tumour only (Spearman ρ / Cox HR, FDR)"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(OUT / "Fig_module_trait_A_B.png", dpi=300); plt.close()

# MM vs GS in key module
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(km.MM_own, km.GS_tumour, s=8, c="#c0392b", alpha=.5)
for _, r in km[km.gene_name.isin(HUBS)].iterrows():
    ax.annotate(r.gene_name, (r.MM_own, r.GS_tumour), fontsize=8, xytext=(3, 3), textcoords="offset points")
rr, pp = pearsonr(km.MM_own, km.GS_tumour)
ax.axvline(.8, ls="--", c="grey", lw=.8); ax.axhline(.5, ls="--", c="grey", lw=.8)
ax.set(xlabel=f"Module membership in {key} (signed)", ylabel="Gene significance for tumour status (signed)",
       title=f"{key}: MM vs GS (r = {rr:.2f}, p = {pp:.1e})")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / f"Fig_MM_vs_GS_{key}.png", dpi=300); plt.close()

# module sizes
fig, ax = plt.subplots(figsize=(8, 4))
vc = pd.Series(labA).value_counts().sort_index()
ax.bar([f"M{i}" if i else "grey" for i in vc.index], vc.values,
       color=["#bbbbbb" if i == 0 else ("#c0392b" if i == key_n else "#5d6d7e") for i in vc.index])
for i, v in enumerate(vc.values): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
ax.set(ylabel="Genes", title="Signed WGCNA module sizes (all samples)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_module_sizes_A.png", dpi=300); plt.close()

PARAMS.update(beta_A=betaA, beta_B=betaB, n_modules_A=int(len(set(labA) - {0})),
              n_grey_A=int((labA == 0).sum()), n_modules_B=int(len(set(labB) - {0})),
              n_grey_B=int((labB == 0).sum()), key_module_A=key,
              compute_backend="CUDA" if DEV else "CPU")
(OUT / "WGCNA_parameters.json").write_text(json.dumps(PARAMS, indent=2))
pd.DataFrame({"gene_id": genes, "gene_name": gnames, "module_A": labA, "module_B": labB}) \
  .to_csv(OUT / "gene_module_assignments_A_B.csv", index=False)
(OUT / "R02_report.txt").write_text("\n".join(report), encoding="utf-8")
log(f"\nAll outputs in {OUT}")
