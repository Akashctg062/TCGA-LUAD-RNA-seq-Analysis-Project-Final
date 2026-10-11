"""
R11f_drugs_depmap.py — Revision step 11f (PI comment M8)
=========================================================
Replaces the original literature-curated drug section (which contained factual errors:
ispinesib/filanesib inhibit KIF11 and GSK923295 inhibits CENP-E — not KIF2C; palbociclib inhibits CDK4/6,
not cyclin B-CDK1; taxanes do not target cyclin B) and the circular "drug-sensitivity signature" correlations.

(1) Drug table: curated drug MECHANISMS from ChEMBL (drug_mechanism table) for each hub protein,
    including protein complexes containing it; max clinical phase (R11e download)
(2) DepMap CRISPR (Chronos) gene effect in lung adenocarcinoma cell lines vs all other lines;
    dependency = gene effect < -1 (per DepMap convention ~ median effect of common-essential genes = -1);
    common-essential status from CRISPRInferredCommonEssentials -> discussion of therapeutic window
    (runs only if the DepMap_* files are present in raw/)
Outputs: analysis/11_extended/drugs/
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
OUT = NEW / "11_extended" / "drugs"; OUT.mkdir(exist_ok=True)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
PARTNER = {"CCNB1": "CDK1 (cyclin B1-CDK1 complex)", "CCNB2": "CDK1 (cyclin B2-CDK1 complex)",
           "CCNA2": "CDK2 / CDK1 (cyclin A2-CDK complexes)"}

# ── (1) ChEMBL drug table ──
ch = pd.read_csv(RAW / "ChEMBL_hub_drug_mechanisms.csv")
ch["drug_name"] = ch.drug_name.map(lambda n: n if any(ch_.isdigit() for ch_ in str(n)) else str(n).capitalize())  # INN capitalised, code names kept
rows = []
for g in HUB7:
    d = ch[ch.hub_gene == g].sort_values("max_phase", ascending=False)
    rows.append(dict(gene=g,
                     n_clinical_stage_agents=d.molecule_chembl_id.nunique(),
                     highest_phase=d.max_phase.max() if len(d) else np.nan,
                     agents_phase3=", ".join(d[d.max_phase >= 3].drug_name.unique()),
                     agents_phase2=", ".join(d[d.max_phase == 2].drug_name.unique()),
                     agents_phase1=", ".join(d[d.max_phase == 1].drug_name.unique()),
                     direct_targets=", ".join(sorted(d.target_name.unique())),
                     note=("No curated drug mechanism; druggable only indirectly via partner kinase " + PARTNER[g])
                          if (len(d) == 0 and g in PARTNER) else ("No curated clinical-stage agent in ChEMBL" if len(d) == 0 else "")))
dt = pd.DataFrame(rows)
dt.to_csv(OUT / "Table_hub_gene_drug_mechanisms_ChEMBL.csv", index=False)
ch.to_csv(OUT / "ChEMBL_all_mechanism_records.csv", index=False)
print(dt.to_string(index=False))

# ── (2) DepMap ──
f_eff, f_mod, f_ce = (RAW / "DepMap_CRISPRGeneEffect.csv", RAW / "DepMap_Model.csv",
                      RAW / "DepMap_CRISPRInferredCommonEssentials.csv")
if not (f_eff.exists() and f_mod.exists()):
    print("\nDepMap files not found yet -> skipped. Place DepMap_CRISPRGeneEffect.csv, DepMap_Model.csv "
          "(and DepMap_CRISPRInferredCommonEssentials.csv) in raw/ and re-run.")
    raise SystemExit
eff = pd.read_csv(f_eff, index_col=0)
eff.columns = [c.split(" (")[0] for c in eff.columns]
eff = eff[[g for g in HUB7 if g in eff.columns]]
mod = pd.read_csv(f_mod).set_index("ModelID")
sub_col = "OncotreeSubtype" if "OncotreeSubtype" in mod else [c for c in mod.columns if "subtype" in c.lower()][0]
luad = mod.index[mod[sub_col].astype(str).str.contains("Lung Adenocarcinoma", case=False)].intersection(eff.index)
other = eff.index.difference(luad)
ce = set()
if f_ce.exists():
    c = pd.read_csv(f_ce)
    ce = set(c.iloc[:, 0].astype(str).str.split(" ").str[0])
rel = (RAW / "DepMap_release.txt").read_text().strip() if (RAW / "DepMap_release.txt").exists() else "see Methods"
dm = []
for g in eff.columns:
    a, b = eff.loc[luad, g].dropna(), eff.loc[other, g].dropna()
    dm.append(dict(gene=g, n_LUAD_lines=len(a), median_effect_LUAD=a.median(), pct_LUAD_dependent=100 * (a < -1).mean(),
                   median_effect_other=b.median(), pct_other_dependent=100 * (b < -1).mean(),
                   p_LUAD_vs_other=mannwhitneyu(a, b).pvalue, common_essential=g in ce))
dm = pd.DataFrame(dm); dm.to_csv(OUT / "DepMap_dependency_LUAD.csv", index=False)
print(f"\nDepMap ({rel}); LUAD lines: {len(luad)}, other lines: {len(other)}")
print(dm.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

fig, ax = plt.subplots(figsize=(8, 4.5))
data = [eff.loc[luad, g].dropna() for g in eff.columns]
ax.boxplot(data, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#e6a19a"), medianprops=dict(color="k"))
for i, d in enumerate(data):
    ax.scatter(np.random.default_rng(1).normal(i + 1, .06, len(d)), d, s=6, color="#8e1b10", alpha=.6)
ax.axhline(-1, ls="--", c="grey", lw=.8); ax.axhline(0, c="k", lw=.5)
for i, g in enumerate(eff.columns):
    if g in ce: ax.text(i + 1, ax.get_ylim()[1] * .9 if ax.get_ylim()[1] > 0 else 0.1, "CE", ha="center", fontsize=8)
ax.set_xticklabels(eff.columns, fontstyle="italic")
ax.set(ylabel="CRISPR gene effect (Chronos)", title=f"{rel}: dependency in {len(luad)} LUAD cell lines\n"
       "(dashed = -1, typical common-essential effect; CE = common essential)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_DepMap_LUAD_dependency.png", dpi=300); plt.close()
