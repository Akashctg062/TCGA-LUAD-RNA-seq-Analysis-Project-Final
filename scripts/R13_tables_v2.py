"""R13_tables_v2.py — tables for revision 2 (limitation analyses).
Runs R12e_tables_json.py (Tables 1-4, S1-S21, Supplementary Data S1-S4) and then
  * Table 1: adds the five additional GEO cohorts
  * Table 2: pooled OS HR per gene over all nine cohorts
  * Table 4: adds the five additional cohorts and the nine-cohort meta-analysis
  * new Supplementary Tables S22-S26 (normal reference, single cell, protein, published signatures, follow-up/treatment/ancestry)
Writes Manuscript_revised/tables_v2.json. Every value is read from result files."""
import json, runpy
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ns = runpy.run_path(str(HERE / "R12e_tables_json.py"))
T, add, hr, p3, fp, f, pct, revkm, cohort_col, NEW, OUT, HUB7 = (ns[k] for k in
    ["T", "add", "hr", "p3", "fp", "f", "pct", "revkm", "cohort_col", "NEW", "OUT", "HUB7"])
L = NEW / "13_limitations"
X = L / "extra_cohorts"
R9 = pd.read_csv(X / "R13g_cohort_results.csv").set_index("cohort")
M9 = pd.read_csv(X / "R13g_meta_analysis.csv").set_index("analysis")
NEWC = ["GSE50081", "GSE30219", "GSE37745", "GSE42127", "GSE13213"]
PLAT = {"GSE50081": "HG-U133 Plus 2.0", "GSE30219": "HG-U133 Plus 2.0", "GSE37745": "HG-U133 Plus 2.0",
        "GSE42127": "Illumina WG-6 v3", "GSE13213": "Agilent 4×44K"}
ORIGIN = {"GSE50081": "Canada", "GSE30219": "France", "GSE37745": "Sweden", "GSE42127": "USA", "GSE13213": "Japan"}

# ── Table 1: add five columns ──
t1 = T["T1"]
for g in NEWC:
    C = pd.read_csv(X / f"{g}_clinical_adeno.csv", index_col=0)
    if "ever_smoker" not in C: C["ever_smoker"] = np.nan
    adj = C.adjuvant
    sec = (f"adjuvant therapy recorded: {int((adj == 1).sum())} treated" if adj.notna().any() else "–")
    col = cohort_col(C, f"{g} ({ORIGIN[g]})", PLAT[g], sec)
    if C.ever_smoker.isna().all(): col[5] = "not available"
    t1["cols"].append(col[0])
    for i, row in enumerate(t1["rows"]): row.append(str(col[i + 1]))
t1["title"] = "Table 1. Characteristics of the discovery and eight validation cohorts (primary adenocarcinomas)."
t1["note"] = t1["note"].replace("*Reverse Kaplan–Meier estimate.", "*Reverse Kaplan–Meier estimate. GSE31210, GSE68465 and GSE72094 were "
    "the original validation cohorts; GSE50081, GSE30219, GSE37745, GSE42127 and GSE13213 were added in this revision (adenocarcinomas only; "
    "GSE30219 stage derived from pathological T, N and M, AJCC 7th edition approximation).")
t1["widths"] = [13] + [9] * (len(t1["cols"]) - 1)

# ── Table 2: nine-cohort pooled HR ──
t2 = T["T2"]
j = t2["cols"].index("Pooled OS HR (95% CI)†")
for row in t2["rows"]:
    m = M9.loc[f"gene {row[0]}, all 9 cohorts, multivariable"]; row[j] = hr(m.HR, m.lo, m.hi)
t2["note"] = t2["note"].replace("random-effects estimate across TCGA-LUAD, GSE31210, GSE68465 and GSE72094 (adjusted for age, sex and stage)",
                                "random-effects estimate across TCGA-LUAD and eight GEO cohorts (2,106 patients; adjusted for age, sex and stage)")

# ── Table 4: add new cohorts and nine-cohort meta ──
t4 = T["T4"]
meta_row = t4["rows"].pop()
for g in NEWC:
    x = R9.loc[g]
    t4["rows"].append([g, "OS", f"{int(x.n)} ({int(x.events)})", hr(x.HR_uni, x.lo_uni, x.hi_uni), p3(x.p_uni),
                       hr(x.HR_multi, x.lo_multi, x.hi_multi), p3(x.p_multi), f"{x.C_clinical:.3f} → {x.C_clinical_plus_score:.3f}",
                       p3(x.LRT_p), f(x.tdAUC_3y), f(x.tdAUC_5y)])
mu, mm = M9.loc["all 9 cohorts, univariable"], M9.loc["all 9 cohorts, multivariable"]
t4["rows"].append(["Random-effects meta-analysis (9 cohorts)", "OS", f"{int(R9.n.sum()):,} ({int(R9.events.sum())})",
                   hr(mu.HR, mu.lo, mu.hi), p3(mu.p), hr(mm.HR, mm.lo, mm.hi), p3(mm.p), f"I² = {mu.I2:.0f}% / {mm.I2:.0f}%", "–", "–", "–"])
t4["bold_rows"] = [len(t4["rows"]) - 1]
t4["title"] = "Table 4. Prognostic performance of the locked seven-gene score in the training cohort and eight validation cohorts."
t4["note"] = t4["note"].replace("(Uno, inverse-probability-of-censoring weighting; not estimable at 5 years in GSE72094)",
    "(Uno, inverse-probability-of-censoring weighting; not estimable at 5 years in GSE72094). GSE50081 to GSE13213 were added in this revision "
    "(adenocarcinomas only)").replace("I² given for the univariate / adjusted pooled estimates.",
    "I² given for the univariate / adjusted pooled estimates; the earlier four-cohort estimate was 1.31 (1.20–1.43).")

# ── S22 normal reference ──
A = pd.read_csv(L / "field_effect" / "R13a_hub_tumour_NAT_GTEx.csv").set_index("gene")
B = pd.read_csv(L / "field_effect" / "R13b_hub_genes_DE_vs_GTEx.csv").set_index("gene_name")
SB = json.load(open(L / "field_effect" / "R13b_summary.json"))
rows = []
for g in HUB7 + ["MKI67", "score"]:
    a = A.loc[g]; b = B.loc[g] if g in B.index else None
    rows.append(["Seven-gene score" if g == "score" else g, f"{a.median_tumour:.2f} / {a.median_NAT:.2f} / {a.median_GTEx:.2f}",
                 f(a.cliff_T_vs_GTEx), p3(a.FDR_T_vs_GTEx), f(a.cliff_NAT_vs_GTEx), p3(a.FDR_NAT_vs_GTEx), f(a.AUC_T_vs_GTEx),
                 "–" if b is None else f(b.log2FoldChange_TvsGTEx), "–" if b is None else f(b.log2FoldChange_NATvsGTEx),
                 "–" if b is None else p3(b.padj_NATvsGTEx)])
n = SB["n_TvsGTEx"]
add("S22", "Table S22. Hub genes in LUAD tumors, TCGA adjacent normal tissue (NAT) and GTEx healthy lung (UCSC Toil, uniform processing).",
    ["Gene", "Median log2 TPM, tumor / NAT / GTEx", "Cliff's δ, tumor vs GTEx", "FDR", "Cliff's δ, NAT vs GTEx", "FDR", "AUC, tumor vs GTEx",
     "DESeq2 log2FC, tumor vs GTEx", "DESeq2 log2FC, NAT vs GTEx", "padj, NAT vs GTEx"], rows,
    f"{int(A.n_tumour.iloc[0])} primary tumors (one per patient), {int(A.n_NAT.iloc[0])} NAT and {int(A.n_GTEx.iloc[0])} GTEx lung samples; "
    "Mann–Whitney tests with BH-FDR. Genome-wide DESeq2 (RSEM expected counts, design ~group; "
    f"{n['Tumour']} tumors, {n['NAT']} NAT, {n['GTEx']} GTEx; {SB['n_genes_compared']:,} genes): of the {SB['orig_Up_n']:,} up- and "
    f"{SB['orig_Down_n']:,} down-regulated DEGs of the original tumor-versus-NAT analysis, {100 * SB['orig_Up_same_direction_vs_GTEx']:.0f}% and "
    f"{100 * SB['orig_Down_same_direction_vs_GTEx']:.0f}% changed in the same direction versus GTEx lung "
    f"({100 * SB['orig_Up_also_DEG_Up_vs_GTEx']:.0f}% and {100 * SB['orig_Down_also_DEG_Down_vs_GTEx']:.0f}% met the DEG criteria); "
    f"all {SB['candidates_n']} candidate genes were higher in tumors than in GTEx lung (FDR < 0.05; {SB['candidates_DEG_up_vs_GTEx']} met the DEG "
    f"criteria) and {SB['candidates_DEG_up_NAT_vs_GTEx']} were up-regulated in NAT versus GTEx; Spearman ρ between the two sets of fold "
    f"changes {SB['spearman_log2FC_TvsNAT_vs_TvsGTEx']:.2f}.", italic_first=True, widths=[9, 14, 8, 7, 8, 7, 7, 9, 9, 8])

# ── S23 single cell ──
G = pd.read_csv(L / "single_cell" / "R13f_gene_by_celltype_tumour.csv")
CY = pd.read_csv(L / "single_cell" / "R13f_cycling_by_group.csv").set_index("group")
SS = pd.read_csv(L / "single_cell" / "R13f_summary.csv", index_col=0)["value"].astype(float)
groups = ["Malignant epithelial", "T/NK cells", "Myeloid cells", "B cells", "Mast cells", "Fibroblasts", "Endothelial cells",
          "Other epithelial", "Other"]
rows = []
for gr in groups:
    sub = G[(G.group == gr) & G.gene.isin(HUB7)].set_index("gene")
    rows.append([gr.replace("Other", "Unassigned") if gr == "Other" else gr, int(CY.loc[gr, "n"])] +
                [f"{sub.loc[g, 'share_of_tumour_transcripts']:.1f} ({sub.loc[g, 'pct_expressing']:.1f})" for g in HUB7] +
                [f"{CY.loc[gr, 'pct_cycling']:.1f}", f"{CY.loc[gr, 'pct_hub_transcripts_from_cycling']:.0f}"])
add("S23", "Table S23. Cellular source of hub-gene expression in primary LUAD (single-cell RNA-seq, GSE131907).",
    ["Cell group", "Cells, n"] + HUB7 + ["Cycling cells, %", "Hub transcripts from cycling cells, %"], rows,
    "Primary-tumor lung samples (tLung and tL/B; 15 samples). Values per gene: share of all hub-gene transcripts (TPM) contributed by the cell "
    "group (% of cells with detectable expression). Malignant epithelial cells as annotated by the original authors (inferCNV). Cycling: MKI67 "
    f"or TOP2A detected. In normal lung samples ({int(SS['pseudobulk_n_normal'])}), {SS['pct_cycling_normal']:.1f}% of {int(SS['n_normal_epi']):,} "
    f"normal epithelial cells were cycling versus {SS['pct_cycling_malignant']:.1f}% of {int(SS['n_malignant']):,} malignant cells; per-sample "
    f"(pseudo-bulk) scores were higher in tumor than in normal lung samples (Mann–Whitney p = {SS['pseudobulk_p']:.4f}).",
    widths=[13, 6] + [7] * 7 + [7, 9])

# ── S24 protein ──
P = pd.read_csv(L / "protein" / "R13h_protein_tumour_vs_NAT.csv").set_index("gene")
SP = pd.read_csv(L / "protein" / "R13h_summary.csv", index_col=0)["value"]
rows = []
for g in HUB7 + ["MKI67", "TOP2A"]:
    x = P.loc[g]
    if not x.protein_quantified:
        rows.append([g, "not quantified", "–", "–", "–", "–", f"{x.mRNA_median_paired_log2_diff:+.2f}"]); continue
    rows.append([g, f"{int(x.n_tumours_detected)} / {int(x.n_NAT_detected)}", int(x.n_pairs), f"{x.median_paired_log2_diff:+.2f}",
                 f"{x.pct_tumour_higher:.0f}", p3(x.FDR_paired), f"{x.mRNA_median_paired_log2_diff:+.2f}", ])
    rows[-1].insert(6, f(x.rho_mRNA_protein))
for r_ in rows:
    if len(r_) == 7: r_.insert(6, "–")
add("S24", "Table S24. Hub proteins in LUAD tumors and paired normal adjacent tissue (CPTAC-LUAD, TMT proteomics).",
    ["Gene", "Detected, tumors / NAT", "Pairs", "Median paired protein log2 difference", "Tumor > NAT, % of pairs", "FDR",
     "mRNA–protein ρ (tumors)", "Median paired mRNA log2 difference"], rows,
    "CPTAC-LUAD (Gillette et al., 2020): 110 tumors and 101 NAT (LinkedOmics gene-level TMT log2 ratios and RNA-seq). Wilcoxon signed-rank "
    "tests with BH-FDR across the quantified hub proteins. Protein score (mean z of the five quantified hub proteins, tumors with ≥ 3 detected): "
    f"higher in tumor than paired NAT in {float(SP['protein_score_pct_tumour_higher']):.0f}% of {int(float(SP['n_pairs']))} pairs "
    f"(p = {float(SP['protein_score_p_paired']):.1e}); Spearman ρ with the mRNA seven-gene score {float(SP['rho_protein_score_vs_mRNA_score']):.2f} "
    f"(n = {int(float(SP['n_mRNA_protein']))}), with MKI67 protein {float(SP['rho_protein_score_vs_MKI67_protein']):.2f}; higher in TP53-mutant "
    f"tumors (p = {float(SP['p_TP53']):.4f}); not associated with stage (ρ = {float(SP['rho_protein_score_vs_stage']):.2f}, "
    f"p = {float(SP['p_protein_score_vs_stage']):.2f}).", italic_first=True)

# ── S25 published signatures ──
PR = pd.read_csv(L / "published_signatures" / "R13c_signature_performance.csv")
PH = pd.read_csv(L / "published_signatures" / "R13c_head_to_head_vs_HUB7.csv")
PM = pd.read_csv(L / "published_signatures" / "R13c_meta_analysis.csv").set_index("signature")
DEF = {"HUB7_mean": ("Seven-gene score (this study)", "7", "unweighted mean z", "TCGA (discovery)"),
       "PAM50_prolif": ("PAM50 proliferation index (Nielsen et al., 2010)", "11", "unweighted mean z", "breast cancer"),
       "CCP31": ("Cell-cycle progression score (Wistuba et al., 2013)", "31", "unweighted mean z", "prostate cancer; qPCR"),
       "KRATZ11": ("Kratz et al. (2012), 14-gene assay", "11", "published direction (±1)", "NSCLC, qPCR (USA)"),
       "CHEN5": ("Chen et al. (2007)", "5", "ln HR", "NSCLC (Taiwan)"),
       "SHUKLA4": ("Shukla et al. (2017)", "4", "published direction (+1)", "TCGA-LUAD"),
       "ALDHERASI7": ("Al-Dherasi et al. (2021)", "7", "published Cox coefficients", "Fudan cohort; validated in TCGA"),
       "LI8": ("Li et al. (2020)", "8", "published LASSO-Cox coefficients", "TCGA-LUAD")}
rows = []
for k, (nm, ng, w, tr) in DEF.items():
    pr = PR[PR.signature == k].set_index("cohort")
    cvals = " / ".join("–" if c not in pr.index else f"{pr.loc[c, 'C_uni']:.2f}" for c in ["TCGA-LUAD", "GSE31210", "GSE68465", "GSE72094"])
    m = PM.loc[k] if k in PM.index else None
    mr = PM.loc[k + "_refit"] if k + "_refit" in PM.index else None
    h = PH[PH.signature == k]
    rows.append([nm, ng, w, tr, cvals, "–" if m is None else hr(m.HR, m.lo, m.hi), "–" if mr is None else hr(mr.HR, mr.lo, mr.hi),
                 "–" if h.empty else f"{h.r_spearman.min():.2f} to {h.r_spearman.max():.2f}",
                 "–" if h.empty else f"{(h.FDR_HUB7_added < 0.05).sum()}/{len(h)}", "–" if h.empty else f"{(h.FDR_sig_added < 0.05).sum()}/{len(h)}"])
add("S25", "Table S25. Comparison of the seven-gene score with published prognostic signatures in four cohorts (overall survival).",
    ["Signature", "Genes, n", "Weights", "Derived in", "C-index: TCGA / GSE31210 / GSE68465 / GSE72094", "Pooled HR per SD, as published (95% CI)*",
     "Pooled HR, re-weighted in TCGA (95% CI)†", "ρ with 7-gene score", "7-gene adds (FDR < 0.05)‡", "Signature adds (FDR < 0.05)‡"], rows,
    "Gene lists, directions and weights were taken from the original publications (Kratz: US patent 9,476,098 B2; Chen: per-gene HRs). Scores were "
    "computed from within-cohort z-scores of the genes measured on each platform (≥ 50% required; LINC00941 and WNT3A not measured on some arrays; "
    "Shukla not evaluable in GSE68465). *DerSimonian–Laird estimate, adjusted for age, sex and stage, excluding cohorts used by the original "
    "authors for training or validation (TCGA for Shukla, Al-Dherasi and Li; GSE31210 and GSE68465 for the CCP score). †Ridge-Cox weights for the signature genes estimated in TCGA and "
    "applied unchanged to the three GEO cohorts. ‡Number of cohorts in which the likelihood-ratio test for adding one score to a clinical model "
    "containing the other was significant (BH-FDR across all comparisons).", widths=[16, 4, 9, 11, 12, 11, 11, 7, 6, 6])

# ── S26 follow-up, treatment, ancestry ──
rows = []
for c, x in R9.iterrows():
    rows.append([c, x.region, f"{int(x.n)} ({int(x.events)})", f"{x.median_followup_y:.1f} / {x.max_followup_y:.1f}", f"{x.stage_I_pct:.0f}",
                 hr(x.HR_multi, x.lo_multi, x.hi_multi), f"{x.HR_0_5y:.2f}",
                 "–" if pd.isna(x.get("HR_after_5y")) else f"{hr(x.HR_after_5y, x.lo_after_5y, x.hi_after_5y)}; {int(x.events_landmark5)} deaths",
                 "–" if pd.isna(x.get("HR_no_adjuvant")) else f"{hr(x.HR_no_adjuvant, x.lo_no_adjuvant, x.hi_no_adjuvant)}; n = {int(x.n_no_adjuvant)}",
                 "–" if pd.isna(x.get("interaction_p")) else f"{x.interaction_HR:.2f} (p = {x.interaction_p:.2f})"])
rows.append(["Pooled analysis", "", "Cohorts, k", "", "", "Pooled HR per SD (95% CI)", "", "", "p", "I²"]); sub_i = len(rows) - 1
for k in ["all 9 cohorts, multivariable", "5 new cohorts, multivariable", "East Asian, multivariable", "Western, multivariable",
          "deaths within 5 years", "5-year landmark (after 5 years)", "no adjuvant therapy"] + [a for a in M9.index if a.startswith("leave-one-out")]:
    m = M9.loc[k]
    rows.append([(lambda t: t[0].upper() + t[1:])(k.replace(", multivariable", "").replace("leave-one-out: ", "Leave-one-out: ")),
                 "", f"{int(m.k)}", "", "", hr(m.HR, m.lo, m.hi), "", "", fp(m.p), f"{m.I2:.0f}%"])
add("S26", "Table S26. Locked seven-gene score by follow-up period, adjuvant treatment and region in nine cohorts (overall survival).",
    ["Cohort", "Region", "Patients (deaths)", "Median / maximum follow-up, years*", "Stage I, %", "Adjusted HR per SD (95% CI)",
     "HR, deaths ≤ 5 y", "HR after 5 y (5-year landmark)", "HR, no adjuvant therapy†", "Score × adjuvant chemotherapy HR (p)"], rows,
    "HR per SD adjusted for age, sex and stage (GSE68465 stratified by institution). *Reverse Kaplan–Meier median. Landmark analyses were run "
    "when ≥ 10 deaths occurred after 5 years. †GSE31210: complete resection and no adjuvant therapy; GSE68465, GSE37745 and GSE42127: patients "
    f"without adjuvant chemotherapy. In the cohorts with > 8 years of maximum follow-up pooled (eight cohorts, stratified by cohort; {int(float(pd.read_csv(X / 'R13g_time_varying.csv', index_col=0).loc['n', 'value'])):,} patients), "
    f"the proportional-hazards test against log(time) gave p = {float(pd.read_csv(X / 'R13g_time_varying.csv', index_col=0).loc['PH_test_p_logtime', 'value']):.1e}, "
    "consistent with an effect that is concentrated in the first years after surgery. Lower part: random-effects (DerSimonian–Laird) pooled "
    "estimates; p for the pooled HR.", widths=[11, 7, 8, 9, 5, 12, 6, 14, 14, 10], bold_rows=[sub_i])

T2 = {k: T[k] for k in sorted(T, key=lambda k: (k[0], int(k[1:])))}
json.dump(T2, open(OUT / "tables_v2.json", "w"), indent=0, ensure_ascii=False)
print({k: len(v["rows"]) for k, v in T2.items()})
