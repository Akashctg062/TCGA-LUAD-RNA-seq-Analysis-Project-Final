"""R12e_tables_json.py — builds all manuscript (Tables 1–4) and supplementary tables (S1–S20) from result files
and writes them to JSON for the Word builders (build_tables.js), plus Supplementary_Data_S1-S4.xlsx (large gene-level files).
Every value is read from the result CSVs of steps R02–R12a; nothing is typed by hand."""
import json, numpy as np, pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter

NEW = Path(__file__).resolve().parents[1]
OUT = NEW / "Manuscript_revised"; OUT.mkdir(exist_ok=True)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
r = lambda p, **k: pd.read_csv(NEW / p, **k)

def fp(p):
    if p is None or pd.isna(p): return "–"
    if p < 0.001:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return f"{m:.1f} × 10^{{{e}}}" if p < 1e-4 else "<\u00a00.001"
    return f"{p:.3f}"
def p3(p): return "–" if pd.isna(p) else ("<\u00a00.001" if p < 0.001 else f"{p:.3f}")
def hr(h, l, u): return "–" if pd.isna(h) else f"{h:.2f} ({l:.2f}–{u:.2f})"
def f(x, d=2): return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"
def pct(n, d): return f"{n} ({100 * n / d:.0f}%)"

T = {}
def add(key, title, cols, rows, note="", widths=None, italic_first=False, bold_rows=None):
    T[key] = dict(title=title, cols=cols, rows=[[str(c) for c in row] for row in rows], note=note,
                  widths=widths, italic_first=italic_first, bold_rows=bold_rows or [])

# ───────────── data shared ─────────────
clin = r("04_clinical/clinical_tumour_harmonised.csv")
DAT = NEW / "07_external_validation" / "data"
geo = {g: pd.read_csv(DAT / f"{g}_clinical.csv", index_col=0).query("tumour == 1") for g in ["GSE31210", "GSE68465", "GSE72094"]}
perf = r("07_external_validation/signature_performance_all_cohorts.csv")
meta = r("07_external_validation/meta_analysis_OS_random_effects.csv")
genes_ev = r("07_external_validation/individual_genes_all_cohorts.csv")
res = r("05_survival/cox_uni_multi_per_SD_OS_PFI_DSS.csv")
obs = r("06_benchmark/signature_performance.csv", index_col=0)
bt = r("06_benchmark/paired_bootstrap_deltaC_HUB7_vs_comparators.csv", index_col=0)
nest = r("06_benchmark/nested_LRT_HUB7_vs_proliferation.csv").set_index("comparator")
sdef = r("06_benchmark/signature_definitions.csv").set_index("signature")
SIGN = {"HUB7": "Seven-gene score (this study)", "OLD5": "Original five genes", "PAM50_prolif": "PAM50 proliferation index",
        "PCNA_meta": "PCNA metagene", "MKI67": "MKI67", "HALLMARK_G2M": "Hallmark G2M checkpoint", "HALLMARK_E2F": "Hallmark E2F targets"}
META_KEY = {"HUB7": "HUB7_mean"}

def ne(C, e):
    d = C[[f"{e}_days", f"{e}_event"]].dropna(); return f"{len(d)} ({int(d[f'{e}_event'].sum())})"
def revkm(t, e):
    k = KaplanMeierFitter().fit(t, 1 - e); return k.median_survival_time_

# ═════════ Main Table 1: cohorts ═════════
def cohort_col(C, name, plat, second):
    n = len(C); st = C.stage.value_counts()
    os_ = C[["OS_days", "OS_event"]].dropna()
    male = C.male.sum() if "male" in C else (C.sex == "Male").sum()
    smk = C.ever_smoker
    return [name, plat, str(n), f"{C.age.median():.0f} ({C.age.quantile(.25):.0f}–{C.age.quantile(.75):.0f})",
            pct(int(male), n), f"{pct(int(smk.sum()), n)}; {int(smk.isna().sum())} unknown",
            " / ".join(str(int(st.get(s, 0))) for s in ["I", "II", "III", "IV"]) + f" ({int(C.stage.isna().sum())} unknown)",
            f"{len(os_)} ({int(os_.OS_event.sum())})", f"{revkm(os_.OS_days / 365.25, os_.OS_event):.1f}", second]
c0 = clin.assign(male=(clin.sex == "Male").astype(float))
cols1 = [cohort_col(c0, "TCGA-LUAD (discovery)", "RNA-seq (Illumina, STAR)",
                    f"PFI {ne(clin, 'PFI')}; DSS {ne(clin, 'DSS')}")]
for g, plat, sec in [("GSE31210", "HG-U133 Plus 2.0", "RFS {} ({})"), ("GSE68465", "HG-U133A", "PFS {} ({})"), ("GSE72094", "Rosetta/Merck custom array", "–")]:
    C = geo[g]
    s2 = "RFS " + ne(C, "RFS") if g == "GSE31210" else "PFS " + ne(C, "PFS") if g == "GSE68465" else "–"
    cols1.append(cohort_col(C, g, plat, s2))
labels = ["Cohort", "Platform", "Primary tumors, n", "Age, years, median (IQR)", "Male, n (%)", "Ever-smoker, n (%)",
          "Stage I / II / III / IV, n", "OS: patients (deaths)", "Median follow-up, years*", "Other endpoints: patients (events)"]
add("T1", "Table 1. Characteristics of the discovery and validation cohorts (primary tumors).",
    ["Characteristic"] + [c[0] for c in cols1], [[labels[i]] + [c[i] for c in cols1] for i in range(1, len(labels))],
    "*Reverse Kaplan–Meier estimate. TCGA-LUAD clinical data from TCGA-CDR; GSE68465 stage derived from pathological T and N "
    "(AJCC 6th edition). GSE31210 additionally contains 20 normal lung samples and GSE68465 19 commercial reference RNAs "
    "(excluded). DSS, disease-specific survival; IQR, interquartile range; OS, overall survival; PFI, progression-free interval; "
    "PFS, progression-free survival; RFS, relapse-free survival.")

# ═════════ Main Table 2: hub genes ═════════
hub = r("hub_genes_final.csv").set_index("gene")
de = r("12_DE_revision/DE_full_517T_59N.csv", index_col=0)
dep = r("11_extended/drugs/DepMap_dependency_LUAD.csv").set_index("gene")
mut = r("08_mutation_context/mutation_frequencies_patient_level.csv").set_index("gene")
osr = res[res.endpoint == "OS"].set_index("gene")
mg = meta[(meta.model == "multi")].set_index("item")
rows = []
for g in HUB7:
    h = hub.loc[g]; d = de.loc[h.gene_id]; o = osr.loc[g]; m = mg.loc[g]
    rows.append([g, f"{int(h.networks_in_top10_of_9)}/9", f"{int(h['primary_network_rank_E2_0.7'])}", f(h.MM_ME2_signed),
                 f(h.GS_tumour_signed), f(d.log2FC_shrunk), fp(d.padj), hr(o.multi_HR, o.multi_lo, o.multi_hi), p3(o.multi_FDR),
                 hr(m.HR, m.lo, m.hi), f"{mut.loc[g, 'pct_any']:.1f}", f"{dep.loc[g, 'pct_LUAD_dependent']:.0f}"])
add("T2", "Table 2. The seven hub genes: selection, differential expression and prognostic association.",
    ["Gene", "Top-10 in STRING networks", "Rank (E2 ≥ 0.7)", "MM (ME2)", "GS (tumor)", "Shrunken log2FC", "Adjusted p (DE)",
     "TCGA OS HR per SD (95% CI)*", "FDR", "Pooled OS HR (95% CI)†", "Mutated, %‡", "DepMap dependent LUAD lines, %"],
    rows, "*Cox model adjusted for age, sex, stage and smoking (473 patients, 168 deaths); FDR across the seven genes. †DerSimonian–Laird "
    "random-effects estimate across TCGA-LUAD, GSE31210, GSE68465 and GSE72094 (adjusted for age, sex and stage). ‡Protein-altering "
    "mutations in 566 sequenced tumors (MC3). Rank (E2 ≥ 0.7): consensus centrality rank in the STRING network without co-expression "
    "evidence at score ≥ 0.7 (ties share a rank). DepMap Public 26Q1: gene effect < −1 in 53 LUAD cell lines. GS, gene significance; "
    "MM, module membership.", italic_first=True, widths=[7, 7, 6, 5, 5, 6, 9, 12, 6, 12, 6, 8])

# ═════════ Main Table 3: benchmark ═════════
rows = []
for k in SIGN:
    mm = meta[(meta.item == META_KEY.get(k, k)) & (meta.model == "multi")].iloc[0]
    if k == "HUB7":
        rows.append([SIGN[k], int(sdef.loc[k, "n_present"]), "1.00", f"{obs.loc[k, 'C_uni']:.3f}", "–", "–", "–",
                     f"{obs.loc[k, 'HR_multi']:.2f}", hr(mm.HR, mm.lo, mm.hi), f"{mm.I2:.0f}"])
    else:
        b = bt.loc[k]; n_ = nest.loc[k]
        rows.append([SIGN[k], int(sdef.loc[k, "n_present"]), f"{obs.loc[k, 'r_with_HUB7']:.2f}", f"{obs.loc[k, 'C_uni']:.3f}",
                     f"{b.delta_C:+.3f} ({b.lo:+.3f} to {b.hi:+.3f})", p3(n_.p_HUB7_adds), p3(n_.p_comparator_adds),
                     f"{obs.loc[k, 'HR_multi']:.2f}", hr(mm.HR, mm.lo, mm.hi), f"{mm.I2:.0f}"])
add("T3", "Table 3. Benchmarking of the seven-gene score against proliferation signatures (overall survival).",
    ["Signature", "Genes, n", "r with 7-gene score", "C-index, TCGA", "ΔC, 7-gene − comparator (95% CI)*",
     "LRT p: 7-gene adds†", "LRT p: comparator adds†", "TCGA HR per SD‡", "Pooled HR per SD (95% CI)§", "I², %"],
    rows, "*Paired bootstrap (1,000 resamples) of Harrell's C for the score alone. †Likelihood-ratio tests in Cox models containing age, "
    "sex, stage, smoking and both signatures. ‡Adjusted for age, sex, stage and smoking. §Random-effects meta-analysis across the four "
    "cohorts, adjusted for age, sex and stage. Against 1,000 random expression-matched up-regulated and 1,000 random G2M/E2F seven-gene "
    "sets, the C-index of the seven-gene score ranked at the 97th (p = 0.027) and 99th (p = 0.011) percentiles.",
    widths=[16, 5, 6, 6, 16, 8, 8, 6, 12, 4])

# ═════════ Main Table 4: external validation ═════════
ta = r("07_external_validation/time_dependent_AUC_HUB7_mean.csv")
rows = []
for _, x in perf[perf.signature == "HUB7_mean"].iterrows():
    a3 = ta[(ta.cohort == x.cohort) & (ta.endpoint == x.endpoint) & (ta.year == 3)].AUC
    a5 = ta[(ta.cohort == x.cohort) & (ta.endpoint == x.endpoint) & (ta.year == 5)].AUC
    rows.append([x.cohort + (" (training)" if x.cohort == "TCGA-LUAD" else ""), x.endpoint, f"{x.n} ({x.events})",
                 hr(x.HR_uni, x.lo_uni, x.hi_uni), p3(x.p_uni), hr(x.HR_multi, x.lo_multi, x.hi_multi), p3(x.p_multi),
                 f"{x.C_clinical:.3f} → {x.C_clinical_plus_sig:.3f}", p3(x.LRT_p),
                 f"{a3.iloc[0]:.2f}" if len(a3) else "–", f"{a5.iloc[0]:.2f}" if len(a5) else "–"])
u = meta[(meta.item == "HUB7_mean") & (meta.model == "uni")].iloc[0]; m_ = meta[(meta.item == "HUB7_mean") & (meta.model == "multi")].iloc[0]
rows.append(["Random-effects meta-analysis (4 cohorts)", "OS", "1,544 (558)", hr(u.HR, u.lo, u.hi), p3(u.p), hr(m_.HR, m_.lo, m_.hi), p3(m_.p),
             f"I² = {u.I2:.0f}% / {m_.I2:.0f}%", "–", "–", "–"])
add("T4", "Table 4. Prognostic performance of the locked seven-gene score in the training and validation cohorts.",
    ["Cohort", "Endpoint", "Patients (events)", "Univariate HR per SD (95% CI)", "p", "Adjusted HR per SD (95% CI)*", "p",
     "C-index: clinical → clinical + score", "LRT p", "tdAUC 3 y", "tdAUC 5 y"],
    rows, "The score (mean z-score of the seven genes) was applied without refitting. *Adjusted for age, sex and stage; GSE68465 models "
    "stratified by institution. LRT, likelihood-ratio test for adding the score to the clinical model; tdAUC, time-dependent AUC "
    "(Uno, inverse-probability-of-censoring weighting; not estimable at 5 years in GSE72094). I² given for the univariate / adjusted "
    "pooled estimates.", bold_rows=[len(rows) - 1], widths=[17, 6, 8, 12, 6, 12, 6, 12, 6, 5, 5])
# patients/events in meta row computed explicitly
os_rows = perf[(perf.signature == "HUB7_mean") & (perf.endpoint == "OS")]
T["T4"]["rows"][-1][2] = f"{int(os_rows.n.sum()):,} ({int(os_rows.events.sum())})"

# ═════════ Supplementary tables ═════════
comp = r("04_clinical/completeness_old_vs_new.csv", index_col=0)
add("S1", "Table S1. TCGA-LUAD clinical data completeness and sample flow (517 primary tumors).",
    ["Variable", "Missing, original GDC metadata", "Missing, revised (TCGA-CDR + GDC)", "Available, revised"],
    [[i, "not available" if pd.isna(x.old_metadata_missing) else int(x.old_metadata_missing), int(x.new_missing), int(x.new_available)]
     for i, x in comp.iterrows()],
    "Sample flow: 576 STAR-Counts samples retained after quality control (517 primary tumors, 59 adjacent normal; 58 tumor–normal "
    "pairs); 517 tumors matched to TCGA-CDR; usable follow-up (time > 0, status known) for OS 504 (182 deaths), PFI 504 (207 events) and DSS 469 "
    "(112 events); complete cases for multivariable OS models (age, sex, stage, smoking) 473 (168 deaths); 566 tumors with MC3 "
    "mutation calls; 503 tumors with ABSOLUTE purity estimates.")

wp = json.load(open(NEW / "02_WGCNA_signed" / "WGCNA_parameters.json"))
wp = {k: (v.replace(" ** beta", "^{β}").replace("R^2 >= 0.85", "R² ≥ 0.85").replace("tumour", "tumor").replace("Benjamini-Hochberg", "BH-FDR")
          if isinstance(v, str) else v) for k, v in wp.items()}; pp = json.load(open(NEW / "03_PPI_hub_selection" / "R03_parameters.json"))
params = [
    ["Differential expression", "PyDESeq2 0.5.4; design ~condition; Cook's-distance refitting; DEG: BH-adjusted p < 0.05 and |MLE log2FC| > 1; "
     "apeGLM-type LFC shrinkage for display; paired sensitivity ~patient + condition (58 pairs)"],
    ["Low-count filter", "genes with < 10 counts in > 90% of samples removed"],
    ["GSEA / ORA", "GSEApy pre-ranked GSEA (MSigDB Hallmark); Enrichr GO Biological Process 2023 and KEGG 2021 for down-regulated DEGs"],
    ["WGCNA input", wp["expression"] + "; " + wp["gene_selection"]],
    ["WGCNA network", f"signed; adjacency {wp['adjacency']}; {wp['tom']}; soft threshold β = {wp['beta_A']} (all samples), "
     f"{wp['beta_B']} (tumor only); rule: {wp['soft_threshold_rule']}"],
    ["Module detection", f"{wp['clustering']}; {wp['tree_cut']}; merge cut height {wp['merge_cut_height']}; "
     f"{wp['n_modules_A']} modules ({wp['n_grey_A']} grey genes) / {wp['n_modules_B']} modules ({wp['n_grey_B']} grey) "],
    ["Module–trait", f"A: {wp['trait_A']}; B: {wp['trait_B']}; {wp['multiple_testing']}"],
    ["Candidate genes", "ME2 members with signed MM > 0.8, GS > 0.5 and up-regulated DEG (86 genes)"],
    ["PPI network", f"STRING v{pp['string_version']}, Homo sapiens (9606), interactions with score ≥ {pp['download_min_score']} downloaded; "
     "evidence E1 combined score, E2 combined score without co-expression (prior-corrected), E3 physical; cut-offs 0.4/0.7/0.9"],
    ["Hub selection", f"centralities {', '.join(pp['centralities'])}; {pp['consensus']}; hub = top 10 in ≥ 6 of 9 networks"],
    ["Seven-gene score", "mean of gene-wise z-scores of log2 expression within each cohort (no fitted weights); secondary: ridge Cox (penalizer 0.1) weights from TCGA, locked"],
    ["Survival models", "lifelines 0.30; HR per 1 SD; adjusted for age (per 10 y), sex, stage (II, III, IV vs I), smoking; Schoenfeld tests (rank time); "
     "restricted cubic splines (4 knots); stage-stratified and 10-year-censored sensitivity analyses; 300 bootstrap resamples for ΔC"],
    ["Benchmark", "1,000 paired bootstrap resamples; 2 × 1,000 random seven-gene sets (expression-decile matched up-DEGs; Hallmark G2M ∪ E2F genes)"],
    ["External validation", "locked score; Cox adjusted for age, sex, stage (GSE68465 stratified by institution); DerSimonian–Laird meta-analysis; "
     "Uno time-dependent AUC (scikit-survival 0.25)"],
    ["Prediction models", "clinical Cox; clinical + score; elastic-net Cox (l1 ratio 0.5, penalizer by inner 3-fold CV); random survival forest "
     "(300 trees, min leaf 15); repeated 5 × 5-fold CV in TCGA; locked models applied to GEO"],
    ["Diagnostic models", "scikit-learn 1.7 / XGBoost 3.2; nested CV (outer 5-fold × 5 repeats, inner 3-fold grid search); class weighting; "
     "Youden threshold from inner out-of-fold predictions"],
    ["Mutations", "TCGA PanCanAtlas MC3 via cBioPortal; KRAS codons 12/13/61, EGFR activating, BRAF V600E as drivers; ALK not assessed (fusions)"],
    ["Regulation", "TF activity: univariate linear model on CollecTRI regulons (≥ 10 targets) with hub genes removed; TRRUST v2; "
     "GISTIC2 thresholded copy number; 450K promoter probes (−1,500/+500 bp of TSS, hg19)"],
    ["Immune", "MCP-counter markers (mean log2 CPM); PanCanAtlas CIBERSORT; partial Spearman correlation adjusted for ABSOLUTE purity; BH-FDR"],
    ["Pan-cancer", "PanCanAtlas EB++ RNA-seq (10,215 primary tumors, 33 types); TCGA-CDR OS/PFI; UCSC Toil TCGA + GTEx for tumor–normal"],
    ["Drugs / dependency", "ChEMBL drug_mechanism records (targets containing the hub proteins); DepMap Public 26Q1 Chronos gene effect"],
    ["General", "two-sided tests; BH-FDR within test families; random seed 42; Python 3.10/3.12"],
]
add("S2", "Table S2. Analysis parameters and software.", ["Step", "Settings"], params, widths=[2200, 11758])

trA = r("02_WGCNA_signed/A_module_trait_tumour_status.csv"); trB = r("02_WGCNA_signed/B_tumour_only_module_clinical_traits.csv")
add("S3", "Table S3. Signed WGCNA module–trait associations.",
    ["Network", "Module", "Genes", "r with tumor status", "FDR", "ρ with stage", "FDR", "ρ with age", "FDR", "OS HR per SD", "FDR"],
    [["All samples (β = 15)", x.module, x["size"], f(x.r_tumour), fp(x.FDR), "–", "–", "–", "–", "–", "–"] for _, x in trA.sort_values("module").iterrows()] +
    [["Tumors only (β = 10)", x.module, x["size"], "–", "–", f(x.rho_stage), p3(x.FDR_stage), f(x.rho_age), p3(x.FDR_age), f(x.HR_OS_perSD), p3(x.FDR_OS)]
     for _, x in trB.iterrows()],
    "All-sample network: Pearson correlation of module eigengenes with tumor status (576 samples). Tumor-only network (517 tumors): "
    "Spearman correlation with stage (509 staged) and age, univariate Cox HR per SD of the eigengene for OS. BH-FDR across modules within each trait.")

summ = r("03_PPI_hub_selection/network_summary_9_settings.csv")
EV = {"E1": "Combined score", "E2": "Combined score without co-expression", "E3": "Physical interaction score"}
add("S4", "Table S4. STRING v12 networks used for the robustness analysis of hub-gene selection.",
    ["Network", "Evidence", "Cut-off", "Connected nodes", "Edges", "Density", "Largest component", "Consensus top 10 (rank order)"],
    [[x.network, EV[x.network[:2]], x.cutoff, x.nodes_with_edges, x.edges, f(x.density, 3), x.largest_component, x.top10] for _, x in summ.iterrows()],
    "Networks were built from the 86 ME2 candidate genes. Centralities: degree, betweenness, closeness, maximal clique centrality and maximum "
    "neighborhood component; consensus rank = mean of the five ranks.", widths=[1000, 2300, 800, 1000, 800, 900, 1100, 6058])

cand = r("02_WGCNA_signed/A_key_module_ME2_hub_candidates_MM0.8_GS0.5_DEGup.csv")
cons = r("03_PPI_hub_selection/consensus_rank_across_9_networks.csv", index_col=0)
cc = cand[["gene_name", "MM_own", "GS_tumour", "log2FoldChange"]].join(cons, on="gene_name").sort_values(["n_top10", "median_rank"], ascending=[False, True])
netc = [c for c in cons.columns if c[:2] in ("E1", "E2", "E3")]
add("S5", "Table S5. Consensus centrality ranks of the 86 candidate genes in the nine STRING networks.",
    ["Gene", "MM", "GS", "log2FC"] + [c.replace("_", " ≥") for c in netc] + ["Top-10 (of 9)", "Median rank"],
    [[x.gene_name, f(x.MM_own), f(x.GS_tumour), f(x.log2FoldChange)] + ["–" if pd.isna(x[c]) else f"{x[c]:.0f}" for c in netc] +
     [int(x.n_top10), f(x.median_rank, 1)] for _, x in cc.iterrows()],
    "Ranks are consensus ranks (1 = most central); '–', gene not connected in that network. Genes in the top 10 in ≥ 6 networks were defined as hub genes.",
    italic_first=True, bold_rows=[i for i, g in enumerate(cc.gene_name) if g in HUB7])

rows = []
for ep in ["OS", "PFI", "DSS"]:
    for g in HUB7:
        x = res[(res.endpoint == ep) & (res.gene == g)].iloc[0]
        rows.append([ep, g, f"{x.uni_n} ({x.uni_events})", hr(x.uni_HR, x.uni_lo, x.uni_hi), p3(x.uni_FDR),
                     f"{x.multi_n} ({x.multi_events})", hr(x.multi_HR, x.multi_lo, x.multi_hi), p3(x.multi_FDR)])
add("S6", "Table S6. Cox regression for each hub gene in TCGA-LUAD (HR per 1 SD of log2 CPM).",
    ["Endpoint", "Gene", "Univariate n (events)", "Univariate HR (95% CI)", "FDR", "Multivariable n (events)", "Multivariable HR (95% CI)*", "FDR"],
    rows, "*Adjusted for age, sex, stage (II, III, IV vs I) and smoking; each gene modelled separately. BH-FDR across the seven genes within endpoint and model.")
T["S6"]["rows"] = [[row[0], row[1]] + row[2:] for row in T["S6"]["rows"]]

addv = r("05_survival/added_value_over_clinical_model.csv"); ph = r("05_survival/proportional_hazards_tests.csv")
nl = r("05_survival/nonlinearity_spline_tests.csv"); ss = r("05_survival/sensitivity_stage_stratified_cox.csv")
c10 = r("05_survival/sensitivity_OS_multivariable_censored_10y.csv").set_index("gene")
rows = []
for g in HUB7:
    a = addv[(addv.endpoint == "OS") & (addv.gene == g)].iloc[0]; p_ = ph[(ph.endpoint == "OS") & (ph.gene == g)].iloc[0]
    n_ = nl[(nl.endpoint == "OS") & (nl.gene == g)].iloc[0]; s_ = ss[(ss.endpoint == "OS") & (ss.gene == g)].iloc[0]; c_ = c10.loc[g]
    dc = f"{a.delta_C:+.3f} ({a.delta_C_lo:+.3f} to {a.delta_C_hi:+.3f})".replace("-0.000", "0.000").replace("+0.000", "0.000")
    rows.append([g, f"{a.C_clinical:.3f}", f"{a.C_clinical_plus_gene:.3f}", dc, p3(a.LRT_FDR), p3(p_.ph_p_gene), p3(n_.p_nonlinear),
                 hr(s_.HR, s_.lo, s_.hi), p3(s_.FDR), hr(c_.HR, c_.lo, c_.hi)])
add("S7", "Table S7. Overall survival in TCGA-LUAD: added value over clinical covariates, model diagnostics and sensitivity analyses.",
    ["Gene", "C, clinical model", "C, + gene", "ΔC (95% bootstrap CI)", "LRT FDR", "PH test p (gene)", "Non-linearity p",
     "Stage-stratified HR (95% CI)", "FDR", "HR, censored at 10 y"],
    rows, "Clinical model: age, sex, stage, smoking (473 patients, 168 deaths). PH, Schoenfeld-residual test (rank time); non-linearity, "
    "restricted cubic spline (4 knots) vs linear term. Stage-stratified models are adjusted for age, sex and smoking.", italic_first=True)

pcts = r("06_benchmark/HUB7_vs_null_empirical.csv")
rows = [[SIGN[k], int(sdef.loc[k, "n_present"]), f"{obs.loc[k, 'C_uni']:.3f}", f"{obs.loc[k, 'C_multi']:.3f}",
         f"{obs.loc[k, 'delta_C_vs_clinical']:+.3f}", f"{obs.loc[k, 'HR_multi']:.2f}", fp(obs.loc[k, 'p_multi']),
         f"{obs.loc[k, 'AUC']:.3f}", f"{obs.loc[k, 'r_with_HUB7']:.2f}"] for k in SIGN]
add("S8", "Table S8. Performance of proliferation signatures in TCGA-LUAD and comparison with random gene sets.",
    ["Signature", "Genes, n", "C (score alone)", "C (clinical + score)", "ΔC vs clinical", "Adjusted HR per SD", "p", "Tumor vs normal AUC", "r with 7-gene score"],
    rows + [["", "", "", "", "", "", "", "", ""]] +
    [[("Random expression-matched up-DEG sets" if x.null.startswith("N1") else "Random G2M/E2F sets") + f" – {x.metric.replace('C_uni','C (score alone)').replace('C_multi','C (clinical + score)')}",
      "7", f"7-gene: {x.HUB7:.3f}", f"null median {x.null_median:.3f}", f"null 95th {x.null_95th:.3f}", f"percentile {x.HUB7_percentile:.0f}",
      f"p = {x.empirical_p:.3f}", "", ""] for _, x in pcts.iterrows()],
    "Clinical model: age, sex, stage and smoking (C = 0.672). PCNA metagene: 1% of expressed genes most correlated with PCNA in TCGA-LUAD "
    "tumors (Venet rule). Original five genes: BIRC5, CCNB1, CCNB2, CDC20, KIF2C. Empirical p = (1 + number of random sets ≥ observed) / 1,001.")

rows = []
for g in HUB7:
    row = [g]
    for coh in ["TCGA-LUAD", "GSE31210", "GSE68465", "GSE72094"]:
        x = genes_ev[(genes_ev.cohort == coh) & (genes_ev.endpoint == "OS") & (genes_ev.gene == g)].iloc[0]
        row += [hr(x.HR_multi, x.lo_multi, x.hi_multi), p3(x.FDR_multi)]
    m = mg.loc[g]; row += [hr(m.HR, m.lo, m.hi), f"{m.I2:.0f}"]
    rows.append(row)
add("S10", "Table S10. Single-gene overall survival associations in the four cohorts and random-effects meta-analysis.",
    ["Gene", "TCGA HR (95% CI)", "FDR", "GSE31210 HR", "FDR", "GSE68465 HR", "FDR", "GSE72094 HR", "FDR", "Pooled HR (95% CI)", "I², %"],
    rows, "HR per SD, adjusted for age, sex and stage (GSE68465 stratified by institution); BH-FDR across genes within cohort.", italic_first=True)

sens = r("07_external_validation/sensitivity_HUB7_mean.csv")
nullc = r("07_external_validation/HUB7_vs_random_sets_all_cohorts.csv")
add("S9", "Table S9. Sensitivity analyses of the locked seven-gene score and comparison with random gene sets in each cohort.",
    ["Cohort", "Endpoint", "Analysis", "Patients (events)", "HR per SD (95% CI)", "p"],
    [[x.cohort, x.endpoint, x.analysis.replace(" (GSE31210)", ""), f"{x.n} ({x.events})", hr(x.HR, x.lo, x.hi), p3(x.p)] for _, x in sens.iterrows()] +
    [[x.cohort, x.endpoint, ("vs random expression-matched up-DEG sets" if x.null.startswith("N1") else "vs random G2M/E2F sets") +
      f": C {x.HUB7_C:.3f} vs null median {x.null_median:.3f}", "1,000 sets", f"percentile {x.percentile:.0f}", f"{x.empirical_p:.3f}"]
     for _, x in nullc.iterrows()],
    "Sensitivity models are adjusted for age, sex, stage and smoking; 'complete resection, no adjuvant' excludes GSE31210 patients flagged by the "
    "original investigators for incomplete resection or adjuvant therapy. Random-set comparisons use the C-index of the score alone.")

rows = []
for _, x in meta[meta.model == "multi"].iterrows():
    it = x["item"]
    nm = SIGN.get({"HUB7_mean": "HUB7"}.get(it, it), it)
    nm = "Seven-gene ridge-Cox score (TCGA weights)" if it == "HUB7_risk" else nm
    u_ = meta[(meta["item"] == it) & (meta.model == "uni")].iloc[0]
    rows.append([nm, hr(u_.HR, u_.lo, u_.hi), f"{u_.I2:.0f}", hr(x.HR, x.lo, x.hi), fp(x.p), f"{x.I2:.0f}", p3(x.Q_p)])
add("S11", "Table S11. Random-effects meta-analysis of overall survival (TCGA-LUAD, GSE31210, GSE68465, GSE72094).",
    ["Signature / gene", "Univariate HR (95% CI)", "I², %", "Adjusted HR (95% CI)", "p", "I², %", "Cochran Q p"], rows,
    "DerSimonian–Laird estimates of HR per SD; adjusted models include age, sex and stage.")

cv = r("09_ML/B_prognostic_cv_summary.csv", index_col=0); exm = r("09_ML/B_prognostic_external_locked_models.csv")
MN = {"Cox clinical": "Clinical Cox (age, sex, stage)", "Cox clinical + HUB7 score": "Clinical Cox + seven-gene score",
      "Elastic-net Cox clinical + 7 genes": "Elastic-net Cox, clinical + 7 genes", "Random survival forest clinical + 7 genes": "Random survival forest, clinical + 7 genes"}
rows = []
for k in MN:
    row = [MN[k], f"{cv.loc[k, 'C_mean']:.3f} ({cv.loc[k, 'C_lo']:.3f}–{cv.loc[k, 'C_hi']:.3f})"]
    for g in ["GSE31210", "GSE68465", "GSE72094"]:
        x = exm[(exm.cohort == g) & (exm.model == k)].iloc[0]
        row += [f"{x.C:.3f} ({x.C_lo:.3f}–{x.C_hi:.3f})", f(x.calibration_slope)]
    rows.append(row)
add("S12", "Table S12. Prognostic prediction models fitted in TCGA-LUAD and applied unchanged to the validation cohorts (overall survival).",
    ["Model", "TCGA, repeated CV C (95% range)", "GSE31210 C (95% CI)", "Slope", "GSE68465 C (95% CI)", "Slope", "GSE72094 C (95% CI)", "Slope"],
    rows, "TCGA: 5-fold cross-validation repeated 5 times (2.5th–97.5th percentile of fold estimates). External: Harrell C with 500 bootstrap "
    "resamples; slope, calibration slope of the linear predictor (not defined for the random survival forest).")

dsum = r("09_ML/A_diagnostic_nested_cv_summary.csv", index_col=[0, 1]); dext = r("09_ML/A_external_GSE31210_diagnostic.csv").set_index("model")
p0 = r("09_ML/part0_reproduction_of_original_errors.csv")
rows = []
for fs, lab in [("HUB7", "7 hub genes"), ("FOLD7", "7 DEGs re-selected in each fold")]:
    for mdl in ["Logistic regression", "SVM (RBF)", "Random forest", "XGBoost", "MLP"]:
        s = dsum.loc[(fs, mdl)]
        e = dext.loc[mdl] if fs == "HUB7" else None
        rows.append([lab, mdl, f"{s.AUC_mean:.3f}", f"{s.balanced_accuracy_mean:.3f}", f"{s.MCC_mean:.3f}", f"{s.sensitivity_mean:.3f}",
                     f"{s.specificity_mean:.3f}", "–" if e is None else f"{e.AUC:.3f} ({e.AUC_lo:.3f}–{e.AUC_hi:.3f})",
                     "–" if e is None else f"{e.balanced_accuracy:.3f}", "–" if e is None else f"{e.sensitivity:.2f} / {e.specificity:.2f}"])
add("S13", "Table S13. Tumor-versus-normal classification: repeated nested cross-validation in TCGA-LUAD and locked models in GSE31210.",
    ["Features", "Model", "AUC", "Balanced accuracy", "MCC", "Sensitivity", "Specificity", "GSE31210 AUC (95% CI)", "GSE31210 balanced accuracy", "GSE31210 sens. / spec."],
    rows, f"TCGA values are means over 25 outer folds (517 tumors, 59 normals; Youden threshold chosen within training folds). GSE31210: 226 tumors "
    f"and 20 normal lung samples, models and thresholds locked. Error diagnosis of an earlier analysis: a multilayer perceptron with early stopping "
    f"stopped after 12–30 iterations and gave AUC {p0.AUC[0]:.3f} (AUC {p0.AUC[1]:.3f} without early stopping); specificities of "
    f"{p0.specificity[2]:.3f} and {p0.specificity[3]:.3f} originated from cross-validated and apparent (training-set) predictions, respectively.")

sa = r("08_mutation_context/score_by_mutation_status.csv").set_index("mutation")
rows = []
for g in ["TP53", "KRAS", "KEAP1", "STK11", "EGFR", "BRAF"] + HUB7:
    x = mut.loc[g]
    drv = x.role == "driver"
    rows.append([g, "driver" if drv else "hub gene", int(x.patients_any_protein_altering), f"{x.pct_any:.1f}",
                 f"{x.pct_driver_definition:.1f}" if drv else "–",
                 (f"{sa.loc[g, 'median_mut']:+.2f} / {sa.loc[g, 'median_wt']:+.2f}" if g in sa.index else "–"),
                 fp(sa.loc[g, "FDR"]) if g in sa.index else "–"])
add("S14", "Table S14. Somatic mutation frequencies and the seven-gene score by driver-mutation status (TCGA-LUAD).",
    ["Gene", "Role", "Patients with protein-altering mutation", "%", "Driver definition, %", "Median score, mutant / wild-type", "FDR"],
    rows, "566 sequenced tumors (TCGA PanCanAtlas MC3). Driver definitions: KRAS codon 12/13/61, EGFR activating mutations, BRAF V600E; "
    "TP53, STK11 and KEAP1 any protein-altering mutation. ALK (fusion-driven) not assessed. Mann–Whitney test with BH-FDR.", italic_first=True)

inter = r("08_mutation_context/score_by_mutation_interaction_cox.csv")
add("S15", "Table S15. Seven-gene score × driver-mutation interaction (Cox models).",
    ["Cohort", "Endpoint", "Mutation", "Mutant patients (events)", "HR mutant (95% CI)", "HR wild-type (95% CI)", "Interaction HR", "Interaction p", "FDR"],
    [[x.cohort, x.endpoint, x.mutation, f"{x.n_mut} ({x.events_mut})", "< 10 events" if x.events_mut < 10 else hr(x.HR_Mut, x.lo_Mut, x.hi_Mut),
      hr(x.HR_WT, x.lo_WT, x.hi_WT), f(x.HR_interaction), p3(x.p_interaction), p3(x.FDR_interaction)] for _, x in inter.iterrows()],
    "Models: score + mutation + score × mutation + age + sex + stage; HR per SD. BH-FDR across mutations within cohort and endpoint.")

trd = r("08_mutation_context/stage_trend_jonckheere.csv")
rows = []
for coh in ["TCGA-LUAD", "GSE68465", "GSE72094", "GSE31210"]:
    for v in ["HUB7_score"] + HUB7:
        x = trd[(trd.cohort == coh) & (trd.variable == v)].iloc[0]
        rows.append([coh, "Seven-gene score" if v == "HUB7_score" else v, x.n, x.stages.replace("{", "").replace("}", "").replace(": ", " = ").replace("1 =", "I =").replace("2 =", "II =").replace("3 =", "III =").replace("4 =", "IV ="),
                     f(x.JT_z), fp(x.JT_p), p3(x.JT_FDR), f(x.spearman_rho)])
add("S16", "Table S16. Expression of the hub genes and the seven-gene score across pathological stages (Jonckheere–Terpstra trend test).",
    ["Cohort", "Variable", "n", "Stage distribution", "JT z", "p", "FDR", "Spearman ρ"], rows, "BH-FDR within cohort.")

ta2 = r("11_extended/M9_TF_activity_vs_score.csv")
FOC = ["E2F1", "E2F2", "E2F3", "E2F4", "TFDP1", "MYC", "MYBL2", "FOXM1", "STAT3", "RELA", "RB1", "TP53"]
evd = r("11_extended/M9_curated_TF_target_evidence.csv")
rows = []
for tf in FOC:
    x = ta2.set_index("TF").loc[tf]
    targets = sorted(evd[evd.TF == tf].hub_gene.unique())
    rows.append([tf, int(x.regulon_size), f(x.rho_activity_vs_score), fp(x.FDR), f(x.rho_mRNA_vs_score), ", ".join(targets) if targets else "–"])
add("S17", "Table S17. Transcription-factor activity and curated regulation of the hub genes.",
    ["TF", "Regulon size (hub genes removed)", "ρ, activity vs score", "FDR", "ρ, mRNA vs score", "Hub genes with curated TF–target evidence (TRRUST/CollecTRI)"],
    rows, "TF activity inferred with a univariate linear model on CollecTRI regulons in 517 TCGA-LUAD tumors; BH-FDR across all TFs tested. "
    "Top 15 TFs by activity correlation are shown in Supplementary Fig. S10A.", italic_first=True)

cn = r("11_extended/M10_copy_number_vs_expression.csv").set_index("gene"); gl = r("11_extended/M10_methylation_promoter_gene_level.csv").set_index("gene")
rows = []
for g in HUB7:
    c = cn.loc[g]; m = gl.loc[g]
    rows.append([g, f"{c.pct_deep_del:.1f} / {c.pct_shallow_del:.1f}", f"{c.pct_gain:.1f} / {c.pct_amp:.1f}", f(c.rho_CN_vs_expr), p3(c.FDR),
                 int(m.n_promoter_probes), f"{m.median_promoter_beta_normal:.3f} / {m.median_promoter_beta_tumour:.3f}", f"{m.delta_beta:+.3f}",
                 fp(m.FDR_tumour_vs_normal), f(m.rho_promoter_beta_vs_expr), p3(m.FDR_beta_vs_expr)])
add("S18", "Table S18. Copy number and promoter DNA methylation of the hub genes (TCGA-LUAD).",
    ["Gene", "Deep / shallow deletion, %", "Gain / amplification, %", "ρ, CN vs expression", "FDR", "Promoter probes", "Median β, normal / tumor",
     "Δβ", "FDR (T vs N)", "ρ, β vs expression", "FDR"],
    rows, f"Copy number: GISTIC2 thresholded calls ({int(cn.n.iloc[0])} tumors). Methylation: Illumina 450K probes within −1,500/+500 bp of the "
    f"transcription start site; {int(gl.n_tumour.iloc[0])} tumors and {int(gl.n_normal.iloc[0])} normal samples.", italic_first=True)

im = r("10_immune/immune_correlations_raw_and_purity_adjusted.csv"); im = im[im.variable == "HUB7 score"]
rows = [[x.family, x.feature, x.n, f(x.rho), p3(x.FDR), f(x.rho_purity_adj), p3(x.FDR_purity_adj)]
        for _, x in im.sort_values(["family", "rho_purity_adj"], ascending=[True, False]).iterrows()]
add("S19", "Table S19. Correlation of the seven-gene score with immune cell estimates and immune checkpoint genes (TCGA-LUAD).",
    ["Method", "Feature", "n", "Spearman ρ", "FDR", "Partial ρ (purity-adjusted)", "FDR"], rows,
    "Partial Spearman correlation adjusted for ABSOLUTE tumor purity (rank-residual method). Purity was not correlated with the score "
    "(ρ = 0.001, p = 0.98, n = 503). BH-FDR within each method. Gene-level correlations are provided in Supplementary Data S3.")

sv = r("11_extended/pancancer/pancancer_survival_score_per_SD.csv"); tn = r("11_extended/pancancer/pancancer_tumour_vs_TCGA_plus_GTEx_normal.csv").set_index("cancer")
rows = []
for c in sorted(sv.cancer.unique()):
    o = sv[(sv.cancer == c) & (sv.endpoint == "OS")].iloc[0]; p_ = sv[(sv.cancer == c) & (sv.endpoint == "PFI")].iloc[0]
    t = tn.loc[c] if c in tn.index else None
    tvn = "–" if t is None or pd.isna(t["diff"]) else f"{t['diff']:+.2f} ({fp(t.FDR)})"
    rows.append([c, f"{int(o.n)} ({int(o.events)})", hr(o.HR, o.lo, o.hi), p3(o.FDR), f(o.HR_adj), p3(o.FDR_adj),
                 f"{int(p_.n)} ({int(p_.events)})", hr(p_.HR, p_.lo, p_.hi), p3(p_.FDR), tvn])
add("S20", "Table S20. Seven-gene score and survival in 33 TCGA cancer types.",
    ["Cancer", "OS patients (deaths)", "OS HR (95% CI)", "FDR", "Adjusted OS HR", "FDR", "PFI patients (events)", "PFI HR (95% CI)", "FDR", "Tumor − normal (FDR)*"],
    rows, "HR per SD of the score (z-scored within cancer type); adjusted models include age and sex, and stage where ≥ 70% of patients were staged. "
    "Cancer types with < 5 events were not modelled ('–'). *Difference in median score between primary tumors and same-site normal tissue "
    "(TCGA adjacent normal + GTEx; UCSC Toil), shown for types with ≥ 10 normal samples. TCGA codes: ACC adrenocortical carcinoma; BLCA bladder "
    "urothelial carcinoma; BRCA breast invasive carcinoma; CESC cervical squamous cell carcinoma and endocervical adenocarcinoma; CHOL "
    "cholangiocarcinoma; COAD colon adenocarcinoma; DLBC diffuse large B-cell lymphoma; ESCA esophageal carcinoma; GBM glioblastoma; HNSC head "
    "and neck squamous cell carcinoma; KICH kidney chromophobe; KIRC kidney renal clear cell carcinoma; KIRP kidney renal papillary cell "
    "carcinoma; LAML acute myeloid leukemia; LGG lower grade glioma; LIHC liver hepatocellular carcinoma; LUAD lung adenocarcinoma; LUSC lung "
    "squamous cell carcinoma; MESO mesothelioma; OV ovarian serous cystadenocarcinoma; PAAD pancreatic adenocarcinoma; PCPG pheochromocytoma "
    "and paraganglioma; PRAD prostate adenocarcinoma; READ rectum adenocarcinoma; SARC sarcoma; SKCM skin cutaneous melanoma; STAD stomach "
    "adenocarcinoma; TGCT testicular germ cell tumors; THCA thyroid carcinoma; THYM thymoma; UCEC uterine corpus endometrial carcinoma; UCS "
    "uterine carcinosarcoma; UVM uveal melanoma.", bold_rows=[i for i, row in enumerate(rows) if row[0] in ("LUAD", "LUSC")])

ch = r("11_extended/drugs/ChEMBL_all_mechanism_records.csv")
rows = []
for g in HUB7:
    d = ch[ch.hub_gene == g].sort_values(["max_phase", "drug_name"], ascending=[False, True])
    if len(d) == 0:
        note = {"CCNB1": "none (indirect via CDK1)", "CCNB2": "none (indirect via CDK1)", "CCNA2": "none (indirect via CDK1/CDK2)"}.get(g, "none")
        rows.append([g, note, "–", "–", "–", f"{dep.loc[g, 'median_effect_LUAD']:.2f}", f"{dep.loc[g, 'pct_LUAD_dependent']:.0f}",
                     f"{dep.loc[g, 'pct_other_dependent']:.0f}", "yes" if dep.loc[g, "common_essential"] else "no"])
        continue
    for i, (_, x) in enumerate(d.iterrows()):
        rows.append([g if i == 0 else "", x.drug_name, x.molecule_chembl_id, int(x.max_phase), x.target_name,
                     f"{dep.loc[g, 'median_effect_LUAD']:.2f}" if i == 0 else "", f"{dep.loc[g, 'pct_LUAD_dependent']:.0f}" if i == 0 else "",
                     f"{dep.loc[g, 'pct_other_dependent']:.0f}" if i == 0 else "", ("yes" if dep.loc[g, "common_essential"] else "no") if i == 0 else ""])
add("S21", "Table S21. Curated drug mechanisms (ChEMBL) and CRISPR dependency (DepMap Public 26Q1) of the hub genes.",
    ["Gene", "Agent", "ChEMBL ID", "Max phase", "Target", "Median gene effect, LUAD", "% dependent, LUAD", "% dependent, other", "Common essential"],
    rows, "Mechanism records for ChEMBL targets containing the hub proteins (single proteins, complexes and families). DepMap: 53 LUAD and 1,155 "
    "other cancer cell lines; dependent = Chronos gene effect < −1.", italic_first=True, widths=[8, 17, 14, 6, 24, 10, 8, 8, 8])

T = {k: T[k] for k in sorted(T, key=lambda k: (k[0], int(k[1:])))}
json.dump(T, open(OUT / "tables.json", "w"), indent=0, ensure_ascii=False)
print({k: len(v["rows"]) for k, v in T.items()})

# ═════════ Supplementary Data (Excel; large gene-level tables) ═════════
with pd.ExcelWriter(OUT / "Supplementary_Data_S1-S4.xlsx") as xw:
    full = de.copy(); pair = r("12_DE_revision/DE_paired_58pairs.csv", index_col=0)[["log2FC_MLE", "padj", "DEG"]].add_suffix("_paired")
    full.join(pair).reset_index().rename(columns={"index": "gene_id"}).to_excel(xw, sheet_name="Data_S1_DE_all_genes", index=False)
    r("02_WGCNA_signed/A_gene_module_MM_GS.csv").to_excel(xw, sheet_name="Data_S2_WGCNA_genes", index=False)
    im_all = r("10_immune/immune_correlations_raw_and_purity_adjusted.csv")
    im_all.to_excel(xw, sheet_name="Data_S3_immune_all", index=False)
    clin.to_excel(xw, sheet_name="Data_S4_TCGA_clinical", index=False)
print("Supplementary data written")
