# A locked seven-gene mitotic prognostic score for lung adenocarcinoma

Code and processed results for the study:

> **Robust mitotic hub genes in lung adenocarcinoma form a locked prognostic score validated across nine cohorts and confirmed at the protein and single-cell level**
> Asaduzzaman, Chang-I Chen, Cheng-Jen Chang, Tzu-Sen Yang (Taipei Medical University). *Manuscript under review.*

The workflow is written entirely in Python. It starts from TCGA-LUAD RNA-seq data, identifies hub genes with signed WGCNA and a consensus centrality analysis across nine STRING networks, builds an unweighted seven-gene score (BUB1B, CCNB1, CDC20, AURKB, PLK1, CCNB2, CCNA2), and evaluates it with locked external validation, proliferation benchmarking and multi-omic context analyses.

## Main results

| Analysis | Result |
|---|---|
| Hub genes | 7 mitotic genes in the consensus top 10 in ≥ 6 of 9 STRING networks |
| TCGA-LUAD, adjusted overall survival | HR 1.37 per SD (95% CI 1.17–1.61) |
| Locked validation, 9 cohorts (2,106 patients) | pooled HR 1.37 (1.27–1.47), I² = 3% |
| Random gene sets (2 × 1,000) | score above the 97th–99th percentile |
| Published signatures | equivalent to PAM50 proliferation and CCP scores; better than three signatures that failed outside their training data |
| Healthy lung (GTEx), protein (CPTAC), single cell | over-expression confirmed; cycling malignant cells are the main source |

## Repository structure

```
TCGA-LUAD-7gene-signature/
├── data/                     small processed TCGA-LUAD tables (see data/README.md); large files are re-downloaded
├── results/                  differential expression and enrichment results used by the analysis scripts
├── scripts/                  original data download, QC, DESeq2 and enrichment scripts (steps 01–04, GEO download)
└── analysis/
    ├── hub_genes_final.csv   the locked seven-gene set
    ├── scripts/              all analysis, figure and table scripts (R01–R13)
    ├── 01_WGCNA_audit/       audit of the original unsigned WGCNA
    ├── 02_WGCNA_signed/      signed WGCNA (all samples and tumour-only networks)
    ├── 03_PPI_hub_selection/ nine STRING networks and consensus centrality
    ├── 04_clinical/          harmonised TCGA-CDR clinical data
    ├── 05_survival/          Cox models, PH tests, splines, stage-stratified models
    ├── 06_benchmark/         proliferation signatures and random gene sets
    ├── 07_external_validation/ GSE31210, GSE68465, GSE72094 and meta-analysis
    ├── 08_mutation_context/  driver mutations, interaction models, stage trend
    ├── 09_ML/                nested cross-validated classifiers and prognostic models
    ├── 10_immune/            purity-adjusted immune context
    ├── 11_extended/          TF activity, CNV, methylation, pan-cancer, ChEMBL and DepMap
    ├── 12_DE_revision/       LFC shrinkage and paired DESeq2 analysis
    ├── 13_limitations/       GTEx healthy lung, single cell, CPTAC protein, published signatures, 5 extra cohorts
    ├── Figures_final/        Main (Fig. 1–5) and Supplementary (Fig. S1–S17) figures, 600 dpi
    └── Manuscript_revised/   Supplementary Data S1–S4 (Excel) and all table contents (tables_v2.json)
```

Raw downloads (GEO series, DepMap, PanCanAtlas, UCSC Toil, CPTAC, single-cell matrices) are not stored in the repository because of their size; the download scripts below recreate them in `raw/` sub-folders.

## Installation

Python 3.10–3.12 is required.

```bash
git clone https://github.com/Akashctg062/TCGA-LUAD-RNA-seq-Analysis-Project.git
cd TCGA-LUAD-RNA-seq-Analysis-Project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is optional: `R02_wgcna_signed.py` uses PyTorch for the correlation and TOM matrices when it is available and falls back to NumPy otherwise.

## Reproducing the analysis

Run the scripts from the repository root, in this order. Scripts named `*download*` need internet access.

| Step | Script(s) | What it does |
|---|---|---|
| 1 | `scripts/01_data_download.py`, `02_preprocessing_qc.py` | TCGA-LUAD STAR-Counts from GDC, QC, filtered count matrix |
| 2 | `scripts/03_differential_expression.py`, `04_enrichment_analysis.py` | PyDESeq2 and enrichment |
| 3 | `scripts/09a_geo_validation.py` | downloads GSE31210, GSE68465, GSE72094 (only the download part is used) |
| 4 | `analysis/scripts/R01`, `R02`, `R02b` | WGCNA audit and signed WGCNA |
| 5 | `R03`, `R03b` | STRING networks, consensus centrality, locked hub genes |
| 6 | `R04a`, `R04b` | TCGA-CDR clinical data |
| 7 | `R05`, `R05b`, `R05c` | survival analysis |
| 8 | `R06a`, `R06` | proliferation benchmarking |
| 9 | `R07a`, `R07b` | external validation and meta-analysis |
| 10 | `R08a`, `R08b` | mutation context |
| 11 | `R09` | machine learning (nested cross-validation) |
| 12 | `R10a`, `R10b` | immune context |
| 13 | `R11a`–`R11f` | TF activity, CNV, methylation, pan-cancer, drugs, DepMap |
| 14 | `R12a` | DESeq2 shrinkage and paired analysis |
| 15 | `R13d`, `R13e` | downloads for the limitation analyses |
| 16 | `R13a`–`R13c`, `R13f`–`R13h` | GTEx, published signatures, single cell, extra cohorts, CPTAC |
| 17 | figure and table scripts | see the table below |

DepMap Public 26Q1 files (`CRISPRGeneEffect.csv`, `Model.csv`, `CRISPRInferredCommonEssentials.csv`) must be downloaded manually from the [DepMap portal](https://depmap.org/portal/data_page/) into `analysis/11_extended/raw/` with the prefix `DepMap_`.

## Figures and tables

| Output | Script |
|---|---|
| Fig. 1 (workflow) | `R13_fig1_workflow_bw.py` |
| Fig. 2 (discovery) | `R12_fig2_discovery.py` |
| Fig. 3 (prognosis and benchmarking) | `R12_fig3_prognosis_benchmark.py` |
| Fig. 4 (external validation) | `R13_fig4_external.py` |
| Fig. 5 (biology, pan-cancer, druggability) | `R12_fig5_biology.py` |
| Fig. S1–S12 | `R12_supp_figures_v2.py` (uses `R12_supp_figures.py`, `R12_supp_helpers.py`) |
| Fig. S13–S17 | `R13_supp_figures.py` |
| Tables 1–4, S1–S26, Supplementary Data S1–S4 | `R12e_tables_json.py`, then `R13_tables_v2.py` |

Figures are written as 600-dpi PNG and TIFF by default. Set `FIG_LOWERCASE=1` to write 600-dpi JPG files with lowercase panel letters (`analysis/Figures_final/JPG_600dpi_lowercase/`):

```bash
FIG_LOWERCASE=1 python analysis/scripts/R12_fig2_discovery.py        # Windows: set FIG_LOWERCASE=1
```

All analyses use random seed 42.

## Data sources

All data are public and de-identified: TCGA-LUAD (GDC), TCGA-CDR and PanCanAtlas, GEO (GSE31210, GSE68465, GSE72094, GSE50081, GSE30219, GSE37745, GSE42127, GSE13213, GSE131907), UCSC Xena and the Toil TCGA/GTEx recompute, CPTAC-LUAD via LinkedOmics, cBioPortal (MC3 mutations), STRING v12, MSigDB, TRRUST, CollecTRI (OmniPath), ChEMBL and DepMap.

## Citation

If you use this code, please cite the article (citation details will be added on publication) and this repository (see `CITATION.cff`).

## License

Code is released under the MIT License (see `LICENSE`). Data obtained from the resources above remain subject to their original terms of use.

## Contact

Asaduzzaman (first author, d825114001@tmu.edu.tw) and Prof. Tzu-Sen Yang (corresponding author, tsyang@tmu.edu.tw), Taipei Medical University, Taiwan.
