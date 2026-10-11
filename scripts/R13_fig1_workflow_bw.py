"""R13_fig1_workflow_bw.py — Main Figure 1 in the black-and-white layout used in the final manuscript
(white boxes, black borders, no step numbers), redrawn as a 600-dpi vector-based figure.
Run with FIG_LOWERCASE=1 to write Figures_final/JPG_600dpi_lowercase/Main/Figure1_workflow.jpg."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from R12_figstyle import FIG_MAIN, W2, save

fig = plt.figure(figsize=(W2, 4.85))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 141.8); ax.set_ylim(0, 96); ax.axis("off")
LW = 1.5

def box(x, y, w, h, title, body, tsize=8.4, bsize=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec="black", lw=LW))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tsize, fontweight="bold")
    ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=bsize, linespacing=1.45)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=LW, color="black",
                                 shrinkA=0, shrinkB=0))

# left column (discovery)
LX, LW_ = 1.4, 67.6
left = [(82.4, 13.0, "Discovery cohort: TCGA-LUAD",
         "STAR-Counts RNA-seq, 576 samples (517 tumor, 59 normal)\nTCGA-CDR curated clinical data and endpoints (OS, PFI, DSS)"),
        (65.6, 13.0, "Differential expression (PyDESeq2)", "8,301 DEGs (5,797 up, 2,504 down)\n|log2FC| > 1, BH-adjusted p < 0.05"),
        (49.6, 13.0, "Signed WGCNA (soft-threshold power 15)", "9 modules; key module ME2 (834 genes)\nr = 0.64 with tumor status"),
        (33.6, 13.0, "Candidate genes", "ME2 membership > 0.8, gene significance > 0.5\nand up-regulated DEGs: 86 genes"),
        (17.6, 13.0, "STRING v12 network centrality",
         "3 evidence types × 3 confidence cut-offs (9 networks)\nconsensus of 5 centralities; top-10 in ≥ 6/9 networks"),
        (1.0, 13.0, "Seven hub genes and locked score",
         "BUB1B, CCNB1, CDC20, AURKB, PLK1, CCNB2, CCNA2\n7-gene score = mean z-score (no fitted weights)")]
for y, h, t, b in left:
    box(LX, y, LW_, h, t, b)
for (y_top, _, _, _), (y_nxt, h_nxt, _, _) in zip(left[:-1], left[1:]):
    arrow(LX + LW_ / 2, y_top, LX + LW_ / 2, y_nxt + h_nxt)

# right column (evaluation)
RX, RW = 76.8, 63.0
box(74.2, 82.4, 66.2, 13.0, "Eight independent validation cohorts (GEO)",
    "GSE31210, GSE68465, GSE72094 (original); GSE50081,\nGSE30219, GSE37745, GSE42127, GSE13213 (additional)")
right = [(64.4, 12.6, RX, 57.6, "Prognosis (TCGA)", "OS, PFI, DSS; Cox HR per SD, age/sex/stage/smoking\nproportional hazards, splines, stage-stratified"),
         (49.4, 12.6, RX, 57.6, "Benchmark against proliferation", "PAM50 + 6 published LUAD signatures (incl. CCP);\nMKI67, PCNA, G2M/E2F; 2 × 1,000 random sets"),
         (33.0, 13.2, RX, RW, "External validation", "locked score in 8 GEO cohorts (1,620 tumors with OS)\nmeta-analysis; follow-up, adjuvant therapy, ancestry"),
         (17.0, 12.6, RX, RW, "Biological context", "GTEx healthy lung, CPTAC protein, single-cell source;\nmutations, TF activity, CNV, methylation, immune"),
         (1.0, 12.6, RX, RW, "Pan-cancer and druggability", "33 TCGA cancer types + GTEx normal tissue\nChEMBL drug mechanisms, DepMap 26Q1 dependency")]
for y, h, x, w, t, b in right:
    box(x, y, w, h, t, b)
# bus from hub-gene box to every evaluation box
BX = 72.4
mid = [y + h / 2 for y, h, *_ in right]
ax.plot([LX + LW_, BX], [mid[-1], mid[-1]], color="black", lw=LW, solid_capstyle="butt")
ax.plot([BX, BX], [mid[-1], mid[0]], color="black", lw=LW, solid_capstyle="butt")
for m in mid:
    arrow(BX, m, RX - 0.1, m)
# GEO cohorts -> external validation (right margin)
GX = 137.4
arrow(GX, 82.4, GX, 33.0 + 13.2)
save(fig, FIG_MAIN / "Figure1_workflow")
print("saved")
