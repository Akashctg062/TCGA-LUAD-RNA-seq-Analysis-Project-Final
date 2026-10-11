"""
R12_figstyle.py — shared publication style for all revised figures.
Double-column width 180 mm (7.09 in); fonts 6.5-8 pt; panel letters 11 pt bold; export 600 dpi PNG + LZW TIFF.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
FIG_MAIN = NEW / "Figures_final" / "Main"
FIG_SUPP = NEW / "Figures_final" / "Supplementary"
for d in (FIG_MAIN, FIG_SUPP):
    d.mkdir(parents=True, exist_ok=True)
# FIG_LOWERCASE=1: lowercase panel letters and 600-dpi JPG output in a separate folder (originals untouched)
LOWER = os.environ.get("FIG_LOWERCASE") == "1"
JPG_DIR = NEW / "Figures_final" / "JPG_600dpi_lowercase"

W2 = 7.09        # double-column width (in)
RED, BLUE, GREY, DARK = "#c0392b", "#2c7bb6", "#7f8c8d", "#2c3e50"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

def panel(ax, letter, x=-0.14, y=1.06):
    if LOWER: letter = letter.lower()
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")

def save(fig, path_stem):
    """Save 600-dpi PNG and LZW-compressed TIFF (or, with FIG_LOWERCASE=1, a 600-dpi JPG in JPG_DIR)."""
    if LOWER:
        from pathlib import Path as _P
        stem = _P(str(path_stem)); sub = JPG_DIR / stem.parent.name; sub.mkdir(parents=True, exist_ok=True)
        fig.savefig(sub / f"{stem.name}.jpg", dpi=600, bbox_inches="tight", pad_inches=0.04,
                    facecolor="white", pil_kwargs={"quality": 95, "optimize": True})
        plt.close(fig); return
    fig.savefig(f"{path_stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{path_stem}.tif", dpi=600, bbox_inches="tight", pad_inches=0.04,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

def qfmt(q):
    return "FDR < 0.001" if q < 0.001 else f"FDR = {q:.3f}"
