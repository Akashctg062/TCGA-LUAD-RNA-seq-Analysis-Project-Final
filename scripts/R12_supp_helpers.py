"""Helpers for supplementary figures: data loaders and a matplotlib table renderer."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from R12_figstyle import NEW, RED, BLUE, GREY

ROOT = NEW.parent
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()

def table(ax, df, title=None, fontsize=5.6, col_widths=None, row_height=1.35, header_color="#e5e8e8", wrap_title=True,
          bold_first_col=False, highlight_rows=None):
    """Render a DataFrame as a clean table filling the axis."""
    ax.axis("off")
    cell = [[("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in row] for row in df.values]
    t = ax.table(cellText=cell, colLabels=list(df.columns), loc="upper center", cellLoc="center",
                 colLoc="center", colWidths=col_widths)
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, row_height)
    for (r, c), cl in t.get_celld().items():
        cl.set_linewidth(0.3); cl.set_edgecolor("#bdc3c7")
        if r == 0:
            cl.set_facecolor(header_color); cl.set_text_props(fontweight="bold")
        elif highlight_rows and (r - 1) in highlight_rows:
            cl.set_facecolor("#fdedec")
        if bold_first_col and c == 0 and r > 0:
            cl.set_text_props(fontstyle="italic")
    if title:
        ax.set_title(title, loc="left", fontsize=7, pad=4)
    return t

def f2(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"

def fp(p):
    if pd.isna(p): return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def hrci(h, l, u):
    return f"{h:.2f} ({l:.2f}–{u:.2f})"

def tcga_expr():
    counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
    gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
    lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
    lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
    return lc[~lc.index.duplicated()]

def km_with_risk(ax, T, E, groups, colors, labels, tmax=10, step=2, risk_y0=-0.30, dy=0.07, show_row_labels=True):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    T = T.copy(); E = E.copy()
    E[T > tmax] = 0; T = T.clip(upper=tmax)
    for g, c, lab in zip(groups, colors, labels):
        KaplanMeierFitter().fit(T[g], E[g], label=lab).plot_survival_function(ax=ax, ci_show=False, color=c, lw=1)
    p = logrank_test(T[groups[0]], T[groups[1]], E[groups[0]], E[groups[1]]).p_value
    ticks = list(range(0, tmax + 1, step)); ax.set_xticks(ticks); ax.set_xlim(0, tmax); ax.set_ylim(0, 1.03)
    for i, (g, c, lab) in enumerate(zip(groups, colors, labels)):
        y = risk_y0 - i * dy
        if show_row_labels:
            ax.text(-0.12, y, lab, transform=ax.transAxes, ha="right", fontsize=5.2, color=c)
        for t_ in ticks:
            ax.text(t_, y, str(int((T[g] >= t_).sum())), transform=ax.get_xaxis_transform(), ha="center", fontsize=5.2, color=c)
    return p
