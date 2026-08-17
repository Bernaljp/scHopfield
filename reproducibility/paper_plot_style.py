#!/usr/bin/env python3
"""
paper_plot_style.py - one shared, submission-hardened matplotlib style.

Import this into every gen_figN_*.py so all data figures match and every one is
submission-ready by construction. It fixes the things a bare rcParams snippet leaves
open: Type-42 (embeddable) fonts, a real sans-serif with graceful fallback, a
colorblind-safe default cycle (validated with the dataviz palette checker), and figure
sizing pinned to journal column widths so text size does not drift figure to figure.

    from paper_plot_style import use_style, figure_for, save, panel_label, PALETTE
    use_style()                                   # once, at import time of your fig script
    fig, ax = plt.subplots(figsize=figure_for("nature", cols=1))
    ax.plot(x, y)                                 # picks Okabe-Ito slot 1 automatically
    save(fig, "figures/fig2_benchmark")           # -> fig2_benchmark.pdf (+ .png at 600 dpi)

Design choices and their reasons live in <research-paper>/references/figures.md.
Deps: matplotlib (numpy optional, only for the __main__ demo).
"""
from __future__ import annotations
import sys, warnings

# --- Colorblind-safe categorical palette ------------------------------------------
# The 6-slot Okabe-Ito subset (pure black and yellow dropped: black leaves the OKLCH
# lightness band, yellow is ~1.3:1 on white). This subset PASSES the dataviz validator
# (lightness band, chroma floor, CVD separation; worst adjacent deutan dE 48.9). The
# lighter hues WARN on contrast-vs-white, which we always relieve with markers +
# direct labels (secondary encoding), so identity is never color-alone or lost in print.
PALETTE = {
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "blue":    "#0072B2",
    "vermillion": "#D55E00",
    "purple":  "#CC79A7",
}
CYCLE = list(PALETTE.values())          # the default series order (fixed, never cycled past 6)
INK = "#222222"                         # text / axis ink (never a series color)
MUTED = "#666666"                       # secondary text
GRID = "#DDDDDD"

# Perceptually-uniform, CVD-safe ramps for the magnitude / polarity jobs (heatmaps).
SEQUENTIAL = "viridis"                  # magnitude: perceptually uniform, grayscale-safe
SEQUENTIAL_MONO = "Blues"               # single-hue alternative when a hue is wanted
DIVERGING = "RdBu_r"                    # polarity: blue-white-red, avoids red-green

# Markers / linestyles used as the secondary (non-color) identity channel on lines.
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

# --- Journal column widths (mm). figure_for() turns these into inches. --------------
# Single / intermediate / double column full-width, from each publisher's author guide.
_COLS_MM = {
    "nature":  {1: 89,  1.5: 120, 2: 183},   # Nature / Nature Methods
    "science": {1: 55,  1.5: 120, 2: 183},
    "cell":    {1: 85,  1.5: 114, 2: 174},
    "pnas":    {1: 87,  1.5: 114, 2: 178},
    "ieee":    {1: 88,  2: 181},              # IEEE two-column
    "generic": {1: 90,  1.5: 120, 2: 180},
}
DEFAULT_ASPECT = 0.72                    # height / width when not given


def _resolve_sans():
    """Best available sans-serif, preferring Helvetica/Arial and their metric clones.
    Returns (family_list, chosen_name, is_last_resort). Warns if only DejaVu is left."""
    from matplotlib import font_manager as fm
    prefer = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans",
              "TeX Gyre Heros", "Liberation Sans", "Arimo", "DejaVu Sans"]
    have = {f.name for f in fm.fontManager.ttflist}
    chosen = next((n for n in prefer if n in have), "DejaVu Sans")
    if chosen == "DejaVu Sans":
        warnings.warn(
            "paper_plot_style: no Helvetica/Arial-class font found; falling back to "
            "DejaVu Sans. For a Nature-class look install a Helvetica clone, e.g. "
            "'fonts-urw-base35' (Nimbus Sans) or 'texlive-fonts-extra' (TeX Gyre Heros).",
            stacklevel=2)
    return prefer, chosen, chosen == "DejaVu Sans"


def use_style(base_fontsize: int = 7):
    """Apply the shared style globally. base_fontsize in pt is the final-print body size
    (Nature wants 5-7pt); 7 is a safe default for single-column figures."""
    import matplotlib as mpl
    fam, chosen, last_resort = _resolve_sans()
    mathset = "dejavusans" if last_resort else "stixsans"   # match math to the body sans
    from cycler import cycler
    mpl.rcParams.update({
        # Fonts: embeddable Type 42 so no journal rejects Type-3 outlines.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": fam,
        "mathtext.fontset": mathset, "mathtext.default": "regular",
        # Sizes (pt at final print size).
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1, "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1, "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1, "figure.titlesize": base_fontsize + 1,
        # Color: colorblind-safe default cycle; ink stays neutral.
        "axes.prop_cycle": cycler(color=CYCLE),
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        # Marks: hairlines that survive downscaling to a column.
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        # Chartjunk off: no top/right spine, no grid by default, borderless legend.
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4,
        # Output: vector by default, exact-width by default (see save()).
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.format": "pdf",
        "savefig.bbox": None, "savefig.pad_inches": 0.01, "savefig.transparent": False,
    })
    return chosen


def figure_for(journal: str = "generic", cols=1, aspect: float | None = None,
               height: float | None = None):
    """(width, height) in inches for a target journal column span.
    cols in {1, 1.5, 2}; aspect = height/width; or pass an explicit height (inches)."""
    table = _COLS_MM.get(journal.lower(), _COLS_MM["generic"])
    if cols not in table:
        cols = min(table, key=lambda c: abs(c - cols))
    w = table[cols] / 25.4
    h = height if height is not None else w * (aspect or DEFAULT_ASPECT)
    return (round(w, 3), round(h, 3))


def panel_label(ax, letter: str, *, x=-0.02, y=1.04, weight="bold", size=None):
    """Bold panel letter (a, b, c) in axes-fraction coords, above the top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontweight=weight, fontsize=size, color=INK, clip_on=False)


def save(fig, path: str, *, formats=("pdf",), png_dpi: int = 600,
         tight: bool = False, tex_includes: str | None = None, tex_width: str = r"\linewidth"):
    """Save vector-first, preserving the figure's exact width by default (tight=False)
    so column sizing does not drift. Pass tight=True only for one-off previews.
    Set tex_includes to a path to append an \\includegraphics snippet for paste-in."""
    import os
    base, _ = os.path.splitext(path)
    bbox = "tight" if tight else None
    written = []
    for fmt in formats:
        out = f"{base}.{fmt}"
        dpi = png_dpi if fmt in ("png", "jpg", "tiff") else None
        fig.savefig(out, format=fmt, bbox_inches=bbox, dpi=dpi,
                    metadata={"Creator": "paper_plot_style"} if fmt == "pdf" else None)
        written.append(out)
    if tex_includes:
        name = os.path.basename(base)
        with open(tex_includes, "a") as fh:
            fh.write(f"\\includegraphics[width={tex_width}]{{{name}}}%% {name}\n")
    return written


def _demo(outdir="out"):
    """Render a 3-panel sample so you can eyeball the style. `python paper_plot_style.py demo`."""
    import os, numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_style()
    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=figure_for("nature", cols=2, aspect=0.32),
                            layout="constrained")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    for i in range(4):
        axs[0].plot(x, np.sin(x + i) + 0.15 * i, marker=MARKERS[i], markevery=8,
                    linestyle=LINESTYLES[i], label=f"m{i+1}")
    axs[0].set(xlabel="step", ylabel="metric"); axs[0].legend(ncol=2)
    cats, vals = ["A", "B", "C"], [0.62, 0.81, 0.74]
    bars = axs[1].bar(cats, vals, color=CYCLE[:3])
    for b, v in zip(bars, vals):
        axs[1].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    axs[1].set(ylabel="AUROC", ylim=(0, 1))
    im = axs[2].imshow(rng.random((6, 6)), cmap=SEQUENTIAL, aspect="auto")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set(xlabel="gene", ylabel="cell")
    for ax, L in zip(axs, "abc"):
        panel_label(ax, L)
    outs = save(fig, os.path.join(outdir, "style_demo"), formats=("pdf", "png"))
    print("wrote", outs, "| font:", _resolve_sans()[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        _demo(sys.argv[2] if len(sys.argv) > 2 else "out")
    else:
        print("usage: paper_plot_style.py [demo [OUTDIR] | --help]")
