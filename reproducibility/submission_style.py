"""Journal page geometry and type rules for the submission, as code.

A written figure specification drifts from what is actually rendered. This module is the
single source of truth instead: every generator imports these constants and calls
``figure_for()`` rather than hardcoding a ``figsize``, so "does it fit the page" is answered
by the code that draws it.

The rules come from Nature Portfolio figure guidelines. Confirm them against the live
guidelines once (task #75) and change them here, in one place, if they differ.

    from submission_style import MM, DOUBLE_COL, figure_for, check_type_sizes

    fig = figure_for("double", height_mm=180)      # never exceeds the page
    ...
    check_type_sizes(fig)                          # raises if any text is below the floor
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Page geometry, in millimeters.
MM = 1 / 25.4                     # millimeters to inches
SINGLE_COL = 89.0                 # one column
DOUBLE_COL = 180.0                # full width, the hard maximum
MAX_HEIGHT = 247.0                # full page, including the legend area
SAFE_HEIGHT = 210.0               # leaves room for a legend under the figure

# Type. Nature's floor is 5 pt at final size; we aim higher and refuse to go below.
TYPE_FLOOR = 5.0
TYPE_BODY = 7.0
TYPE_LABEL = 8.0
TYPE_PANEL_LETTER = 8.0           # bold, lowercase, top left of each panel

WIDTHS = {"single": SINGLE_COL, "double": DOUBLE_COL}


def use_submission_style() -> None:
    """Vector output with embedded (Type-42) fonts, at the submission type sizes."""
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": TYPE_BODY,
        "axes.titlesize": TYPE_LABEL,
        "axes.labelsize": TYPE_BODY,
        "xtick.labelsize": TYPE_FLOOR + 1,
        "ytick.labelsize": TYPE_FLOOR + 1,
        "legend.fontsize": TYPE_FLOOR + 1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        # House style, inherited from paper_plot_style.py: only the two spines that carry an
        # axis are drawn. A submission path that sets its own rcParams and forgets these puts
        # a full box around every panel, which reads as four axes instead of two.
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def figure_for(column: str = "double", height_mm: float = SAFE_HEIGHT, **kwargs):
    """A figure at the journal's page width, in inches, refusing anything off-page."""
    width_mm = WIDTHS[column]
    if height_mm > MAX_HEIGHT:
        raise ValueError(
            f"height {height_mm:.0f} mm exceeds the {MAX_HEIGHT:.0f} mm page. "
            "Move panels to Extended Data rather than scaling the type down.")
    use_submission_style()
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kwargs)


def panel_letter(ax, letter: str, dx: float = -0.02, dy: float = 1.02) -> None:
    """One panel letter, placed identically everywhere: bold, lowercase, above top left."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=TYPE_PANEL_LETTER,
            fontweight="bold", ha="right", va="bottom")


def check_type_sizes(fig, floor: float = TYPE_FLOOR) -> list[tuple[str, float]]:
    """Every Text artist below the floor, as (text, size). Empty means the figure passes.

    Call this before saving. A figure that renders beautifully at 16 inches and is then
    scaled to 180 mm is the failure this catches.
    """
    bad = []
    for t in fig.findobj(match=plt.Text):
        s = t.get_fontsize()
        if t.get_text().strip() and s < floor:
            bad.append((t.get_text()[:40], s))
    return bad


def save(fig, path: str, *, check: bool = True) -> None:
    """Save as vector PDF with embedded fonts, after the type check."""
    if check:
        bad = check_type_sizes(fig)
        if bad:
            raise ValueError(
                f"{len(bad)} text items below the {TYPE_FLOOR} pt floor, e.g. {bad[:3]}")
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)


# --------------------------------------------------------------------------------------
# Color registry
#
# NOT a shared palette. The figures show different quantities and must not be forced onto
# one scheme: two different quantities in the same colors misleads a reader far more than
# two different quantities in different colors. What is shared is the RULE, which is that a
# quantity owns an encoding.
#
#   - one encoding per quantity, and no two quantities share one;
#   - the same quantity keeps its encoding wherever it appears;
#   - a lineage decision is part of the quantity's identity, so "toward alpha vs beta" and
#     "toward differentiated vs progenitor" are different quantities with different maps;
#   - prediction and result never share an encoding (Jacobian push vs propagated fate);
#   - categorical identity (cell type) is consistent per dataset, because it is identity
#     rather than magnitude.
#
# The reasoning, and the one accepted reuse, are recorded with the manuscript sources.
# --------------------------------------------------------------------------------------

COLOR_REGISTRY = {
    "interaction_weight_signed": "RdBu_r",
    "expression": "Blues",
    "unspliced": "Purples",
    "pseudotime": "cividis",
    "energy_quasipotential": "viridis",
    "rotation_vorticity": "magma",
    "simulation_time": "viridis",          # ground-truth time, only in the dyngen figure
    "fate_shift_decision0": "DEC_CMAPS[0]",
    "fate_shift_decision1": "DEC_CMAPS[1]",
    "synergy_vs_additive": "PRGn",
    "jacobian_commitment_push": "PuOr",
    "flow_alignment_development": "BrBG",
    "node_degree_signed": "NODE_CMAP",
    "cell_type_identity": "per-dataset categorical",
}

# Reuses that are known, accepted, and recorded rather than silently tolerated. Each entry
# is a colormap carrying more than one meaning, with why it is allowed to.
ACCEPTED_REUSE = {
    "RdBu_r": ("interaction_weight_signed", "fate_shift_decision0",
               "never co-occur in a panel; every colorbar labels its poles by meaning"),
    "viridis": ("energy_quasipotential", "simulation_time",
                "simulation time appears only in the dyngen figure, which shows no energy map"),
}


def check_registry() -> list[str]:
    """Colormaps carrying more than one meaning that are not in ACCEPTED_REUSE.

    Guards the rule that different quantities look different. Empty means the registry is
    clean; anything returned is a collision to resolve or to record as accepted.
    """
    seen: dict[str, list[str]] = {}
    for quantity, cmap in COLOR_REGISTRY.items():
        seen.setdefault(cmap, []).append(quantity)
    return [f"{cmap} means {' and '.join(qs)}"
            for cmap, qs in seen.items()
            if len(qs) > 1 and cmap not in ACCEPTED_REUSE]
