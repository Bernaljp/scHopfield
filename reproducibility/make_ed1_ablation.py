"""Extended Data Fig. 1: what the restricted nonlinear model class buys over a linear field.

v24 cites Extended Data Fig. 1 once, for the sentence:

    "For the repressilator, Hill activation reproduced the velocity field almost perfectly
     whereas a linear field reached only 0.44. For the toggle switch, a linear autonomous
     model both failed to reconstruct the dynamics and, by admitting at most one fixed
     point, could not represent multistability."

so the figure carries exactly those two claims and nothing else:
  a  velocity reconstruction, Hill against linear, on both circuits
  b  stable fixed points recovered against the truth, which is where the linear model fails
     qualitatively rather than quantitatively

Drawn from reproducibility/data/ablations/hill_vs_linear.json, the same numbers the text quotes.
Vector throughout; the existing hill_vs_linear.png is a raster of an older layout.

Run:  python reproducibility/make_ed1_ablation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import PALETTE                                    # noqa: E402
from submission_style import figure_for, panel_letter, save, TYPE_BODY  # noqa: E402

SRC = os.path.join(paths.ABLATIONS, "hill_vs_linear.json")
OUT = os.path.join(paths.FIGURES_SPEC, "ExtendedDataFig1.pdf")

# Model class is a new categorical distinction, not one of the quantities in the colormap
# registry, so it takes two unused Okabe-Ito hues rather than borrowing a reserved encoding.
HILL, LINEAR, TRUTH = PALETTE["blue"], PALETTE["vermillion"], "0.55"
LABEL = {"toggle_bistable": "toggle switch", "repressilator": "repressilator"}


def main() -> int:
    d = json.load(open(SRC))
    circuits = [c for c in ("toggle_bistable", "repressilator") if c in d]
    x = np.arange(len(circuits))
    w = 0.34

    fig = figure_for("double", height_mm=62.0)
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.86, bottom=0.20, wspace=0.30)

    # ---- a: velocity reconstruction --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for k, (key, col, name) in enumerate([("hill_recon_r2", HILL, "Hill activation"),
                                          ("linear_recon_r2", LINEAR, "linear field")]):
        v = [d[c][key] for c in circuits]
        ax.bar(x + (k - 0.5) * w, v, w, color=col, edgecolor="0.25", linewidth=0.4, label=name)
        for xi, vi in zip(x + (k - 0.5) * w, v):
            # 0.0001 is invisible as a bar, so every value is printed.
            ax.annotate(f"{vi:.4g}", (xi, vi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("velocity reconstruction $R^2$", fontsize=TYPE_BODY)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in circuits], fontsize=6.5)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.16), handlelength=1.2, columnspacing=1.0)
    panel_letter(ax, "a")

    # ---- b: stable fixed points against the truth ------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    wb = 0.26
    for k, (key, col, name) in enumerate([("true_stable_fixedpoints", TRUTH, "ground truth"),
                                          ("hill_stable_fixedpoints", HILL, "Hill activation"),
                                          ("linear_stable_fixedpoints", LINEAR, "linear field")]):
        v = [d[c][key] for c in circuits]
        ax.bar(x + (k - 1) * wb, v, wb, color=col, edgecolor="0.25", linewidth=0.4, label=name)
        for xi, vi in zip(x + (k - 1) * wb, v):
            ax.annotate(f"{vi:g}", (xi, vi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.5)
    ax.set_ylim(0, 3.9)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("stable fixed points recovered", fontsize=TYPE_BODY)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in circuits], fontsize=6.5)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.16), handlelength=1.2, columnspacing=0.9)
    panel_letter(ax, "b")
    # The repressilator has no stable fixed point by construction: it is a limit cycle, so
    # zero is the correct answer there and is not a failure of either model.
    ax.annotate("limit cycle:\nno stable fixed point", (x[-1], 0.06), xytext=(0, 12),
                textcoords="offset points", ha="center", va="bottom", fontsize=5.2,
                color="0.35", style="italic")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save(fig, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
