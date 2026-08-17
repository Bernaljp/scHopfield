"""Composite validation figure: toggle switch + repressilator (small circuits).

One mosaic figure that assembles the small-circuits validation story, panels lettered
a-j:

  Toggle switch          a circuit   b phase portrait   c bifurcation diagram
                         d W-recovery (ground truth large, 3 recovered small)
                         e Jacobian eigenvalue maps (couplings c = -2.5 and -8)
  Repressilator          f circuit   g 3D limit cycle + oscillations
                         h W-recovery (ground truth large, 3 recovered small)
                         i PCA energy landscape (3D) + 2D energy contour with flow
                         j Jacobian spectrum on the PCA grid (Re / Im of the 3 eigenvalues)

Every panel is regenerated natively (vector) reusing the cached circuit fits in
``reproducibility/data/small_circuits/fits.npz`` and the compute helpers in
``reproducibility/build_circuits_report.py``. Submission style from
``reproducibility/paper_plot_style.py``.

Two renderings, same content:

  default       the large report canvas (317 x 498 mm), written to
                reproducibility/figures/small-circuits-validation.{pdf,png}
  --submission  the journal page (180 mm wide, one page tall, no type below 5 pt),
                written to reproducibility/figures/submission/Figure2.pdf

The journal version keeps all ten panels and their letters. It fits by redistributing
them, not by scaling the type down: the toggle Jacobian maps (e) and the repressilator
spectrum (j) are laid out as single rows instead of grids, which trades unused width for
the height the page does not have.

Run:  python reproducibility/make_small_circuits_validation.py
      python reproducibility/make_small_circuits_validation.py --submission
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, PALETTE      # noqa: E402
import build_circuits_report as B                          # noqa: E402
from scHopfield.validation.circuits import ToggleCircuit, OscillatorCircuit  # noqa: E402
from scHopfield.validation.simulate import simulate_circuit  # noqa: E402

OUT = paths.FIGURES
NAME = "small-circuits-validation"
SUB_OUT = os.path.join(paths.FIGURES_SPEC, "Figure2.pdf")
SUB_H_MM = 233.0          # page height used by the journal rendering, under the 247 mm cap
SUB_W_MM = 180.0          # the double-column width figure_for() hands back
ORANGE, BLUE, GREEN = PALETTE["orange"], PALETTE["blue"], PALETTE["green"]
VERM, SKY, PURPLE = PALETTE["vermillion"], PALETTE["sky"], PALETTE["purple"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def letter(ax, s, x=-0.16, y=1.06):
    txt = ax.text2D if hasattr(ax, "text2D") else ax.text   # 3D axes need text2D
    txt(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=12,
        va="bottom", ha="right")


def heat_W(ax, W, labels, vmax, title, annot=True, fs=8, title_fs=8, pad=3):
    ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    if annot:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                t = f"{W[i, j]:.0f}"
                if t == "-0":
                    t = "0"
                ax.text(j, i, t, ha="center", va="center", color="black", fontsize=fs)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=fs); ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=title_fs, pad=pad)


def heat_W_small(ax, W, vmax, title, title_fs=7, pad=1.5):
    ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=title_fs, pad=pad)


# --------------------------------------------------------------------------- #
# circuit diagrams (a, f) -- rendered with TikZ (clean Bar-tip repression and real
# self-loops), embedded as high-DPI images; matplotlib fallback if LaTeX is missing
# --------------------------------------------------------------------------- #
TIKZ_PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage{tikz}\usetikzlibrary{arrows.meta}
\definecolor{gnblue}{HTML}{0072B2}\definecolor{gnbluefill}{HTML}{CFE8FF}
\definecolor{gnorfill}{HTML}{FFE0CF}\definecolor{gnverm}{HTML}{D55E00}
\definecolor{gngreen}{HTML}{009E73}
\begin{document}
\begin{tikzpicture}[
  gene/.style={circle,draw=gnblue,fill=gnbluefill,line width=0.9pt,minimum size=11mm,font=\large},
  rgene/.style={circle,draw=gnverm,fill=gnorfill,line width=0.9pt,minimum size=11mm,font=\large},
  repress/.style={-{Bar[width=7pt]},gnverm,line width=1.1pt,shorten >=3pt,shorten <=3pt},
  activate/.style={-{Latex[length=4pt,width=3.5pt]},gngreen,line width=1.1pt}]
"""
TIKZ_TOGGLE = r"""
\node[gene] (A) at (0,0) {$x_1$};
\node[gene] (B) at (2.6,0) {$x_2$};
\draw[repress] (A) to[bend left=13] (B);
\draw[repress] (B) to[bend left=13] (A);
\draw[activate] (A) to[out=115,in=65,looseness=5] (A);
\draw[activate] (B) to[out=115,in=65,looseness=5] (B);
\node[font=\small] at (1.3,-1.4) {$\begin{aligned}
  \dot{x}_1 &= \textcolor{gngreen}{a}\,\sigma(x_1) - \textcolor{gnverm}{b}\,\sigma(x_2) + \textcolor{gnverm}{b} - \gamma x_1\\
  \dot{x}_2 &= \textcolor{gngreen}{a}\,\sigma(x_2) - \textcolor{gnverm}{b}\,\sigma(x_1) + \textcolor{gnverm}{b} - \gamma x_2
\end{aligned}$};
"""
TIKZ_REPRESSILATOR = r"""
\node[rgene] (X) at (90:2.7) {$x$};
\node[rgene] (Y) at (210:2.7) {$y$};
\node[rgene] (Z) at (330:2.7) {$z$};
\draw[repress] (X) to[bend right=16] (Y);
\draw[repress] (Y) to[bend right=16] (Z);
\draw[repress] (Z) to[bend right=16] (X);
\node[font=\small] at (0,0.15) {$\begin{aligned}
  \dot{x} &= \textcolor{gnverm}{\alpha} - x - \textcolor{gnverm}{\alpha}\,\varphi(z)\\
  \dot{y} &= \textcolor{gnverm}{\alpha} - y - \textcolor{gnverm}{\alpha}\,\varphi(x)\\
  \dot{z} &= \textcolor{gnverm}{\alpha} - z - \textcolor{gnverm}{\alpha}\,\varphi(y)
\end{aligned}$};
"""


# Journal-page variants. The TikZ text is baked into the image, so at page size the only
# way to keep the equations above the 5 pt floor is to shrink the drawing around them: the
# ring radius comes in, the equations stay the size they were.
TIKZ_TOGGLE_SUB = r"""
\node[gene] (A) at (0,0) {$x_1$};
\node[gene] (B) at (2.4,0) {$x_2$};
\draw[repress] (A) to[bend left=13] (B);
\draw[repress] (B) to[bend left=13] (A);
\draw[activate] (A) to[out=115,in=65,looseness=5] (A);
\draw[activate] (B) to[out=115,in=65,looseness=5] (B);
\node[font=\small] at (1.2,-1.5) {$\begin{aligned}
  \dot{x}_1 &= \textcolor{gngreen}{a}\,\sigma(x_1) - \textcolor{gnverm}{b}\,\sigma(x_2)
              + \textcolor{gnverm}{b} - \gamma x_1\\
  \dot{x}_2 &= \textcolor{gngreen}{a}\,\sigma(x_2) - \textcolor{gnverm}{b}\,\sigma(x_1)
              + \textcolor{gnverm}{b} - \gamma x_2
\end{aligned}$};
"""
TIKZ_REPRESSILATOR_SUB = r"""
\node[rgene] (X) at (90:2.15) {$x$};
\node[rgene] (Y) at (210:2.15) {$y$};
\node[rgene] (Z) at (330:2.15) {$z$};
\draw[repress] (X) to[bend right=16] (Y);
\draw[repress] (Y) to[bend right=16] (Z);
\draw[repress] (Z) to[bend right=16] (X);
\node[font=\small] at (0,0.1) {$\begin{aligned}
  \dot{x} &= \textcolor{gnverm}{\alpha} - x - \textcolor{gnverm}{\alpha}\,\varphi(z)\\
  \dot{y} &= \textcolor{gnverm}{\alpha} - y - \textcolor{gnverm}{\alpha}\,\varphi(x)\\
  \dot{z} &= \textcolor{gnverm}{\alpha} - z - \textcolor{gnverm}{\alpha}\,\varphi(y)
\end{aligned}$};
"""


def render_tikz(body, dpi=600):
    """Compile a standalone TikZ snippet to a high-DPI image array (or None on failure)."""
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "c.tex"), "w") as f:
                f.write(TIKZ_PREAMBLE + body + "\\end{tikzpicture}\n\\end{document}\n")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "c.tex"],
                               cwd=td, capture_output=True)
            if r.returncode != 0 or not os.path.exists(os.path.join(td, "c.pdf")):
                return None
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                            os.path.join(td, "c.pdf"), os.path.join(td, "c")], capture_output=True)
            png = os.path.join(td, "c.png")
            return plt.imread(png) if os.path.exists(png) else None
    except Exception:
        return None


def draw_toggle_circuit(ax, img):
    if img is None:
        return draw_toggle_circuit_mpl(ax)
    ax.imshow(img)
    ax.axis("off")


def draw_repressilator_circuit(ax, img):
    if img is None:
        return draw_repressilator_circuit_mpl(ax)
    ax.imshow(img)
    ax.axis("off")


def draw_toggle_circuit_mpl(ax):
    pos = {"x1": (-1, 0), "x2": (1, 0)}
    # positive autoregulation: a self-loop on TOP of each node (drawn first, high zorder)
    for cx, cy in pos.values():
        loop = FancyArrowPatch((cx - 0.13, cy + 0.31), (cx + 0.13, cy + 0.31),
                               connectionstyle="arc3,rad=-2.6", arrowstyle="-|>",
                               mutation_scale=11, color=GREEN, lw=1.9, zorder=5)
        ax.add_patch(loop)
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.34, fc="#cfe8ff", ec=BLUE, lw=1.8, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=11, zorder=4)
    # mutual repression: two flat-headed (T-bar) links between the nodes
    ax.annotate("", xy=(-0.62, 0.11), xytext=(0.62, 0.11),
                arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.5", color=VERM, lw=1.8))
    ax.annotate("", xy=(0.62, -0.11), xytext=(-0.62, -0.11),
                arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.5", color=VERM, lw=1.8))
    ax.text(0, 0.95, "autoactivation", ha="center", color=GREEN, fontsize=7.5)
    ax.text(0, -0.52, "mutual repression", ha="center", color=VERM, fontsize=7.5)
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-0.85, 1.25); ax.set_aspect("equal"); ax.axis("off")


def draw_repressilator_circuit_mpl(ax):
    ang = {"x": 90, "y": 210, "z": 330}
    P = {k: (np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for k, a in ang.items()}
    for name, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.3, fc="#ffe0cf", ec=VERM, lw=1.8, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=11, zorder=4)
    for a, b in [("x", "y"), ("y", "z"), ("z", "x")]:
        xa, ya = P[a]; xb, yb = P[b]
        v = np.array([xb - xa, yb - ya]); v = v / np.linalg.norm(v)
        s = (xa + v[0] * 0.34, ya + v[1] * 0.34); e = (xb - v[0] * 0.36, yb - v[1] * 0.36)
        ax.annotate("", xy=e, xytext=s,
                    arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.6", color=VERM,
                                    lw=1.8, connectionstyle="arc3,rad=0.18"))
    ax.text(0, -1.5, "cyclic repression", ha="center", color=VERM, fontsize=7.5)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.7, 1.4); ax.set_aspect("equal"); ax.axis("off")


# --------------------------------------------------------------------------- #
# toggle phase portrait (b)
# --------------------------------------------------------------------------- #
def draw_phase(ax, circ, res, *, fs_leg=6.5, fs_cb=6, fs_cbtick=5.5, fs_title=8,
               star_ms=13, saddle_ms=6, leg_star_ms=9, stream_lw=0.5, arrowsize=0.7,
               title="energy landscape + flow ($b=4$)"):
    # learned energy landscape (viridis, same encoding as the repressilator energy in i);
    # the toggle's W is symmetric so the energy is a true Lyapunov function whose wells sit
    # at the stable states.
    W, I, gamma = res["full"]["W_inferred"], res["full"]["I_inferred"], res["full"]["gamma_inferred"]
    gv = np.linspace(0.01, 5, 120); GX, GY = np.meshgrid(gv, gv)
    E = B._energy_grid_2d(["x1", "x2"], "toggle_circuit", W, I, gamma,
                          np.vstack([GX.ravel(), GY.ravel()]).T, GX.shape)
    cf = ax.contourf(GX, GY, E, levels=30, cmap="viridis", zorder=0)
    # flow of the ground-truth circuit, white streamlines on the energy
    val = np.linspace(0.01, 5, 22); X, Y = np.meshgrid(val, val)
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j], V[i, j] = circ.rhs(np.array([X[i, j], Y[i, j]]))
    ax.streamplot(X, Y, U, V, color="white", density=0.85, linewidth=stream_lw,
                  arrowsize=arrowsize, zorder=1)
    # fixed points: color = state type (orange symmetric / blue asymmetric), marker = stability
    # (filled star = stable, open diamond = saddle); white edge so they read on the dark wells.
    stable, saddle = B.all_fixed_points(circ)
    for x in stable:
        sym = abs(x[0] - x[1]) < 0.15
        ax.plot(*x, "*", ms=star_ms, color=(ORANGE if sym else BLUE), mec="white", mew=1.0,
                zorder=6)
    for x in saddle:
        ax.plot(*x, "D", ms=saddle_ms, mfc="white", mec="black", mew=1.0, zorder=6)
    handles = [Line2D([0], [0], ls="", marker="*", ms=leg_star_ms, color=ORANGE, mec="white",
                      label="symmetric"),
               Line2D([0], [0], ls="", marker="*", ms=leg_star_ms, color=BLUE, mec="white",
                      label="asymmetric"),
               Line2D([0], [0], ls="", marker="D", ms=saddle_ms, mfc="white", mec="black",
                      label="saddle")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=fs_leg, handletextpad=0.2,
                    borderpad=0.3, frameon=True, framealpha=0.92, facecolor="white",
                    edgecolor="0.7")
    leg.set_zorder(10)
    cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("energy", fontsize=fs_cb)
    cb.ax.tick_params(labelsize=fs_cbtick)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_title(title, fontsize=fs_title)


# --------------------------------------------------------------------------- #
# toggle bifurcation diagram (c)
# --------------------------------------------------------------------------- #
def draw_bifurcation(ax, circ, *, lw=1.8, fs_leg=6, fs_mark=7, fs_title=8, leg_ncol=1):
    a, Ib, gm = float(circ.a), float(circ.b), float(circ.gamma)

    def s1(x):
        xc = max(x, 0.0); return xc ** 4 / (1.0 + xc ** 4)

    def sp1(x):
        return 0.0 if x <= 0 else 4.0 * x ** 3 / (1.0 + x ** 4) ** 2

    def stable(x1, x2, c):
        J = np.array([[a * sp1(x1) - gm, c * sp1(x2)], [c * sp1(x1), a * sp1(x2) - gm]])
        return bool(np.all(np.real(np.linalg.eigvals(J)) < 0))

    sym = []
    for c in np.linspace(-6.0, 1.0, 600):
        f = lambda x: (a + c) * s1(x) + Ib - gm * x
        xs = np.linspace(-0.6, 4.0, 600); fv = np.array([f(x) for x in xs])
        for k in range(len(xs) - 1):
            if fv[k] * fv[k + 1] < 0:
                r = brentq(f, xs[k], xs[k + 1]); sym.append((c, r, stable(r, r, c)))
    asy = []
    for x2 in np.linspace(-0.95, 2.2, 900):
        g = lambda x1: (a * s1(x1) ** 2 + (gm * x2 - Ib - a * s1(x2)) * s1(x2)
                        + (Ib - gm * x1) * s1(x1))
        xs = np.linspace(x2 + 0.03, 3.8, 450); gv = np.array([g(x) for x in xs])
        for k in range(len(xs) - 1):
            if gv[k] * gv[k + 1] < 0:
                x1 = brentq(g, xs[k], xs[k + 1])
                if x1 > x2 + 0.03:
                    c = (gm * x2 - Ib - a * s1(x2)) / s1(x1)
                    if -6.5 <= c <= 1.0:
                        asy.append((c, x1, x2, stable(x1, x2, c)))
    for st, ls in [(True, "-"), (False, "--")]:
        pts = sorted((c, x) for c, x, s in sym if s == st)
        if pts:
            cc, xx = zip(*pts); ax.plot(cc, xx, ls=ls, color=ORANGE, lw=lw)
    for st, ls in [(True, "-"), (False, "--")]:
        sub = sorted((x2, c, x1) for c, x1, x2, s in asy if s == st)
        if sub:
            x2o = [p[0] for p in sub]; co = [p[1] for p in sub]; x1o = [p[2] for p in sub]
            ax.plot(co, x1o, ls=ls, color=BLUE, lw=lw)
            ax.plot(co, x2o, ls=ls, color=BLUE, lw=lw)
    ax.axvline(-4.0, color="0.55", lw=0.9, ls=":")
    ax.text(-4.0, 3.45, " $b{=}4$", color="0.35", fontsize=fs_mark, va="top")
    handles = [Line2D([0], [0], color=ORANGE, lw=lw, label="symmetric"),
               Line2D([0], [0], color=BLUE, lw=lw, label="asymmetric"),
               Line2D([0], [0], color="0.3", lw=lw, ls="-", label="stable"),
               Line2D([0], [0], color="0.3", lw=lw, ls="--", label="unstable")]
    ax.legend(handles=handles, loc="lower right", fontsize=fs_leg, handlelength=1.3,
              borderpad=0.3, labelspacing=0.25, ncol=leg_ncol, columnspacing=0.8)
    ax.set_xlim(-6, 1); ax.set_ylim(-1.1, 3.7)
    ax.set_xlabel("mutual inhibition $c$"); ax.set_ylabel("steady state $x_1$")
    ax.set_title("bifurcation diagram", fontsize=fs_title)


# --------------------------------------------------------------------------- #
# W recovery block (d, h): ground truth large + 3 recovered small
# --------------------------------------------------------------------------- #
def draw_W_recovery(fig, spec, res, labels, vmax, annot_fs=9, *, width_ratios=(2.0, 0.7),
                    hspace=1.0, wspace=0.1, tick_fs=None, title_fs=8, small_title_fs=7,
                    small_pad=1.5):
    inner = spec.subgridspec(3, 2, width_ratios=list(width_ratios), hspace=hspace, wspace=wspace)
    axGT = fig.add_subplot(inner[:, 0])
    heat_W(axGT, res["full"]["W_true"], labels, vmax, "ground-truth $W$", annot=True,
           fs=annot_fs, title_fs=title_fs)
    if tick_fs is not None:
        axGT.tick_params(labelsize=tick_fs)
    for k, name in enumerate(["full", "partial", "none"]):
        axs = fig.add_subplot(inner[k, 1])
        heat_W_small(axs, res[name]["W_inferred"], vmax,
                     ("recovered $\\hat W$\n" if k == 0 else "") + name,
                     title_fs=small_title_fs, pad=small_pad)


# --------------------------------------------------------------------------- #
# toggle Jacobian maps (e): couplings -2.5 and -8, rows lambda1 / lambda2
# --------------------------------------------------------------------------- #
def toggle_jac_fields(res, couplings=(-2.5, -8.0), n=55):
    """Real parts of the two Jacobian eigenvalues over state space, per coupling."""
    W_base = res["full"]["W_inferred"].copy(); gamma = res["full"]["gamma_inferred"]
    val = np.linspace(0.01, 5, n); X, Y = np.meshgrid(val, val)
    gp = np.vstack([X.ravel(), Y.ravel()]).T
    E = {c: [None, None] for c in couplings}
    for c in couplings:
        W = W_base.copy(); W[0, 1] = c; W[1, 0] = c
        e1 = np.zeros(len(gp)); e2 = np.zeros(len(gp))
        for j, pt in enumerate(gp):
            sp = B.hill_prime(pt)
            ev = np.sort(np.real(np.linalg.eigvals(W * sp[None, :] - np.diag(gamma))))[::-1]
            e1[j], e2[j] = ev[0], ev[1]
        E[c] = [e1.reshape(X.shape), e2.reshape(X.shape)]
    return X, Y, E


def draw_jac_toggle(fig, spec, res):
    couplings = [-2.5, -8.0]
    X, Y, E = toggle_jac_fields(res, couplings=tuple(couplings))
    inner = spec.subgridspec(2, 3, width_ratios=[1, 1, 0.09], hspace=0.32, wspace=0.28)
    names = ["$\\lambda_1$", "$\\lambda_2$"]
    for row in range(2):
        m = max(np.max(np.abs(E[c][row])) for c in couplings)
        for col, c in enumerate(couplings):
            ax = fig.add_subplot(inner[row, col])
            ax.contourf(X, Y, E[c][row], levels=30, cmap="RdBu_r", vmin=-m, vmax=m)
            if E[c][row].min() < 0 < E[c][row].max():
                ax.contour(X, Y, E[c][row], levels=[0.0], colors="k", linewidths=1.0)
            ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4])
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"$c={c:g}$", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{names[row]}\n$x_2$", fontsize=7)
            if row == 1:
                ax.set_xlabel("$x_1$", fontsize=7)
        cax = fig.add_subplot(inner[row, 2])
        sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-m, m))
        cb = fig.colorbar(sm, cax=cax); cb.ax.tick_params(labelsize=5.5)
        cb.set_label("Re $\\lambda$", fontsize=6)


# --------------------------------------------------------------------------- #
# repressilator limit cycle (g) + oscillations
# --------------------------------------------------------------------------- #
def _osc_traj(circ):
    return solve_ivp(lambda t, x: circ.rhs(x), (0, 80), [1.5, 0.5, 1.0],
                     t_eval=np.linspace(0, 80, 2000), method="LSODA", rtol=1e-8, atol=1e-10)


def draw_limitcycle(ax, circ, *, fs_lab=7, fs_tick=5, labelpad=-9, tickpad=-3, lw=1.4,
                    fs_title=8, nbins=None, zoom=None):
    sol = _osc_traj(circ); tail = sol.y[:, sol.t > 25]
    ax.plot(tail[0], tail[1], tail[2], color=BLUE, lw=lw)   # default view + auto limits
    ax.set_xlabel("$x$", fontsize=fs_lab, labelpad=labelpad)
    ax.set_ylabel("$y$", fontsize=fs_lab, labelpad=labelpad)
    ax.set_zlabel("$z$", fontsize=fs_lab, labelpad=labelpad)
    ax.tick_params(labelsize=fs_tick, pad=tickpad)
    if nbins is not None:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(mpl.ticker.MaxNLocator(nbins))
    if zoom is not None:
        ax.set_box_aspect(None, zoom=zoom)      # fill the small cell, not the empty cube
    ax.set_title("3D limit cycle", fontsize=fs_title)


def draw_oscillations(ax, circ, *, lw=1.1, fs_leg=6.5, fs_title=8, t_hi=58):
    sol = _osc_traj(circ); keep = (sol.t > 25) & (sol.t < t_hi)   # ~3 periods, less clutter
    for i, (name, col) in enumerate(zip("xyz", [BLUE, VERM, GREEN])):
        ax.plot(sol.t[keep], sol.y[i][keep], color=col, lw=lw, label=f"${name}$")
    ax.set_ylim(0, 9)
    ax.set_xlabel("time"); ax.set_ylabel("expression")
    ax.legend(ncol=3, fontsize=fs_leg, handlelength=1.1, columnspacing=0.9,
              loc="upper center", bbox_to_anchor=(0.5, 1.02), borderpad=0.2, framealpha=0.9)
    ax.set_title("phase-shifted oscillations", fontsize=fs_title)


# --------------------------------------------------------------------------- #
# repressilator PCA fields (energy, flow, jacobian eigenvalues) on one grid
# --------------------------------------------------------------------------- #
def repr_pca_fields(ao, res, n=70):
    W, I, gamma = res["W_inferred"], res["I_inferred"], res["gamma_inferred"]
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(ao.layers["Ms"])
    c1 = (emb[:, 0].max() + emb[:, 0].min()) / 2; r1 = emb[:, 0].max() - emb[:, 0].min()
    c2 = (emb[:, 1].max() + emb[:, 1].min()) / 2; r2 = emb[:, 1].max() - emb[:, 1].min()
    half = 0.72 * max(r1, r2)                       # square domain so maps are not distorted
    p1 = np.linspace(c1 - half, c1 + half, n); p2 = np.linspace(c2 - half, c2 + half, n)
    P1, P2 = np.meshgrid(p1, p2)
    gexpr = pca.inverse_transform(np.vstack([P1.ravel(), P2.ravel()]).T)
    Egrid = B._energy_grid_2d(ao.var_names, "oscillator_circuit", W, I, gamma, gexpr, P1.shape)
    vel = B.hill(gexpr) @ W.T + I - gamma * gexpr
    vpca = vel @ pca.components_.T
    U = vpca[:, 0].reshape(P1.shape); V = vpca[:, 1].reshape(P2.shape)
    re = np.zeros((len(gexpr), 3)); im = np.zeros((len(gexpr), 3))
    for k, x in enumerate(gexpr):
        sp = B.hill_prime(x)
        ev = np.linalg.eigvals(W * sp[None, :] - np.diag(gamma))
        idx = np.argsort(np.real(ev))[::-1]
        re[k] = np.real(ev[idx]); im[k] = np.imag(ev[idx])
    return dict(pca=pca, emb=emb, p1=p1, p2=p2, P1=P1, P2=P2, E=Egrid, U=U, V=V,
                re=[re[:, i].reshape(P1.shape) for i in range(3)],
                im=[im[:, i].reshape(P1.shape) for i in range(3)])


def draw_energy3d(ax, F, tarr, *, fs_lab=7, fs_tick=5, labelpad=-9, tickpad=-3, s=3,
                  fs_title=8, nbins=None, zoom=None):
    ax.plot_surface(F["P1"], F["P2"], F["E"], cmap="viridis", alpha=0.9, linewidth=0,
                    antialiased=True, rcount=60, ccount=60)
    z = griddata((F["P1"].ravel(), F["P2"].ravel()), F["E"].ravel(),
                 (F["emb"][:, 0], F["emb"][:, 1]), method="cubic")
    ax.scatter(F["emb"][:, 0], F["emb"][:, 1], z, color="white", s=s, depthshade=True)
    ax.view_init(elev=34, azim=-58)
    ax.set_xlabel("PC1", fontsize=fs_lab, labelpad=labelpad)
    ax.set_ylabel("PC2", fontsize=fs_lab, labelpad=labelpad)
    ax.set_zlabel("energy", fontsize=fs_lab, labelpad=labelpad)
    ax.tick_params(labelsize=fs_tick, pad=tickpad)
    if nbins is not None:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(mpl.ticker.MaxNLocator(nbins))
    if zoom is not None:
        ax.set_box_aspect(None, zoom=zoom)
    ax.set_title("PCA energy landscape", fontsize=fs_title)


def draw_energy_flow(ax, F, tarr, *, density=1.2, stream_lw=0.6, arrowsize=0.8, s=3,
                     fs_title=8):
    cf = ax.contourf(F["P1"], F["P2"], F["E"], levels=28, cmap="viridis")
    ax.streamplot(F["p1"], F["p2"], F["U"], F["V"], color="white", density=density,
                  linewidth=stream_lw, arrowsize=arrowsize)
    ax.scatter(F["emb"][:, 0], F["emb"][:, 1], color="white", s=s, alpha=0.5,
               edgecolor="none", zorder=3)   # data extent, neutral (no extra colormap)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("energy contour + flow", fontsize=fs_title)
    return cf


def draw_jac_pca(fig, spec, F):
    inner = spec.subgridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.06, hspace=0.22)
    # signed -> diverging RdBu_r (same encoding as W and the toggle Jacobian); all-blue here
    # correctly reads as "stable everywhere" (the oscillation is in the imaginary parts).
    vmax_r = max(np.max(np.abs(m)) for m in F["re"])
    vmax_i = max(np.max(np.abs(m)) for m in F["im"])   # imag: diverging, rotation frequency
    for i in range(3):
        axr = fig.add_subplot(inner[0, i])
        axr.contourf(F["P1"], F["P2"], F["re"][i], levels=30, cmap="RdBu_r",
                     vmin=-vmax_r, vmax=vmax_r)
        axr.scatter(F["emb"][:, 0], F["emb"][:, 1], c="k", s=1.2, alpha=0.35)
        axr.set_aspect("equal"); axr.set_xticklabels([]); axr.tick_params(labelsize=6)
        axr.set_title(f"Re $\\lambda_{i+1}$", fontsize=8)
        if i == 0:
            axr.set_ylabel("PC2", fontsize=7)
        axi = fig.add_subplot(inner[1, i])
        axi.contourf(F["P1"], F["P2"], F["im"][i], levels=30, cmap="PuOr",
                     vmin=-vmax_i, vmax=vmax_i)
        axi.scatter(F["emb"][:, 0], F["emb"][:, 1], c="k", s=1.2, alpha=0.35)
        axi.set_aspect("equal"); axi.tick_params(labelsize=6)
        axi.set_title(f"Im $\\lambda_{i+1}$", fontsize=8)
        axi.set_xlabel("PC1", fontsize=7)
        if i == 0:
            axi.set_ylabel("PC2", fontsize=7)
    car = fig.add_subplot(inner[0, 3])
    smr = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-vmax_r, vmax_r))
    cb = fig.colorbar(smr, cax=car); cb.set_label("Re $\\lambda$ (stability)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cai = fig.add_subplot(inner[1, 3])
    smi = mpl.cm.ScalarMappable(cmap="PuOr", norm=mpl.colors.Normalize(-vmax_i, vmax_i))
    cb2 = fig.colorbar(smi, cax=cai); cb2.set_label("Im $\\lambda$ (rotation)", fontsize=6)
    cb2.ax.tick_params(labelsize=5.5)


# --------------------------------------------------------------------------- #
# shared inputs
# --------------------------------------------------------------------------- #
def load_inputs():
    """Cached fits, the two circuits, the simulated oscillator cells and the PCA fields."""
    z = np.load(f"{B.DATA}/fits.npz", allow_pickle=True)
    res_t, res_o = z["res_t"].item(), z["res_o"].item()
    toggle = ToggleCircuit(a=5.0, b=4.0)
    osc = OscillatorCircuit(alpha=10.0, n=4)
    ao = simulate_circuit(osc, transient_fraction=0.0, n_trajectories=50, points_per_trajectory=50)
    ppt = ao.uns["ground_truth"]["points_per_trajectory"]
    tarr = np.tile(np.linspace(0, 1, ppt), ao.n_obs // ppt)
    F = repr_pca_fields(ao, res_o["full"])
    return res_t, res_o, toggle, osc, tarr, F


# --------------------------------------------------------------------------- #
# journal-page rendering (--submission)
#
# Same ten panels, same letters, same colormaps; only the arrangement changes. Panels are
# placed by an explicit millimeter table rather than by nested gridspecs, because the two
# spectral blocks (e and j) hold square maps whose height is set by their width, and a
# colorbar has to end up exactly as tall as the maps it belongs to.
# --------------------------------------------------------------------------- #
SUB_LEFT = 9.5             # left edge of the panel column, mm
SUB_RIGHT = 170.0          # right edge; the 10 mm beyond it carries colorbar tick labels
SUB_FS = dict(title=7.0, tick=6.0, small=6.0, cb=6.0, cbtick=5.5, letter=8.0, header=8.0)


def _rect(fig, left, right, top, bottom):
    """A SubplotSpec covering a rectangle of the page given in millimeters from top left."""
    return fig.add_gridspec(1, 1, left=left / SUB_W_MM, right=right / SUB_W_MM,
                            top=1 - top / SUB_H_MM, bottom=1 - bottom / SUB_H_MM)[0, 0]


def _ax(fig, left, right, top, bottom, **kw):
    return fig.add_subplot(_rect(fig, left, right, top, bottom), **kw)


def _text_mm(fig, x, y, s, **kw):
    fig.text(x / SUB_W_MM, 1 - y / SUB_H_MM, s, **kw)


def sub_letter(fig, x_mm, y_mm, s):
    """Panel letter: bold, lowercase, 8 pt, just above and left of the panel."""
    _text_mm(fig, x_mm, y_mm, s, fontweight="bold", fontsize=SUB_FS["letter"],
             va="bottom", ha="left")


def draw_jac_toggle_row(fig, left, right, top, res, row_bottom=None):
    """e on one row: the two eigenvalues, two couplings each, one colorbar per eigenvalue.

    Returns the row's bottom edge in mm. The grid is [map map cbar | map map cbar], which
    buys back the height the two-row version spends and keeps the maps square.
    """
    couplings = (-2.5, -8.0)
    X, Y, E = toggle_jac_fields(res, couplings=couplings)
    # column 3 is an empty spacer: it holds the first colorbar's tick labels and title away
    # from the second pair's y axis.
    ratios = [1, 1, 0.10, 0.26, 1, 1, 0.10]
    wspace = 0.16
    n = len(ratios)
    unit = (right - left) / (sum(ratios) * (1 + (n - 1) * wspace / n))
    # Square maps set the block height; centre that block in the row so it lines up with the
    # taller panel beside it instead of hanging from the top edge.
    if row_bottom is not None and row_bottom - top > unit:
        top = top + (row_bottom - top - unit) / 2.0
    bottom = top + unit
    inner = _rect(fig, left, right, top, bottom).subgridspec(
        1, n, width_ratios=ratios, wspace=wspace)
    cols = [(0, 1, 2), (4, 5, 6)]
    for k in range(2):                                   # k = 0 -> lambda1, 1 -> lambda2
        m = max(np.max(np.abs(E[c][k])) for c in couplings)
        c0, c1, ccb = cols[k]
        for col, c in zip((c0, c1), couplings):
            ax = fig.add_subplot(inner[0, col])
            ax.contourf(X, Y, E[c][k], levels=30, cmap="RdBu_r", vmin=-m, vmax=m)
            if E[c][k].min() < 0 < E[c][k].max():
                ax.contour(X, Y, E[c][k], levels=[0.0], colors="k", linewidths=0.7)
            ax.set_aspect("equal")
            ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4])
            ax.tick_params(labelsize=SUB_FS["tick"], pad=1.5)
            ax.set_title(f"$\\lambda_{k+1}$, $c={c:g}$", fontsize=SUB_FS["title"], pad=2.5)
            ax.set_xlabel("$x_1$", labelpad=1)
            if col == c0:
                ax.set_ylabel("$x_2$", labelpad=1)
            else:
                ax.set_yticklabels([])
        cax = fig.add_subplot(inner[0, ccb])
        sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-m, m))
        cb = fig.colorbar(sm, cax=cax)
        cax.set_title(f"Re $\\lambda_{k+1}$", fontsize=SUB_FS["cb"], pad=2.0)
        cb.ax.tick_params(labelsize=SUB_FS["cbtick"], pad=1.2, length=1.5)
        cb.outline.set_linewidth(0.4)
    return bottom


def draw_jac_pca_row(fig, left, right, top, F):
    """j on one row: Re then Im of the three eigenvalues, one colorbar per part."""
    ratios = [1, 1, 1, 0.09, 0.55, 1, 1, 1, 0.09]     # column 4 is the spacer, as in e
    wspace = 0.13
    n = len(ratios)
    unit = (right - left) / (sum(ratios) * (1 + (n - 1) * wspace / n))
    bottom = top + unit
    inner = _rect(fig, left, right, top, bottom).subgridspec(
        1, n, width_ratios=ratios, wspace=wspace)
    # signed -> diverging RdBu_r (same encoding as W and the toggle Jacobian); all-blue here
    # correctly reads as "stable everywhere" (the oscillation is in the imaginary parts).
    vmax_r = max(np.max(np.abs(m)) for m in F["re"])
    vmax_i = max(np.max(np.abs(m)) for m in F["im"])   # imag: diverging, rotation frequency
    blocks = [(F["re"], "RdBu_r", vmax_r, "Re", (0, 1, 2), 3, "Re $\\lambda$ (stability)"),
              (F["im"], "PuOr", vmax_i, "Im", (5, 6, 7), 8, "Im $\\lambda$ (rotation)")]
    for maps, cmap, vmax, part, cols, ccb, cblabel in blocks:
        for i, col in enumerate(cols):
            ax = fig.add_subplot(inner[0, col])
            ax.contourf(F["P1"], F["P2"], maps[i], levels=30, cmap=cmap,
                        vmin=-vmax, vmax=vmax)
            ax.scatter(F["emb"][:, 0], F["emb"][:, 1], c="k", s=0.5, alpha=0.3,
                       linewidths=0)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=SUB_FS["tick"], pad=1.5)
            ax.set_title(f"{part} $\\lambda_{i+1}$", fontsize=SUB_FS["title"], pad=2.5)
            ax.set_xlabel("PC1", labelpad=1)
            if i == 0:
                ax.set_ylabel("PC2", labelpad=1)
            else:
                ax.set_yticklabels([])
        cax = fig.add_subplot(inner[0, ccb])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(-vmax, vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(cblabel, fontsize=SUB_FS["cb"], labelpad=1.5)
        cb.ax.tick_params(labelsize=SUB_FS["cbtick"], pad=1.2, length=1.5)
        cb.outline.set_linewidth(0.4)
    return bottom


def main_submission(out_path=SUB_OUT):
    """The journal rendering: 180 mm wide, one page tall, nothing below 5 pt."""
    from submission_style import figure_for, save as save_submission

    use_style(7)                       # font family and palette; sizes come from figure_for
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res_t, res_o, toggle, osc, tarr, F = load_inputs()

    img_toggle = render_tikz(TIKZ_TOGGLE_SUB)
    img_repr = render_tikz(TIKZ_REPRESSILATOR_SUB)
    if img_toggle is None or img_repr is None:
        print("   note: TikZ/LaTeX render unavailable, using matplotlib circuit fallback")

    fig = figure_for("double", height_mm=SUB_H_MM)
    L, R = SUB_LEFT, SUB_RIGHT

    # ---- toggle switch -------------------------------------------------------------
    _text_mm(fig, L, 6.0, "Toggle switch", ha="left", va="bottom",
             fontsize=SUB_FS["header"], fontweight="bold")

    r1_top, r1_bot = 13.0, 46.0
    ax_a = _ax(fig, L, L + 40, r1_top, r1_bot); draw_toggle_circuit(ax_a, img_toggle)
    ax_b = _ax(fig, 57.0, 108.0, r1_top, r1_bot)
    draw_phase(ax_b, toggle, res_t, fs_leg=5.5, fs_cb=SUB_FS["cb"], fs_cbtick=SUB_FS["cbtick"],
               fs_title=SUB_FS["title"], star_ms=7, saddle_ms=3.6, leg_star_ms=6,
               stream_lw=0.35, arrowsize=0.5, title="energy landscape + flow ($b{=}4$)")
    ax_c = _ax(fig, 130.0, R, r1_top, r1_bot)
    draw_bifurcation(ax_c, toggle, lw=1.1, fs_leg=5.5, fs_mark=6, fs_title=SUB_FS["title"])
    ax_b.tick_params(labelsize=SUB_FS["tick"]); ax_c.tick_params(labelsize=SUB_FS["tick"])

    r2_top, r2_bot = 59.0, 92.0
    draw_W_recovery(fig, _rect(fig, L, L + 42, r2_top, r2_bot), res_t, ["$x_1$", "$x_2$"], 5.0,
                    annot_fs=7.5, width_ratios=(2.0, 0.72), hspace=0.55, wspace=0.12,
                    tick_fs=SUB_FS["tick"], title_fs=SUB_FS["title"], small_title_fs=6.0,
                    small_pad=1.2)
    e_bot = draw_jac_toggle_row(fig, 58.0, R, r2_top, res_t, row_bottom=r2_bot)

    # ---- repressilator -------------------------------------------------------------
    _text_mm(fig, L, 101.0, "Repressilator", ha="left", va="bottom",
             fontsize=SUB_FS["header"], fontweight="bold")

    r3_top, r3_bot = 109.0, 142.0
    ax_f = _ax(fig, L, L + 38, r3_top, r3_bot); draw_repressilator_circuit(ax_f, img_repr)
    ax_g1 = _ax(fig, 52.0, 94.0, r3_top - 2, r3_bot + 2, projection="3d")
    draw_limitcycle(ax_g1, osc, fs_lab=6.5, fs_tick=5.5, labelpad=-5, tickpad=-2, lw=1.0,
                    fs_title=SUB_FS["title"], nbins=3, zoom=1.12)
    ax_g2 = _ax(fig, 110.0, R, r3_top, r3_bot)
    draw_oscillations(ax_g2, osc, lw=0.9, fs_leg=5.5, fs_title=SUB_FS["title"], t_hi=40)
    ax_g2.tick_params(labelsize=SUB_FS["tick"])

    r4_top, r4_bot = 155.0, 189.0
    draw_W_recovery(fig, _rect(fig, L, L + 44, r4_top, r4_bot), res_o, ["$x$", "$y$", "$z$"],
                    10.0, annot_fs=7.0, width_ratios=(2.0, 0.72), hspace=0.55, wspace=0.12,
                    tick_fs=SUB_FS["tick"], title_fs=SUB_FS["title"], small_title_fs=6.0,
                    small_pad=1.2)
    ax_i1 = _ax(fig, 58.0, 102.0, r4_top - 2, r4_bot + 2, projection="3d")
    draw_energy3d(ax_i1, F, tarr, fs_lab=6.5, fs_tick=5.5, labelpad=-5, tickpad=-2, s=1.5,
                  fs_title=SUB_FS["title"], nbins=3, zoom=1.12)
    ax_i2 = _ax(fig, 116.0, 116.0 + (r4_bot - r4_top), r4_top, r4_bot)
    cf = draw_energy_flow(ax_i2, F, tarr, density=0.9, stream_lw=0.45, arrowsize=0.6, s=1.2,
                          fs_title=SUB_FS["title"])
    ax_i2.tick_params(labelsize=SUB_FS["tick"], pad=1.5)
    cax_i = _ax(fig, 153.0, 155.5, r4_top, r4_bot)
    cb = fig.colorbar(cf, cax=cax_i); cb.set_label("energy", fontsize=SUB_FS["cb"], labelpad=1.5)
    cb.ax.tick_params(labelsize=SUB_FS["cbtick"], pad=1.2, length=1.5)
    cb.outline.set_linewidth(0.4)

    # ---- j Jacobian-in-PCA spectral maps -------------------------------------------
    r5_top = 202.0
    r5_bot = draw_jac_pca_row(fig, L, R, r5_top, F)

    for x_mm, y_mm, s in [(L - 2.0, r1_top - 1.5, "a"), (55.0, r1_top - 1.5, "b"),
                          (128.0, r1_top - 1.5, "c"),
                          (L - 2.0, r2_top - 1.5, "d"), (56.0, r2_top - 1.5, "e"),
                          (L - 2.0, r3_top - 1.5, "f"), (50.0, r3_top - 1.5, "g"),
                          (L - 2.0, r4_top - 1.5, "h"), (56.0, r4_top - 1.5, "i"),
                          (L - 2.0, r5_top - 1.5, "j")]:
        sub_letter(fig, x_mm, y_mm, s)

    save_submission(fig, out_path)
    print(f"wrote {out_path}  ({SUB_W_MM:.0f} x {SUB_H_MM:.0f} mm canvas, "
          f"panels e and j end at {e_bot:.1f} / {r5_bot:.1f} mm)")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def main():
    use_style(9)
    os.makedirs(OUT, exist_ok=True)

    res_t, res_o, toggle, osc, tarr, F = load_inputs()

    img_toggle = render_tikz(TIKZ_TOGGLE)
    img_repr = render_tikz(TIKZ_REPRESSILATOR)
    if img_toggle is None or img_repr is None:
        print("   note: TikZ/LaTeX render unavailable, using matplotlib circuit fallback")

    fig = plt.figure(figsize=(12.5, 19.6))
    # three gridspecs: toggle section, repressilator section, and panel j. Keeping them
    # separate lets the section-to-section spacing and j's height be set independently
    # (a single gridspec would let j's large height inflate every row gap). j is given a
    # modest height and a slightly narrower span so its square maps stay compact.
    gs_tog = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.0], hspace=0.26,
                              top=0.95, bottom=0.655, left=0.05, right=0.93)
    gs_rep = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.0], hspace=0.26,
                              top=0.61, bottom=0.335, left=0.05, right=0.93)
    gs_j = fig.add_gridspec(1, 1, top=0.29, bottom=0.03, left=0.13, right=0.87)

    # ---- toggle row 1: a circuit | b phase | c bifurcation ----
    r0 = gs_tog[0].subgridspec(1, 3, width_ratios=[0.72, 1.2, 1.12], wspace=0.3)
    ax_a = fig.add_subplot(r0[0]); draw_toggle_circuit(ax_a, img_toggle)
    ax_b = fig.add_subplot(r0[1]); draw_phase(ax_b, toggle, res_t)
    ax_c = fig.add_subplot(r0[2]); draw_bifurcation(ax_c, toggle)

    # ---- toggle row 2: d W-recovery | e Jacobians ----
    r1 = gs_tog[1].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.26)
    draw_W_recovery(fig, r1[0], res_t, ["$x_1$", "$x_2$"], 5.0)
    draw_jac_toggle(fig, r1[1], res_t)

    # ---- repressilator row 1: f circuit | g 3D limit cycle | g oscillations ----
    r2 = gs_rep[0].subgridspec(1, 3, width_ratios=[0.8, 1.0, 1.2], wspace=0.3)
    ax_f = fig.add_subplot(r2[0]); draw_repressilator_circuit(ax_f, img_repr)
    ax_g1 = fig.add_subplot(r2[1], projection="3d"); draw_limitcycle(ax_g1, osc)
    ax_g2 = fig.add_subplot(r2[2]); draw_oscillations(ax_g2, osc)

    # ---- repressilator row 2: h W-recovery | i 3D energy | i 2D flow ----
    r3 = gs_rep[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.28)
    draw_W_recovery(fig, r3[0], res_o, ["$x$", "$y$", "$z$"], 10.0)
    ax_i1 = fig.add_subplot(r3[1], projection="3d"); draw_energy3d(ax_i1, F, tarr)
    ax_i2 = fig.add_subplot(r3[2]); cf = draw_energy_flow(ax_i2, F, tarr)
    cb = fig.colorbar(cf, ax=ax_i2, fraction=0.046, pad=0.04); cb.set_label("energy", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    # ---- j Jacobian-in-PCA spectral maps ----
    draw_jac_pca(fig, gs_j[0], F)

    # aligned panel letters: each at its cell's top-left in figure coords, so letters line
    # up in height within a row and the leftmost letters line up in x across rows.
    for spec, s in [(r0[0], "a"), (r0[1], "b"), (r0[2], "c"), (r1[0], "d"), (r1[1], "e"),
                    (r2[0], "f"), (r2[1], "g"), (r3[0], "h"), (r3[1], "i"), (gs_j[0], "j")]:
        bb = spec.get_position(fig)
        lx = 0.044 if s == "j" else bb.x0 - 0.006   # keep j's letter in the aligned left column
        fig.text(lx, bb.y1 + 0.004, s, fontweight="bold", fontsize=12, va="bottom", ha="right")
    fig.text(0.05, 0.976, "Toggle switch", ha="left", va="bottom", fontsize=14, fontweight="bold")
    fig.text(0.05, 0.632, "Repressilator", ha="left", va="bottom", fontsize=14, fontweight="bold")

    save(fig, f"{OUT}/{NAME}", formats=("pdf", "png"))
    print(f"wrote {OUT}/{NAME}.pdf + .png")
    plt.close(fig)


INK_HDR = "#222222"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", action="store_true",
                    help=f"render the journal page version to {SUB_OUT}")
    args = ap.parse_args()
    main_submission() if args.submission else main()
