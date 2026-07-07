"""Convert the assembled markdown manuscript to a compilable LaTeX document,
placing figures at the [Figure N] section markers."""
import re

MD = "../ManuscriptIdeaJesper/scHopfield_Manuscript_assembled.md"
OUT = "main.tex"

FIGS = {
    1: ("figures/fig1.png", "Overview of scHopfield. Single-cell expression and RNA velocity are used to fit Hill activation functions; a cell-type-specific interaction matrix is inferred either unconstrained (pseudoinverse) or scaffold-guided; the resulting dynamical system yields an energy landscape, Jacobian-based stability, and in-silico perturbation.", 0.9),
    2: ("figures/fig2.png", "Continuous Hopfield dynamics capture canonical behaviors. Toggle-switch energy landscape and flow across increasing mutual inhibition, showing the bifurcation from a monostable to a bistable regime. On these circuits scHopfield recovers the ground-truth network with edge-sign accuracy 1.00 and outperforms GENIE3 on larger synthetic networks (AUROC 0.98 vs 0.70; benchmarks M3, M7, M8).", 0.95),
    3: ("figures/fig3.png", "Learned perturbational dynamics recover established lineage regulators. In-silico knockout expression-shift projected onto the embedding. Across a panel of hematopoietic master regulators scHopfield predicted the correct lineage-shift direction for 10 of 10 factors (benchmark M4).", 0.7),
    4: ("figures/fig4.png", "Driver identification is robust to network and regularization choices. Perturbation-based drivers are stable across two base networks and three scaffold-regularization regimes (Jaccard 0.67) while the static network score is unstable (0.20); the no-scaffold pseudoinverse is an outlier that loses canonical drivers (benchmarks M5, M6).", 0.95),
    5: ("figures/fig5.png", "Lineage-specific stability during pancreatic endocrinogenesis. Distribution of positive real Jacobian eigenvalues by cell type: Delta and Epsilon cells are least stable, whereas Pre-endocrine, Alpha, and Beta cells occupy more stable regimes.", 0.8),
    6: ("figures/fig6.png", "Higher-order perturbational simulations. Dose-response of lineage bias to graded perturbation magnitude, showing coherent nonlinear responses. scHopfield nominates candidate higher-order interactions (for example Stat3) as experimentally testable hypotheses.", 0.8),
}


UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "−": "-", "→": r"$\to$",
       "·": r"$\cdot$", "∈": r"$\in$", "∑": r"$\sum$", "∫": r"$\int$", "∇": r"$\nabla$",
       "⊙": r"$\odot$", "≈": r"$\approx$", "∼": r"$\sim$", "√": r"$\surd$", "∞": r"$\infty$",
       "σ": r"$\sigma$", "γ": r"$\gamma$", "φ": r"$\varphi$", "τ": r"$\tau$", "λ": r"$\lambda$",
       "ρ": r"$\rho$", "Δ": r"$\Delta$", "α": r"$\alpha$", "β": r"$\beta$", "μ": r"$\mu$",
       "θ": r"$\theta$", "π": r"$\pi$", "Σ": r"$\Sigma$", "Φ": r"$\Phi$", "ε": r"$\varepsilon$",
       "“": "``", "”": "''", "‘": "`", "’": "'", "–": "--", "…": r"\ldots{}", "≠": r"$\neq$"}


NOTATION = [
    (r"W\^\(C\)", "W^{(C)}"), (r"R-squared", "R^2"),
    (r"sigma_j\(x_j\)", r"\sigma_j(x_j)"), (r"\bsigma_j\b", r"\sigma_j"),
    (r"\bn_j\b", "n_j"), (r"\bk_j\b", "k_j"), (r"\bgamma_i\b", r"\gamma_i"),
    (r"\bI_i\b", "I_i"), (r"\bx_j\b", "x_j"), (r"\bx_i\b", "x_i"),
]


def esc(t):
    # escape LaTeX specials in body text (no math mode here)
    t = t.replace("\\", "")  # drop stray backslashes
    # protect known inline notation as math placeholders (survive escaping)
    store = []
    def stash(latex):
        store.append("$" + latex + "$"); return f"@@N{len(store)-1}@@"
    for pat, latex in NOTATION:
        t = re.sub(pat, lambda m, l=latex: stash(l), t)
    for u, r in UNI.items():
        t = t.replace(u, r)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("^", r"\textasciicircum{}"), ("~", r"\textasciitilde{}"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$"), ("+/-", r"$\pm$")]:
        t = t.replace(a, b)
    # markdown bold/italic
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    # restore protected inline math
    t = re.sub(r"@@N(\d+)@@", lambda m: store[int(m.group(1))], t)
    return t


def fig_block(n):
    path, cap, w = FIGS[n]
    return (f"\n\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width={w}\\linewidth]{{{path}}}\n"
            f"\\caption{{{esc(cap)}}}\n\\label{{fig:{n}}}\n\\end{{figure}}\n")


def methods_tex(path="../docs/methods/scHopfield - Methods Final.md"):
    """Math-aware markdown->LaTeX for the Methods (preserves $...$ and $$...$$)."""
    import os
    if not os.path.exists(path):
        return ""
    raw = open(path, encoding="utf-8").read()
    # protect math
    store = []
    def stash(m):
        store.append(m.group(0)); return f"@@M{len(store)-1}@@"
    raw = re.sub(r"\$\$.*?\$\$", stash, raw, flags=re.S)
    raw = re.sub(r"\$[^$\n]+\$", stash, raw)
    out = ["\n\\clearpage\n\\section*{Methods}\n"]
    for ln in raw.split("\n"):
        s = ln.rstrip()
        if s.startswith("# Methods"):
            continue
        if s.startswith("### "):
            out.append(f"\n\\subsubsection*{{{esc(s[4:])}}}\n")
        elif s.startswith("## "):
            out.append(f"\n\\subsection*{{{esc(s[3:])}}}\n")
        elif s.startswith("---") or s.startswith("## References"):
            if s.startswith("## References"):
                break
            continue
        else:
            out.append(esc(s) + "\n" if s.strip() else "\n")
    text = "".join(out)
    # restore math: $$..$$ -> equation; $..$ stays inline
    def restore(m):
        i = int(m.group(1)); frag = store[i]
        if frag.startswith("$$"):
            body = frag.strip("$").strip()
            body = re.sub(r"\\tag\{[^}]*\}", "", body)
            return "\n\\begin{equation}\n" + body + "\n\\end{equation}\n"
        return frag
    text = re.sub(r"@@M(\d+)@@", restore, text)
    return text


def supp_tex():
    return r"""
\clearpage
\section*{Supplementary Information}
\subsection*{S1. Reproducibility}
Cluster-specific inference is seeded. On the pancreatic dataset (3696 cells, 300 genes),
two fits with the same seed were bit-identical (interaction-matrix and out-strength
centrality correlations 1.000); unseeded fits were numerically stable in $W$ (Pearson
0.999) but their gene rankings drifted (centrality Spearman 0.73--0.93 across five seeds).
\begin{figure}[htbp]\centering
\includegraphics[width=0.7\linewidth]{figures/supp_repro.png}
\caption{Reproducibility of seeded vs unseeded inference (benchmark M1).}\end{figure}

\subsection*{S2. GRN recovery vs GENIE3}
On four synthetic 40-gene Hopfield networks with known ground truth, scHopfield recovered
edges at AUROC $0.975\pm0.018$ / AUPRC $0.970$ versus GENIE3 (expression-only ExtraTrees)
AUROC $0.701\pm0.025$ / AUPRC $0.240$ (benchmark M8).
\begin{figure}[htbp]\centering
\includegraphics[width=0.55\linewidth]{figures/supp_genie3.png}
\caption{scHopfield vs GENIE3 GRN edge recovery.}\end{figure}

\subsection*{S3. Robustness of driver identification}
Across two mouse base networks and three scaffold-regularization regimes, perturbation-based
drivers were stable (mean pairwise Jaccard of top-15 genes 0.67) while the static network
score was unstable (0.20); an unconstrained pseudoinverse fit was an outlier (0.36) that
dropped Gata1/Klf1 (benchmarks M5, M6).
\begin{figure}[htbp]\centering
\includegraphics[width=0.9\linewidth]{figures/supp_sensitivity.png}
\caption{Network and regularization sensitivity of driver identification.}\end{figure}
Why the scaffold matters is clarified by an identifiability analysis on the same data
(benchmark M12): at a fixed cell count, adding neighbouring (off-manifold) cells raises the
effective rank of the sigmoid design matrix (participation ratio 6.0 to 7.1), confirming that
broader state-space coverage improves identifiability. However, real single-cell expression
is intrinsically low-rank (participation ratio only about 6--7 for 100 genes), so the
unconstrained interaction matrix remains underdetermined (split-half correlation near zero)
regardless of coverage. The data alone therefore cannot determine the network, which is the
concrete reason a transcription-factor scaffold prior is required for identifiable inference.
\begin{figure}[htbp]\centering
\includegraphics[width=0.6\linewidth]{figures/supp_identifiability.png}
\caption{Real-data identifiability: neighbour augmentation raises the effective rank of
$\sigma(X)$, but the data is too low-rank to determine $W$ without a scaffold (benchmark M12).}
\end{figure}

\subsection*{S4. Biophysical circuits and identifiability}
The dissertation oscillator is Hopfield-form and recovers at correlation 1.000. The Novak
cell-cycle and Adlung JAK-STAT circuits are not of Hopfield form (no true interaction
matrix) but the Hill model represents their velocity fields at $R^2=1.000$ (Hill-only basis
$R^2=0.98$/$0.99$; benchmark M9). For these systems we asked whether the fitted interaction
matrix recovers a meaningful \emph{effective} regulatory network, defined as the sign of the
average off-diagonal Jacobian of the true system. On data confined to a low-dimensional
trajectory, the effective-GRN sign-accuracy was at chance (0.47/0.49), but with broad
state-space sampling it rose to 0.85 (Novak) and 0.98 (Adlung) (benchmark M10). Thus the
limiting factor for these systems is \emph{identifiability} of the interaction matrix, not
the expressiveness of the additive-Hill model: trajectory-confined data underdetermine the
network, whereas broad coverage of state space (which the neighbour-augmented inference
provides on real data) resolves it, as does a transcription-factor scaffold prior (S3). We
also implemented an optional Jacobian-consistency regulariser that pulls the model's local
sensitivity toward a neighbour-estimated velocity Jacobian; on these circuits it did not
improve recovery (benchmark M11), because neighbour-estimated Jacobian targets on limited,
noisy data are themselves unreliable, so it is provided as an off-by-default option.

\subsection*{S5. Note on the Methods-equation Hill derivative}
The Hill derivative used in the Jacobian is $\varphi'(x)=n\,\varphi(1-\varphi)/x$; the
factor $n$ (present in the implementation) should be shown explicitly in Eq. (4) and
Eq. (21) (benchmark M2).

\subsection*{S6. Generalization across developmental systems}
The main findings reproduce across independent developmental datasets (hematopoiesis,
pancreatic endocrinogenesis, murine neural crest, human limb).
\emph{Identifiability (M13).} The neighbour-augmentation effect of S3 holds in every system:
adding off-manifold neighbour cells raises the effective rank of $\sigma(X)$ (hematopoiesis
6.0 to 7.1, pancreas 13.9 to 15.5, murine neural crest 19.4 to 23.1, human limb 18.2 to
19.5), while the split-half correlation of the unconstrained $W$ stays near zero in all
four, confirming that real single-cell data are intrinsically low-rank and require a
scaffold prior.
\emph{Stability (M14).} The energy-landscape and Jacobian-stability analysis reproduces the
progenitor-instability to terminal-stability ordering in murine neural crest: progenitor
states (PNS glia and neurons) have positive leading Jacobian eigenvalues, whereas terminal
states (myelinating Schwann cells, melanocytes) are more stable with deeper energy wells.
\emph{Perturbation (M15).} The known-driver knockout validation generalizes to neural crest:
for bona-fide transcription-factor masters the predicted glia-versus-neuron shift direction
is correct in 4 of 4 cases (Sox10 toward neuron; Neurod1, Isl1, Pou4f1 toward glia); the full
eight-gene panel scores 5 of 8, the three misses being non-transcription-factor genes
(a receptor and two myelin structural proteins) that are not regulators in the model and
produce near-zero effects. Overall accuracy is lower than in hematopoiesis, reflecting the
less cleanly separable glia-neuron programme and the use of a genome-wide scaffold prior.
\begin{figure}[htbp]\centering
\includegraphics[width=0.9\linewidth]{figures/supp_multi_ident.png}
\caption{Identifiability across four developmental systems (M13): neighbour augmentation
raises the effective rank of $\sigma(X)$ everywhere, but split-half $W$ stability stays near
zero, so a scaffold prior is required in all systems.}\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=0.95\linewidth]{figures/supp_murine_energy.png}
\caption{Energy and Jacobian-stability analysis on murine neural crest (M14): progenitor
states are less stable than terminal states.}\end{figure}

\subsection*{S7. Robustness to the velocity estimator}
scHopfield regresses against an estimated RNA velocity, so we tested whether its downstream
outputs depend on the upstream estimator rather than on how similar the velocity estimates
themselves are. On pancreatic endocrinogenesis, murine neural crest, and a dynamo-processed
hematopoiesis dataset, we refit the model against scVelo, dynamo, and pseudotime-derived
velocities on identical cells and genes, with the activation functions and scaffold held
fixed (benchmarks M25--M28). For every estimator the fitted velocity projected to the same
progenitor-to-terminal differentiation streams (Fig. S7), and the top structural drivers were
largely shared where the estimators agreed (top-25 Jaccard 0.90 on neural crest), though less
so on hematopoiesis where the estimators disagreed most. The estimator-induced variation fell
on the individual interaction weights rather than on the projected flow or the well-determined
driver set, so we report driver-level and landscape-level conclusions rather than individual
edge weights. Because the pseudotime estimator requires only a cell ordering, scHopfield also
applies to datasets without spliced and unspliced counts.
\begin{figure}[htbp]\centering
\includegraphics[width=0.95\linewidth]{figures/supp_velocity_sources.png}
\caption{Robustness to the velocity estimator on pancreatic endocrinogenesis (benchmarks
M25--M28). The fitted velocity field (bottom row) projects to the same progenitor-to-terminal
flow whether the model is trained on scVelo, dynamo, or pseudotime-derived velocity (top row),
while the estimator-induced differences fall on the low-level interaction weights rather than
the qualitative dynamics.}\end{figure}
"""


def main():
    lines = open(MD, encoding="utf-8").read().split("\n")
    body = []
    i = 0
    # skip until Abstract
    while i < len(lines) and not lines[i].startswith("## Abstract"):
        i += 1
    para = []

    def flush():
        if para:
            text = " ".join(x.strip() for x in para).strip()
            if text:
                body.append(esc(text) + "\n")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## Figure pointers") or ln.startswith("## Outstanding") or ln.startswith("## References"):
            flush()
            break
        if ln.startswith("### "):
            flush()
            m = re.search(r"\[Figure (\d+)\]", ln)
            title = re.sub(r"\*?\[Figure \d+\]\*?", "", ln[4:]).strip().rstrip("*").strip()
            body.append(f"\n\\subsection*{{{esc(title)}}}\n")
            if m:
                body.append(fig_block(int(m.group(1))))
        elif ln.startswith("## "):
            flush()
            title = ln[3:].strip()
            if title == "Abstract":
                body.append("\n\\begin{abstract}\n")
                para.append("__ABSTRACT_OPEN__")
                para.clear()
                body.append("__ABSTRACT__")
            elif title == "Main":
                body.append("\n\\section*{Introduction}\n")
            else:
                body.append(f"\n\\section*{{{esc(title)}}}\n")
        elif ln.strip() == "---":
            flush()
        elif ln.strip().startswith("dx_i/dt"):
            flush()
            body.append(r"\begin{equation}\frac{dx_i}{dt} = \sum_j W_{ij}^{(C)}\,\sigma_j(x_j) - \gamma_i x_i + I_i,\end{equation}" + "\n")
        else:
            para.append(ln)
        i += 1
    flush()

    text = "".join(body)
    # close abstract: the abstract paragraphs were emitted right after the marker; wrap them
    text = text.replace("__ABSTRACT__", "", 1)
    # find abstract content: between \begin{abstract} and next \section
    text = re.sub(r"(\\begin\{abstract\})\n(.*?)(\n\\section)", r"\1\n\2\n\\end{abstract}\3", text, count=1, flags=re.S)

    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{times}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\title{\textbf{scHopfield: Interpretable dynamical systems learning of regulatory landscapes and perturbational responses from single-cell data}}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-2em}
"""
    refs = r"""
\section*{References (working list)}
\small
Hopfield JJ (1982) \textit{PNAS} 79:2554. Hopfield JJ (1984) \textit{PNAS} 81:3088.
Bastidas-Ponce A et al. (2019) \textit{Development} 146:dev173849.
Paul F et al. (2015) \textit{Cell} 163:1663.
Qiu X et al. (2022) \textit{Cell} 185:690 (dynamo).
Kamimoto K et al. (2023) \textit{Nature} 614:742 (CellOracle).
Huynh-Thu VA et al. (2010) \textit{PLoS ONE} 5:e12776 (GENIE3).
Aibar S et al. (2017) \textit{Nat Methods} 14:1083 (SCENIC).
Bergen V et al. (2020) \textit{Nat Biotechnol} 38:1408 (scVelo).
La Manno G et al. (2018) \textit{Nature} 560:494.
Elowitz MB, Leibler S (2000) \textit{Nature} 403:335.
Kingma DP, Ba J (2015) \textit{ICLR} (Adam).
Cannoodt R et al. (2021) \textit{Nat Commun} 12:3942 (dyngen).
"""
    full = preamble + text + refs + methods_tex() + supp_tex() + "\n\\end{document}\n"
    open(OUT, "w", encoding="utf-8").write(full)
    print(f"wrote {OUT} (main {len(text)} chars + Methods + Supplementary)")


if __name__ == "__main__":
    main()
