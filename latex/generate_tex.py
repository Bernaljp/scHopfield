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


def esc(t):
    # escape LaTeX specials in body text (no math mode here)
    t = t.replace("\\", "")  # drop stray backslashes
    for u, r in UNI.items():
        t = t.replace(u, r)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("^", r"\textasciicircum{}"), ("~", r"\textasciitilde{}"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$"), ("+/-", r"$\pm$")]:
        t = t.replace(a, b)
    # markdown bold/italic
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
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

\subsection*{S4. Biophysical circuits}
The dissertation oscillator is Hopfield-form and recovers at correlation 1.000. The Novak
cell-cycle and Adlung JAK-STAT circuits are not of Hopfield form (no true interaction
matrix) but the Hill model represents their velocity fields at $R^2=1.000$ (Hill-only basis
$R^2=0.98$/$0.99$); benchmark M9.

\subsection*{S5. Note on the Methods-equation Hill derivative}
The Hill derivative used in the Jacobian is $\varphi'(x)=n\,\varphi(1-\varphi)/x$; the
factor $n$ (present in the implementation) should be shown explicitly in Eq. (4) and
Eq. (21) (benchmark M2).
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
