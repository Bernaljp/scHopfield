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


def esc(t):
    # escape LaTeX specials in body text (no math mode here)
    t = t.replace("\\", "")  # drop stray backslashes
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
    open(OUT, "w", encoding="utf-8").write(preamble + text + refs + "\n\\end{document}\n")
    print(f"wrote {OUT} ({len(text)} chars body)")


if __name__ == "__main__":
    main()
