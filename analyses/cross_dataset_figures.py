"""Cross-dataset exploratory results from the fitted pipeline outputs.

Reads benchmark_results/pipeline/<dataset>/adata_fitted.h5ad (5 developmental
systems, already fit: energies, Jacobian stats, per-cluster W) and generates
cross-dataset result figures into benchmark_results/cross_dataset/plots/.
No fitting; CPU-only.
"""
import os

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [
    ("hematopoiesis", "paul15_clusters", "mouse"),
    ("pancreas", "clusters", "mouse"),
    ("murine_nc", "celltype_update", "mouse"),
    ("human_limb", "leiden_R_celltype", "human"),
    ("schwann", "location", "mouse"),
]
COL = dict(zip([d[0] for d in DATASETS],
               ["#2a6f97", "#e76f51", "#2a9d8f", "#8338ec", "#fb8500"]))
OUT = "benchmark_results/cross_dataset/plots"
MINCELLS = 20


def collect():
    """Return per-cell obs frames and per-cluster summary rows across datasets."""
    percell, summary = {}, []
    for name, ck, species in DATASETS:
        p = f"benchmark_results/pipeline/{name}/adata_fitted.h5ad"
        if not os.path.exists(p):
            continue
        a = ad.read_h5ad(p)
        lab = a.obs[ck].astype(str)
        keep = [c for c in lab.value_counts().index if (lab == c).sum() >= MINCELLS]
        df = a.obs[["energy_total", "energy_interaction", "energy_degradation",
                    "energy_bias", "jacobian_eig1_real", "jacobian_positive_evals",
                    "jacobian_trace"]].copy()
        df["cluster"] = lab.values
        df = df[df["cluster"].isin(keep)]
        percell[name] = df
        for c in keep:
            m = df["cluster"] == c
            W = np.asarray(a.varp[f"W_{c}"]) if f"W_{c}" in a.varp else None
            spec = float(np.max(np.linalg.eigvals(W).real)) if W is not None else np.nan
            sym = float(np.linalg.norm((W + W.T) / 2) / (np.linalg.norm(W) + 1e-9)) if W is not None else np.nan
            summary.append({
                "dataset": name, "species": species, "cluster": c, "n": int(m.sum()),
                "energy": float(df.loc[m, "energy_total"].median()),
                "eig": float(df.loc[m, "jacobian_eig1_real"].median()),
                "frac_unstable": float((df.loc[m, "jacobian_positive_evals"] > 0).mean()),
                "spectral_radius": spec, "W_symmetry": sym,
                "e_int": float(df.loc[m, "energy_interaction"].abs().median()),
                "e_deg": float(df.loc[m, "energy_degradation"].abs().median()),
                "e_bias": float(df.loc[m, "energy_bias"].abs().median()),
            })
        del a
        print(f"  loaded {name}: {len(keep)} clusters", flush=True)
    return percell, pd.DataFrame(summary)


def facet_box(percell, col, title, fname, hline=None):
    ds = list(percell.keys())
    fig, axes = plt.subplots(1, len(ds), figsize=(4 * len(ds), 4.5), squeeze=False)
    for ax, name in zip(axes[0], ds):
        df = percell[name]
        order = df.groupby("cluster")[col].median().sort_values().index
        data = [df.loc[df["cluster"] == c, col].values for c in order]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=COL[name], alpha=0.6)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=6)
        ax.set_title(name, color=COL[name], fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="crimson", ls="--", lw=0.8)
    axes[0][0].set_ylabel(col)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading fitted datasets ...", flush=True)
    percell, S = collect()

    # 1. stability structure (heterogeneous; magnitudes not comparable across fits)
    facet_box(percell, "jacobian_eig1_real",
              "Local stability (leading Jacobian eigenvalue, Re) per cell type. Sign = stable(-)/unstable(+); "
              "magnitudes are NOT comparable across systems (fit-scale dependent)",
              "1_stability_by_cluster.png", hline=0.0)
    # 2. energy landscape
    facet_box(percell, "energy_total",
              "Energy landscape depth per cell type, across systems",
              "2_energy_by_cluster.png")

    # 3. energy-stability coupling, z-scored WITHIN each dataset (scales differ)
    fig, ax = plt.subplots(figsize=(8, 6))
    Sz = S.copy()
    for name in percell:
        m = Sz["dataset"] == name
        for c in ["energy", "eig"]:
            v = Sz.loc[m, c]
            Sz.loc[m, c] = (v - v.mean()) / (v.std() + 1e-9)
        ax.scatter(Sz.loc[m, "energy"], Sz.loc[m, "eig"], s=60, color=COL[name], label=name, alpha=0.8, edgecolor="k", lw=0.4)
    r = np.corrcoef(Sz["energy"], Sz["eig"])[0, 1]
    ax.axhline(0, color="crimson", ls="--", lw=0.8); ax.axvline(0, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("per-cluster energy (z-scored within dataset)")
    ax.set_ylabel("per-cluster leading Jacobian eig (z-scored within dataset)")
    ax.set_title(f"Energy-stability coupling (within-dataset z-scores, pooled r={r:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/3_energy_stability_coupling.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 4. energy composition (which term dominates)
    fig, ax = plt.subplots(figsize=(9, 5))
    comp = S.groupby("dataset")[["e_int", "e_deg", "e_bias"]].median()
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp = comp.reindex([d[0] for d in DATASETS if d[0] in comp.index])
    bottom = np.zeros(len(comp))
    for lab, c in [("e_int", "#2a6f97"), ("e_deg", "#e9c46a"), ("e_bias", "#d1495b")]:
        ax.bar(comp.index, comp[lab], bottom=bottom, label={"e_int": "interaction", "e_deg": "degradation", "e_bias": "bias"}[lab], color=c)
        bottom += comp[lab].values
    ax.set_ylabel("fraction of |energy| (median per cluster)")
    ax.set_title("The bias term dominates the energy in the UNPENALIZED (pseudoinverse) fits -- the\n"
                 "'bias takeover' problem, cross-dataset. Only the penalized scaffold fit (hematopoiesis) controls it")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/4_energy_composition.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 5. GRN spectral radius per cluster
    facet_box_summary(S, "spectral_radius",
                      "GRN amplification: max real eigenvalue of W per cell type",
                      "5_spectral_radius.png")
    # 6. fraction unstable
    facet_box_summary(S, "frac_unstable",
                      "Fraction of cells with a positive real eigenvalue (locally unstable)",
                      "6_fraction_unstable.png")

    # 7. cross-dataset metric heatmap
    agg = S.groupby("dataset").agg(
        median_energy=("energy", "median"), median_eig=("eig", "median"),
        frac_unstable=("frac_unstable", "mean"), spectral_radius=("spectral_radius", "median"),
        W_symmetry=("W_symmetry", "median"), n_clusters=("cluster", "nunique")).reindex(
        [d[0] for d in DATASETS if d[0] in S["dataset"].values])
    z = (agg - agg.mean()) / (agg.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(agg.columns))); ax.set_xticklabels(agg.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(agg.index))); ax.set_yticklabels(agg.index)
    for i in range(len(agg.index)):
        for j in range(len(agg.columns)):
            ax.text(j, i, f"{agg.values[i, j]:.2g}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="z-score"); ax.set_title("Cross-dataset summary metrics (annotated with raw values)")
    fig.tight_layout(); fig.savefig(f"{OUT}/7_metric_heatmap.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 8. leading-eigenvalue distribution per dataset
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [percell[n]["jacobian_eig1_real"].clip(-50, 50).values for n in percell]
    parts = ax.violinplot(data, showmedians=True)
    for pc, n in zip(parts["bodies"], percell):
        pc.set_facecolor(COL[n]); pc.set_alpha(0.6)
    ax.axhline(0, color="crimson", ls="--", lw=0.8)
    ax.set_xticks(range(1, len(percell) + 1)); ax.set_xticklabels(list(percell.keys()), rotation=20, ha="right")
    ax.set_ylabel("leading Jacobian eig (Re), clipped")
    ax.set_title("Distribution of local stability across cells, per system")
    fig.tight_layout(); fig.savefig(f"{OUT}/8_eig_distribution.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    S.to_csv(f"{OUT}/../cross_dataset_summary.csv", index=False)
    print(f"wrote 8 cross-dataset figures to {OUT}/", flush=True)


def facet_box_summary(S, col, title, fname):
    """Faceted bar of a per-cluster summary column, one panel per dataset."""
    ds = [d[0] for d in DATASETS if d[0] in S["dataset"].values]
    fig, axes = plt.subplots(1, len(ds), figsize=(4 * len(ds), 4.5), squeeze=False)
    for ax, name in zip(axes[0], ds):
        sub = S[S["dataset"] == name].sort_values(col)
        ax.bar(range(len(sub)), sub[col].values, color=COL[name], alpha=0.75)
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub["cluster"], rotation=55, ha="right", fontsize=6)
        ax.set_title(name, color=COL[name], fontweight="bold")
    axes[0][0].set_ylabel(col)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
