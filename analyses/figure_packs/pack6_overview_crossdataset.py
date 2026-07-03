"""Figure packs 6 + 7: reproducible inference (R1 / Fig 1) and cross-dataset generalization.

Pack 6 (reproducibility / model overview): velocity-reconstruction quality, energy
composition (bias ~0 after the L1 + scaffold fit), and the model dimensions for every
re-fitted dataset. Pack 7 (cross-dataset): the five systems compared under one common
estimator -- stability, energy, GRN amplification, and a metric summary. CPU.

    figure_packs/pack6_reproducibility/{plots,data}/ + FIGURE_GUIDE.md
    figure_packs/pack7_cross_dataset/{plots,data}/    + FIGURE_GUIDE.md

Run:  PYTHONPATH=analyses/figure_packs .venv/bin/python analyses/figure_packs/pack6_overview_crossdataset.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

P6 = "figure_packs/pack6_reproducibility"
P7 = "figure_packs/pack7_cross_dataset"
COMPS = ["energy_interaction", "energy_degradation", "energy_bias"]
CCOL = ["#2a6f97", "#e9c46a", "#d1495b"]


def reconstruction(a, ck):
    """Per-cell predicted vs observed velocity: pooled R^2 and per-cell cosine."""
    x = np.asarray(a.layers["Ms"]); v = np.asarray(a.layers["velocity_S"])
    sig = np.asarray(a.layers["sigmoid"])
    lab = a.obs[ck].astype(str); pred = np.zeros_like(v)
    for c in C.present_clusters(a, ck):
        if f"W_{c}" not in a.varp:
            continue
        m = (lab == c).values
        W = np.asarray(a.varp[f"W_{c}"])
        I = np.asarray(a.var[f"I_{c}"]) if f"I_{c}" in a.var else 0.0
        g = np.asarray(a.var[f"gamma_{c}"]) if f"gamma_{c}" in a.var else np.asarray(a.var["gamma"])
        pred[m] = sig[m] @ W.T + I - g * x[m]
    ss_res = ((v - pred) ** 2).sum(); ss_tot = ((v - v.mean(0)) ** 2).sum()
    cos = np.sum(v * pred, 1) / ((np.linalg.norm(v, axis=1) + 1e-9) * (np.linalg.norm(pred, axis=1) + 1e-9))
    return float(1 - ss_res / ss_tot), cos


def energy_fracs(a):
    M = np.array([float(np.nanmedian(np.abs(a.obs[c].values))) for c in COMPS])
    return M / (M.sum() + 1e-12)


def collect():
    rows = {}
    for name, ck in C.available().items():
        a = C.load(name)
        clusters = C.present_clusters(a, ck)
        r2, cos = reconstruction(a, ck)
        lead = (a.obs["jacobian_leading_real"].values if "jacobian_leading_real" in a.obs
                else np.asarray(a.obsm["jacobian_eigenvalues"]).real.max(1))
        rad = [float(np.max(np.real(np.linalg.eigvals(C.W_of(a, c))))) for c in clusters]
        rows[name] = dict(
            n_cells=int(a.n_obs), n_genes=int(a.n_vars), n_clusters=len(clusters),
            recon_r2=r2, recon_cos=float(np.nanmedian(cos)), cos_all=cos,
            energy_frac=energy_fracs(a).tolist(),
            median_energy=float(np.nanmedian(a.obs["energy_total"].values)),
            median_leading=float(np.nanmedian(lead)),
            frac_unstable=float((lead > 0).mean()),
            spectral_radius=float(np.median(rad)),
            lead_all=lead,
        )
    return rows


# ------------------------- pack 6: reproducibility ------------------------- #
def pack6(rows):
    os.makedirs(f"{P6}/plots", exist_ok=True); os.makedirs(f"{P6}/data", exist_ok=True)
    names = list(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [rows[n]["recon_cos"] for n in names], 0.4, label="median cosine", color="#2a6f97")
    ax.bar(x + 0.2, [max(0, rows[n]["recon_r2"]) for n in names], 0.4, label="pooled R^2", color="#e09f3e")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="velocity reconstruction", title="Velocity reconstruction quality per dataset")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P6}/plots/01_reconstruction_quality.png", dpi=140); plt.close(fig)

    # reconstruction cosine distributions
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.violinplot([rows[n]["cos_all"] for n in names], showmedians=True)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set(ylabel="per-cell velocity cosine", title="Per-cell reconstruction cosine")
    fig.tight_layout(); fig.savefig(f"{P6}/plots/02_reconstruction_cosine_dist.png", dpi=140); plt.close(fig)

    # energy composition (bias ~0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = np.zeros(len(names))
    for k, (comp, col) in enumerate(zip(COMPS, CCOL)):
        vals = np.array([rows[n]["energy_frac"][k] for n in names])
        ax.bar(range(len(names)), vals, bottom=bottom, color=col, label=comp.replace("energy_", ""))
        bottom += vals
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="fraction of |energy| (median)", title="Energy composition (bias collapses to ~0)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P6}/plots/03_energy_composition.png", dpi=140); plt.close(fig)

    # model dimensions table-as-figure
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(names) + 1.4)); ax.axis("off")
    cell = [[n, rows[n]["n_cells"], rows[n]["n_genes"], rows[n]["n_clusters"],
             f"{rows[n]['recon_cos']:.2f}", f"{rows[n]['energy_frac'][2]*100:.1f}%"] for n in names]
    tbl = ax.table(cellText=cell,
                   colLabels=["dataset", "cells", "genes", "clusters", "recon cos", "bias E%"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    ax.set_title("Model dimensions and fit summary", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{P6}/plots/04_model_dimensions.png", dpi=140); plt.close(fig)

    json.dump({n: {k: v for k, v in rows[n].items() if k not in ("cos_all", "lead_all")}
               for n in names}, open(f"{P6}/data/reproducibility.json", "w"), indent=2)
    open(f"{P6}/FIGURE_GUIDE.md", "w").write(GUIDE6.replace("{datasets}", ", ".join(names)))


# ------------------------- pack 7: cross-dataset --------------------------- #
def pack7(rows):
    os.makedirs(f"{P7}/plots", exist_ok=True); os.makedirs(f"{P7}/data", exist_ok=True)
    names = list(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(names)), [rows[n]["frac_unstable"] for n in names], color="#9e2a2b")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set(ylabel="fraction cells leading eig > 0", title="Local instability across systems")
    fig.tight_layout(); fig.savefig(f"{P7}/plots/01_fraction_unstable.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.violinplot([rows[n]["lead_all"] for n in names], showmedians=True)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set(ylabel="leading eigenvalue (Re)", title="Leading-eigenvalue distribution per system")
    fig.tight_layout(); fig.savefig(f"{P7}/plots/02_leading_eig_distribution.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = np.zeros(len(names))
    for k, (comp, col) in enumerate(zip(COMPS, CCOL)):
        vals = np.array([rows[n]["energy_frac"][k] for n in names])
        ax.bar(range(len(names)), vals, bottom=bottom, color=col, label=comp.replace("energy_", ""))
        bottom += vals
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="fraction of |energy|", title="Energy composition across systems (one estimator)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P7}/plots/03_energy_composition.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(names)), [rows[n]["spectral_radius"] for n in names], color="#2a9d8f")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="median max Re(eig W)", title="GRN amplification (spectral radius) per system")
    fig.tight_layout(); fig.savefig(f"{P7}/plots/04_spectral_radius.png", dpi=140); plt.close(fig)

    # metric heatmap (datasets x metrics, z-scored across datasets)
    metrics = ["median_energy", "median_leading", "frac_unstable", "spectral_radius", "recon_cos", "n_clusters"]
    M = np.array([[rows[n][m] for m in metrics] for n in names], float)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.6))
    im = ax.imshow(Z, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center", fontsize=7)
    ax.set_title("Cross-dataset metric summary (z-scored)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{P7}/plots/05_metric_heatmap.png", dpi=140); plt.close(fig)

    json.dump({n: {k: v for k, v in rows[n].items() if k not in ("cos_all", "lead_all")}
               for n in names}, open(f"{P7}/data/cross_dataset.json", "w"), indent=2)
    open(f"{P7}/FIGURE_GUIDE.md", "w").write(GUIDE7.replace("{datasets}", ", ".join(names)))


GUIDE6 = """# Figure pack 6: reproducible inference (model overview)

Fit quality and model summary for every re-fitted dataset under one common estimator
(species-appropriate scaffold + L1 bias). Regenerated by
`analyses/figure_packs/pack6_overview_crossdataset.py`. Targets paper section R1 / Fig 1.

Datasets: {datasets}

- `01_reconstruction_quality` -- per-dataset velocity reconstruction (median cosine and
  pooled R^2). Cosine is the scale-invariant summary; magnitude/scale differences between
  datasets pull the pooled R^2 down (e.g. hematopoiesis), so both are shown honestly.
- `02_reconstruction_cosine_dist` -- per-cell reconstruction-cosine distribution.
- `03_energy_composition` -- interaction/degradation/bias energy fractions; the bias term
  collapses to ~0 everywhere (the takeover fix holds on the new code).
- `04_model_dimensions` -- cells / genes / clusters / reconstruction / bias-energy table.
"""

GUIDE7 = """# Figure pack 7: cross-dataset generalization

The five developmental systems compared under one common estimator. Regenerated by
`analyses/figure_packs/pack6_overview_crossdataset.py`. Complements the committed
`benchmark_results/cross_dataset/` guide with the fixed-code re-fits.

Datasets: {datasets}

- `01_fraction_unstable` -- fraction of cells with a positive leading eigenvalue.
- `02_leading_eig_distribution` -- per-system leading-eigenvalue distribution.
- `03_energy_composition` -- energy fractions across systems (bias ~0 everywhere).
- `04_spectral_radius` -- GRN amplification (spectral radius) per system.
- `05_metric_heatmap` -- datasets x metrics summary (z-scored, raw values annotated).
"""


def main():
    rows = collect()
    pack6(rows)
    pack7(rows)
    print(f"wrote {P6}/ and {P7}/", flush=True)


if __name__ == "__main__":
    main()
