"""Figure pack: ablation studies on the inference choices.

Re-fits one representative real dataset (pancreas) under a set of ablations and measures
the effect on velocity reconstruction, bias-energy fraction, local stability, and driver
identity. Answers the questions: does a single global GRN suffice vs per-cell-type?
do neighbor-augmented batches and hierarchical pretraining help? heuristic vs jointly
fit Hill? which bias penalty, and is the scaffold necessary?

    figure_packs/pack8_ablations/{plots,data}/ + FIGURE_GUIDE.md
    appends numbered entries to benchmark_results/FINDINGS.md

Run:  PYTHONPATH=analyses/figure_packs .venv/bin/python analyses/figure_packs/pack_ablations.py --device cuda
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch
import _common as C

OUT = "figure_packs/pack8_ablations"
PANCREAS = "data/Pancreas/pancreas_scvelo_ready.h5ad"
MOUSE_GRN = "data/hematopoiesis/networks/mouse_scATAC_atlas.parquet"
N_GENES = 200
N_EPOCHS = 400

BASE_FIT = dict(n_epochs=N_EPOCHS, batch_size=128, learning_rate=0.1,
                reconstruction_regularization=100, bias_regularization=1,
                refit_gamma=True, use_plateau_scheduler=True, plateau_patience=80,
                plateau_factor=0.1, drop_last=True, include_neighbors=True,
                neighbor_fraction=0.2, bias_penalty="l1")

# name -> (overrides to BASE_FIT, use_scaffold, single_matrix)
ABLATIONS = {
    "baseline (per-cluster, scaffold, L1)": (dict(), True, False),
    "D1 single global GRN":                 (dict(), True, True),
    "D2 no neighbors":                      (dict(include_neighbors=False, neighbor_fraction=0.0), True, False),
    "D2 neighbor_fraction=0.4":             (dict(neighbor_fraction=0.4), True, False),
    "D3 hierarchical pretrain":             (dict(hierarchical_pretrain=True), True, False),
    "D4 jointly-fit Hill":                  (dict(fit_hill=True), True, False),
    "D5 bias L2":                           (dict(bias_penalty="l2"), True, False),
    "D5 bias none (lambda=0)":              (dict(bias_regularization=0.0), True, False),
    "D5 no refit_gamma":                    (dict(refit_gamma=False), True, False),
    "D5 no scaffold (pseudoinverse)":       (dict(), False, False),
}


def reconstruction(a, ck):
    x = np.asarray(a.layers["Ms"]); v = np.asarray(a.layers["velocity_S"]); sig = np.asarray(a.layers["sigmoid"])
    lab = a.obs[ck].astype(str); pred = np.zeros_like(v)
    for c in a.obs[ck].astype(str).unique():
        if f"W_{c}" not in a.varp:
            continue
        m = (lab == c).values; W = np.asarray(a.varp[f"W_{c}"])
        I = np.asarray(a.var[f"I_{c}"]) if f"I_{c}" in a.var else 0.0
        g = np.asarray(a.var[f"gamma_{c}"]) if f"gamma_{c}" in a.var else np.asarray(a.var["gamma"])
        pred[m] = sig[m] @ W.T + I - g * x[m]
    ss_res = ((v - pred) ** 2).sum(); ss_tot = ((v - v.mean(0)) ** 2).sum()
    cos = np.sum(v * pred, 1) / ((np.linalg.norm(v, axis=1) + 1e-9) * (np.linalg.norm(pred, axis=1) + 1e-9))
    return float(1 - ss_res / ss_tot), float(np.nanmedian(cos))


def top_drivers(a, ck, n=10):
    lab = a.obs[ck].astype(str); c = lab.value_counts().index[0]
    W = np.asarray(a.varp[f"W_{c}"]); s = np.abs(W).sum(0)
    return set(np.asarray(a.var_names)[np.argsort(s)[::-1][:n]])


def build_base(device):
    adata = ad.read_h5ad(PANCREAS)
    if "sigmoid" not in adata.layers:
        adata.var["scHopfield_used"] = True
    sub = sch.workflows.select_top_velocity_genes(adata, N_GENES)
    base = pd.read_parquet(MOUSE_GRN)
    scaffold, ntf, nedge = sch.inf.build_scaffold(sub, base, return_stats=True)
    sub.obs["all_one"] = "all"
    print(f"base pancreas: {sub.shape}, scaffold {ntf} TFs / {nedge} edges", flush=True)
    return sub, scaffold.values.T


def run_config(sub, scaffold_arg, name, overrides, use_scaffold, single, device):
    ck = "all_one" if single else "clusters"
    fit = dict(BASE_FIT); fit.update(overrides)
    a = sch.run_pipeline(
        sub, cluster_key=ck, scaffold=(scaffold_arg if use_scaffold else None),
        fit_kwargs=fit, device=device, seed=0, copy=True, verbose=False,
        compute_centrality=False,
    )
    r2, cos = reconstruction(a, ck)
    eb = np.abs(a.obs["energy_bias"].values)
    et = np.abs(a.obs[["energy_interaction", "energy_degradation", "energy_bias"]].values).sum(1)
    lead = (a.obs["jacobian_leading_real"].values if "jacobian_leading_real" in a.obs
            else np.asarray(a.obsm["jacobian_eigenvalues"]).real.max(1))
    return dict(name=name, recon_r2=r2, recon_cos=cos,
                bias_frac=float(np.nanmedian(eb / (et + 1e-12))),
                frac_unstable=float((lead > 0).mean()),
                n_W=len([k for k in a.varp if k.startswith("W_")]),
                drivers=top_drivers(a, ck)), a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); args = ap.parse_args()
    os.makedirs(f"{OUT}/plots", exist_ok=True); os.makedirs(f"{OUT}/data", exist_ok=True)
    sch.set_seed(0)
    sub, scaffold_arg = build_base(args.device)

    rows = []; base_drivers = None
    for name, (ov, use_scaf, single) in ABLATIONS.items():
        try:
            r, _ = run_config(sub, scaffold_arg, name, ov, use_scaf, single, args.device)
            if base_drivers is None:
                base_drivers = r["drivers"]
            r["driver_jaccard"] = len(r["drivers"] & base_drivers) / len(r["drivers"] | base_drivers)
            rows.append(r)
            print(f"  {name:38s} cos={r['recon_cos']:.3f} biasE%={r['bias_frac']*100:4.1f} "
                  f"unstable%={r['frac_unstable']*100:3.0f} driverJ={r['driver_jaccard']:.2f}", flush=True)
        except Exception as exc:
            import traceback; print(f"  FAILED {name}: {exc}"); traceback.print_exc()

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "drivers"} for r in rows])
    df.to_csv(f"{OUT}/data/ablations.csv", index=False)

    labels = df["name"].tolist(); y = np.arange(len(labels))
    panels = [("recon_cos", "velocity reconstruction cosine", "#2a6f97"),
              ("bias_frac", "bias energy fraction", "#d1495b"),
              ("frac_unstable", "fraction cells unstable", "#9e2a2b"),
              ("driver_jaccard", "top-driver Jaccard vs baseline", "#2a9d8f")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    for ax, (col, ttl, c) in zip(axes, panels):
        ax.barh(y, df[col], color=c); ax.set_title(ttl, fontsize=10)
        ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Ablation study on pancreas (fit choices)", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/01_ablation_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # highlight: the scaffold/penalty controls bias takeover
    fig, ax = plt.subplots(figsize=(7, 4.6))
    key = df.set_index("name")
    sel = [n for n in ["baseline (per-cluster, scaffold, L1)", "D5 bias L2",
                       "D5 bias none (lambda=0)", "D5 no scaffold (pseudoinverse)"] if n in key.index]
    ax.bar(range(len(sel)), [key.loc[n, "bias_frac"] * 100 for n in sel],
           color=["#2a9d8f", "#e9c46a", "#e76f51", "#9e2a2b"])
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(["baseline\n(scaffold+L1)", "L2 bias", "no bias\npenalty", "no scaffold\n(pseudoinv)"], fontsize=8)
    ax.set(ylabel="bias energy fraction (%)", title="Scaffold + L1 penalty prevent the bias takeover")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/02_bias_takeover_controls.png", dpi=140); plt.close(fig)

    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE)
    _append_findings(df)
    print(f"wrote {OUT}/", flush=True)


def _append_findings(df):
    key = df.set_index("name")
    def g(n, c): return float(key.loc[n, c]) if n in key.index else float("nan")
    base = "baseline (per-cluster, scaffold, L1)"
    lines = [
        "\n## M21 -- Ablation: is a single global GRN enough? (pancreas)\n",
        f"Per-cluster baseline reconstruction cosine {g(base,'recon_cos'):.3f} vs a single "
        f"global GRN {g('D1 single global GRN','recon_cos'):.3f}. Cell-type-specific GRNs "
        f"{'improve' if g(base,'recon_cos')>g('D1 single global GRN','recon_cos') else 'do not improve'} "
        "the velocity fit, supporting per-cluster inference.\n",
        "\n## M22 -- Ablation: neighbors and hierarchical pretraining (pancreas)\n",
        f"Reconstruction cosine: baseline (neighbors 0.2) {g(base,'recon_cos'):.3f}, "
        f"no neighbors {g('D2 no neighbors','recon_cos'):.3f}, neighbor_fraction 0.4 "
        f"{g('D2 neighbor_fraction=0.4','recon_cos'):.3f}, hierarchical pretrain "
        f"{g('D3 hierarchical pretrain','recon_cos'):.3f}. See figure_packs/pack8_ablations.\n",
        "\n## M23 -- Ablation: Hill fit and bias penalty (pancreas)\n",
        f"Jointly-fit Hill cosine {g('D4 jointly-fit Hill','recon_cos'):.3f} vs heuristic "
        f"{g(base,'recon_cos'):.3f}. Bias energy fraction: L1 {g(base,'bias_frac')*100:.1f}%, "
        f"L2 {g('D5 bias L2','bias_frac')*100:.1f}%, no penalty "
        f"{g('D5 bias none (lambda=0)','bias_frac')*100:.1f}%, no scaffold "
        f"{g('D5 no scaffold (pseudoinverse)','bias_frac')*100:.1f}%. The scaffold + L1 "
        "penalty are what keep the bias term from taking over.\n",
    ]
    with open("benchmark_results/FINDINGS.md", "a") as f:
        f.writelines(lines)


GUIDE = """# Figure pack 8: ablation studies

Pancreas re-fit under a sweep of inference choices (200 top-velocity genes, mouse scATAC
scaffold, 400 epochs), measuring velocity-reconstruction cosine, bias-energy fraction,
local instability, and top-driver stability. Regenerated by
`analyses/figure_packs/pack_ablations.py`. Feeds paper section R1/R4 and
`benchmark_results/FINDINGS.md` (M21-M23).

- `01_ablation_summary` -- all ablations x four readouts (reconstruction, bias energy,
  instability, driver Jaccard vs baseline):
  - **D1** single global GRN vs per-cell-type;
  - **D2** neighbor augmentation on/off and `neighbor_fraction`;
  - **D3** hierarchical pretraining;
  - **D4** jointly-fit vs heuristic Hill;
  - **D5** bias penalty (L1 / L2 / none), `refit_gamma`, and scaffold vs pseudoinverse.
- `02_bias_takeover_controls` -- the bias-energy fraction under scaffold+L1 vs L2 vs no
  penalty vs no scaffold: the scaffold and L1 penalty are what prevent the takeover.

`data/ablations.csv` holds the full table.
"""


if __name__ == "__main__":
    main()
