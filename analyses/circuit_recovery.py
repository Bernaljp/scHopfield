"""Synthetic-circuit GRN-recovery benchmark for scHopfield.

For each canonical circuit with a known Hopfield-form interaction matrix
(toggle switch, repressilator, dissertation oscillator), we simulate expression
+ analytic velocity, fit scHopfield's scaffold-guided optimizer under three
scaffold priors (full / partial / none) across several seeds, and measure how
well the inferred W recovers the ground truth:

  - edge_sign_accuracy : fraction of true edges with correct sign
  - edge_correlation   : Pearson r between vec(W_hat) and vec(W_true)
  - frobenius_distance : ||W_hat - W_true|| / ||W_true||
  - edge_auroc/auprc   : detection of nonzero edges (|W_hat| as score), when the
                         circuit has both edges and non-edges (sparse circuits)

This supports the honest claim: scHopfield recovers ground-truth network
structure on systems where the answer is known, and degrades gracefully as the
scaffold prior weakens and as observation noise increases.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from scHopfield.validation.circuits import (
    ToggleCircuit,
    OscillatorCircuit,
    DissertationOscillatorCircuit,
)
from scHopfield.validation.simulate import simulate_circuit
from scHopfield.validation.fit_validation import fit_circuit
from scHopfield.validation.metrics import summarize_recovery


def edge_detection_scores(W_inferred, W_true, thr=1e-9):
    """AUROC/AUPRC for detecting nonzero edges; off-diagonal only.

    Returns (auroc, auprc) or (nan, nan) if only one class is present.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    n = W_true.shape[0]
    off = ~np.eye(n, dtype=bool)
    y = (np.abs(W_true[off]) > thr).astype(int)
    score = np.abs(W_inferred[off])
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def build_circuits(include_hard=False):
    # Toggle + repressilator are the natively-Hopfield circuits with an exactly
    # known W; these are the validation headline. The dissertation oscillator and
    # other biophysical models are NOT faithfully Hopfield-representable and
    # recover poorly (documented as a limitation, off by default).
    circuits = {
        "toggle_bistable": ToggleCircuit(b=4.0),      # 2 genes, dense (no non-edges)
        "repressilator": OscillatorCircuit(),          # 3 genes, sparse
    }
    if include_hard:
        circuits["diss_oscillator"] = DissertationOscillatorCircuit()
    return circuits


def circuit_hill_params(circuit):
    k = getattr(circuit, "k", getattr(circuit, "K", 1.0))
    n = getattr(circuit, "n", 4)
    return float(k), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scaffolds", nargs="+", default=["full", "partial", "none"])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--n-epochs", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--include-hard", action="store_true",
                    help="also run non-Hopfield circuits (documented limitation)")
    ap.add_argument("--out", default="benchmark_results/circuit_recovery")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device={dev} cuda={torch.cuda.is_available()}", flush=True)

    circuits = build_circuits(include_hard=args.include_hard)
    rows = []           # per-run records
    agg = defaultdict(list)  # (circuit,scaffold,noise) -> list of metric dicts

    # Cache simulations: they depend on (circuit, noise, seed) but NOT scaffold,
    # so we simulate once and reuse across scaffold regimes (the stiff ODE solve
    # is the bottleneck, not the fit).
    sim_cache = {}

    def get_sim(cname, circuit, noise, seed, k, n):
        key = (cname, noise, seed)
        if key not in sim_cache:
            adata = simulate_circuit(circuit, n_trajectories=40,
                                     points_per_trajectory=40,
                                     noise_sigma=noise, seed=seed)
            adata.uns["ground_truth"]["k"] = k
            adata.uns["ground_truth"]["n"] = n
            sim_cache[key] = adata
        return sim_cache[key]

    for cname, circuit in circuits.items():
        k, n = circuit_hill_params(circuit)
        for noise in args.noise:
            for scaffold in args.scaffolds:
                for seed in args.seeds:
                    adata = get_sim(cname, circuit, noise, seed, k, n)
                    res = fit_circuit(adata, scaffold_mode=scaffold,
                                      n_epochs=args.n_epochs, device=dev, seed=seed)
                    m = summarize_recovery(res["W_inferred"], res["W_true"])
                    auroc, auprc = edge_detection_scores(res["W_inferred"], res["W_true"])
                    m["edge_auroc"] = auroc
                    m["edge_auprc"] = auprc
                    rec = {"circuit": cname, "scaffold": scaffold, "noise": noise,
                           "seed": seed, **m}
                    rows.append(rec)
                    agg[(cname, scaffold, noise)].append(m)
                    print(f"[{cname} sc={scaffold} noise={noise} seed={seed}] "
                          f"sign={m['edge_sign_accuracy']:.2f} corr={m['edge_correlation']:.2f} "
                          f"fro={m['frobenius_distance']:.2f} auroc={auroc:.2f}", flush=True)

    # aggregate mean +/- sd across seeds
    summary = []
    for (cname, scaffold, noise), ms in agg.items():
        entry = {"circuit": cname, "scaffold": scaffold, "noise": noise,
                 "n_seeds": len(ms)}
        for key in ms[0]:
            vals = np.array([mm[key] for mm in ms], dtype=float)
            entry[f"{key}_mean"] = float(np.nanmean(vals))
            entry[f"{key}_sd"] = float(np.nanstd(vals))
        summary.append(entry)

    with open(os.path.join(args.out, "runs.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # human-readable CSV
    import csv
    cols = ["circuit", "scaffold", "noise", "n_seeds",
            "edge_sign_accuracy_mean", "edge_sign_accuracy_sd",
            "edge_correlation_mean", "edge_correlation_sd",
            "frobenius_distance_mean", "frobenius_distance_sd",
            "edge_auroc_mean", "edge_auprc_mean"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in sorted(summary, key=lambda r: (r["circuit"], r["noise"], r["scaffold"])):
            w.writerow(e)

    print("\n=== RECOVERY SUMMARY (mean over seeds) ===", flush=True)
    for e in sorted(summary, key=lambda r: (r["circuit"], r["noise"], r["scaffold"])):
        print(f"{e['circuit']:16s} sc={e['scaffold']:8s} noise={e['noise']:.2f} | "
              f"sign={e['edge_sign_accuracy_mean']:.3f} "
              f"corr={e['edge_correlation_mean']:.3f} "
              f"relFro={e['frobenius_distance_mean']:.3f} "
              f"AUROC={e['edge_auroc_mean']:.3f}", flush=True)
    print(f"wrote {args.out}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
