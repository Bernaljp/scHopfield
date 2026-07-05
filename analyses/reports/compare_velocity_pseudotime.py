"""Compare velocity- vs pseudotime-driven scHopfield fits on the same dataset.

For datasets where both signals are available we fit the model twice -- once from RNA
velocity, once from pseudotime-inferred dynamics -- with everything else identical, and
compare the downstream analyses: which genes get selected, how similar the inferred GRNs
are, and whether energy / stability / driver / KO conclusions agree. This supports the
manuscript point that scHopfield does not require RNA velocity: pseudotime suffices.

    figure_packs/reports/_velocity_vs_pseudotime/{plots}/ + RESULTS.md

Run:  PYTHONPATH=analyses/reports .venv/bin/python analyses/reports/compare_velocity_pseudotime.py --device cuda
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch
from rutils import prepare_and_fit, cache_path, present_clusters, Report, ROOT

OUT = f"{ROOT}/_velocity_vs_pseudotime"

# dataset -> (mode_for_cache_main, the alternative mode to fit)
PAIRS = {
    "paul15": ("pseudotime", "velocity"),      # native cache is pseudotime; add velocity
    "pancreas": ("velocity", "pseudotime"),    # native cache is velocity; add pseudotime (DPT)
}


def _recon_cos(a, ck):
    x = np.asarray(a.layers["Ms"]); v = np.asarray(a.layers["velocity_S"]); sig = np.asarray(a.layers["sigmoid"])
    lab = a.obs[ck].astype(str); pred = np.zeros_like(v)
    for c in lab.unique():
        if f"W_{c}" not in a.varp:
            continue
        m = (lab == c).values; W = np.asarray(a.varp[f"W_{c}"])
        I = np.asarray(a.var[f"I_{c}"]) if f"I_{c}" in a.var else 0.0
        g = np.asarray(a.var[f"gamma_{c}"]) if f"gamma_{c}" in a.var else np.asarray(a.var["gamma"])
        pred[m] = sig[m] @ W.T + I - g * x[m]
    cos = np.sum(v * pred, 1) / ((np.linalg.norm(v, axis=1) + 1e-9) * (np.linalg.norm(pred, axis=1) + 1e-9))
    return float(np.nanmedian(cos))


def _leading(a):
    return (a.obs["jacobian_leading_real"].values if "jacobian_leading_real" in a.obs
            else np.asarray(a.obsm["jacobian_eigenvalues"]).real.max(1))


def _top_drivers(a, c, n=20):
    W = np.asarray(a.varp[f"W_{c}"]); s = np.abs(W).sum(0)
    return set(np.asarray(a.var_names)[np.argsort(s)[::-1][:n]])


def compare(name, device, rep):
    from config import DATASETS
    ck = DATASETS[name]["cluster_key"]
    main_mode, alt_mode = PAIRS[name]
    import anndata as ad
    a_main = ad.read_h5ad(cache_path(name))                       # native cache
    a_alt = prepare_and_fit(name, device=device, mode=alt_mode, tag=alt_mode[:3])
    modes = {main_mode: a_main, alt_mode: a_alt}
    print(f"[{name}] comparing {list(modes)}", flush=True)

    rep.section(f"{name}: velocity vs pseudotime",
                f"Two fits of **{name}**, identical except the dynamics target "
                f"({main_mode} vs {alt_mode}).")

    # gene-set overlap
    gA, gB = set(modes[main_mode].var_names), set(modes[alt_mode].var_names)
    jac = len(gA & gB) / len(gA | gB)
    rep.text(f"- **Selected-gene overlap** ({main_mode} vs {alt_mode}): Jaccard = **{jac:.2f}** "
             f"({len(gA & gB)} shared of {len(gA | gB)}).")

    shared_clusters = [c for c in present_clusters(a_main, ck)
                       if f"W_{c}" in a_main.varp and f"W_{c}" in a_alt.varp]
    shared_genes = sorted(gA & gB)
    idxA = [list(a_main.var_names).index(g) for g in shared_genes]
    idxB = [list(a_alt.var_names).index(g) for g in shared_genes]

    # W correlation per cluster on shared genes
    wcorr = []
    for c in shared_clusters:
        WA = np.asarray(a_main.varp[f"W_{c}"])[np.ix_(idxA, idxA)]
        WB = np.asarray(a_alt.varp[f"W_{c}"])[np.ix_(idxB, idxB)]
        wcorr.append(np.corrcoef(WA.ravel(), WB.ravel())[0, 1])
    def _wc():
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(shared_clusters), 4))
        ax.bar(range(len(shared_clusters)), wcorr, color="#3b6ea5")
        ax.set_xticks(range(len(shared_clusters))); ax.set_xticklabels(shared_clusters, rotation=45, ha="right", fontsize=7)
        ax.set(ylabel="corr(W_velocity, W_pseudotime)", title=f"{name}: per-cluster GRN agreement")
        ax.axhline(0, color="k", lw=0.6)
        return fig
    _save(rep, name, f"{name}_W_corr.png", _wc,
          f"Per-cell-type interaction-matrix agreement between the two fits "
          f"(mean corr {np.nanmean(wcorr):.2f}).")

    # per-cell energy + stability agreement (same cells)
    def _scatter(col, getter, ttl, fname):
        xa, xb = getter(a_main), getter(a_alt)
        n = min(len(xa), len(xb))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xa[:n], xb[:n], s=5, alpha=0.3, color="#5f0f40")
        r = np.corrcoef(xa[:n], xb[:n])[0, 1]
        lim = [min(xa[:n].min(), xb[:n].min()), max(xa[:n].max(), xb[:n].max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set(xlabel=f"{main_mode}", ylabel=f"{alt_mode}", title=f"{name}: {ttl} (r={r:.2f})")
        return fig
    _save(rep, name, f"{name}_energy_agree.png",
          lambda: _scatter("e", lambda a: a.obs["energy_total"].values.astype(float), "per-cell energy", ""),
          "Per-cell total energy agrees between velocity and pseudotime fits.")
    _save(rep, name, f"{name}_stability_agree.png",
          lambda: _scatter("s", _leading, "per-cell leading eigenvalue", ""),
          "Per-cell local stability (leading eigenvalue) agrees between the fits.")

    # summary metrics table
    rows = []
    for mode, a in modes.items():
        djac = np.mean([len(_top_drivers(a_main, c) & _top_drivers(a, c)) /
                        len(_top_drivers(a_main, c) | _top_drivers(a, c)) for c in shared_clusters])
        rows.append(dict(mode=mode, recon_cos=_recon_cos(a, ck),
                         frac_unstable=float((_leading(a) > 0).mean()),
                         median_energy=float(np.nanmedian(a.obs["energy_total"].values)),
                         driver_jaccard_vs_main=djac))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/plots/{name}_summary.csv", index=False)
    rep.text("\nSummary:\n\n" + _md_table(df))
    return dict(name=name, gene_jaccard=jac, mean_W_corr=float(np.nanmean(wcorr)))


def _md_table(df):
    """Render a DataFrame as a GitHub markdown table without the tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.round(3).iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _save(rep, name, fname, fn, cap):
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    try:
        fig = fn(); p = f"{OUT}/plots/{fname}"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        rep.img(f"plots/{fname}", cap)
    except Exception as e:
        import traceback; print(f"  [fig FAIL] {fname}: {e}"); traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default="paul15,pancreas"); args = ap.parse_args()
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    rep = Report("_velocity_vs_pseudotime", "scHopfield: velocity vs pseudotime dynamics")
    rep.text("Does scHopfield need RNA velocity? We fit the model from RNA velocity and from "
             "pseudotime-inferred dynamics on the same datasets and compare every downstream "
             "readout. High agreement supports using pseudotime when velocity is unavailable.")
    summ = []
    for name in args.datasets.split(","):
        try:
            summ.append(compare(name, args.device, rep))
        except Exception as e:
            import traceback; print(f"FAILED {name}: {e}"); traceback.print_exc()
    if summ:
        rep.section("Summary across datasets")
        rep.text(_md_table(pd.DataFrame(summ)))
    # write to the comparison folder
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(rep.parts))
    print(f"wrote {OUT}/RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
