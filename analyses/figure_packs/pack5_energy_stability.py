"""Figure pack 5: energy landscape, Jacobians, and local stability.

For every re-fitted dataset: the Hopfield energy and its decomposition per cell type,
the energy landscape and stability over the embedding, and the Jacobian spectra that
quantify local (in)stability along the differentiation trajectory. CPU.

    figure_packs/pack5_energy_stability/<dataset>/{plots,data}/ + FIGURE_GUIDE.md

Run:  .venv/bin/python analyses/figure_packs/pack5_energy_stability.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

OUT = "figure_packs/pack5_energy_stability"
STAB = "jacobian_leading_real"   # true leading eigenvalue (max real part)


def _leading(adata):
    if STAB in adata.obs:
        return np.asarray(adata.obs[STAB].values, float)
    return np.asarray(adata.obsm["jacobian_eigenvalues"]).real.max(1)


def fig_energy_decomposition(adata, clusters, ck, out):
    """1: median |energy| decomposition per cell type (bias should be ~0)."""
    comps = ["energy_interaction", "energy_degradation", "energy_bias"]
    cols = ["#2a6f97", "#e9c46a", "#d1495b"]
    M = np.array([[float(np.nanmedian(np.abs(adata.obs.loc[adata.obs[ck].astype(str) == c, comp].values)))
                   for comp in comps] for c in clusters])
    frac = M / (M.sum(1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(clusters), 4.4))
    bottom = np.zeros(len(clusters))
    for k, (comp, col) in enumerate(zip(comps, cols)):
        ax.bar(range(len(clusters)), frac[:, k], bottom=bottom, color=col,
               label=comp.replace("energy_", ""))
        bottom += frac[:, k]
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="fraction of |energy| (median)", title="Energy decomposition per cell type")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/plots/01_energy_decomposition.png", dpi=140); plt.close(fig)
    return dict(clusters=clusters, fraction=frac.tolist(),
                bias_fraction_median=float(np.median(frac[:, 2])))


def fig_energy_box(adata, clusters, ck, out):
    """2: total energy depth per cell type."""
    data = [adata.obs.loc[adata.obs[ck].astype(str) == c, "energy_total"].values for c in clusters]
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(clusters), 4.2))
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="total Hopfield energy", title="Energy depth per cell type")
    fig.tight_layout(); fig.savefig(f"{out}/plots/02_energy_boxplot.png", dpi=140); plt.close(fig)


def fig_landscape(adata, out):
    """3+4: energy and stability over the embedding."""
    basis = C.basis_of(adata)
    emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
    lead = _leading(adata)
    for idx, (vals, ttl, cmap, kw) in [
        ("03", (adata.obs["energy_total"].values, "Energy landscape", "viridis", {})),
        ("04", (lead, "Local stability (leading eig, Re)", "RdBu_r",
                dict(vmin=-np.nanpercentile(np.abs(lead), 95), vmax=np.nanpercentile(np.abs(lead), 95)))),
    ]:
        v, ttl, cmap, kw = vals, ttl, cmap, kw
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=v, s=6, cmap=cmap, **kw)
        ax.set(xticks=[], yticks=[], title=f"{ttl} ({basis})")
        fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(f"{out}/plots/{idx}_{'energy' if idx=='03' else 'stability'}_umap.png", dpi=140)
        plt.close(fig)


def fig_fraction_unstable(adata, clusters, ck, out):
    """5: fraction of cells carrying an unstable mode, per cell type."""
    lead = _leading(adata)
    fr = []
    for c in clusters:
        m = (adata.obs[ck].astype(str) == c).values
        fr.append(float((lead[m] > 0).mean()) if m.sum() else np.nan)
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(clusters), 4.2))
    ax.bar(range(len(clusters)), fr, color="#9e2a2b")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="fraction with leading eig > 0", title="Fraction of cells locally unstable")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    fig.tight_layout(); fig.savefig(f"{out}/plots/05_fraction_unstable.png", dpi=140); plt.close(fig)
    return dict(clusters=clusters, fraction_unstable=fr)


def fig_leading_dist(adata, clusters, ck, out):
    """6: distribution of the leading eigenvalue per cell type."""
    lead = _leading(adata)
    data = [lead[(adata.obs[ck].astype(str) == c).values] for c in clusters]
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(clusters), 4.2))
    parts = ax.violinplot(data, showmedians=True)
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(clusters) + 1)); ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="leading eigenvalue (Re)", title="Leading-eigenvalue distribution per cell type")
    fig.tight_layout(); fig.savefig(f"{out}/plots/06_leading_eig_distribution.png", dpi=140); plt.close(fig)


def fig_spectrum(adata, out, n=4000):
    """7: pooled Jacobian eigenvalue spectrum in the complex plane."""
    ev = np.asarray(adata.obsm["jacobian_eigenvalues"])
    rng = np.random.default_rng(0)
    idx = rng.choice(ev.shape[0], min(n, ev.shape[0]), replace=False)
    pts = ev[idx].ravel()
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.15, color="#3b0f70")
    ax.axvline(0, color="crimson", lw=0.8, ls="--")
    ax.set(xlabel="Re", ylabel="Im", title="Jacobian eigenvalue spectrum (pooled)")
    fig.tight_layout(); fig.savefig(f"{out}/plots/07_eig_spectrum.png", dpi=140); plt.close(fig)


def fig_coupling(adata, out):
    """8: energy vs stability coupling."""
    e = adata.obs["energy_total"].values.astype(float)
    lead = _leading(adata)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(e, lead, s=5, alpha=0.25, color="#0f4c5c")
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set(xlabel="total energy", ylabel="leading eigenvalue (Re)", title="Energy vs local stability")
    m = np.isfinite(e) & np.isfinite(lead)
    if m.sum() > 10:
        r = np.corrcoef(e[m], lead[m])[0, 1]
        ax.text(0.02, 0.96, f"r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out}/plots/08_energy_stability_coupling.png", dpi=140); plt.close(fig)


GUIDE = """# Figure pack 5: energy landscape, Jacobians, stability

Hopfield energy and local-stability structure of each re-fitted dataset. Regenerated by
`analyses/figure_packs/pack5_energy_stability.py`. Targets paper Fig 6 (energy/stability
and higher-order predictions). Stability uses the true leading eigenvalue
(`jacobian_leading_real`), not the arbitrary index-0 eigenvalue.

Datasets: {datasets}

Per dataset (`<dataset>/plots/`):
1. `01_energy_decomposition` -- interaction/degradation/bias fractions per cell type
   (bias ~0 after the L1 + scaffold fit).
2. `02_energy_boxplot` -- total energy depth per cell type.
3. `03_energy_umap` -- energy landscape over the embedding.
4. `04_stability_umap` -- leading Jacobian eigenvalue over the embedding.
5. `05_fraction_unstable` -- fraction of cells with a positive leading eigenvalue.
6. `06_leading_eig_distribution` -- per-cell-type leading-eigenvalue violins.
7. `07_eig_spectrum` -- pooled Jacobian spectrum in the complex plane.
8. `08_energy_stability_coupling` -- energy vs leading eigenvalue.
"""


def run_dataset(name, cluster_key):
    adata = C.load(name)
    clusters = C.present_clusters(adata, cluster_key)
    out = f"{OUT}/{name}"
    os.makedirs(f"{out}/plots", exist_ok=True); os.makedirs(f"{out}/data", exist_ok=True)
    print(f"[pack5] {name}: {len(clusters)} clusters", flush=True)
    data = {}
    data["decomposition"] = fig_energy_decomposition(adata, clusters, cluster_key, out)
    fig_energy_box(adata, clusters, cluster_key, out)
    fig_landscape(adata, out)
    data["stability"] = fig_fraction_unstable(adata, clusters, cluster_key, out)
    fig_leading_dist(adata, clusters, cluster_key, out)
    fig_spectrum(adata, out)
    fig_coupling(adata, out)
    json.dump(data, open(f"{out}/data/pack5_{name}.json", "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = C.available()
    for name, ck in avail.items():
        try:
            run_dataset(name, ck)
        except Exception as exc:
            import traceback; print(f"FAILED {name}: {exc}"); traceback.print_exc()
    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE.format(datasets=", ".join(avail)))
    print(f"wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
