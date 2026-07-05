"""Shared machinery for the comprehensive per-dataset scHopfield reports.

``prepare_and_fit`` runs the whole notebook 01-04 pipeline (preprocess -> velocity
[RNA or pseudotime] -> sigmoid -> scaffold GRN fit -> energies -> energy-gene
correlation -> network correlations/centrality/eigenanalysis -> Jacobians/stats/
rotational) and caches the analyzed AnnData so the figure sections can be regenerated
without re-fitting. Outputs live under ``figure_packs/reports/<dataset>/``.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch
from config import DATASETS, N_GENES, FIT_KWARGS, HILL_N_MAX

ROOT = "figure_packs/reports"


def out_dirs(name):
    base = f"{ROOT}/{name}"
    for sub in ("plots", "data"):
        os.makedirs(f"{base}/{sub}", exist_ok=True)
    return base


def cache_path(name):
    return f"{ROOT}/{name}/data/adata_analyzed.h5ad"


# --------------------------------------------------------------------------- #
# preprocessing + fit + full analysis set
# --------------------------------------------------------------------------- #
def prepare_and_fit(name, device="cuda", force=False, mode=None, tag=""):
    """Return the fully analyzed AnnData for a dataset (cached).

    ``mode`` overrides the dataset's velocity_mode ('velocity' or 'pseudotime'); ``tag``
    suffixes the cache path so both fits of the same dataset coexist (for the
    velocity-vs-pseudotime comparison).
    """
    cp = cache_path(name) if not tag else cache_path(name).replace(".h5ad", f"_{tag}.h5ad")
    if os.path.exists(cp) and not force:
        print(f"[{name}{('/'+tag) if tag else ''}] loading cached {cp}", flush=True)
        return ad.read_h5ad(cp)

    cfg = DATASETS[name]
    out_dirs(name)
    ck = cfg["cluster_key"]
    vmode = mode or cfg["velocity_mode"]
    print(f"[{name}] loading {cfg['path']} (mode={vmode})", flush=True)
    adata = ad.read_h5ad(cfg["path"])

    if cfg.get("prepare"):
        sch.pp.prepare_dataset(adata)

    # some datasets (e.g. dynamo-processed) lack an 'Ms' layer; map from a configured
    # smoothed-expression layer (e.g. dynamo's 'M_t').
    if "Ms" not in adata.layers:
        ms = cfg.get("ms_layer")
        if ms and ms in adata.layers:
            L = adata.layers[ms]
            adata.layers["Ms"] = np.asarray(L.todense()) if hasattr(L, "todense") else np.asarray(L)
        elif "spliced" in adata.layers:
            adata.layers["Ms"] = np.asarray(
                adata.layers["spliced"].todense() if hasattr(adata.layers["spliced"], "todense")
                else adata.layers["spliced"])

    # neighbor graph (needed for pseudotime velocity + neighbor-augmented fit)
    if "connectivities" not in adata.obsp:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
        sc.pp.neighbors(adata, n_neighbors=15)

    # ---- dynamics: RNA velocity or pseudotime-inferred ----
    if vmode == "pseudotime":
        pt_key = cfg.get("pseudotime_key", "Pseudotime")
        if pt_key not in adata.obs:
            # compute a diffusion pseudotime with a data-driven root (extreme DC1 cell)
            if "X_diffmap" not in adata.obsm:
                sc.tl.diffmap(adata)
            adata.uns["iroot"] = int(np.argmin(adata.obsm["X_diffmap"][:, 1]))
            sc.tl.dpt(adata)
            pt_key = "dpt_pseudotime"
        sch.pp.estimate_velocity_from_pseudotime(adata, pseudotime_key=pt_key, store_key="velocity_S")
        print(f"[{name}] velocity from pseudotime '{pt_key}'", flush=True)
    else:
        vk = cfg.get("velocity_key", "velocity_S")
        if vk != "velocity_S":
            L = adata.layers[vk]
            adata.layers["velocity_S"] = np.asarray(L.todense()) if hasattr(L, "todense") else np.asarray(L)
        if "velocity_S" not in adata.layers:
            raise ValueError(f"{name}: no velocity layer '{vk}'")

    # sanitize expression + velocity globally *before* gene selection: dynamo-processed
    # and raw datasets can carry NaN/inf that would corrupt the top-velocity ranking and
    # produce NaN weights downstream.
    for L in ("Ms", "velocity_S"):
        if L in adata.layers:
            X = adata.layers[L]
            X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X, dtype=float)
            n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
            if n_bad:
                print(f"[{name}] sanitized {n_bad} NaN/inf in layer '{L}'", flush=True)
            adata.layers[L] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    adata.var["scHopfield_used"] = True

    # ---- top-velocity genes, keeping lineage + anchor genes ----
    keep = []
    lin = cfg.get("lineages")
    for g in (cfg.get("anchors") or []):
        keep.append(g)
    sub = sch.workflows.select_top_velocity_genes(
        adata, N_GENES, keep_genes=[g for g in keep if g in adata.var_names]) \
        if N_GENES < adata.n_vars else adata
    sub.var["scHopfield_used"] = True

    # sanitize (dynamo-processed / raw datasets can carry NaN/inf -> NaN W -> SVD crash)
    for L in ("Ms", "velocity_S"):
        if L in sub.layers:
            sub.layers[L] = np.nan_to_num(np.asarray(sub.layers[L]), nan=0.0, posinf=0.0, neginf=0.0)

    # ---- sigmoids (raised exponent ceiling + multi-start refine handle sharp/double
    #      sigmoid genes) ----
    sch.pp.fit_all_sigmoids(sub, spliced_key="Ms", n_max=HILL_N_MAX)
    sch.pp.compute_sigmoid(sub, spliced_key="Ms")

    # ---- scaffold + GRN fit ----
    base = pd.read_parquet(cfg["base_grn"])
    scaffold, ntf, nedge = sch.inf.build_scaffold(sub, base, return_stats=True)
    print(f"[{name}] scaffold {ntf} TFs / {nedge} edges; fitting {sub.shape}", flush=True)
    sch.inf.fit_interactions(sub, cluster_key=ck, w_scaffold=scaffold.values.T,
                             device=device, seed=0, **FIT_KWARGS)

    # ---- downstream analyses (notebooks 02-04) ----
    print(f"[{name}] energies / correlations / networks / jacobians", flush=True)

    def _safe(label, fn):
        try:
            fn()
        except Exception as e:
            print(f"[{name}] analysis '{label}' failed: {type(e).__name__}: {e}", flush=True)

    _safe("energies", lambda: sch.tl.compute_energies(sub, cluster_key=ck))
    _safe("energy_gene_correlation", lambda: sch.tl.energy_gene_correlation(sub, cluster_key=ck))
    _safe("network_correlations", lambda: sch.tl.network_correlations(sub, cluster_key=ck))
    _safe("centrality", lambda: sch.tl.compute_network_centrality(sub, cluster_key=ck))
    _safe("eigenanalysis", lambda: sch.tl.compute_eigenanalysis(sub, cluster_key=ck))
    _safe("jacobians", lambda: sch.tl.compute_jacobians(sub, cluster_key=ck, device=device))
    _safe("jacobian_stats", lambda: sch.tl.compute_jacobian_stats(sub))
    _safe("rotational", lambda: sch.tl.compute_rotational_part(sub, cluster_key=ck, device=device))

    # record provenance
    sub.uns["report"] = dict(name=name, velocity_mode=vmode,
                             cluster_key=ck, n_tfs=int(ntf), n_edges=int(nedge))
    _clean_for_write(sub)
    sub.write(cp)
    print(f"[{name}] cached -> {cp}", flush=True)
    return sub


MODEL_UNS = ("models", "jacobian_eigenvectors_temp")


def _clean_for_write(adata):
    uns = adata.uns.get("scHopfield", {})
    for k in MODEL_UNS:
        uns.pop(k, None); adata.uns.pop(k, None)


# --------------------------------------------------------------------------- #
# cluster / lineage helpers
# --------------------------------------------------------------------------- #
def present_clusters(adata, ck, min_cells=20):
    lab = adata.obs[ck].astype(str)
    vc = lab.value_counts()
    return [c for c in vc.index if vc[c] >= min_cells]


def resolve_lineages(adata, name):
    """Return (A, B, A_name, B_name). Explicit if configured, else data-driven:
    the two clusters whose networks are most different (proxy for divergent fates)."""
    cfg = DATASETS[name]; ck = cfg["cluster_key"]
    lin = cfg.get("lineages")
    present = present_clusters(adata, ck)
    if lin:
        A = [c for c in lin["A"] if c in present]
        B = [c for c in lin["B"] if c in present]
        if A and B:
            return A, B, lin["A_name"], lin["B_name"]
    # data-driven: pick the pair of clusters with the least W correlation
    Ws = {c: np.asarray(adata.varp[f"W_{c}"]) for c in present if f"W_{c}" in adata.varp}
    cl = list(Ws)
    best = (cl[0], cl[-1]); worst = 2.0
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            r = np.corrcoef(Ws[cl[i]].ravel(), Ws[cl[j]].ravel())[0, 1]
            if r < worst:
                worst, best = r, (cl[i], cl[j])
    return [best[0]], [best[1]], str(best[0]), str(best[1])


# --------------------------------------------------------------------------- #
# figure + markdown helpers
# --------------------------------------------------------------------------- #
def _as_figure(obj):
    """Resolve a Figure from whatever a plotting fn returns (Figure / Axes / ndarray
    of Axes / (fig, ...) tuple / None). The sch.pl.* return types are inconsistent."""
    if obj is None:
        return plt.gcf()
    if hasattr(obj, "savefig"):            # Figure
        return obj
    if hasattr(obj, "figure"):             # Axes
        return obj.figure
    if isinstance(obj, (list, tuple)) and len(obj):
        return _as_figure(obj[0])
    if isinstance(obj, np.ndarray) and obj.size:
        return _as_figure(obj.flat[0])
    return plt.gcf()


def save(fig_or_ax, name, fname):
    p = f"{ROOT}/{name}/plots/{fname}"
    fig = _as_figure(fig_or_ax)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"plots/{fname}"


class Report:
    """Accumulates markdown sections with embedded images, writes RESULTS.md."""
    def __init__(self, name, title):
        self.name = name
        self.parts = [f"# {title}\n"]

    def text(self, md):
        self.parts.append(md.rstrip() + "\n")

    def section(self, title, intro=""):
        self.parts.append(f"\n## {title}\n")
        if intro:
            self.parts.append(intro.rstrip() + "\n")

    def sub(self, title, intro=""):
        self.parts.append(f"\n### {title}\n")
        if intro:
            self.parts.append(intro.rstrip() + "\n")

    def img(self, relpath, caption=""):
        if relpath is None:
            return
        self.parts.append(f"\n![{caption}]({relpath})\n")
        if caption:
            self.parts.append(f"*{caption}*\n")

    def write(self):
        p = f"{ROOT}/{self.name}/RESULTS.md"
        open(p, "w").write("\n".join(self.parts))
        print(f"[{self.name}] wrote {p}", flush=True)
        return p
