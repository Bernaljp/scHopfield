"""High-level, reproducible scHopfield pipeline.

:func:`run_pipeline` chains the canonical steps that every scHopfield analysis
runs in the same order:

    prepare -> (optional) gene subset -> (optional) scaffold -> fit GRN
            -> energies -> jacobians -> jacobian stats -> centrality

It is a thin, transparent orchestration of the public API (no hidden magic): each
step is an ordinary ``sch.*`` call with sensible defaults, and the function
returns a summary ``dict`` describing what was computed. Use it to run the same
pipeline reproducibly across many datasets; drop down to the individual functions
whenever you need finer control.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from anndata import AnnData

from . import inference as inf
from . import preprocessing as pp
from . import tools as tl

__all__ = ["run_pipeline", "select_top_velocity_genes"]


def select_top_velocity_genes(
    adata: AnnData,
    n_genes: int,
    velocity_key: str = "velocity_S",
    keep_genes: Optional[Sequence[str]] = None,
) -> AnnData:
    """Subset to the ``n_genes`` genes with the largest mean absolute velocity.

    Optionally force-keep extra genes (e.g. perturbation targets) even if they
    fall outside the top set. Returns a copy.
    """
    vmag = np.abs(np.asarray(adata.layers[velocity_key])).mean(0).ravel()
    top = set(np.argsort(vmag)[::-1][:n_genes].tolist())
    for g in keep_genes or []:
        if g in adata.var_names:
            top.add(adata.var_names.get_loc(g))
    return adata[:, sorted(top)].copy()


def run_pipeline(
    adata: AnnData,
    cluster_key: str,
    *,
    prepare: bool = False,
    prepare_kwargs: Optional[Dict] = None,
    n_top_genes: Optional[int] = None,
    keep_genes: Optional[Sequence[str]] = None,
    scaffold: Optional["np.ndarray"] = None,
    fit_kwargs: Optional[Dict] = None,
    compute_centrality: bool = True,
    device: str = "cpu",
    seed: Optional[int] = 0,
    spliced_key: str = "Ms",
    velocity_key: str = "velocity_S",
    degradation_key: str = "gamma",
    verbose: bool = True,
    copy: bool = True,
) -> AnnData:
    """Run the end-to-end scHopfield pipeline on one dataset.

    Parameters
    ----------
    adata
        Input data. If ``prepare=True`` it may be raw (spliced/unspliced counts);
        otherwise it must already be scHopfield-ready (``Ms``, ``velocity_S``,
        ``sigmoid``, ``gamma``, ``connectivities``).
    cluster_key
        ``adata.obs`` column with cell-type / cluster labels.
    prepare
        If ``True``, run :func:`~scHopfield.preprocessing.prepare_dataset` first.
    prepare_kwargs
        Extra kwargs for ``prepare_dataset``.
    n_top_genes
        If set, subset to the top-``n_top_genes`` genes by mean absolute velocity
        (keeps the model tractable and comparable across datasets).
    keep_genes
        Genes to force-keep through the ``n_top_genes`` subset.
    scaffold
        Optional prior-knowledge mask passed to ``fit_interactions`` as
        ``w_scaffold``. Build one with :func:`~scHopfield.inference.build_scaffold`
        and pass ``scaffold.values.T``. If ``None``, an unconstrained
        (pseudoinverse) GRN is fit.
    fit_kwargs
        Extra kwargs forwarded to :func:`~scHopfield.inference.fit_interactions`.
        Defaults use the fast pseudoinverse path (``skip_all=True``) unless a
        scaffold is supplied.
    compute_centrality
        If ``True``, also compute per-cluster network centrality.
    device
        Torch device for the GRN fit and Jacobian computation.
    seed
        Reproducibility seed threaded into the fit.
    verbose
        Print a one-line progress marker per step.
    copy
        If ``True`` (default), operate on a copy and return it; if ``False``,
        modify ``adata`` in place and return it.

    Returns
    -------
    :class:`~anndata.AnnData`
        The processed object with GRN, energies, Jacobian stats, and (optionally)
        centrality populated. A summary is stored in
        ``adata.uns['scHopfield_pipeline']``.
    """
    a = adata.copy() if copy else adata
    prepare_kwargs = dict(prepare_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})
    log: Dict[str, object] = {"steps": []}

    def _mark(step):
        log["steps"].append(step)
        if verbose:
            print(f"[scHopfield] {step}", flush=True)

    if prepare:
        pp.prepare_dataset(a, spliced_key=spliced_key, velocity_key=velocity_key,
                           degradation_key=degradation_key, **prepare_kwargs)
        _mark("prepare_dataset")

    if "scHopfield_used" not in a.var:
        a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values, spliced_key=spliced_key)
        pp.compute_sigmoid(a, spliced_key=spliced_key)
        _mark("fit_sigmoids")

    if n_top_genes is not None and n_top_genes < a.n_vars:
        a = select_top_velocity_genes(a, n_top_genes, velocity_key=velocity_key,
                                      keep_genes=keep_genes)
        a.var["scHopfield_used"] = True
        _mark(f"select_top_velocity_genes(n={n_top_genes})")

    # --- GRN inference ---
    fit_defaults = dict(
        spliced_key=spliced_key, velocity_key=velocity_key,
        degradation_key=degradation_key, device=device, seed=seed,
    )
    if scaffold is not None:
        fit_defaults.update(w_scaffold=scaffold, only_TFs=True,
                            scaffold_regularization=fit_kwargs.pop("scaffold_regularization", 0.1),
                            skip_all=True, w_threshold=1e-12)
    else:
        fit_defaults.update(w_scaffold=None, skip_all=True, w_threshold=1e-12)
    fit_defaults.update(fit_kwargs)
    inf.fit_interactions(a, cluster_key=cluster_key, **fit_defaults)
    _mark("fit_interactions" + (" (scaffold)" if scaffold is not None else " (pseudoinverse)"))

    # --- energy landscape ---
    tl.compute_energies(a, cluster_key=cluster_key, spliced_key=spliced_key,
                        degradation_key=degradation_key)
    _mark("compute_energies")

    # --- jacobian stability ---
    tl.compute_jacobians(a, cluster_key=cluster_key, spliced_key=spliced_key,
                         degradation_key=degradation_key, device=device)
    tl.compute_jacobian_stats(a)
    _mark("compute_jacobians + stats")

    # --- network centrality ---
    if compute_centrality:
        try:
            tl.compute_network_centrality(a, cluster_key=cluster_key)
            _mark("compute_network_centrality")
        except Exception as exc:  # centrality is non-critical; keep the pipeline going
            _mark(f"compute_network_centrality FAILED: {type(exc).__name__}: {exc}")

    log["n_cells"], log["n_genes"] = int(a.n_obs), int(a.n_vars)
    log["clusters"] = [str(c) for c in a.obs[cluster_key].astype(str).unique()]
    log["scaffold"] = scaffold is not None
    log["seed"] = seed
    a.uns["scHopfield_pipeline"] = log
    if verbose:
        print(f"[scHopfield] pipeline complete: {a.n_obs} cells x {a.n_vars} genes, "
              f"{len(log['clusters'])} clusters", flush=True)
    return a
