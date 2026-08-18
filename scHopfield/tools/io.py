"""Save and load fitted model parameters."""

import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
from anndata import AnnData


# adata.var columns written by fit_all_sigmoids (gene-level, not per-cluster).
# All seven sigmoid columns are written by either fit mode: a single-Hill fit sets the
# second component equal to the first and the mixing weight to 1, so persisting the whole
# set is what lets compute_sigmoid rebuild the activation the model was actually fit with.
_VAR_BASE_KEYS = [
    'scHopfield_used',
    'sigmoid_threshold',
    'sigmoid_exponent',
    'sigmoid_threshold2',
    'sigmoid_exponent2',
    'sigmoid_mix',
    'sigmoid_offset',
    'sigmoid_mse',
]

# The second Hill component: absent from files written before it was persisted
_SIGMOID_BIMODAL_KEYS = {'sigmoid_mix', 'sigmoid_threshold2', 'sigmoid_exponent2'}

# Primitive-valued keys in uns['scHopfield'] to persist
_UNS_META_KEYS = ['spliced_key', 'velocity_key', 'degradation_key', 'cluster_key',
                  'sigmoid_bimodal']


def _is_fitted(adata: AnnData) -> bool:
    """Return True if at least one W matrix is present in adata.varp."""
    return any(k.startswith('W_') for k in adata.varp)


def save_model(
    adata: AnnData,
    filename: str,
    overwrite: bool = False,
    compression: str = 'gzip',
) -> None:
    """
    Save fitted model parameters to an HDF5 file.

    Saves the parameters that define the Hopfield network:

    - Interaction matrices W (per cluster) from adata.varp
    - Bias vectors I and degradation rates gamma (per cluster) from adata.var
    - Sigmoid parameters from adata.var: both Hill components (threshold,
      exponent, threshold2, exponent2), their mixing weight (mix), and the
      offset and mse
    - Gene mask (scHopfield_used) from adata.var
    - Scalar metadata (spliced_key, cluster_key, etc.) from adata.uns['scHopfield']

    Cell-level derived quantities (energies, UMAP, Jacobians) are NOT saved
    here — use the dedicated save_embedding() and save_jacobians() for those.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted model (after fit_interactions).
    filename : str
        Path for the HDF5 output file.
    overwrite : bool, optional (default: False)
        If False, skip saving when the file already exists (prints a warning).
        Pass True to overwrite an existing file.
    compression : str, optional (default: 'gzip')
        HDF5 compression algorithm ('gzip', 'lzf', or None to disable).

    Raises
    ------
    ValueError
        If no W matrices are found in adata.varp (model not fitted yet).

    Examples
    --------
    >>> sch.tl.save_model(adata, 'model.h5sch')
    >>> # Next session:
    >>> sch.tl.load_model(adata, 'model.h5sch')
    """
    import h5py

    path = Path(filename)
    if path.exists() and not overwrite:
        warnings.warn(
            f"'{filename}' already exists. Pass overwrite=True to replace it.",
            stacklevel=2,
        )
        return

    if not _is_fitted(adata):
        raise ValueError(
            "No fitted W matrices found in adata.varp. Run fit_interactions() first."
        )

    clusters = sorted(k[2:] for k in adata.varp if k.startswith('W_'))

    with h5py.File(filename, 'w') as f:
        # Gene names — used on load to verify compatibility
        f.create_dataset(
            'gene_names',
            data=np.array(adata.var_names, dtype=h5py.string_dtype()),
        )
        f.attrs['n_vars'] = adata.n_vars
        f.attrs['clusters'] = json.dumps(clusters)

        # Primitive uns['scHopfield'] metadata
        sch_uns = adata.uns.get('scHopfield', {})
        meta = {
            k: v for k, v in sch_uns.items()
            if k in _UNS_META_KEYS and isinstance(v, (str, int, float))
        }
        f.attrs['uns_scHopfield'] = json.dumps(meta)

        # Gene-level parameters (adata.var columns)
        var_grp = f.create_group('var')
        for key in _VAR_BASE_KEYS:
            if key in adata.var:
                var_grp.create_dataset(
                    key, data=adata.var[key].values, compression=compression
                )
        for cluster in clusters:
            for prefix in ('I_', 'gamma_'):
                col = f'{prefix}{cluster}'
                if col in adata.var:
                    var_grp.create_dataset(
                        col, data=adata.var[col].values, compression=compression
                    )

        # Interaction matrices (adata.varp)
        varp_grp = f.create_group('varp')
        for cluster in clusters:
            key = f'W_{cluster}'
            varp_grp.create_dataset(
                key, data=np.array(adata.varp[key]), compression=compression
            )

    n_genes = int(adata.var['scHopfield_used'].sum()) if 'scHopfield_used' in adata.var else '?'
    print(f"Model saved to '{filename}'  |  clusters={clusters}  |  genes={n_genes}")


def load_model(
    adata: AnnData,
    filename: str,
    overwrite: bool = False,
) -> Optional[AnnData]:
    """
    Load fitted model parameters from an HDF5 file into adata.

    Restores W matrices, bias vectors, sigmoid parameters, degradation rates,
    and scalar metadata from a file created by save_model().

    **Two behaviours depending on gene compatibility:**

    * **Exact gene match** — model parameters are written directly into the
      ``adata`` object that was passed (in-place).  Returns ``None``.
    * **adata is a superset** — a subsetted copy of adata is created that
      contains only the model genes, parameters are loaded into that copy, and
      the copy is returned.  The original ``adata`` is *not* modified.
      Reassign the return value: ``adata = sch.tl.load_model(adata, file)``.

    Parameters
    ----------
    adata : AnnData
        Annotated data object to load parameters into.  Must contain at least
        all genes present in the saved model.
    filename : str
        Path to the HDF5 model file created by save_model().
    overwrite : bool, optional (default: False)
        If False, skip loading when fitted parameters are already present in
        adata (W matrices found in adata.varp). Pass True to always reload.

    Returns
    -------
    AnnData
        The AnnData with the model loaded. This is the same object when the gene
        sets match (modified in place) or a gene-subsetted copy when they differ,
        so ``adata = load_model(adata, file)`` is always safe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the model needs genes that are absent from adata.var_names.

    Warns
    -----
    UserWarning
        If the file stores only the first Hill component of the activation, which
        checkpoints written before save_model persisted the second one do.

    Examples
    --------
    >>> adata = sch.tl.load_model(adata, 'model.h5sch')
    """
    import h5py

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: '{filename}'")

    if _is_fitted(adata) and not overwrite:
        warnings.warn(
            "Fitted model parameters already present in adata. "
            "Pass overwrite=True to reload from file.",
            stacklevel=2,
        )
        return None

    with h5py.File(filename, 'r') as f:
        saved_genes = np.array(f['gene_names']).astype(str)
        current_genes = np.array(adata.var_names, dtype=str)

        # target is what we write into; replaced by a copy when subsetting
        target = adata

        if not np.array_equal(saved_genes, current_genes):
            missing = saved_genes[~np.isin(saved_genes, current_genes)]
            if missing.size > 0:
                preview = ', '.join(missing[:5].tolist())
                suffix = ', ...' if missing.size > 5 else ''
                raise ValueError(
                    f"Gene names in '{filename}' are not a subset of adata.var_names. "
                    f"{missing.size} missing gene(s): {preview}{suffix}. "
                    "The model was fitted on a different gene set."
                )
            # Build ordered index: saved gene i → position in current adata
            lookup = {g: i for i, g in enumerate(current_genes)}
            ordered_idx = np.array([lookup[g] for g in saved_genes])
            warnings.warn(
                f"adata has {len(current_genes)} genes but the model was trained on "
                f"{len(saved_genes)}.  A subsetted copy is being returned; "
                "the original adata is NOT modified.  Reassign the return value:\n"
                "    adata = sch.tl.load_model(adata, filename)",
                stacklevel=2,
            )
            target = adata[:, ordered_idx].copy()

        clusters = json.loads(f.attrs['clusters'])

        # Restore uns['scHopfield'] primitive metadata
        meta = json.loads(f.attrs.get('uns_scHopfield', '{}'))
        if 'scHopfield' not in target.uns:
            target.uns['scHopfield'] = {}
        target.uns['scHopfield'].update(meta)

        # Restore var columns
        var_grp = f['var']
        for key in var_grp:
            target.var[key] = var_grp[key][:]

        # Restore W matrices
        varp_grp = f['varp']
        for key in varp_grp:
            target.varp[key] = varp_grp[key][:]

        # A checkpoint written before save_model persisted the second Hill component
        # carries the first one alone. compute_sigmoid would then rebuild an ordinary
        # single-Hill activation, which is not the one a bimodal fit was made with, so
        # say it here, where the file is what shows the loss.
        var_keys = set(var_grp)
        if 'sigmoid_threshold' in var_keys and not _SIGMOID_BIMODAL_KEYS <= var_keys:
            warnings.warn(
                f"'{filename}' stores only the first Hill component of the activation "
                f"({', '.join(sorted(_SIGMOID_BIMODAL_KEYS - var_keys))} absent). It was "
                "written before save_model persisted the second one. If the model was "
                "fitted with bimodal=True, which is the default, rebuilding the "
                "activation from this file alone gives a single Hill the model was not "
                "fitted with. Re-run sch.pp.fit_all_sigmoids(adata, bimodal=True) on the "
                "expression data, or write the checkpoint again from a current fit.",
                stacklevel=2,
            )

    n_genes = int(target.var['scHopfield_used'].sum()) if 'scHopfield_used' in target.var else '?'
    print(f"Model loaded from '{filename}'  |  clusters={clusters}  |  genes={n_genes}")

    # Always return the loaded object so ``adata = load_model(adata, file)`` is
    # safe whether the model was applied in place (gene sets match) or a
    # gene-subsetted copy was created.
    return target


