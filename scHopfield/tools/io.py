"""Save and load fitted model parameters."""

import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
from anndata import AnnData


# adata.var columns written by fit_all_sigmoids (gene-level, not per-cluster)
_VAR_BASE_KEYS = [
    'scHopfield_used',
    'sigmoid_threshold',
    'sigmoid_exponent',
    'sigmoid_offset',
    'sigmoid_mse',
]

# Primitive-valued keys in uns['scHopfield'] to persist
_UNS_META_KEYS = ['spliced_key', 'velocity_key', 'degradation_key', 'cluster_key']


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
    - Sigmoid parameters (threshold, exponent, offset, mse) from adata.var
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
    AnnData or None
        Returns a subsetted copy of adata when the gene sets differ, otherwise
        returns ``None`` (parameters written in-place).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the model needs genes that are absent from adata.var_names.

    Examples
    --------
    >>> # Gene sets match exactly — in-place, no return value needed:
    >>> sch.tl.load_model(adata, 'model.h5sch')
    >>>
    >>> # adata has more genes than the model — reassign the return value:
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

    n_genes = int(target.var['scHopfield_used'].sum()) if 'scHopfield_used' in target.var else '?'
    print(f"Model loaded from '{filename}'  |  clusters={clusters}  |  genes={n_genes}")

    if target is not adata:
        return target
    return None
