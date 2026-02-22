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
) -> None:
    """
    Load fitted model parameters from an HDF5 file into adata (in-place).

    Restores W matrices, bias vectors, sigmoid parameters, degradation rates,
    and scalar metadata from a file created by save_model().

    Parameters
    ----------
    adata : AnnData
        Annotated data object to load parameters into. Must have the same
        gene ordering as the adata used when the model was saved.
    filename : str
        Path to the HDF5 model file created by save_model().
    overwrite : bool, optional (default: False)
        If False, skip loading when fitted parameters are already present in
        adata (W matrices found in adata.varp). Pass True to always reload.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the gene names in the file do not match adata.var_names.

    Examples
    --------
    >>> sch.tl.load_model(adata, 'model.h5sch')
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
        return

    with h5py.File(filename, 'r') as f:
        # Verify gene compatibility before touching adata
        saved_genes = np.array(f['gene_names']).astype(str)
        current_genes = np.array(adata.var_names, dtype=str)
        if not np.array_equal(saved_genes, current_genes):
            raise ValueError(
                f"Gene names in '{filename}' do not match adata.var_names. "
                "The model was fitted on a different gene set or gene ordering."
            )

        clusters = json.loads(f.attrs['clusters'])

        # Restore uns['scHopfield'] primitive metadata
        meta = json.loads(f.attrs.get('uns_scHopfield', '{}'))
        if 'scHopfield' not in adata.uns:
            adata.uns['scHopfield'] = {}
        adata.uns['scHopfield'].update(meta)

        # Restore var columns
        var_grp = f['var']
        for key in var_grp:
            adata.var[key] = var_grp[key][:]

        # Restore W matrices
        varp_grp = f['varp']
        for key in varp_grp:
            adata.varp[key] = varp_grp[key][:]

    n_genes = int(adata.var['scHopfield_used'].sum()) if 'scHopfield_used' in adata.var else '?'
    print(f"Model loaded from '{filename}'  |  clusters={clusters}  |  genes={n_genes}")
