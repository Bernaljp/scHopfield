"""Common documentation templates for scHopfield."""

ADATA_PARAM = """\
adata : AnnData
    Annotated data object containing single-cell expression data."""

GENES_PARAM = """\
genes : None, list of str, list of int, or array of bool, optional (default: None)
    Gene subset to use. If None, uses all genes. Can be gene names,
    indices, or boolean mask."""

CLUSTER_KEY_PARAM = """\
cluster_key : str, optional
    Key in adata.obs containing cluster/cell type annotations."""

COPY_PARAM = """\
copy : bool, optional (default: False)
    If True, return a copy of adata instead of modifying in-place."""

RETURNS_NONE = """\
Returns
-------
None
    Modifies adata in-place."""

RETURNS_COPY = """\
Returns
-------
AnnData or None
    Returns adata if copy=True, otherwise None."""
