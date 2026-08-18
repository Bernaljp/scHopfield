"""Build a prior-knowledge scaffold (regulator -> target mask) for GRN inference.

A *scaffold* is a binary (n_genes x n_genes) matrix that restricts which
gene-gene interactions :func:`scHopfield.inference.fit_interactions` is allowed to
learn (or penalizes away from). It is typically derived from a base GRN such as a
CellOracle motif-scan parquet, an ATAC-derived TF->peak->gene map, or any
long-format edge list. :func:`scHopfield.inference.fetch_base_grn` downloads one of
the three CellOracle tables scHopfield was developed against.

The same scaffold-construction logic was previously copy-pasted across several
analysis scripts; it now lives here so every workflow builds the scaffold the
same way.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

__all__ = ["build_scaffold", "scaffold_from_edges"]


def _resolve_genes(adata: AnnData, genes: Optional[Sequence[str]], used_key: str) -> pd.Index:
    """Return the ordered gene index the scaffold should span."""
    if genes is not None:
        return pd.Index(list(genes))
    if used_key in adata.var and adata.var[used_key].any():
        return adata.var_names[adata.var[used_key].values]
    return adata.var_names


def build_scaffold(
    adata: AnnData,
    base_grn: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    gene_col: str = "gene_short_name",
    tf_columns: Optional[Sequence[str]] = None,
    drop_columns: Sequence[str] = ("peak_id",),
    used_key: str = "scHopfield_used",
    case_insensitive: bool = True,
    return_stats: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int, int]]:
    """Build a regulator-by-target scaffold from a wide base-GRN table.

    The base GRN is the CellOracle-style *wide* format: one column of target gene
    names (``gene_col``) plus one binary column per transcription factor, where a
    ``1`` marks a putative TF -> target edge. This is converted into a square
    ``genes x genes`` scaffold restricted to the genes scHopfield is modelling.

    Parameters
    ----------
    adata
        Annotated data. Used only to determine the gene universe (via ``genes``
        or ``adata.var[used_key]``).
    base_grn
        Wide base-GRN table (targets in rows, TFs in columns, plus ``gene_col``).
    genes
        Explicit ordered gene list for the scaffold. If ``None``, uses the genes
        flagged by ``adata.var[used_key]`` (falling back to all ``var_names``).
    gene_col
        Column in ``base_grn`` holding the target gene symbol.
    tf_columns
        TF columns to consider. If ``None``, every column except ``gene_col`` and
        ``drop_columns`` is treated as a candidate TF.
    drop_columns
        Columns to ignore (e.g. ``peak_id``). Missing columns are skipped.
    used_key
        ``adata.var`` boolean column marking modelled genes (used when ``genes`` is
        ``None``).
    case_insensitive
        Match TF and target names ignoring case (recommended: base GRNs and
        AnnData often differ in capitalization).
    return_stats
        If ``True``, also return ``(n_tfs, n_edges)``.

    Returns
    -------
    scaffold : :class:`pandas.DataFrame`
        Square ``genes x genes`` matrix. ``scaffold.loc[tf, target] == 1`` marks an
        allowed regulator -> target edge. Pass ``scaffold.values.T`` as the
        ``w_scaffold`` argument of :func:`fit_interactions`, whose ``W`` is indexed
        ``[target, regulator]``.
    n_tfs, n_edges : int
        Only if ``return_stats=True``.

    Examples
    --------
    >>> import scHopfield as sch
    >>> base = sch.fetch_base_grn("mouse")
    >>> scaffold = sch.inf.build_scaffold(adata, base)
    >>> sch.inf.fit_interactions(adata, cluster_key="celltype",
    ...                          w_scaffold=scaffold.values.T,
    ...                          scaffold_regularization=0.1, only_TFs=True)
    """
    gene_index = _resolve_genes(adata, genes, used_key)
    base = base_grn.copy()
    for col in drop_columns:
        if col in base.columns:
            base = base.drop(columns=col)
    if gene_col not in base.columns:
        raise KeyError(
            f"base_grn has no target column '{gene_col}'. "
            f"Available columns: {list(base.columns)[:8]}..."
        )

    if tf_columns is None:
        tf_columns = [c for c in base.columns if c != gene_col]
    scaffold = pd.DataFrame(0, index=gene_index, columns=gene_index, dtype=np.int8)

    if case_insensitive:
        row_map = {g.lower(): g for g in scaffold.index}
        col_map = {g.lower(): g for g in scaffold.columns}
        tf_lut = {c.lower(): c for c in tf_columns}
        shared_tfs = [tf_lut[k] for k in (set(tf_lut) & set(row_map))]

        def _target_key(name):
            return str(name).lower()
    else:
        row_map = {g: g for g in scaffold.index}
        col_map = {g: g for g in scaffold.columns}
        shared_tfs = [c for c in tf_columns if c in row_map]

        def _target_key(name):
            return str(name)

    for tf_col in shared_tfs:
        tf_gene = row_map[tf_col.lower()] if case_insensitive else tf_col
        targets = base.loc[base[tf_col] == 1, gene_col]
        for tgt in targets:
            key = _target_key(tgt)
            if key in col_map:
                scaffold.loc[tf_gene, col_map[key]] = 1

    n_tfs = len(shared_tfs)
    n_edges = int(scaffold.values.sum())
    if return_stats:
        return scaffold, n_tfs, n_edges
    return scaffold


