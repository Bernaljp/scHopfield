# scHopfield plotting library — review

Assessment after driving ~15 `sch.pl.*` / `sch.tl.*` functions across 6 datasets to build
the per-dataset reports. Overall the plotting library is **well-suited to programmatic and
notebook use**; the issues below are mostly consistency/ergonomics, with one real
robustness bug (fixed).

## What is good
- **No side effects for batch use.** No `plt.show()` and no hardcoded `savefig` anywhere,
  so headless generation is safe (the caller owns the figure).
- **Composable.** Most functions accept an `ax=` (flow.py, networks.py, energy landscape,
  genes) so panels can be assembled into grids.
- **Consistent analysis params.** `cluster_key`, `order`, `colors`, `figsize` are uniform;
  colors default to `adata.uns[f'{cluster_key}_colors']`.

## Problems / improvements
1. **Inconsistent return types (main issue).** Functions variously return a `Figure`
   (`plot_energy_boxplots(..., plot_energy='all')`), an `Axes`
   (`plot_network_centrality_rank`, `plot_eigenvalue_spectrum`, `plot_grn_network`), an
   `ndarray` of Axes (grid plotters), or `None`. A caller that wants to save/compose must
   handle every case (the report uses a `_as_figure()` shim). **Recommend:** standardize on
   returning `fig` (or `(fig, axes)`), or at minimum document the return per function.
2. **Same function returns different types by argument.** `plot_energy_boxplots` returns a
   `Figure` for `plot_energy='all'` but an `Axes` otherwise; likewise `plot_energy_scatters`.
   Branch-dependent return types are surprising.
3. **No `save=` / `path=` convenience.** Every caller re-implements "capture return ->
   `savefig`". A thin `save=`/`dpi=` kwarg (or a `sch.pl.save(obj, path)` helper) would
   remove boilerplate.
4. **Uneven `ax` support.** `jacobian.py` grid plotters and a couple of energy functions do
   not accept `ax`, so they cannot be embedded in a shared figure like the others.
5. **Robustness to non-finite weights (fixed).** `network_correlations` called
   `np.linalg.svd(W)` directly; a dataset with NaN velocities (dynamo hematopoiesis: 244k
   NaN velocities) yields NaN `W` and crashes with "SVD did not converge". Guarded with
   `np.nan_to_num` before the SVD (`scHopfield/tools/networks.py`). Upstream, the report
   pipeline now also sanitizes `Ms`/`velocity_S` before fitting.

## Suggested priorities
- (High) Standardize return types to `fig`; keep `ax=` for composition.
- (Med) Add a `save=`/`dpi=` convenience or a `sch.pl.save` helper.
- (Med) Add `ax=` to the jacobian grid plotters.
- (Done) Guard `network_correlations` against non-finite `W`.
