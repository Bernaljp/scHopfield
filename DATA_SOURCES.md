# Data sources

scHopfield distributes no third-party data. This file records what the package and the
figure code fetch from elsewhere, where it comes from, and what terms it carries.

## Base regulatory network scaffolds

`sch.inf.fit_interactions` learns a gene regulatory network restricted by a
prior-knowledge scaffold, and that scaffold is built from a *base GRN*: a wide table with
one row per target gene and one binary column per transcription factor. Fitting without a
scaffold is a different, and measurably worse, method, so the base GRN is not optional
for reproducing the paper.

**scHopfield does not distribute a base GRN.** `sch.fetch_base_grn` downloads one from
the CellOracle repository on demand and caches it locally:

```python
import scHopfield as sch

base_grn = sch.fetch_base_grn("mouse")          # or "human"
scaffold = sch.inf.build_scaffold(adata, base_grn)
```

The first call downloads; later calls read the cache, so a notebook is offline-safe after
its first run. The cache is `~/.cache/scHopfield/base_grn`, or `$SCHOPFIELD_CACHE/base_grn`
if that variable is set.

### License, which is not what CellOracle's packaging says

**These tables are not covered by scHopfield's MIT license.** CellOracle distributes them
under a modified Apache License 2.0 whose header reads:

> The software is provided under a modified Apache License Version 2.0.
> The software may be used under the Apache License below for NON-COMMERCIAL ACADEMIC
> PURPOSES only. For any other use of the Work, including commercial use, please contact
> Morris lab.

Complying with those terms is the responsibility of whoever fetches the table. Note that
CellOracle's own PyPI metadata declares plain `Apache-2.0`, which contradicts the
`LICENSE` file shipped in the same distribution; anyone who reads only the package
classifier will reach the wrong conclusion.

This restriction is the reason for the fetch-on-demand design. scHopfield is MIT, which
permits commercial use and sublicensing, so committing a CellOracle-derived table into
this repository would offer it onward under terms we do not hold. Downloading a file from
its publisher is not redistribution, so no copy is ever made by this project.

The upstream chain does not lift the restriction. The mouse sci-ATAC-seq atlas posts no
license at GEO, on its portal, or in its code repository, and the `gimme.vertebrate.v5.0`
motif set behind the promoter tables ships with no license either. Every layer is
unlicensed rather than permissively licensed, and silence is not a grant.

### The three tables

Each is addressed by a pinned commit rather than by a branch, so a reorganization
upstream cannot change what is fetched, and every download is checked against the sha256
below before it is cached or returned.

| Name | Table | Pinned commit | Bytes | sha256 |
| :--- | :--- | :--- | ---: | :--- |
| `mouse_atlas` | mouse sci-ATAC-seq atlas, mm9, `v202204` | `77cd39fdf77ec931d55437b8b728830bf0b38ee5` | 9,448,146 | `adb95f68f3b03c8522eb9cf55a0a4235bc663c991a10dc655902c1f1cd854129` |
| `human_promoter` | human promoter, hg19, gimmemotifs v5, `fpr2` | `22b29abc469a361ce14e6249ca1413bb91e327fc` | 5,520,100 | `2be7c71cc37906805e8fdc8a717f4d31406e1b0b566cbfcc1564acddbc0caa8c` |
| `mouse_promoter` | mouse promoter, mm10, gimmemotifs v5, `fpr2` | `22b29abc469a361ce14e6249ca1413bb91e327fc` | 4,633,084 | `32de319419199f53bbdbf7c8545394a70052c7d14918d59b54c96593000dea40` |

The species aliases `"mouse"` and `"human"` resolve to `mouse_atlas` and
`human_promoter`, which are the priors those species were fit with throughout the paper.
Five of the seven datasets use the mouse atlas and two use the human promoter table. The
mouse promoter table is the second arm of the driver-stability analysis, which compares a
fit under the atlas prior against one under the promoter prior.

The genome and threshold in each row are load-bearing. The human table is hg19, not hg38,
and both promoter tables are `fpr2`, not `fpr1`. A different choice gives a different
scaffold and different fitted parameters.

### Getting a table without this helper

Both alternatives are named in the error message whenever a fetch fails, because there is
no local copy to fall back on.

1. Download it by hand from
   `https://raw.githubusercontent.com/morris-lab/CellOracle/<commit>/<path>` and put it in
   the cache directory under its original filename. The paths under `celloracle/data/`
   are `TFinfo_data/mm9_mouse_atac_atlas_data_TSS_and_cicero_0.9_accum_threshold_10.5_DF_peaks_by_TFs_v202204.parquet`,
   `promoter_base_GRN/hg19_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet`
   and `promoter_base_GRN/mm10_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet`.
2. Install CellOracle and call its own loader, which fetches the same file:
   `celloracle.data.load_mouse_scATAC_atlas_base_GRN()`,
   `celloracle.data.load_human_promoter_base_GRN(version="hg19_gimmemotifsv5_fpr2")` or
   `celloracle.data.load_mouse_promoter_base_GRN(version="mm10_gimmemotifsv5_fpr2")`.

A table that does not match the recorded sha256 is rejected rather than used, since a
silently different prior would produce a plausible scaffold from the wrong information
and nothing downstream would notice. The tables in this project's own working tree were
re-serialized by pandas and so do not match byte for byte, although their contents are
identical; a copy dropped into the cache has to be the file CellOracle distributes.

### Citation

If you use a base GRN, please cite the resources it is built from. CellOracle asks only
for its own paper, so crediting the rest is a scientific norm here rather than a
documented requirement of the distributor.

- Kamimoto, K., et al. Dissecting cell identity via network inference and in silico gene
  perturbation. *Nature* 614, 742-751 (2023). doi:10.1038/s41586-022-05688-9
- Cusanovich, D. A., et al. A single-cell atlas of in vivo mammalian chromatin
  accessibility. *Cell* 174, 1309-1324 (2018). doi:10.1016/j.cell.2018.06.052 (the mouse
  sci-ATAC-seq atlas underlying the mouse base GRN)
- Pliner, H. A., et al. Cicero predicts cis-regulatory DNA interactions from single-cell
  chromatin accessibility data. *Molecular Cell* 71, 858-871 (2018).
  doi:10.1016/j.molcel.2018.06.044 (the co-accessibility linkage used to build it)
- Bruse, N. and van Heeringen, S. J. GimmeMotifs: an analysis framework for transcription
  factor motif analysis. https://github.com/vanheeringen-lab/gimmemotifs (the
  `gimme.vertebrate.v5.0` motif set behind the promoter base GRNs)
- Weirauch, M. T., et al. Determination and inference of eukaryotic transcription factor
  sequence specificity. *Cell* 158, 1431-1443 (2014). doi:10.1016/j.cell.2014.08.009
  (CIS-BP, the principal source of that motif set)

## Single-cell datasets

The seven dataset entries in `reproducibility/config.py` read six objects: `paul15` and
`paul15_coarse` are the same object under a fine and a coarse annotation. All are public
under their own accession and none is redistributed here. They are not fetched
automatically: point the figure code at your own copies with `SCHOPFIELD_DATA` and
`SCHOPFIELD_DYNAMISC_DATA`, as described in `README.md` and `reproducibility/paths.py`.

An accession says where the raw data is. It does not say how to get the object that was
actually fit, which is a preprocessed one, and for four of the six objects that
preprocessing happened before the data reached this project. The table says which is
which, so nothing here implies a rebuild that does not exist.

| `config.py` entry | Object the fit reads | Source | Accession | First step |
| :--- | :--- | :--- | :--- | :--- |
| `pancreas` | `Pancreas/pancreas_scvelo_ready.h5ad` | Bastidas-Ponce et al. 2019 | GEO GSE132188 | `scvelo.datasets.pancreas()`, then `reproducibility/prep_pancreas.py` |
| `paul15`, `paul15_coarse` | `hematopoiesis/base_preprocessed.h5ad` | Paul et al. 2015 | GEO GSE72857 | CellOracle's own tutorial object, plus the steps below |
| `dynamo_hematopoiesis` | `<SCHOPFIELD_DYNAMISC_DATA>/hematopoiesis.h5ad` | Qiu et al. 2022 | GEO GSE193517 | `dynamo.sample_data.hematopoiesis()` |
| `murine_nc` | `generalize/murine_nc.h5ad` | Qiu et al. 2024 | GEO GSE186069, GSE228590 | preprocessed elsewhere, then `reproducibility/prep_dataset.py` |
| `schwann` | `<SCHOPFIELD_DYNAMISC_DATA>/schwann.h5ad` | Kastriti et al. 2022 | GEO GSE201257 | preprocessed elsewhere, then prepared on load by `prepare=True` |
| `human_limb` | `generalize/human_limb.h5ad` | Zhang et al. 2024 | ArrayExpress E-MTAB-8813 | preprocessed elsewhere, then `reproducibility/prep_dataset.py` |

### What an object has to carry

A fit does not need a particular provenance, it needs particular fields. Any object with
these can be added to `config.py` and fitted:

- `layers['Ms']`, smoothed expression, or the name of a layer to map onto it (`ms_layer`,
  which is how the Dynamo object's `M_t` is used)
- dynamics, either `layers['velocity_S']` or a pseudotime column named by
  `pseudotime_key`, per the entry's `velocity_mode`
- `var['gamma']`, per-gene degradation rates, refit during inference
- a cell-type column in `obs`, named by `cluster_key`
- `obsp['connectivities']`, computed on load when absent

`sch.pp.prepare_dataset` produces all of them from spliced and unspliced counts, which is
what both preparation scripts call. One absence is worth knowing about before it bites: a
pseudotime entry whose named column is missing does not fail, it falls back to a diffusion
pseudotime computed on the spot, which fits different dynamics from the published ones.

### Pancreatic endocrinogenesis, which rebuilds exactly

This is the dataset behind five of the six figures that read the report tree, and the one
used in the tutorials. scVelo downloads the raw object and
`reproducibility/prep_pancreas.py` applies the recipe the fitted object was made with:

```bash
python reproducibility/prep_pancreas.py     # about a minute, CPU
```

Verified, not assumed: run into an empty data root from an unrelated working directory,
the result is the object the published pancreas fits were run on, bit for bit. Same 3,696
cells and 2,000 genes in the same order, and `X`, `layers['Ms']`, `layers['velocity_S']`
and `var['gamma']` all identical to zero difference.

### Mouse hematopoiesis, which rebuilds except for one column

This object is CellOracle's own Paul et al. tutorial object with our layers added, so its
first step is a CellOracle call rather than a GEO download:

```python
import celloracle as co
adata = co.data.load_Paul2015_data()        # anndata/Paul_etal_v202204.h5ad, 66,291,080 bytes
```

The same file can be downloaded directly from
`https://raw.githubusercontent.com/morris-lab/CellOracle/<commit>/celloracle/data/anndata/Paul_etal_v202204.h5ad`,
which needs no CellOracle install. Either way the terms in the base GRN section above
apply, since this is CellOracle's material too.

What that object gives you, measured against the object we fit: the same 2,671 cells and
1,999 genes in the same order, `layers['raw_count']` and `obsm['X_pca']` identical bit for
bit, and `obs['paul15_clusters']` and `obs['cell_type']` identical. Two fields are
recomputed rather than shipped and both come back within float32 rounding: the
log-normalized matrix, which is `sc.pp.normalize_per_cell` then `sc.pp.log1p` of
`raw_count`, to 7.2e-06; and `layers['Ms']`, which is that matrix assigned to `spliced`
and `unspliced` followed by `scv.pp.moments(n_pcs=30, n_neighbors=30)` over the object's
own PCA, to 2.4e-06.

**One field does not rebuild.** `obs['Pseudotime']`, which is what this dataset is fitted
from, is CellOracle's lineage pseudotime, produced by their pseudotime tutorial rather than
by any code here. The downloadable object carries `obs['dpt_pseudotime']` instead, and the
two are not the same ordering: Pearson 0.59, rank correlation 0.72. Substituting it, or
letting the fallback above compute a fresh diffusion pseudotime, gives a fit that is not
the published one. There is no preparation script for this dataset for that reason: one
that produced an object without that column would look complete and fit something else.

### Human hematopoiesis, which is a download

The object is the processed one Dynamo distributes, not a reprocessing of the GEO series:

```python
import dynamo as dyn
adata = dyn.sample_data.hematopoiesis()     # hematopoiesis.h5ad, 219,336,191 bytes
```

Verified against the copy this project fit: identical size, and identical md5 over the
first and last mebibyte. This is also why the accession's sample count and the object's
1,947 cells do not agree, and are not expected to.

### Murine neural crest, Schwann-cell lineage and human limb, which were preprocessed elsewhere

These three arrived already filtered, normalized, smoothed into `layers['Ms']` and given a
neighbor graph, from a preprocessing pipeline that is not part of this project and is not
reproducible from it. Two of them were then prepared with `reproducibility/prep_dataset.py`
and written out, and the Schwann-cell object is prepared on load instead, by `prepare=True`
in its `config.py` entry. Both routes call the same `sch.pp.prepare_dataset`.

For an object in that state the preparation adds only the velocity layer, the degradation
rates and the sigmoid fits: the smoothing and the neighbor graph are the upstream ones,
which is visible in the neural crest object's 50-neighbor graph where this project's own
setting is 30. Verified on that object: `layers['Ms']`, `layers['velocity_S']` and
`var['gamma']` reproduce bit for bit from its preprocessed input. Its sigmoid columns do
not, because the package now fits the two-component Hill by default and these objects were
prepared when the single-component fit was the default. Nothing downstream depends on it,
and both consumers refit rather than read: the report pipeline at `config.HILL_N_MAX` and
`config.BIMODAL_HILL`, and `reproducibility/identifiability_multi.py` at the pinned recipe
in its own `SIGMOID_FIT`. That second one is deliberate rather than incidental. The sweep
reads four objects, three of which arrive carrying a `sigmoid` layer, so using them would
make a published number depend on which era each object on disk was prepared in. Refitting
every dataset removes that dependence and reproduces the committed JSON exactly.

A reader starting from the accession has to reprocess to that state, and the result will
resemble the fits reported here rather than match them.

### What this costs a reader

- The pancreas branch of the pipeline rebuilds end to end from a package call, which
  covers the tutorials and five of the six figures that read the report tree.
- The sixth, the framework figure, is drawn on mouse hematopoiesis, whose object rebuilds
  in every field except its pseudotime column.
- The Dynamo object needs one package call and no reprocessing.
- The remaining three datasets, which are the neural crest, Schwann-cell and limb
  reproductions, cannot be rebuilt from this repository. Their accessions are above; the
  field contract they have to satisfy is above; the fits will be close rather than equal.

An eighth accession appears in the paper's data availability statement, GEO GSE122662
(Schiebinger et al. 2019), for the MEF-to-iPSC reprogramming control. No code in this
repository reads it.

### Citations

Please cite the source paper of any dataset you use.

- Bastidas-Ponce, A., et al. Comprehensive single cell mRNA profiling reveals a detailed
  roadmap for pancreatic endocrinogenesis. *Development* 146, dev173849 (2019).
  doi:10.1242/dev.173849
- Paul, F., et al. Transcriptional heterogeneity and lineage commitment in myeloid
  progenitors. *Cell* 163, 1663-1677 (2015). doi:10.1016/j.cell.2015.11.013
- Qiu, X., et al. Mapping transcriptomic vector fields of single cells. *Cell* 185,
  690-711 (2022). doi:10.1016/j.cell.2021.12.045
- Qiu, C., et al. A single-cell time-lapse of mouse prenatal development from gastrula to
  birth. *Nature* 626, 1084-1093 (2024). doi:10.1038/s41586-024-07069-w
- Kastriti, M. E., et al. Schwann cell precursors represent a neural crest-like state with
  biased multipotency. *The EMBO Journal* 41, e108780 (2022).
  doi:10.15252/embj.2021108780
- Zhang, B., et al. A human embryonic limb cell atlas resolved in space and time. *Nature*
  635, 668-678 (2024). doi:10.1038/s41586-023-06806-x
