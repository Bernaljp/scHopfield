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

The seven datasets behind the paper's figures are public, each under its own accession,
and none is redistributed here either. They are not fetched automatically: point the
figure code at your own copies with `SCHOPFIELD_DATA` and `SCHOPFIELD_DYNAMISC_DATA`, as
described in `README.md` and `reproducibility/paths.py`. The per-dataset accessions and
source papers belong with the reproducibility documentation rather than here.

The one exception is the pancreatic endocrinogenesis dataset used in the tutorials, which
scVelo downloads itself through `scvelo.datasets.pancreas()`.
