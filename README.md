# scHopfield

Fit an interpretable dynamical model of gene regulation to single-cell data, and read
biology off it.

scHopfield learns a signed gene-gene interaction matrix per cell type from expression and
RNA velocity, in the Hopfield-like form

```
dx/dt = W sigma(x) - Gamma x + I
```

where `sigma` is a per-gene Hill activation fitted from the data, `Gamma` holds the
degradation rates and `I` the basal input. Because the fitted object is a vector field
rather than a black box, one fit yields an energy landscape, per-cell-type attractors and
their local stability from the Jacobian, and in-silico knockouts and overexpressions run
as trajectories of the fitted dynamics.

Prior knowledge enters as a scaffold: a base GRN restricts which transcription factor to
target edges may be nonzero. This is not optional dressing. In the paper's ablations the
scaffold is what makes the inference problem well posed, and dropping it costs accuracy on
both synthetic ground truth and real data.

## Install

Python 3.12 or newer. The floor comes from the dependencies rather than the syntax: scanpy
requires 3.12 and anndata requires 3.11, so anything lower cannot be satisfied.

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install .
```

Two extras are available:

```bash
pip install '.[velocity]'         # scVelo, needed by prepare_dataset
pip install '.[reproducibility]'  # what the figure scripts need on top of the package
```

`torch` is a hard dependency. Fitting runs on CPU; pass `device='cuda'` to use a GPU when
one is present.

## A minimal worked example

The input is an AnnData object that already carries spliced counts, an RNA velocity layer
and degradation rates. If you do not have them, `sch.pp.prepare_dataset` produces them
from spliced and unspliced counts through scVelo.

```python
import scanpy as sc
import scHopfield as sch

adata = sc.read_h5ad("my_data.h5ad")
# needs adata.layers['Ms'], adata.layers['velocity_S'], adata.var['gamma'],
# and a cell-type column in adata.obs

# 1. Fit the per-gene Hill activation.
sch.pp.fit_all_sigmoids(adata)
sch.pp.compute_sigmoid(adata)

# 2. Restrict the network with a base GRN, then fit it per cell type.
# Downloaded from CellOracle on first use and cached; see DATA_SOURCES.md for its terms.
base_grn = sch.fetch_base_grn("mouse")
scaffold = sch.inf.build_scaffold(adata, base_grn)
sch.inf.fit_interactions(
    adata,
    cluster_key="cell_type",
    w_scaffold=scaffold.values.T,   # W is indexed [target, regulator]
)

# 3. Read biology off the fitted model.
sch.tl.compute_energies(adata, cluster_key="cell_type")
sch.tl.compute_jacobians(adata, cluster_key="cell_type")
sch.tl.compute_jacobian_stats(adata)

sch.pl.plot_energy_landscape(adata, cluster="my_cell_type")
```

`sch.run_pipeline(adata, cluster_key="cell_type", base_grn=base_grn)` runs that whole
sequence in one call and returns the processed object.

**The defaults are the paper's method.** Every fitting parameter defaults to the value
used throughout the paper, `seed=0` included, so a call that tunes nothing reproduces the
published configuration. The scaffold is the one exception, because it depends on a base
GRN that is not ours to distribute: `sch.fetch_base_grn` downloads one from CellOracle at
a pinned commit, checks it against a recorded sha256 and caches it under
`~/.cache/scHopfield`. **That table carries CellOracle's license, not ours**, which
restricts use to non-commercial academic purposes. See
[DATA_SOURCES.md](DATA_SOURCES.md).

Persist a fit with `sch.tl.save_model`, not with `adata.write_h5ad`: the fitted optimizers
live in `adata.uns` and AnnData cannot serialize them.

## Documentation

API reference, data conventions and tutorials: https://schopfield.readthedocs.io

The public API is what each submodule's `__all__` declares, reachable as `sch.pp`,
`sch.inf`, `sch.tl`, `sch.pl`, `sch.dyn` and `sch.validation`.

## What is in this repository

```
scHopfield/         the package
tests/              its tests
docs/               the documentation source
reproducibility/    the figure code for the paper, and the inputs that can ship
```

`sch.validation` is worth calling out: it holds synthetic circuits whose interaction
matrix is known exactly, so the recovery claims can be checked without downloading
anything.

```bash
pytest -m slow    # fits the two circuits and pins them against their ground truth
```

## reproducibility/

The ten scripts that draw the paper's figures, flattened into one directory, together with
the configuration they were run with. `reproducibility/config.py` holds the exact
per-dataset settings and fit parameters behind every figure, and
`reproducibility/paths.py` documents the environment variables that locate everything the
scripts read.

What this does and does not give you:

**Four of the ten run from a clean clone.** Their inputs are committed here.

| Script | Figure | Input |
| :--- | :--- | :--- |
| `make_small_circuits_validation.py` | Figure 2 | a 41 KB fit cache |
| `make_dyngen_benchmark.py` | Figure 3 | committed ground truth, plus one rebuild step |
| `make_ed1_ablation.py` | Extended Data Fig. 1 | a small JSON, regenerated by `hill_vs_linear.py` |
| `make_cross_dataset.py --submission` | Extended Data Fig. 3 | a committed scalar cache, and one JSON regenerated by `identifiability_multi.py` |

Each script takes `--submission`, which lays the same panels out on one journal page and is
the variant that appears in the paper. Extended Data Fig. 3 runs from a clean clone only in
that form: the committed cache holds the per-dataset scalars, and the default poster
variant recomputes them from the report tree instead.

Figure 3 is a ground-truth benchmark, so its ground truth is committed: the dyngen
simulator's own CSV exports, from which every matrix is rebuilt bit for bit with no R and
no dyngen.

```bash
python reproducibility/dyngen/02_build_h5ad.py      # about 15 s per backbone, CPU
python reproducibility/make_dyngen_benchmark.py --submission
```

Output lands under `reproducibility/figures/`, which is ignored by git, so a run never
dirties the working tree. Scripts resolve their own location, so the working directory does
not matter.

**The other six do not.** They read a fitted `adata_analyzed.h5ad` from the per-dataset
report tree, which runs from 400 MB to 1.2 GB per dataset and is regenerated rather than
distributed. Figures 5 and 6 additionally need pickled perturbation caches that are not
committed; the scripts that produce them, `reproducibility/compute/`'s
`_perturb_dynamics_compute.py` and `_double_ko_compute.py`, are here, but they need the
fitted objects first. A script whose input is missing stops with a `FileNotFoundError`
naming the exact path it wanted.

**Neither the datasets nor the base GRN scaffolds are redistributed here.** The seven
datasets are public, each under its own accession, and have to be fetched before any fit
can be run. The base GRN tables are third-party material under their own terms, which do
not permit redistribution under this repository's license, so `config.py` names each one
by registry name and `sch.fetch_base_grn` downloads it on first use. See
[DATA_SOURCES.md](DATA_SOURCES.md).

**A committed input is not a reproducible one** unless something here writes it, so the two
producers behind the Extended Data figures ship as well. `reproducibility/hill_vs_linear.py`
simulates its own circuits and needs no data at all, so it rebuilds Extended Data Fig. 1's
input on a clean clone in seconds. `reproducibility/identifiability_multi.py` rebuilds the
split-half sweep that Extended Data Fig. 3 panel g draws, and needs four of the seven
datasets. Both were checked by regenerating the committed JSON from an unrelated working
directory: each comes back byte for byte identical.

**An accession is not the object that was fit**, which is a preprocessed one, so the
preparation step ships too. `reproducibility/prep_pancreas.py` rebuilds the pancreas input
from scVelo's own download, verified identical to the object the published pancreas fits
were run on, and `reproducibility/prep_dataset.py` prepares an object that arrives already
normalized. [DATA_SOURCES.md](DATA_SOURCES.md) records for each of the seven what its
first step is, which objects rebuild from a package call, and which do not.

Point the scripts at your own copies with:

| Variable | What it locates |
| :--- | :--- |
| `SCHOPFIELD_DATA` | the datasets |
| `SCHOPFIELD_REPORTS` | the per-dataset report tree |
| `SCHOPFIELD_DYNAMISC_DATA` | two datasets read from a separate data directory |
| `SCHOPFIELD_CACHE` | where fetched base GRN tables are cached, if not `~/.cache` |

## Citation

A manuscript describing scHopfield is in preparation. Citation details will be added here
on publication.

## License

MIT. See [LICENSE](LICENSE).

The license covers the code in this repository. It does not extend to the single-cell
datasets or the base GRN tables, which carry their own terms and none of which are
distributed here. [DATA_SOURCES.md](DATA_SOURCES.md) states the base GRN terms, which
restrict use to non-commercial academic purposes, and lists the works to cite.

## Contributing

Issues and pull requests are welcome at https://github.com/Bernaljp/scHopfield/issues.

```bash
pip install '.[dev]'
pytest              # the fast suite
pytest -m slow      # the circuit-recovery regression, about a minute per fit
```
