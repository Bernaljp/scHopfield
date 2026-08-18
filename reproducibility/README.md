# reproducibility/

The code that draws the paper's figures, the configuration it was run with, and the inputs
small enough to ship.

The repository's top-level [README](../README.md) summarizes what runs from a clean clone.
This file is the detail behind that summary: which script owns which figure, exactly what
each one reads and how large it is, how a fitted dataset is regenerated and what that
costs, what the configuration means, and which parts will not work on someone else's
machine without extra effort.

Every number below was measured against this tree rather than carried over from an earlier
note. Where a measurement contradicts something stated elsewhere in the repository, the
measurement is given and the contradiction is named.

## The ten figures and their generators

Verified against the submitted manuscript, which declares six main and four Extended Data
figures. The right-hand column names the Results section that cites the figure, quoted as
the manuscript heads it.

| Figure | Generator | Command | Cited in |
| :--- | :--- | :--- | :--- |
| Fig. 1 | `make_framework_figure.py` | `python make_framework_figure.py` | Learning a constrained dynamical system from observational data |
| Fig. 2 | `make_small_circuits_validation.py` | `... --submission` | Restricted non-linear dynamics retains multistability and oscillatory behavior |
| Fig. 3 | `make_dyngen_benchmark.py` | `... --submission` | Structural constraints makes regulatory interactions reoverable |
| Fig. 4 | `make_energy_jacobian.py` | `... --submission` | Interaction-energy depth generalizes across development, whereas local stability is system-specific |
| Fig. 5 | `make_perturbation_dynamics.py` | `... --submission` | Constrained dynamical extrapolation predicts single-gene fate responses |
| Fig. 6 | `make_double_perturbation.py` | `... --submission` | Combinatorial perturbations reveal structured departures from additivity |
| Extended Data Fig. 1 | `make_ed1_ablation.py` | `python make_ed1_ablation.py` | Restricted non-linear dynamics retains multistability and oscillatory behavior |
| Extended Data Fig. 2 | `make_sigmoid_activation.py` | `... --submission` | Learning a constrained dynamical system from observational data |
| Extended Data Fig. 3 | `make_cross_dataset.py` | `... --submission` | cited from four sections: panel a and panel d from the energy-depth section, panel b from the single-gene section, panel c from the structural-constraints section, and the whole figure from the opening section |
| Extended Data Fig. 4 | `make_network_figure.py` | `... --submission` | Interaction-energy depth generalizes across development, whereas local stability is system-specific |

Two scripts are exceptions to the `--submission` pattern, and both are exceptions in the
manuscript's direction rather than accidents:

- **Figure 1 has no `--submission` flag.** It renders one layout, which is the published
  one; the paper's numbered PDF is taken from that single output by hand.
- **Extended Data Fig. 1 has no flag at all**, and no argument parser. It exists only in
  the journal-page form and writes `ExtendedDataFig1.pdf` directly.

Figure 4 writes a second file, `Figure4e_circuits.pdf`, under `--circuits`.

**There is no command that builds all ten.** No orchestrator ships here, and none existed
in the tree these scripts came from: the build script there listed all ten, printed a skip
line for each, and built only the manuscript's text documents. Run the generators
individually, in any order. They share no state and none depends on another's output.

## What each script reads

Measured by tracing the actual file opens of each default submission run, not by reading
import lists. Style modules, source and the interpreter are excluded.

| Figure | Required input bytes | Ships in this repository? |
| :--- | ---: | :--- |
| Extended Data Fig. 1 | 376 | yes |
| Fig. 2 | 41,728 | yes |
| Extended Data Fig. 3 | 5,318 | yes |
| Fig. 3 | 7,183,370 | yes, after one rebuild step |
| Fig. 6 | 615,199 | no |
| Fig. 5 | 2,944,317 | no |
| Fig. 1 | 701,133,656 | no |
| Fig. 4 | 736,999,008 | no |
| Extended Data Fig. 4 | 736,999,008 | no |
| Extended Data Fig. 2 | 1,473,998,016 | no |

That splits three ways.

### Four run from a clean clone

Nothing to download, no dataset, no fitted model. Measured wall times on the development
workstation with a warm page cache:

| Figure | Reads | Time |
| :--- | :--- | ---: |
| Extended Data Fig. 1 | `data/ablations/hill_vs_linear.json`, 376 bytes | 1 s |
| Extended Data Fig. 3 | `cache/_cross_dataset_cache.json`, `cache/_recon_cache.json` and `data/real_identifiability/multi.json`, 5,318 bytes together | 2 s |
| Fig. 2 | `data/small_circuits/fits.npz`, 41,728 bytes | 13 s |
| Fig. 3 | the committed dyngen exports, 7,183,370 bytes | 3 s, after a 15 s rebuild |

Extended Data Fig. 3 runs from a clean clone **only under `--submission`**. That path reads
a committed cache of per-dataset scalars. Without the flag the same script recomputes those
scalars from all seven fitted datasets, which is roughly 5 GB of reads.

### Figure 3 needs one rebuild step first

Figure 3 is a ground-truth benchmark, so its ground truth ships. The dyngen simulator's own
CSV exports are committed for all six backbones, and `dyngen/02_build_h5ad.py` assembles an
AnnData from them, with no R and no dyngen involved:

```bash
python reproducibility/dyngen/02_build_h5ad.py
python reproducibility/make_dyngen_benchmark.py --submission
```

The rebuild takes **15 seconds for all six backbones together**, not 15 seconds each. The
top-level README and the `.gitignore` comment both say "about 15 s per backbone"; that
overstates it by roughly sixfold, and this measurement supersedes them.

The rebuild also rewrites `W_true.npy`, `tf_mask.npy` and `gene_names.npy`, which are
tracked. Running it leaves `git status` clean, so the committed ground truth is exactly
what the committed CSVs produce. The `adata.h5ad` files it writes, about 2.2 MB each, are
ignored by git by name.

Figure 3's submission layout reads three of the six backbones directly, bifurcating, cycle
and linear, and takes the other three through the aggregate in `benchmark_summary.json`.

### Six need a fitted dataset that is not distributed

They read from the per-dataset report tree, located by `SCHOPFIELD_REPORTS`. The fitted
objects there are large. Measured across the seven datasets:

| Dataset | `adata_analyzed.h5ad` | Whole `data/` directory |
| :--- | ---: | ---: |
| human_limb | 406 MB | 824 MB |
| murine_nc | 543 MB | 1.1 GB |
| dynamo_hematopoiesis | 590 MB | 1.8 GB |
| schwann | 618 MB | 1.3 GB |
| paul15_coarse | 668 MB | 1.4 GB |
| pancreas | 703 MB | 2.2 GB |
| paul15 | 1.2 GB | 2.5 GB |

**These are not distributed in this repository and will not be.** They run from 406 MB to
1.2 GB each, they are regenerated output rather than source, and they go stale against the
method. Regenerate them with the recipe below.

What each of the six wants:

| Figure | Default dataset | Reads |
| :--- | :--- | :--- |
| Fig. 1 | paul15_coarse | `adata_analyzed.h5ad` and `perturb_dynamics.pkl` |
| Fig. 4 | pancreas | `adata_analyzed.h5ad` |
| Extended Data Fig. 4 | pancreas | `adata_analyzed.h5ad` |
| Extended Data Fig. 2 | pancreas | two fits, `adata_analyzed_bimodal.h5ad` and `adata_analyzed_singlehill.h5ad` |
| Fig. 5 | pancreas | `perturb_dynamics.pkl`, `driver_scores_1.csv`, `driver_scores_2.csv`, and optionally `plots/A2_input_velocity.png` |
| Fig. 6 | pancreas | `double_ko_screen.pkl` |

Two of these need more than the fit itself:

- **Extended Data Fig. 2 compares two fits**, the two-component activation against the
  single Hill. The second is not produced by a default run: re-fit with
  `BIMODAL_HILL = False` and save the result as `adata_analyzed_singlehill.h5ad`. If only
  one fit is present the script stops rather than comparing a file against itself and
  drawing a flat zero.
- **Figures 5 and 6 read pickled perturbation caches**, not the fit. The scripts that write
  them are here, `compute/_perturb_dynamics_compute.py` and `compute/_double_ko_compute.py`,
  but they need the fitted object first. The all-pairs screen behind Figure 6 is written
  only by `_double_ko_compute.py --only screen`.

Figure 5's panel a prefers the report's pre-rendered streamline PNG when it is present and
draws the same field natively when it is not, saying so on stderr. It does this in **both**
layouts, so the PNG is an optional input to the journal figure as well as the poster one.
Tracing the submission run is what established this: `render_submission` used to document
panel a as always redrawn natively, which described an earlier version, and that docstring
has been corrected to match the code.

Every one of the six stops with a `FileNotFoundError` naming the exact path it wanted.
Verified by running all six against an empty report tree: each exited non-zero and none
drew a partial figure.

## Regenerating a fitted dataset

Two steps: prepare the object, then fit it.

**Step 1, the raw data to a prepared object.** An accession is not the object that was fit.
What that step is differs per dataset, and `DATA_SOURCES.md` records it for all seven. Only
pancreas rebuilds end to end from a package call:

```bash
export SCHOPFIELD_DATA=/path/to/datasets
export SCHOPFIELD_REPORTS=/path/to/reports

# pancreas: scVelo downloads the raw object and this applies the published recipe,
# writing the exact path config.py names. 14 s once the raw object is cached,
# about a minute on a cold run, which is the download. CPU.
python reproducibility/prep_pancreas.py

# an object that already arrived normalized, written out in the same form:
python reproducibility/prep_dataset.py --inp raw.h5ad --out prepared.h5ad
```

Verified here by writing to a scratch path and comparing: `prep_pancreas.py` reproduces the
pancreas input **byte for byte**, same md5 as the object the published pancreas fits were
run on.

The other five were preprocessed before they reached this project, so reprocessing from the
accession gives a close fit rather than an equal one. One field, mouse hematopoiesis's
lineage pseudotime, does not rebuild at all. `DATA_SOURCES.md` is explicit about which is
which; do not assume an accession alone reproduces a published fit.

**Step 2, the prepared object to a fitted one.** `rutils.prepare_and_fit` is the whole path
to `<SCHOPFIELD_REPORTS>/<dataset>/data/adata_analyzed.h5ad`, and it is the function that
produced every fitted object the figures read. It caches: a second call returns the existing
file unless `force=True`.

```bash
cd reproducibility
python -c "import rutils; rutils.prepare_and_fit('pancreas', device='cuda')"
```

The steps it runs, in order, are: load the configured object; drop excluded clusters; run
`sch.pp.prepare_dataset` if the entry asks for it; map a smoothed-expression layer to `Ms`
if the object has none; build a neighbor graph if it has none; take the velocity from the
configured layer or estimate it from pseudotime; drop genes whose velocity is not finite;
select the top `N_GENES` by velocity, keeping the configured anchors and perturbation
genes; fit the per-gene Hill activation; fetch the base GRN and build the scaffold; fit the
interactions per cell type; then run the downstream analyses, which are energies, gene
correlations, network correlations, centrality, eigenanalysis, Jacobians, Jacobian
statistics and the rotational part.

**What it costs.** Measured end to end on pancreas, 3,696 cells by 2,000 genes, 8 cell
types, on one RTX 3090:

| Stage | Time |
| :--- | ---: |
| Load, layers, neighbors, activation fit, scaffold | 42 s |
| Interaction fit, 8 cell types plus a global model | 150 s |
| Downstream analyses | 94 s |
| **Total** | **288 s** |

Output: 703 MB. The other datasets scale with cell count and cell-type count; paul15, with
19 clusters, is the slowest and the largest.

The base GRN is downloaded on the first fit on a machine, from a pinned CellOracle commit,
checked against a recorded sha256 and cached. It carries CellOracle's terms, which are
non-commercial academic only. See [DATA_SOURCES.md](../DATA_SOURCES.md).

**What re-running reproduces.** The fit is seeded, so a re-run reproduces the published
numbers to numerical precision rather than approximately. Re-fitting pancreas and comparing
against the object the published figures were drawn from: identical gene set, identical
file size, per-cell energies agreeing to a relative 1e-16, Jacobian summaries to 3e-6. The
files are not byte-identical, because the HDF5 container is not, so compare the contents
rather than a checksum.

`device='cpu'` works and is slower. Nothing silently falls back: the device is passed
through to the fit.

## The configuration

`config.py` holds the per-dataset settings and fit parameters behind every figure in the
paper. Everything in it is also stated in Methods or in a figure legend.

### Module-level constants

| Name | Value | What it does |
| :--- | :--- | :--- |
| `N_GENES` | 2000 | genes kept, ranked by velocity magnitude, with anchors and perturbation genes forced in |
| `HILL_N_MAX` | 20.0 | ceiling on the fitted Hill exponent |
| `BIMODAL_HILL` | `True` | fit the two-component activation by default; a dataset can override with `bimodal_hill` |
| `PSEUDOTIME_RATE_TARGET` | 1.0 | fixes the arbitrary time unit of pseudotime-derived velocity, applied after gene selection |
| `FIT_KWARGS` | see below | everything passed to `fit_interactions` |
| `PROGENITORS` | 7 entries | progenitor-side cluster names for the energy-depth comparison, read by Extended Data Fig. 3 panel a |

`FIT_KWARGS` carries the boundedness configuration that the paper uses,
`boundedness_lambda=0.1` with `gamma_min=0.01`, and `only_TFs=True`, which hard-zeroes
gene-to-gene edges and soft-penalizes transcription-factor edges that are off the scaffold.
The file's own comments record why each was chosen and what the alternatives cost.

### Per-dataset keys

| Key | Required | Meaning |
| :--- | :--- | :--- |
| `path` | yes | the prepared object, relative to `SCHOPFIELD_DATA` |
| `cluster_key` | yes | the `obs` column holding the cell-type annotation |
| `base_grn` | yes | a registry name, `mouse_atlas` or `human_promoter`, not a path |
| `velocity_mode` | yes | `velocity` to use a velocity layer, `pseudotime` to infer dynamics from an ordering |
| `prepare` | yes | run `sch.pp.prepare_dataset` on load |
| `velocity_key` | with `velocity` | the layer holding the velocity |
| `pseudotime_key` | with `pseudotime` | the `obs` column holding the ordering |
| `ms_layer` | if no `Ms` | smoothed-expression layer to map onto `Ms` |
| `velocity_embedding_key` | no | a precomputed velocity embedding, used instead of reprojecting |
| `exclude_clusters` | no | clusters dropped before anything else runs |
| `order` | no | left-to-right cluster order for the plots |
| `lineages` | no | one A-versus-B grouping for the perturbation analyses |
| `lineage_pairs` | no | several such groupings, when one axis is not enough |
| `anchors` | no | genes forced through gene selection |
| `perturb_genes` | no | genes featured in the perturbation panels, also forced through selection |
| `species` | no | descriptive only; nothing reads it, because `base_grn` names the table |

`lineages` left as `None` means the A-versus-B split is derived from the data, from the two
most pseudotime-terminal and network-distinct clusters.

### The seven datasets as worked examples

| Dataset | Cluster key | Prior | Dynamics | Notable |
| :--- | :--- | :--- | :--- | :--- |
| `pancreas` | `clusters` | mouse_atlas | `velocity_S` | the primary system; two lineage pairs, because a multifurcation has more than one meaningful axis |
| `paul15` | `paul15_clusters` | mouse_atlas | pseudotime, from `Pseudotime` | 19 clusters; shows that no velocity layer is required |
| `paul15_coarse` | `cell_type` | mouse_atlas | pseudotime, from `Pseudotime` | the same object at a 7-type annotation; the only difference from `paul15` is the cluster key |
| `dynamo_hematopoiesis` | `cell_type` | human_promoter | `velocity_alpha_minus_gamma_s` | human, not mouse; needs `ms_layer='M_t'` because the object carries moment layers rather than `Ms` |
| `schwann` | `assignments` | mouse_atlas | `velocity_S` | the one entry prepared on load; drops an unassigned `none` cluster |
| `murine_nc` | `celltype_update` | mouse_atlas | `velocity_S` | minimal entry: no order, no lineages, both derived |
| `human_limb` | `leiden_R_celltype` | human_promoter | `velocity_S` | minimal entry, human prior |

### Adding an eighth dataset

A minimal entry is five keys, as `murine_nc` shows:

```python
"my_dataset": dict(
    path=f"{DATA}/my_dataset/prepared.h5ad",
    cluster_key="cell_type",
    base_grn="mouse_atlas",       # or "human_promoter"
    prepare=False,
    velocity_mode="velocity", velocity_key="velocity_S",
    lineages=None, anchors=None,
),
```

The object must carry spliced counts, a velocity layer or a pseudotime column, and a
cell-type column. `sch.pp.prepare_dataset` produces the first from spliced and unspliced
counts. [DATA_SOURCES.md](../DATA_SOURCES.md) states what an object has to carry in full.

Adding the entry is enough to fit the dataset and to draw the figures whose curated tables
already cover every dataset. It is **not** enough for Figure 4 or Extended Data Fig. 2. See
the next section.

## What will not work off the shelf

Named rather than glossed, so a reader can tell how far they will get.

### Four figures are curated for pancreas alone

Several panels are drawn against gene and cluster lists curated by hand. Their coverage is
uneven, and the difference matters:

| Registry | Where | Covers | Feeds |
| :--- | :--- | :--- | :--- |
| `LINEAGE_BY_DATASET` | `make_sigmoid_activation.py` | pancreas only | Extended Data Fig. 2 panels e and g |
| `JAC_PAIRS`, `NET_GENES` | `make_energy_jacobian.py` | pancreas only | Figure 4's Jacobian mini-networks |
| `TFS_BY_DATASET` | `compute/_perturb_dynamics_compute.py` | all seven | Figure 5, Figure 6, Extended Data Fig. 3 panel b |
| `PROGENITORS` | `config.py` | all seven | Extended Data Fig. 3 panel a |

Passing `--dataset` something the registry does not cover **stops the run** with a message
naming the registry, the datasets it does cover and what the panel needed. It used to draw
an empty panel under a caption asserting a result. Extending Figure 4 or Extended Data
Fig. 2 to a new dataset therefore means curating those lists, not only adding a config
entry.

Figure 1 defaults to `paul15_coarse` and Extended Data Fig. 4 carries a pancreas cell-type
default in `--celltypes`, which has to be changed for another dataset.

### The dyngen simulation needs R, and is not on the reproduction path

`dyngen/01_simulate.R` and `dyngen/run.sh` are provenance. Re-running them needs an R
installation with dyngen, and on the machine this was run on it also needed a glpk shim on
`LD_LIBRARY_PATH`; `run.sh` takes both locations from environment variables because both
are machine-specific.

**None of that is needed to reproduce Figure 3.** The simulator's CSV exports are
committed, and `02_build_h5ad.py` rebuilds every ground-truth matrix from them in Python.
The R scripts are kept so the simulation behind those CSVs can be audited and re-run, not
because the figure depends on them.

### TikZ panels want a LaTeX installation

Figures 1 and 4 and Extended Data Fig. 4 draw regulatory circuits by compiling a standalone
TikZ snippet with `pdflatex` and rasterizing it with `pdftoppm`. Both binaries must be on
`PATH`, and the LaTeX installation needs the `standalone` class and `pgf`.

Without them the package falls back to drawing the same network in matplotlib and warns on
stderr about which panel fell back. The fallback carries the same encoding and is honest
about being a different rendering; it is not a silent substitution. The figures in this
repository's own verification runs were rendered with LaTeX present, so the fallback path
was not exercised here.

### Two of the seven datasets sit outside the main data root

`dynamo_hematopoiesis` and `schwann` are read from `SCHOPFIELD_DYNAMISC_DATA`, which
defaults to `<SCHOPFIELD_DATA>/DynamiSC`. On the machine the analyses were run on that
default does not exist and the variable was set explicitly. If you keep all seven datasets
together, point it at the same directory as `SCHOPFIELD_DATA`.

### Neither the datasets nor the scaffolds are here

The seven datasets are public, each under its own accession, and have to be fetched. The
base GRN tables are third-party material whose terms do not permit redistribution under
this repository's license. `DATA_SOURCES.md` records, per dataset, what its first step is,
which objects rebuild from a package call, and which single field does not rebuild at all.

## Environment variables

All are optional and all fall back to a path under the repository. Set the first three to
work against data that lives elsewhere.

| Variable | Locates | Default |
| :--- | :--- | :--- |
| `SCHOPFIELD_DATA` | the datasets and, historically, the scaffolds | `<repo>/data` |
| `SCHOPFIELD_REPORTS` | the per-dataset report tree | `<repo>/reports` |
| `SCHOPFIELD_DYNAMISC_DATA` | two of the seven datasets | `<SCHOPFIELD_DATA>/DynamiSC` |
| `SCHOPFIELD_FIGURES` | where figures are written | `reproducibility/figures` |
| `SCHOPFIELD_OUTPUT` | where report pages are written | `reproducibility/output` |
| `SCHOPFIELD_CACHE` | where fetched base GRN tables are cached | `~/.cache/scHopfield/base_grn` |
| `SCHOPFIELD_DYNAMO_PYTHON`, `SCHOPFIELD_DYN_STREAMLINE` | an optional dynamo streamline renderer for one report panel | unset, and the caller falls back to scVelo |

Output goes under `reproducibility/figures/`, which is ignored by git, so a run never
dirties the working tree. Every script anchors its paths to its own location, so the
working directory does not matter.

## Files

| Path | What it is |
| :--- | :--- |
| `make_*.py` | the ten figure generators |
| `compute/` | the caches the figures read: perturbation dynamics, the double-knockout screen, the dyngen refits, the reconstruction cache |
| `data/`, `cache/` | the committed inputs |
| `dyngen/` | the R simulation, its exports and the Python assembly step |
| `config.py` | per-dataset settings and fit parameters |
| `paths.py` | the filesystem layout and the environment variables above |
| `guards.py` | the loud-failure helpers the scripts call instead of degrading silently |
| `rutils.py`, `sections.py` | the fitting path and the report sections |
| `prep_pancreas.py`, `prep_dataset.py` | dataset preparation, from an accession to a prepared object |
| `hill_vs_linear.py`, `identifiability_multi.py` | the producers behind Extended Data Figs. 1 and 3g |
| `paper_plot_style.py`, `submission_style.py` | figure style |
