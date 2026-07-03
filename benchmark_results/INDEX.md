# benchmark_results index

Catalog of the committed scHopfield results. Start with **`PAPER_FIGURE_GUIDE.md`** (the
master figure guide) and **`FINDINGS.md`** (numbered findings M1-M23). Raw per-experiment
plots and data live in the themed subfolders below. `logs/` and `*_sanity/` are gitignored.

> The comprehensive **figure packs** (all-figures-from-one-dataset, the way
> `network_reg_sensitivity` is built) are generated on the `figure-packs-and-fixes`
> branch under the gitignored `figure_packs/` tree; see `figure_packs/INDEX.md` and the
> tracked scripts in `analyses/figure_packs/`.

## Master guides
- `PAPER_FIGURE_GUIDE.md` -- master figure guide (all paper sections)
- `FINDINGS.md` -- numbered findings log (M1-M23)
- `audit_table.csv` -- experiment -> claim/figure disposition

## Per-dataset pipeline
- `pipeline/` -- end-to-end fits per dataset (README.md, pipeline_summary.json, per-dataset
  adata + figures)

## Cross-dataset / generalization
- `cross_dataset/` -- 5-system comparison (CROSS_DATASET_FIGURE_GUIDE.md)

## The bias term (L1) study
- `bias_penalty/` -- synthetic + reprogramming bias study (BIAS_FIGURE_GUIDE.md)

## GRN recovery, circuits, baselines
- `circuit_recovery/`, `circuit_sanity/` -- synthetic circuit recovery
- `grn_baseline/` -- GENIE3 baseline comparison

## Robustness / sensitivity / reproducibility
- `network_reg_sensitivity/` -- network + regularization sensitivity (FIGURE_GUIDE.md, 22 figs)
- `real_identifiability/`, `jacobian_reg/` -- identifiability of the fit
- `seed_sensitivity/`, `seed_sensitivity_real/`, `seed_sanity/` -- determinism-when-seeded
- `ablations/` -- earlier ablations (hill-vs-linear); newer ablations in `figure_packs/pack8_ablations/`

## Perturbation / energy
- `hemato_ko/`, `nc_ko/` -- in-silico KO validation
- `energy_stability/` -- energy + stability

## Composed / misc
- `figures/` -- composed paper panels (fig1-6, comparison table)
- `pancreas/` -- notebook-08 figures
