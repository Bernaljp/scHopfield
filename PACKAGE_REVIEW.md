# scHopfield package review

Covers (1) the reproducible end-to-end pipeline, (2) boilerplate promoted into the
package, (3) a duplication audit + deprecations, (4) bugs fixed, and (5) an honest
overall assessment. Written 2026-07-01.

## 1. Reproducible end-to-end pipeline

`sch.run_pipeline` runs the canonical sequence in one call, and
`analyses/run_full_pipeline.py` runs it identically across datasets, storing data
and figures under `benchmark_results/pipeline/<dataset>/`:

    prepare -> gene subset -> (scaffold) -> fit GRN -> energies
            -> Jacobians -> Jacobian stats -> network centrality
            -> drivers -> in-silico KO

Ran on five developmental systems (mouse + human), each producing
`adata_fitted.h5ad`, `summary.json`, and three figures:

| Dataset | Species | Cells | Clusters | Fit | Top drivers | KO target |
|---|---|---|---|---|---|---|
| Hematopoiesis (Paul 2015) | mouse | 2,671 | 16 | scaffold (1045 edges) | Myc, Nfe2, Myb | Myc |
| Pancreas endocrinogenesis | mouse | 3,696 | 8 | pseudoinverse | Rfx6, Pam, Ptprn2 | Pam |
| Murine neural crest | mouse | 6,788 | 9 | pseudoinverse | Wwtr1, Rtn1, Pcdh1 | Wwtr1 |
| Human limb | human | 12,207 | 10 | pseudoinverse | MEST, SHD, COBL | COBL |
| Schwann cell development | mouse | 8,821 | 7 | pseudoinverse | Egr1, Uchl1, Plekhb1 | Egr1 |

Face validity is strong: the recovered drivers are dominated by regulators of the
system in question (Rfx6 for pancreatic endocrine fate, Nfe2 for the erythroid
branch, Egr1 for Schwann myelination). The Schwann run also exercises
`prepare_dataset` end-to-end (raw spliced/unspliced -> velocity -> fit).

## 2. Boilerplate promoted into the package

Logic that had been copy-pasted across analysis scripts now lives in the package:

| New API | Replaces (duplicated in) | Notes |
|---|---|---|
| `sch.inf.build_scaffold` | `build_scaffold` in `hemato_pipeline.py`, `nc_ko_panel.py`, `network_reg_sensitivity.py` (3 verbatim copies) | + `scaffold_from_edges` for long-format edge lists |
| `sch.pp.prepare_dataset` | `prep_dataset.py`, `prep_pancreas.py` velocity/sigmoid prep | lazy scVelo import |
| `sch.dyn.score_ko_panel` | KO directional-scoring loop in `hemato_ko_panel.py`, `nc_ko_panel.py` | returns a table + accuracy |
| `sch.run_pipeline` | the ad-hoc fit->energy->jacobian sequence in every analysis | thin, transparent orchestration |

All four analysis scripts were refactored to call the package functions; behavior
is preserved (the scaffold builders were byte-for-byte identical).

## 3. Duplication audit

**Method.** AST-normalized function-body hashing across every module in
`scHopfield/`, plus manual review of the modules most likely to overlap
(velocity, io, the three perturbation modules).

**Finding: the importable package API has no dead duplicate functions.** The
apparent name collisions (`simulate`, `jacobian`, `forward`, `state_names`, ...)
are per-class methods of the synthetic-circuit classes, not redundant code. The
genuinely different-looking triples are genuinely different:

- `preprocessing/velocity.py` (input velocity from pseudotime) vs
  `tools/velocity.py` (reconstruct velocity from the fitted model): different jobs.
- `simulate_perturbation` (CellOracle-style propagation) vs
  `simulate_perturbation_ode` (single-cluster ODE) vs `simulate_shift_ode`
  (dataset-wide ODE): different simulators, all used.

So the duplication the request anticipated was real, but it lived in the
`analyses/` scripts (section 2), not the package.

**Consolidation opportunity (noted, not changed):** the eight
`validation/run_{cell_cycle,jakstat}_figure*.py` figure drivers share heavy
structure across their `base` / `scalenorm` / `trainable_hill` / `sweep` variants.
They are standalone `__main__` scripts (not importable API), so a deprecation
warning does not apply; they could be unified behind one parametrized entry point
in a later cleanup.

### Deprecation applied

- **`sch.dyn.simulate_shift`** -> use **`sch.dyn.simulate_perturbation`**. It was a
  self-described "backward compatibility alias" for the identical function; it now
  emits a `DeprecationWarning` naming its replacement and will be removed in a
  future release. (`DeprecationWarning` is silent by default, so existing notebooks
  keep working without noise.)

No other function warranted a deprecation; manufacturing one would have been
dishonest.

## 4. Bugs fixed during the review

- **`compute_network_centrality` failed on 3 of 5 real datasets.** Two distinct
  causes, both fixed: (a) `reset_index()` produced a column named after a *named*
  var index, breaking `melt(id_vars='index')` (KeyError on human_limb, schwann);
  (b) igraph `eigenvector_centrality` raised an ARPACK/LAPACK error on
  disconnected graphs (pancreas). Now works on all five.
- **Invalid escape sequences** (`\|`, `\g`) in three `validation/run_*` module
  docstrings (LaTeX in non-raw strings) -> raw docstrings. These were
  `SyntaxWarning`s today and would become `SyntaxError`s in a future Python.
  `compileall` is now clean.
- **Unused locals** in `compute_epistasis` (ruff F841) removed.

## 5. Overall assessment

**Strengths**

- Clean scanpy/scverse-style layout (`pp` / `inf` / `tl` / `pl` / `dyn`), everything
  stored back into `AnnData`; a newcomer can guess the API.
- Good separation of concerns and mostly thorough NumPy-style docstrings.
- Reproducibility is threaded through (`sch.set_seed`, `seed=` on the fit).
- The synthetic-circuit `validation/` suite with known ground-truth `W` is a real
  asset for regression testing.

**Weaknesses / recommendations**

- **Tests are thin.** There is `tests/` and the circuit validators, but no unit
  coverage of the core tools (energy, jacobian, centrality) that would have caught
  the centrality bugs above. Recommend a small `pytest` suite that runs
  `run_pipeline` on a tiny synthetic AnnData and asserts the expected keys exist.
- **Optional heavy deps** (`torch`, `scvelo`, `igraph`) are imported at module
  load in a few places; `prepare_dataset` now imports scVelo lazily, and the docs
  mock the heavy deps. Worth extending lazy imports so `import scHopfield` is light.
- **`pyproject.toml` vs `requirements.txt` drift** (scVelo is in one, not the
  other; igraph is commented out). Consolidate into `pyproject` optional-dependency
  groups (`[project.optional-dependencies]`: `velocity`, `graph`, `docs`).
- **Two overlapping io modules** (`_utils/io.py` low-level helpers vs `tools/io.py`
  model checkpointing) are fine but under-documented; a one-line module docstring
  each would clarify the split.
- **`validation/run_*` scripts** (see section 3) are the main remaining
  consolidation target.

**Verdict:** the package is in good shape, structurally consistent, and now runs a
documented, reproducible pipeline across many datasets. The main gap is automated
test coverage, not design.
