# scHopfield - findings log

Numbered, append-only record of every trustworthy result. Each finding survives
compaction and is the unit you commit against ("M12: ..."). Every headline number
here must be reproducible from a committed script + config/git hash.

Calibration floors: W Pearson between two unrelated random matrices approx 0;
Spearman rank correlation trivial baseline approx 0. "Reproducible" target = 1.0
(bit-exact) for a fixed seed.

---

## M1 - Seeding makes GRN inference bit-reproducible; unseeded rankings are seed-sensitive
- Setup: `analyses/seed_sensitivity.py`, real pancreas (scVelo steady-state,
  `data/Pancreas/pancreas_scvelo_ready.h5ad`), 3696 cells x 300 genes, scaffold-guided
  optimizer (all-ones scaffold, 300 epochs), device=cuda (RTX 3090). Compares the
  interaction matrix W_all and per-gene out-strength centrality across runs.
  Results: `benchmark_results/seed_sensitivity_real/results.json`.
- Result:
  - SEEDED (same seed, two runs): W Pearson = **1.0000**, relative Frobenius = **0.0000**,
    centrality Spearman = **1.0000** (bit-exact reproducibility, on GPU).
  - UNSEEDED (two runs): W Pearson = 0.9988, rel Frobenius = 0.049, centrality Spearman = **0.931**.
  - CROSS-SEED (5 seeds): W Pearson = 0.998 +/- 0.000 (stable), but centrality Spearman
    mean = **0.807** (min 0.733, max 0.887) -- rankings vary substantially by seed.
  - Synthetic control (under-determined, 300x200, GPU): seeded = 1.000000; unseeded W
    Pearson 0.999; same pattern. (`benchmark_results/seed_sensitivity/results.json`)
- What it means: the inferred W is numerically well-determined (near-identical up to a
  few % Frobenius regardless of seed), which is reassuring for the method's core. But
  *ranking-level* readouts (centrality, and by extension top-eigenvector / top-gene
  selection) are seed-sensitive (Spearman 0.73-0.89 across seeds). This is the mechanism
  behind the collaborator's "different genes in the plot / different results" across
  reruns (nb03 3.3-3.4, nb04 4.5). It does NOT show the method is unstable in W; it shows
  unseeded pipelines are not reproducible at the figure level. Fix = seed the pipeline
  (`sch.set_seed` / `fit_interactions(seed=...)` / `compute_umap(random_state=...)`),
  which now yields bit-exact results. Recommendation for the paper: fix a seed AND, for
  ranking figures, report stability (or aggregate over seeds).
- Disposition: feeds Methods "Reproducibility" note + a supplementary reproducibility
  figure (`benchmark_results/seed_sensitivity_real/reproducibility.png`); resolves
  red-team item C (determinism FAIL). audit_table row updated? y.
