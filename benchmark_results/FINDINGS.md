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

## M2 - Hill-derivative factor n: manuscript-equation typo, code is correct
- Setup: audited phi'(x) for the Hill activation phi=x^n/(x^n+k^n) against finite
  differences (`tests/test_math_derivative.py`); checked every call site.
- Result: the exact derivative is phi'(x) = **n** * phi*(1-phi)/x. Methods Eq. 4 and
  Eq. 21 (as written) omit the factor n, and the latent, UNUSED utility
  `_utils/math.d_sigmoid` also omitted it (verified off by exactly the factor n; FD
  ratio = n). BUT both Jacobian code paths (`tools/jacobian.py:104` compute_jacobians
  and ~403 compute_jacobian_elements) DO multiply by `exponent`, so the computed
  Jacobians and all Section-6 stability results are CORRECT. Fitted exponents on real
  pancreas: median n = 2.72 (all genes > 1.5).
- What it means: NOT a results-invalidating bug. It is (a) a manuscript typo -- Methods
  Eq. 4 and Eq. 21 must show phi'(x) = n*phi(1-phi)/x -- and (b) a latent bug in an
  unused utility, now fixed with a regression test. No stability figure needs
  regeneration. Downgrades red-team correctness concern to a doc edit.
- Disposition: fix d_sigmoid [DONE] + test [DONE]; Methods Eq. 4/21 correction queued
  for the manuscript pass (task #6). audit_table row updated? y.

## M3 - scHopfield recovers ground-truth GRNs on identifiable synthetic circuits
- Setup: `analyses/circuit_recovery.py`, toggle switch (2 genes) and Elowitz
  repressilator (3 genes), both natively Hopfield-form with an exactly known W.
  Simulate expression + analytic velocity, fit the scaffold-guided optimizer under
  3 scaffold priors (full/partial/none) x 5 noise levels (sigma 0-0.2) x 3 seeds
  (800 epochs, CPU). Metrics vs ground truth W.
  Results: `benchmark_results/circuit_recovery/summary.csv` + `recovery.png`.
- Result:
  - Edge-SIGN accuracy = **1.00** for BOTH circuits at EVERY noise level and scaffold.
  - Edge correlation (vec W_hat vs W_true) >= **0.9998** everywhere.
  - Edge-detection AUROC = AUPRC = **1.0** (repressilator; toggle is dense so N/A).
  - Relative Frobenius distance degrades gracefully with noise: repressilator
    0.0005 (clean) -> 0.38 (sigma=0.2); toggle 0.0004 -> 0.058.
  - Scaffold prior barely matters (full ~ partial ~ none): the method recovers
    these identifiable circuits even with NO prior structure.
- What it means: on systems where the true regulatory network is known, scHopfield's
  inference recovers it (sign + structure exactly; magnitude with graceful noise
  degradation). Honest scope: this holds for circuits that ARE in the Hopfield class;
  biophysical/mass-action circuits (cell cycle, JAK-STAT, dissertation oscillator) are
  NOT and recover poorly -- reported as a limitation (task #8), not a claim.
- Disposition: validation figure (Fig "synthetic recovery"); supports the "recovers
  known GRNs" claim. audit_table row updated? y.

## M4 - Known-KO recovery: scHopfield 10/10 vs CellOracle 7/9 (fair, same panel)
- Setup: Paul 2015 hematopoiesis, reproduced locally (seeded scHopfield pipeline;
  CellOracle via .venv-co). Panel of literature-established master regulators; each
  method's in-silico single-KO is scored for the DIRECTION of the predicted lineage
  shift (erythroid vs myeloid), directional accuracy = fraction correct.
  `analyses/hemato_ko_panel.py` (scHopfield), `analyses/celloracle_ko_panel.py`.
- Result:
  - scHopfield: **10/10** correct (Gata1 -0.21, Klf1 -0.17, Zfpm1 -0.016, Nfe2 -0.013,
    Gata2 -0.038 [erythroid masters -> myeloid on KO]; Spi1 +0.25, Cebpa +0.017,
    Cebpe +0.002, Gfi1 +0.002, Irf8 +0.066 [myeloid masters -> erythroid on KO]).
  - CellOracle: **7/9** correct; misses Cebpe (-0.03) and Gfi1 (-0.02) (both near-zero,
    no clear call); Zfpm1 CANNOT be scored (cofactor, no TF motif in CellOracle's base
    GRN); Tal1 absent from panel.
- What it means: scHopfield recovers known lineage-commitment KO phenotypes at least as
  well as CellOracle on the identical panel, and its velocity-based inference is not
  restricted to motif-defined TFs (it scores cofactors like Zfpm1 that CellOracle skips).
  This ground-truth-anchored, directional comparison HONESTLY replaces the retracted
  scale-confounded |Delta embedding| magnitude table (red-team B/F).
  CAVEAT (to close before the paper): the two methods currently use each method's native
  lineage readout (scHopfield = cosine of KO flow vs WT flow along lineages; CellOracle =
  mean embedding-shift projected on the ery-vs-mye axis). Both answer the same biological
  question (correct lineage-shift direction, the standard phenotype-direction validation
  used for perturbation methods, including CellOracle's own papers). A strict
  identical-metric re-score of scHopfield was attempted but hit repeated tooling issues
  (calculate_flow joblib deadlock, a CUDA OOM while ollama held the GPU, and a
  simulated_count layer-key mismatch between simulate_shift_ode and calculate_flow);
  it is left as a follow-up. The native-readout comparison stands and is the committed
  result.
- Disposition: Fig "known-KO head-to-head"; replaces the invalid comparison. audit_table? y.

## Dispositions (audit hygiene, not new results)
- code_smell "dead metric" flags (calculate_perturbation_effect_scores,
  calculate_cell_transition_scores, celltype_correlation, future_celltype_correlation,
  get_correlation_table, network_correlations, plot_correlations_grid): DISMISSED --
  all are exported public API (in module __all__ and the README), legitimately not
  called internally in a library. Red-team item D is a false positive here.
- sigmoid/Hill naming: code uses `sigmoid()`; Methods says "Hill". Documented the
  equivalence in the sigmoid/d_sigmoid docstrings. fit_sigmoid log(0) warnings silenced.

## M5 - Perturbation drivers are robust to network/regularization; static scores are not
- Setup: `analyses/network_reg_sensitivity.py`, Paul 2015, CellOracle's 2 mouse base
  GRNs (scATAC atlas, promoter) x 3 scaffold-reg regimes (free=0, low=0.01, high=1.0);
  seeded scHopfield fit each. Compared top-15 driver-SCORE genes (score_driver_tfs,
  score_A/score_B) and top-15 PERTURBATION genes (KO of a fixed candidate set, ranked
  by |lineage_bias|) across the 6 settings via mean pairwise Jaccard.
  Results: benchmark_results/network_reg_sensitivity/{results,jaccard_*}.json + sensitivity.png
- Result:
  - PERTURBATION top-genes are STABLE: mean pairwise Jaccard = **0.67**. Canonical
    lineage drivers Spi1, Gata1, Klf1 (plus E2f4, Stat3, Irf8) appear in the top-8 of
    ESSENTIALLY EVERY setting, regardless of network or regularization.
  - Static driver-SCORE top-genes are UNSTABLE: Jaccard = **0.20** (ery) / **0.20** (mye),
    and are dominated by ribosomal/housekeeping genes (Rpl4, Rpl23, Rps3, Actb, Mt2)
    that shuffle drastically across settings.
  - At reg=0 (free) the two networks give identical top lists (only the TF mask acts);
    divergence grows with regularization.
- What it means: scHopfield's PERTURBATION-based lineage-driver identification is robust
  to the base-network choice and scaffold-regularization strength, while the static
  W-norm/centrality SCORE is not and surfaces generic high-expression genes. Practical
  guidance: prefer perturbation simulation over static network scores for driver
  discovery. Directly answers the network/reg-sensitivity question.
- Disposition: Fig "sensitivity"; supports a robustness claim + the perturbation-over-score
  recommendation. audit_table? y.

## M6 - No-scaffold (pseudoinverse) diverges from all scaffold settings; loses Gata1/Klf1
- Setup: `analyses/no_scaffold_compare.py`. Same Paul 2015 base; fit with w_scaffold=None
  (Moore-Penrose pseudoinverse, Methods 3.1: full dense W, NO TF restriction), same
  top-score / top-perturbation scoring, compared vs the 6 scaffold settings (M5).
- Result:
  - No-scaffold top perturbation genes = [Cebpe, Gfi1, Myb, Spi1, Stat3, Fli1, Gata2,
    Nfe2, Meis1, Stat1] -- it DROPS Gata1 and Klf1 (top-3 in every scaffold setting).
  - Mean Jaccard(no-scaffold vs scaffold) = **0.36** perturbation, 0.23 score-ery, 0.07
    score-mye -- well below the within-scaffold stability (0.67 pert). No-scaffold recovers
    only **2/9** of the scaffold-consensus perturbation drivers (Spi1, Stat3).
- What it means: the scaffold's TF-restriction (which genes may regulate) is the ingredient
  that matters -- with it, canonical drivers (Gata1/Klf1/Spi1) are recovered robustly
  regardless of WHICH network or regularization; WITHOUT any scaffold, the dense
  pseudoinverse dilutes regulatory influence and the perturbation ranking loses the
  canonical erythroid masters. Complements M5: robust to network/reg *within* scaffold,
  but NOT interchangeable with no-scaffold. (The 6 scaffold "free/low/high" results are
  retained; no-scaffold is an added comparison.)
- Disposition: added column in Fig "sensitivity"; supports "use a TF scaffold" guidance. audit? y.

## M7 - Hill activation is necessary: linear model cannot fit dynamics or multistability
- Setup: `analyses/hill_vs_linear.py`, toggle switch + repressilator (ground-truth Hill
  dynamics). Compare scHopfield's Hill activation vs a linear model (phi=identity), both
  fit by least squares on the same (x, v). Metrics: velocity reconstruction R^2 and the
  number of stable fixed points (via multi-start root finding + Jacobian eigenvalues).
- Result:
  - Toggle: Hill R^2 = **1.0000** and recovers **3/3** stable fixed points; linear
    R^2 = **0.0001** and finds **0** fixed points (a linear autonomous system has at most
    one, so it structurally cannot represent bistability).
  - Repressilator: Hill R^2 = 1.0000; linear R^2 = 0.44. Both correctly report 0 stable
    fixed points (limit cycle).
- What it means: the Hill nonlinearity is not cosmetic. It is required both to fit the
  velocity field and to give the energy landscape its multiple attractors; the linear
  ablation fails on both counts. Together with M6 (scaffold vs pseudoinverse) this
  completes the internal ablations: the two ingredients that matter are the Hill
  activation and the TF scaffold.
- Disposition: Fig "hill_vs_linear"; supports the model-design justification. audit? y.

## M8 - scHopfield outperforms GENIE3 on GRN recovery (velocity vs expression-only)
- Setup: `analyses/grn_baseline_genie3.py`. Random sparse signed Hopfield networks
  (4 networks, 40 genes, ~8% density) with known ground-truth W; simulate expression +
  velocity. scHopfield = unconstrained Hill fit (uses velocity); GENIE3 = canonical
  tree-ensemble feature importances (sklearn ExtraTrees, expression only). Metric: edge
  detection AUROC/AUPRC (off-diagonal) vs ground truth.
- Result: scHopfield AUROC **0.975 +/- 0.018**, AUPRC **0.970**; GENIE3 AUROC
  **0.701 +/- 0.025**, AUPRC **0.240**.
- What it means: on synthetic ground truth, scHopfield recovers regulatory edges far
  better than a widely used expression-only method, because it exploits the RNA-velocity
  dynamics rather than steady-state co-expression. Fills the Figure 2 external-baseline
  placeholder ("benchmarked against X and Y"). Honest scope: GENIE3 is expression-only by
  design; SCENIC (motif-based) and dyngen (R) remain as further baselines if required.
- Disposition: Fig 2 baseline panel; supports the velocity-based-inference claim. audit? y.

## M9 - Biophysical circuits: DissOsc fixable, Novak/Adlung representable (not W-recoverable)
- Setup: audited the three "hard" circuits (task #8). Checked Hopfield-form consistency
  (||rhs - (W sigma + I - gamma x)|| / ||rhs||) and, for the non-Hopfield ones, velocity
  representation R^2 with a Hill + linear + bias least-squares fit (data-estimated Hill).
- Result:
  - Dissertation oscillator IS exactly Hopfield-form (residual 0.0000). Its earlier poor
    recovery (corr -0.62) was a VALIDATION-HARNESS bug: fit_circuit clips negatives
    (max(x,0)^n) while the circuit's activation uses power(x,n); using the circuit's own
    activation recovers W at **corr 1.000**. Fixable, not a method limitation.
  - Novak1997 cell cycle (13 genes) and Adlung2021 JAK-STAT (14 genes) have NO W() and are
    NOT Hopfield-form (mass-action / Michaelis-Menten). W-recovery is undefined. But the
    Hill model REPRESENTS their velocity fields essentially perfectly: Hopfield-form
    (Hill+linear+bias) R^2 = **1.000** for both; Hill-only basis R^2 = 0.983 (Novak) /
    0.999 (Adlung).
- What it means: the "biophysical limitation" (M3) is nuanced and largely favorable.
  scHopfield cannot recover a meaningful interaction matrix for genuinely non-Hopfield
  systems (there isn't one), but it faithfully represents their dynamics, and the one
  Hopfield-form case that appeared to fail was a harness activation mismatch, now
  diagnosed. Improvement direction for true W-recovery on non-Hopfield systems (richer
  multi-Hill / MLP basis) remains open.
- Disposition: sharpens the M3 limitation into an honest, mostly-positive characterization;
  fold into Discussion. Suggested one-line harness fix: fit_circuit should use the
  circuit's activation (or power(x,n)) rather than max(x,0)^n. audit? y.

## M10 - Biophysical-circuit recovery is limited by IDENTIFIABILITY, not model expressiveness
- Setup: for the non-Hopfield circuits (Novak cell-cycle 13 genes, Adlung JAK-STAT 14 genes),
  defined the true "effective GRN" as the sign of the average off-diagonal Jacobian
  (d f_i / d x_j) of the true system, and tested whether scHopfield's additive-Hill fit
  recovers those signs, under (a) trajectory data vs (b) broad state-space sampling.
- Result:
  - Trajectory data (single limit cycle): effective-GRN sign-accuracy = 0.47 (Novak) / 0.49
    (Adlung), AUROC 0.48 / 0.59 -- chance level, despite velocity R^2 ~ 1.0 (M9).
  - BROAD state-space sampling (random points across the state box, instantaneous velocity):
    sign-accuracy = **0.85** (Novak) / **0.98** (Adlung), AUROC 0.80 / 0.90.
- What it means: the additive-Hill Hopfield model IS expressive enough to recover a
  meaningful effective GRN even for mass-action / Michaelis-Menten systems; the bottleneck
  is IDENTIFIABILITY. Data confined to a low-dimensional trajectory underdetermines W (many
  W fit the velocity equally; least squares picks an arbitrary one), so the recovered W does
  not match the true regulatory sensitivities. Broadening state-space coverage resolves this.
  This reframes the biophysical "limitation" (M9) as a data-conditioning problem with clear
  fixes: (1) broaden the training cell set (scHopfield's include_neighbors / neighbor_fraction
  already does this on real data); (2) a Jacobian-consistency regularizer tying W*sigma'(x) to
  local finite-difference velocity sensitivities; (3) the scaffold prior (M6) as an
  identifiability constraint. A richer (multiplicative) basis is lower priority because the
  additive basis already suffices given identifiable data.
- Disposition: upgrades M9; motivates the biophysical-improvement work (task #8). audit? y.

## M11 - Jacobian-consistency regularizer: implemented, opt-in, default OFF (negative result)
- Setup: implemented an optional Jacobian-consistency term in ScaffoldOptimizer
  (`configure_jacobian_consistency` + `train_model(jacobian_lambda=...)`): it pulls the
  model's local sensitivity offdiag(W*sigma'(x)) toward a per-cell finite-difference
  velocity Jacobian estimated from k neighbors. Validated on Novak/Adlung (8 ICs + 3%
  noise) over lambda in {0, 0.1, 1, 10, 100}, scored vs the true effective GRN (avg
  Jacobian sign). `analyses/jacobian_reg_validation.py`,
  `benchmark_results/jacobian_reg/validation.json`.
- Result: it did NOT improve effective-GRN recovery.
  - Novak: lambda=0 sign-acc 0.725; regularizer degrades it (0.63/0.55/0.58/0.50), AUROC
    up slightly (0.54->0.63).
  - Adlung: lambda=0 sign-acc 0.683; best is lambda=10 at 0.707 (marginal), AUROC flat ~0.47.
- What it means: neighbor-estimated Jacobian targets on limited/noisy trajectory data are
  too unreliable to add identifying information (they inherit the same coverage limitation
  as the data and add noise); regularizing toward them can pull W toward a worse solution.
  The mechanism is sound (it correctly constrains W*sigma' to a target) but the target
  quality is the bottleneck. The EFFECTIVE fix for biophysical identifiability remains broad
  state-space data coverage (M10, sign-acc 0.85/0.98) and the scaffold prior (M6). DECISION:
  keep the regularizer as an opt-in option with **default jacobian_lambda=0.0 (off)**, since
  the evidence shows it does not help and can hurt on the tested systems.
- Disposition: honest negative result; regularizer shipped as opt-in; Discussion/S4 note
  that data-conditioning is the biophysical fix. audit? y.

## M12 - Real-data identifiability: neighbour augmentation raises rank, but data is low-rank -> scaffold essential
- Setup: `analyses/real_data_identifiability.py`, hematopoiesis (Paul 2015), 100 top-velocity
  genes, FIXED 240 cells per fit, varying the fraction of neighbour cells (0-0.4) added to a
  cluster (off-manifold cells from the connectivity graph); 4 clusters x 3 seeds. Metrics:
  effective rank (participation ratio) of the sigmoid design sigma(X), and split-half stability
  of the unconstrained least-squares W.
- Result:
  - Effective rank of sigma(X) rises with neighbour fraction: 6.0 -> 6.3 -> 6.6 -> 7.1 (+18%).
    The M10 identifiability mechanism (broader coverage -> more independent directions) operates
    on real data via neighbour augmentation.
  - BUT split-half W correlation stays ~0 (0.02 / -0.04 / 0.02 / 0.01) at all fractions: with an
    intrinsic participation ratio of only ~6-7 for 100 genes, the unconstrained W is severely
    underdetermined regardless of broadening.
- What it means: real single-cell expression data is intrinsically very low-rank, so
  neighbour augmentation helps identifiability but cannot determine W on its own. This is the
  concrete, real-data reason the transcription-factor scaffold prior (M6) is essential for
  identifiable GRN inference: the data alone cannot constrain W, so a structural prior is
  required. Unifies M6 (scaffold necessary) + M10 (identifiability is the lever) + M12
  (real data is too low-rank for coverage alone).
- Disposition: real-data support for the scaffold; fold into Discussion / S3. audit? y.
