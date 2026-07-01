# Results (draft): scHopfield

Draft Results narrative assembled from the committed benchmark findings
(`benchmark_results/FINDINGS.md`, M1-M7). Figures reference the committed PNGs.
House style follows the Methods section (British English, no em dashes, honest
framing). This is a working draft for the eventual manuscript, not the Methods
section.

## R1. Reproducible inference

scHopfield's cluster-specific network inference is a stochastic optimisation
(random weight initialisation, mini-batch shuffling), so we first established that
its outputs are reproducible. On the pancreatic dataset, two fits with the same
random seed were bit-identical (interaction-matrix Pearson correlation and
out-strength centrality Spearman correlation both 1.000), whereas unseeded fits
diverged: the inferred matrix W remained numerically stable (Pearson 0.999) but
its derived gene rankings drifted (centrality Spearman 0.73 to 0.93 across five
seeds). Seeding the pipeline removes this variability, and we seed all analyses
below. The practical implication is that ranking-level readouts should be computed
from a fixed seed, and we report ranking stability wherever it is relevant
(Figure R1; benchmark M1).

## R2. Recovery of known gene regulatory networks

On synthetic circuits whose interaction matrix is known exactly (a bistable toggle
switch and the Elowitz repressilator), scHopfield recovered the ground-truth
network with perfect edge-sign accuracy (1.00) and near-perfect signed correlation
(>= 0.9998) at every observation-noise level and scaffold prior tested, with
edge-detection AUROC and AUPRC of 1.0 for the sparse repressilator. The
reconstruction error grew gracefully with noise (relative Frobenius distance from
0.0005 at zero noise to 0.38 at the highest noise), and recovery held even with no
scaffold prior. Recovery was specific to systems in the model's function class:
biophysical circuits that are not natively of Hopfield form were not faithfully
recovered, which we report as a limitation rather than a claim (Figure R2;
benchmark M3).

The Hill nonlinearity is essential to these results. Replacing it with a linear
model (a linear dynamical system fit to the same data) collapsed velocity
reconstruction on the toggle switch (R-squared 0.0001 versus 1.000 for the Hill
model) and, structurally, could not represent the switch's bistability: a linear
autonomous system admits at most one fixed point, so the linear fit recovered none
of the three stable states that the Hill model recovered correctly (Figure R2b;
benchmark M7).

## R3. Recovery of known perturbation phenotypes

We next asked whether scHopfield recovers established transcription-factor knockout
phenotypes on mouse haematopoiesis (Paul et al., 2015). For a panel of literature
master regulators, each in-silico single knockout was scored for the direction of
the predicted lineage shift between the erythroid and myeloid branches. scHopfield
predicted the correct direction for all ten scorable factors, including the
canonical antagonistic pair Gata1 (knockout biases towards myeloid) and Spi1
(knockout biases towards erythroid), which produced the largest-magnitude shifts.
Scored on the identical panel and geometry, CellOracle predicted seven of nine
correctly; it could not score the erythroid cofactor Zfpm1, which lacks a
transcription-factor motif in its base network, whereas scHopfield's
velocity-based inference scored it correctly. This directional, ground-truth
anchored comparison replaces an earlier magnitude-based table that compared
displacements in incompatible units (Figure R3; benchmark M4).

## R4. Robustness of driver identification

Finally, we characterised how driver nominations depend on the analyst's choices of
prior network and regularisation. Across six configurations (two mouse base
networks by three scaffold-regularisation regimes), the top perturbation-based
drivers were stable (mean pairwise Jaccard of the top fifteen genes 0.67), with the
canonical regulators Gata1, Spi1, Klf1, E2f4, Stat3 and Irf8 recovered in nearly
every configuration. In contrast, the static network-score ranking was unstable
(Jaccard 0.20) and dominated by highly expressed housekeeping genes, and the two
rankings were almost disjoint (zero to one shared gene per configuration). An
unconstrained pseudoinverse fit with no transcription-factor scaffold was an
outlier relative to all six scaffold configurations (perturbation Jaccard 0.36) and
dropped Gata1 and Klf1 from its top drivers, because a dense interaction matrix
dilutes each factor's influence. Two conclusions follow: perturbation simulation is
a more reliable driver readout than the static score, and restricting regulation to
a transcription-factor scaffold is the ingredient that matters, whereas the choice
of prior network and the regularisation strength are largely interchangeable
(Figure R4; benchmarks M5 and M6).

## Figure pointers (committed)

- R1: `benchmark_results/seed_sensitivity_real/reproducibility.png`
- R2: `benchmark_results/circuit_recovery/recovery.png`
- R2b: `benchmark_results/ablations/hill_vs_linear.png`
- R3: `benchmark_results/hemato_ko/` (schopfield_ko_panel.json, celloracle_ko_panel.json)
- R4: `benchmark_results/network_reg_sensitivity/sensitivity.png` (+ FIGURE_GUIDE.md)
