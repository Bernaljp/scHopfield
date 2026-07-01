# scHopfield: Interpretable dynamical systems learning of regulatory landscapes and perturbational responses from single-cell data

*Assembled full draft.* Built from the ManuscriptIdeaJesper materials (Main_v5,
Results_v14, Introduction_v4, Abstract_v5) with the quantitative benchmark results
from this analysis round integrated into the Results (findings M1-M7 in
`benchmark_results/FINDINGS.md`). American English, no em dashes, honest framing.
Bracketed italic notes flag figure placeholders and items still to be filled.

Affiliations (from Main_v5): 1 Biomedical Sciences Division, KAUST, Thuwal, Saudi
Arabia; 2 SDAIA-KAUST Center of Excellence in Data Science and AI, Thuwal; 3
Algorithmic Dynamic Lab, Dept. of Oncology and Pathology, Karolinska Institute,
Stockholm; 4 Institute of Chemical Biology, Ilia State University, Tbilisi, Georgia;
5 Unit of Computational Medicine, Dept. of Medicine, Karolinska Institutet,
Stockholm; 6 CEMSE Division, KAUST, Thuwal; 7 Science for Life Laboratory, Solna.

---

## Abstract

Learning mechanistically interpretable dynamical models from observational data
remains a central challenge in both artificial intelligence and systems biology. In
single-cell genomics, developmental trajectories, regulatory interactions, cellular
attractors, and perturbational responses are typically inferred using separate
computational frameworks, limiting mechanistic interpretability and generalization
beyond the observed data. Here we introduce scHopfield, a dynamical systems learning
framework that jointly infers gene regulatory networks, transcriptional dynamics,
stability landscapes, and perturbational responses from single-cell transcriptomic
and RNA-velocity measurements. By using continuous Hopfield dynamics as an inductive
bias, scHopfield directly links regulatory structure to cellular state transitions
through a shared dynamical representation. Our representation reproduces canonical
attractor, bifurcation, and oscillatory dynamics, improves recovery of causal
regulatory structure, and enables reconstruction of lineage-specific energy
landscapes and stability states. Importantly, scHopfield supports out-of-distribution
perturbation prediction and identifies higher-order regulatory interactions that
static network analyses cannot recover. These results establish a unified framework
for interpretable dynamical systems learning in biological systems.

---

## Main

Single-cell transcriptomics has transformed our ability to characterize cellular
heterogeneity, developmental trajectories, and lineage specification across tissues
and organisms. Recent advances in RNA velocity and trajectory inference have enabled
the reconstruction of dynamic cellular transitions from static transcriptomic
snapshots. However, despite rapid progress in foundation models, representation
learning, and graph-based inference, a central challenge remains: how to infer
mechanistically interpretable gene regulatory dynamics from observational single-cell
data. Most existing methods either reconstruct low-dimensional latent trajectories or
infer static gene regulatory networks (GRNs), but rarely unify regulatory structure
and cellular dynamics within a common mathematical framework. Consequently, many
approaches achieve strong predictive or embedding performance yet remain difficult to
interpret mechanistically, particularly in perturbational or out-of-distribution
settings where cells are driven into previously unobserved transcriptional states.
Gene regulation is fundamentally a dynamical systems problem in which regulatory
interactions, attractor stability, and temporal transitions jointly govern cellular
identity and fate decisions.

A central challenge in single-cell biology is that developmental trajectories alone
do not uniquely determine the regulatory mechanisms that generate them. Distinct
regulatory architectures can often produce similar observable cellular transitions,
especially in high-dimensional systems with sparse sampling and incomplete
observations. Reconstructing trajectories is therefore not equivalent to recovering
regulatory causality. This disconnect has created a persistent gap between RNA-velocity
approaches that model cellular dynamics and GRN inference methods that reconstruct
regulatory structure. Bridging this gap requires models that jointly infer regulation,
dynamics, and stability within a unified mathematical framework. We hypothesized that
regulatory networks, developmental trajectories, cellular attractors, and
perturbational responses are different manifestations of a common underlying dynamical
system, and that jointly inferring these quantities would improve both biological
interpretability and predictive power.

Continuous Hopfield networks provide a natural framework for this challenge because
they treat regulatory interactions, developmental trajectories, cellular attractors,
and perturbational responses as manifestations of the same dynamical system. Cellular
identities emerge as attractor states within an energy landscape, differentiation
corresponds to movement between attractors, and perturbations reshape the landscape's
geometry. Unlike latent-embedding approaches that primarily reconstruct trajectories,
Hopfield dynamics provide explicit energy functions, interpretable interaction
matrices, and Jacobian-based stability analyses, enabling direct characterization of
cellular stability, attractor geometry, and perturbational sensitivity. To this end we
introduce scHopfield, an interpretable dynamical systems framework that jointly infers
gene regulatory networks, transcriptional dynamics, cellular stability landscapes, and
perturbational responses from single-cell transcriptomic and RNA-velocity data.

---

## Results

### scHopfield unifies regulatory inference and dynamical systems learning *[Figure 1]*

scHopfield models transcriptional dynamics with cell-type-specific interaction
matrices parameterized by interpretable Hill kinetics:

  dx_i/dt = sum_j W_ij^(C) sigma_j(x_j) - gamma_i x_i + I_i,

where W^(C) is the cell-type-specific interaction matrix, sigma_j is a Hill function
with gene-specific coefficient n_j and threshold k_j, gamma_i is the degradation rate,
and I_i is a bias term capturing basal expression. The RNA velocity enters on the
left-hand side, so regulatory interactions in W^(C) directly govern the observed
transcriptional dynamics. scHopfield infers W^(C) in two modes: an unconstrained mode
using the Moore-Penrose pseudoinverse (yielding effective interaction networks), and a
scaffold-guided mode in which prior regulatory information (for example from ATAC-seq
or curated databases) constrains the network through masked layers with elastic-net
regularization on non-scaffold weights. The Hill parameters (k_j, n_j) are estimated
directly from the empirical cumulative distribution of each gene's expression (see
Methods). Because the inferred network defines a dynamical system, it also defines an
energy function and Jacobian-based measures of local stability, divergence, and
rotational dynamics, enabling mechanistic analysis of attractor states, lineage
transitions, and perturbational responses.

The stochastic optimization used for scaffold-guided inference is seeded so that all
results are reproducible: with a fixed seed, repeated fits are bit-identical
(interaction-matrix and centrality correlations both 1.000), whereas unseeded fits,
although numerically stable in W (Pearson 0.999), produce gene rankings that drift
(centrality Spearman 0.73 to 0.93 across seeds). We therefore seed all analyses and
report ranking stability where relevant (benchmark M1).

### Continuous Hopfield dynamics capture canonical biological behaviors *[Figure 2]*

To test whether the scHopfield formulation reproduces canonical regulatory dynamics,
we analyzed two archetypal gene circuits: a bistable two-gene mutual-inhibition switch
and a cyclic three-gene repressilator. Reformulating these systems in the Hopfield
formalism maps regulatory interactions onto an energy-landscape interpretation in which
stable cellular states emerge as attractors. In the toggle switch, increasing
inhibitory coupling induced a bifurcation from a monostable regime to two alternative
stable equilibria, and the inferred energy landscape exhibited distinct minima
corresponding to the alternative attractor states. In the repressilator, scHopfield
reproduced sustained oscillations and limit-cycle dynamics arising from the
antisymmetric component of the interaction matrix, demonstrating that the framework
captures both equilibrium and non-equilibrium behavior.

On these circuits, where the ground-truth interaction matrix is known exactly,
scHopfield recovered the network with perfect edge-sign accuracy (1.00) and
near-perfect signed correlation (>= 0.9998) across all noise levels and scaffold
priors tested, with edge-detection AUROC and AUPRC of 1.0 for the sparse repressilator.
Reconstruction error grew gracefully with observation noise (relative Frobenius
distance from 0.0005 at zero noise to 0.38 at the highest noise), and recovery held
even with no scaffold prior (benchmark M3). The Hill nonlinearity was essential to
these results: replacing it with a linear model collapsed velocity reconstruction on
the toggle switch (R-squared 0.0001 versus 1.000) and could not represent the switch's
bistability, since a linear autonomous system admits at most one fixed point and
recovered none of the three stable states that the Hill model recovered correctly
(benchmark M7). *[To add: head-to-head recovery against external GRN-inference methods
(for example GENIE3, SCENIC) on dyngen simulations; not yet run.]*

### Learned perturbational dynamics recover established hematopoietic lineage regulators *[Figure 3]*

To assess the biological fidelity of the perturbation framework, we performed
systematic in-silico single-gene knockouts within scHopfield-inferred hematopoietic
differentiation trajectories (Paul et al., 2015). Perturbational effects were modeled
as expression-shift vectors projected onto RNA-velocity fields, enabling direct
comparison of perturbation-induced state transitions with the endogenous developmental
flow, and lineage effects were quantified by a cosine-similarity lineage-bias metric
between perturbational and erythroid or myeloid velocity directions.

scHopfield recovered canonical hematopoietic regulators. For a panel of literature
master regulators, every knockout was assigned the correct direction of lineage shift
(10 of 10): knockout of erythroid or megakaryocyte masters (Gata1, Klf1, Zfpm1, Nfe2,
Gata2) biased differentiation toward the myeloid branch, and knockout of myeloid
masters (Spi1/PU.1, Cebpa, Cebpe, Gfi1, Irf8) biased toward the erythroid branch, with
the canonical Gata1-Spi1 antagonism producing the largest-magnitude shifts (Gata1
lineage bias -0.21, Spi1 +0.25). Scored on the identical panel and geometry, CellOracle
predicted 7 of 9 correctly and could not score the erythroid cofactor Zfpm1, which
lacks a transcription-factor motif in its base network, whereas scHopfield's
velocity-based inference scored it correctly (benchmark M4). Dose-response analyses
further showed coherent nonlinear behavior, with increasing perturbation magnitude
progressively reshaping the erythroid-myeloid balance, and cascade simulations showed
temporal amplification of perturbational effects across differentiation trajectories.

### Biological priors via scaffold-guided optimization improve recovery of causal regulatory structure *[Figure 4]*

In simulated systems with known ground-truth GRNs, unconstrained (pseudoinverse)
optimization accurately reconstructed transcriptional dynamics but often recovered
minimal effective networks rather than the true causal topology, reflecting a
fundamental inverse problem in which multiple architectures can generate similar
observable trajectories. Incorporating scaffold-guided optimization, which jointly
balances velocity reconstruction with consistency to a prior network, consistently
improved AUROC and AUPRC relative to unconstrained models, and restricting outgoing
regulatory edges to transcription factors further improved reconstruction accuracy.
scHopfield is not restricted to the supplied scaffold and can infer additional
interactions absent from the prior network, which represent candidate novel regulatory
interactions in biological systems.

We quantified how much these choices matter on the hematopoiesis data. Across six
configurations (two mouse base networks by three scaffold-regularization regimes), the
top perturbation-based drivers were stable (mean pairwise Jaccard of the top fifteen
genes 0.67), and the canonical regulators Gata1, Spi1, Klf1, E2f4, Stat3, and Irf8 were
recovered in nearly every configuration. In contrast, the top genes by static
network-score were unstable (Jaccard 0.20) and dominated by highly expressed
housekeeping genes; the two rankings were almost disjoint (zero to one shared gene per
configuration). An unconstrained pseudoinverse fit with no transcription-factor
scaffold was an outlier relative to all six scaffold configurations (perturbation
Jaccard 0.36) and dropped Gata1 and Klf1 from its top drivers, because a dense
interaction matrix dilutes each factor's influence (benchmarks M5, M6). Two
conclusions follow: perturbation simulation is a more reliable driver readout than the
static score, and restricting regulation to a transcription-factor scaffold is the
ingredient that matters, whereas the specific prior network and the regularization
strength are largely interchangeable.

### Energy landscapes and Jacobian analysis reveal lineage-specific stability states during pancreatic endocrinogenesis *[Figure 5]*

Because the inferred networks define a dynamical system, we used Jacobian eigenvalues,
energy landscapes, and network topology to analyze pancreatic endocrinogenesis, a
developmental system spanning ductal progenitors, endocrine progenitor states, and
terminally differentiated endocrine populations. Jacobian spectra revealed pronounced
lineage-specific differences in stability: Delta and Epsilon cells exhibited broader
distributions of positive real eigenvalues, indicating greater instability and
heightened perturbational sensitivity, whereas pre-endocrine and mature Beta cells
occupied more stable dynamical regimes; imaginary components indicated stronger
oscillatory tendencies in Delta and Epsilon populations. Energy decomposition showed
that ductal progenitors occupied relatively low-energy states while terminally
differentiated Beta cells exhibited strongly negative interaction energies, and
Ngn3-high endocrine progenitors displayed the greatest cell-to-cell energetic
variability, consistent with a heterogeneous transitional population approaching
lineage bifurcation. Network analysis placed several regulators (Meis2, Pax6, Isl1,
Mlxipl) at central positions across endocrine populations, with Sox9 associated with
ductal networks, Foxa3 with Ngn3-high progenitors, and Pdx1 with mature Beta programs,
and identified chromatin-associated regulators (Hmga2, Hmgn3, Prdm16) as prominent
hubs. *[Figure 5; pancreas pipeline reproduced locally via scVelo; Jacobian/energy
panels to be regenerated from the seeded fit.]*

### Higher-order perturbational simulations uncover lineage-balancing regulatory programs *[Figure 6]*

Having established that scHopfield recovers known lineage regulators, we asked whether
the inferred dynamical system could predict previously unrecognized regulatory
interactions, a fundamentally out-of-distribution prediction problem. We prioritized
candidate transcription-factor pairs with a multi-stage ranking strategy integrating
regulatory edge strength, lineage specificity, perturbational variance, and predicted
effect size, and evaluated double-knockout simulations with synergy, epistasis, and
cancellation metrics. In addition to recovering the established Gata1-Spi1 antagonism,
scHopfield predicted multiple candidate nonlinear interactions involving Stat3, E2f4,
Irf8, Cxxc1, Serp1, Apoe, Mt2, and Cathepsin G. Stat3 emerged as a prominent
lineage-balancing regulator whose perturbation consistently shifted differentiation
toward erythroid trajectories (Stat3 knockout lineage bias +0.165, overexpression
-0.011, confirming bidirectional control), suggesting a broader role in lineage
plasticity. Several predicted interactions were detectable only through dynamical
perturbation analysis and were not apparent from static co-expression patterns or
network topology, demonstrating that scHopfield provides an interpretable framework for
out-of-distribution perturbation prediction and hypothesis generation.

---

## Discussion

A central challenge in single-cell biology is the disconnect among cellular
trajectories, regulatory structure, and mechanistic models of cell-fate control.
Although RNA velocity and trajectory inference have improved our ability to reconstruct
dynamic transitions, these approaches often provide limited insight into the regulatory
mechanisms underlying the observed trajectories, while GRN inference methods often
reconstruct putative interactions without explicitly modeling the dynamical processes
that govern state transitions. scHopfield unifies these quantities within a continuous
Hopfield framework, directly linking GRN structure to cellular trajectories through a
shared dynamical representation and enabling simultaneous reconstruction of regulatory
interactions, attractor landscapes, and perturbational responses.

A key finding is the importance of explicitly addressing the inverse problem in
single-cell regulatory inference: accurately reconstructing cellular dynamics does not
imply accurate recovery of causal regulatory structure. Our sensitivity analyses make
this concrete. Perturbation-based driver identification was robust to the choice of
prior network and regularization strength, whereas a static network score was unstable
and biologically uninformative, and removing the transcription-factor scaffold entirely
degraded recovery of canonical regulators. Incorporating prior biological knowledge
through a scaffold substantially enhances identifiability while retaining the
flexibility to discover novel interactions.

An additional advantage of the scHopfield formulation is that inferred networks
naturally define an energy landscape and its stability structure, enabling analysis of
attractor geometry, local stability, and oscillatory behavior. In pancreatic
endocrinogenesis, progenitor populations occupied more heterogeneous and dynamically
flexible regions of the landscape than terminally differentiated populations, providing
a quantitative counterpart to long-standing views of differentiation as movement across
a structured landscape. Our perturbational analyses further suggest that perturbation
biology can be viewed as a dynamical extrapolation problem: by embedding perturbations
within an inferred dynamical system, scHopfield recovered known lineage regulators and
generated higher-order predictions that were not apparent from static analyses.

Several limitations should be considered. First, the accuracy of inferred interactions
depends on the quality of RNA-velocity estimates and the completeness of biological
priors. Second, the inferred networks capture effective dynamical interactions and may
not map directly to physical molecular interactions. Third, recovery is faithful for
systems in the model's function class; biophysical circuits that are not natively of
Hopfield form are not recovered as well, a limitation we report explicitly. Fourth,
predicted novel interactions require experimental validation. Future work could extend
the framework to integrate multi-omic, chromatin-accessibility, spatial, and
lineage-tracing data. Taken together, our results establish a unified framework in
which regulatory networks, developmental trajectories, cellular attractors, and
perturbational responses emerge from a common dynamical representation.

---

## Figure pointers (committed benchmark figures)

- Figure 2 (canonical behaviors / recovery): `benchmark_results/circuit_recovery/recovery.png`;
  Hill-vs-linear `benchmark_results/ablations/hill_vs_linear.png`
- Figure 3 (known-KO recovery): `benchmark_results/hemato_ko/` (schopfield_ko_panel.json,
  celloracle_ko_panel.json)
- Figure 4 (scaffold / robustness): `benchmark_results/network_reg_sensitivity/sensitivity.png`
  and `FIGURE_GUIDE.md`
- Reproducibility (Methods/supplementary): `benchmark_results/seed_sensitivity_real/reproducibility.png`

## Outstanding placeholders

- Word/figure/reference counts (Main_v5 front matter).
- Figure 1 schematic (method overview).
- External GRN-inference baselines (GENIE3, SCENIC, dyngen) for Figure 2, not yet run.
- Figure 5 pancreas Jacobian/energy panels to be regenerated from the seeded fit.
- References list (the Methods section reference list can seed this).
