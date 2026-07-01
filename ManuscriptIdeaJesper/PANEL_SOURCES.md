# Figure panel sources (where each panel comes from)

Mapping the figure-design slides (`ManuscriptIdeaJesper/Figures/Slide*.png`) to the
actual code that produces each panel. The slide numbers do NOT match final figure
numbers (sections were reordered). Slides 1 and 4 are text plans; Slides 2, 3, 6 are
rendered panel banks. Recognized sources below (notebook = `notebooks/...`).

## Figure 1 - framework overview (Slide 3, full 5-band render; Slide 1 = plan)

| Panel | Content | Source |
|---|---|---|
| Input a | Single-cell lineage tree | BioRender-style illustration (not a code plot) |
| Input b | Expression matrix | schematic |
| Input c | RNA velocity streams (pancreas) | scVelo velocity + nb `experiments/08_pancreas` |
| Input d | Pseudotime (pancreas) | nb `experiments/08_pancreas` |
| Training a | Hill CDF fit sigma(x)=x^n/(x^n+k^n) | nb `core/01` sigmoid fitting (`pl.plot_sigmoid_fit`) |
| Training b | Model-structure NN diagram | custom diagram (WÎ˜(1-G), Wsigma-Î³x+I) |
| Training c | Scaffold circular network | nb `core/03`/`08` (`pl.plot_grn_network`) |
| Inference | Hill fit (Gebf/Ctrb1), scaffold soft/hard loss, unconstrained opt convergence + W heatmap + circular net | nb `core/01`,`05`,`06`; `scripts/cell_oracle_vs_schopfield.py` |
| Downstream 1 | Simulation expr-vs-time | nb `core/01`,`05` (`dyn.simulate_trajectory`) |
| Downstream 1 | Cell-specific Jacobian heatmap | nb `core/04`,`08` (`tl.compute_jacobians`) |
| Downstream 1 | Celltype-specific GRN (circular) | nb `core/03`,`08` |
| Downstream 1 | Flow reconstruction (energy + flow) | `experiments/small_circuits`, nb `core/02` energy |
| Downstream 1 | Hopfield energy 3D landscape | `experiments/small_circuits` (3D energy surface) |
| Downstream 2 | Perturbation Delta-x grid (Gata1/Stat3/Spi1/Sox3) | nb `core/06`,`07` perturbation flow (`hopfield_perturbed_flow`) |
| Downstream 2 | KO ranking bar chart | nb `core/06` (single-KO lineage-bias bar) |
| Downstream 2 | Model capabilities table | rebuilt: `benchmark_results/figures/comparison_table.png` |

## Figure 2 - dynamical fidelity + benchmarking (Slides 4 plan, 6 render)

All illustrative panels from `notebooks/experiments/small_circuits.ipynb`
(extracted to `benchmark_results/figures/fig2_panels/`):
- toggle circuit diagram + Cherry-Adler ODEs; toggle phase portrait / dynamics-recovery (c11);
- 2D vector-field flow (c13); pitchfork bifurcation diagram (steady state vs inhibition);
- 3D energy landscape; energy-landscape bifurcation + flow (c14, c15);
- complete Jacobian stability maps lambda1/lambda2 (c18); repressilator dynamics-recovery (c24);
  repressilator energy landscape (c31); eigenvalues (c32).
Quantitative panels (this round): `circuit_recovery/recovery.png` (M3),
`ablations/hill_vs_linear.png` (M7), `grn_baseline/genie3_vs_schopfield.png` (M8).

## Figure 3 - perturbation recovers lineage regulators

Illustrative: nb `core/05` (WT/KO/OE trajectories, perturbation heatmaps, perturbed
flow on embedding), nb `core/06` (KO ranking, lineage bias).
Quantitative: `benchmark_results/hemato_ko/` (scHopfield 10/10 vs CellOracle 7/9, M4).

## Figure 4 - robustness of driver identification

`benchmark_results/network_reg_sensitivity/` (M5 sensitivity, M6 no-scaffold) + the
22-figure `FIGURE_GUIDE.md` pack.

## Figure 5 - pancreatic endocrinogenesis stability

nb `experiments/08_pancreas` (extracted to `benchmark_results/pancreas/nb08_figures/`):
energy on UMAP, Jacobian eigenvalue spectra, positive-eigenvalue boxplots (Delta/Epsilon
unstable), rotational dynamics.

## Figure 6 - higher-order perturbation

nb `core/07` (extracted to `benchmark_results/figures/fig6/`): dose-response, double-KO
recipe, dense top-10 shifters, STAT-family circuit, phase portraits.

## Composites built this round (in the compiled PDF)

- `latex/figures/fig1.png` = overview schematic + comparison table.
- `latex/figures/fig2.png` = toggle+repressilator dynamics/energy 4-panel (from small_circuits).
- Fig 3-6 use the harvested illustrative panels + the quantitative benchmark panels.
