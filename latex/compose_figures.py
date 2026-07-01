"""Compose improved multi-panel paper figures from harvested real panels.
Layouts follow the ManuscriptIdeaJesper/Figures slide designs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = "/home/bernaljp/Documents/SCH/benchmark_results"
OUT = "/home/bernaljp/Documents/SCH/latex/figures"
os.makedirs(OUT, exist_ok=True)


def panel(ax, path, label, title=None):
    ax.axis("off")
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    if title:
        ax.set_title(title, fontsize=9)
    ax.text(-0.01, 1.03, label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="right")


def grid(name, panels, nrow, ncol, figsize, suptitle=None):
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for k, (path, lab, title) in enumerate(panels):
        panel(axes[k // ncol][k % ncol], path, lab, title)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 1: schematic (a) + 3 real panels (b,c,d) + comparison table (e) ----
f1 = f"{R}/figures/fig1_panels"
fig = plt.figure(figsize=(11, 9.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1.55, 1.35, 0.85], hspace=0.18, wspace=0.05)
panel(fig.add_subplot(gs[0, :]), f"{R}/figures/fig1_schematic.png", "a")
panel(fig.add_subplot(gs[1, 0]), f"{f1}/hill_cdf.png", "b", "Hill activation fit")
panel(fig.add_subplot(gs[1, 1]), f"{f1}/grn_circular.png", "c", "Cell-type-specific GRN")
panel(fig.add_subplot(gs[1, 2]), f"{R}/pancreas/nb08_figures/energy_components_umap.png", "d", "Energy on embedding")
panel(fig.add_subplot(gs[2, :]), f"{R}/figures/comparison_table.png", "e")
fig.suptitle("Figure 1. scHopfield framework and conceptual overview", y=0.995, fontsize=12, fontweight="bold")
fig.savefig(f"{OUT}/fig1.png", dpi=155, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: toggle + repressilator dynamics/energy ----
p2 = f"{R}/figures/fig2_panels"
grid("fig2", [
    (f"{p2}/c11_Dynamics_and_Recovery_Plots_for_Toggle_S.png", "a", "Toggle switch: dynamics + recovery"),
    (f"{p2}/c24_Dynamics_and_Recovery_Plots_for_Repressi.png", "b", "Repressilator: dynamics + recovery"),
    (f"{p2}/c15_Matched_the_couplings_to_the_Jacobian_st.png", "c", "Toggle energy-landscape bifurcation + flow"),
    (f"{p2}/c31_cell31.png", "d", "Repressilator energy landscape"),
], 2, 2, (12, 8))

# ---- Figure 3: perturbation recovers regulators ----
p3 = f"{R}/figures/fig3"
grid("fig3", [
    (f"{p3}/wt_ko_oe_trajectories.png", "a", "WT / KO / OE trajectories"),
    (f"{p3}/perturbation_effect_heatmaps.png", "b", "Perturbation effect (per cluster)"),
    (f"{p3}/top_affected_genes.png", "c", "Top affected genes"),
    (f"{p3}/hopfield_perturbed_flow_KO.png", "d", "Perturbed flow on embedding"),
], 2, 2, (12, 8.5))

# ---- Figure 4: robustness (sensitivity + clustered jaccard) ----
p4 = f"{R}/network_reg_sensitivity"
grid("fig4", [
    (f"{p4}/sensitivity.png", "a", "Driver stability vs network / regularization"),
    (f"{p4}/plots/C_jaccard_top_pert.png", "b", "Clustered perturbation-driver Jaccard"),
], 1, 2, (14, 5))

# ---- Figure 5: pancreatic stability + energy ----
p5 = f"{R}/pancreas/nb08_figures"
grid("fig5", [
    (f"{p5}/energy_components_umap.png", "a", "Energy components on embedding"),
    (f"{p5}/jacobian_eigenvalue_spectrum.png", "b", "Jacobian eigenvalue spectra"),
    (f"{p5}/jacobian_positive_evals_boxplot.png", "c", "Instability by cell type"),
    (f"{p5}/rotational_dynamics.png", "d", "Rotational dynamics"),
], 2, 2, (12, 8.5))

# ---- Figure 6: higher-order perturbation ----
p6 = f"{R}/figures/fig6"
grid("fig6", [
    (f"{p6}/dose_response.png", "a", "Dose-response of lineage bias"),
    (f"{p6}/double_ko_recipe.png", "b", "Double-KO recipe / synergy"),
    (f"{p6}/dense_top10_shifters.png", "c", "Top double-KO shifters"),
    (f"{p6}/stat_family_circuit.png", "d", "STAT-family regulatory circuit"),
], 2, 2, (12, 8.5))

print("composed fig1 (5 panels), fig2-fig6 (multi-panel)")
