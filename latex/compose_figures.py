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


def panel(ax, path, label):
    ax.axis("off")
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="right")


# ---- Figure 1: schematic (a) + comparison table (b) ----
fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.05)
panel(fig.add_subplot(gs[0]), f"{R}/figures/fig1_schematic.png", "a")
panel(fig.add_subplot(gs[1]), f"{R}/figures/comparison_table.png", "b")
fig.suptitle("Figure 1. scHopfield framework and conceptual overview", y=0.99, fontsize=12, fontweight="bold")
fig.savefig(f"{OUT}/fig1.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: toggle + repressilator dynamics/energy (canonical behaviors) ----
p = f"{R}/figures/fig2_panels"
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panel(axes[0, 0], f"{p}/c11_Dynamics_and_Recovery_Plots_for_Toggle_S.png", "a")
panel(axes[0, 1], f"{p}/c24_Dynamics_and_Recovery_Plots_for_Repressi.png", "b")
panel(axes[1, 0], f"{p}/c15_Matched_the_couplings_to_the_Jacobian_st.png", "c")
panel(axes[1, 1], f"{p}/c31_cell31.png", "d")
axes[0, 0].set_title("Toggle switch: dynamics + recovery", fontsize=10)
axes[0, 1].set_title("Repressilator: dynamics + recovery", fontsize=10)
axes[1, 0].set_title("Toggle energy-landscape bifurcation + flow", fontsize=10)
axes[1, 1].set_title("Repressilator energy landscape", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: perturbation flow (Gata1) + known-KO panels ----
# (illustrative perturbed flow already at figures/fig3; pair with recovery bench in caption)
print("composed fig1.png (schematic + comparison table), fig2.png (toggle+repressilator 4-panel)")
