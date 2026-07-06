"""Per-dataset configuration for the comprehensive scHopfield reports.

Each dataset is analyzed the same way (notebooks 01-08 pipeline) and written to
``figure_packs/reports/<dataset>/{data,plots}/`` + ``RESULTS.md`` (the whole
``figure_packs/`` tree is gitignored). The generating code lives in
``analyses/reports/`` (tracked).

velocity_mode:
  'velocity'    use the RNA-velocity layer (velocity_key)
  'pseudotime'  infer dynamics from pseudotime (estimate_velocity_from_pseudotime) --
                the point we want to emphasize: scHopfield does not require velocity.

lineages: A vs B cell-type groups for the perturbation analyses (driver scoring, KO
bias, epistasis). Where the biology is clear we set them explicitly; otherwise None and
they are derived data-drivenly (the two most pseudotime-terminal, network-distinct
clusters).
"""

DYN = "/home/bernaljp/Documents/DynamiSC/Data"
MOUSE_GRN = "data/hematopoiesis/networks/mouse_scATAC_atlas.parquet"
HUMAN_GRN = "data/human_promoter_base_GRN.parquet"

DATASETS = {
    # paul15 -- the pseudotime showcase (has Pseudotime, no reliance on RNA velocity).
    "paul15": dict(
        path="data/hematopoiesis/base_preprocessed.h5ad",
        cluster_key="paul15_clusters", species="mouse",
        base_grn="data/hematopoiesis/base_GRN.parquet",
        prepare=False, velocity_mode="pseudotime", pseudotime_key="Pseudotime",
        lineages=dict(A=["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"],
                      B=["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo",
                         "16Neu", "17Neu", "18Eos"],
                      A_name="erythroid", B_name="myeloid"),
        anchors=["Gata1", "Stat3"],
        # feature these genes in the perturbation panels (Gata1, Stat3 + two others)
        perturb_genes=["Gata1", "Stat3", "Klf1", "Cebpa"],
    ),
    # Dynamo hematopoiesis -- the genuinely missing one. Dynamo-processed object
    # (moment layers M_t/M_n..., no spliced/Ms), so map M_t -> Ms and use its velocity
    # layer directly rather than running scVelo prepare.
    "dynamo_hematopoiesis": dict(
        path=f"{DYN}/hematopoiesis.h5ad",
        cluster_key="cell_type", species="mouse", base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="velocity",
        velocity_key="velocity_alpha_minus_gamma_s", ms_layer="M_t",
        # center-to-sides (HSC in the middle), matching the notebook order
        order=["Meg", "Ery", "MEP-like", "HSC", "GMP-like", "Mon", "Bas", "Neu"],
        lineage_pairs=[dict(A=["Ery", "Meg", "MEP-like"], B=["Neu", "Mon", "Bas", "GMP-like"],
                            A_name="erythroid", B_name="myeloid")],
        lineages=dict(A=["Ery", "Meg", "MEP-like"], B=["Neu", "Mon", "Bas", "GMP-like"],
                      A_name="erythroid", B_name="myeloid"),
        anchors=None,
    ),
    "pancreas": dict(
        path="data/Pancreas/pancreas_scvelo_ready.h5ad",
        cluster_key="clusters", species="mouse", base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="velocity", velocity_key="velocity_S",
        # differentiation left->right
        order=["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine",
               "Alpha", "Beta", "Delta", "Epsilon"],
        # multifurcation: the meaningful axis is differentiated vs progenitor (does a KO
        # block differentiation / push cells back to progenitors?), plus alpha-vs-beta.
        lineage_pairs=[
            dict(A=["Alpha", "Beta", "Delta", "Epsilon"], B=["Ductal", "Ngn3 low EP", "Ngn3 high EP"],
                 A_name="differentiated", B_name="progenitor"),
            dict(A=["Alpha"], B=["Beta"], A_name="alpha", B_name="beta"),
        ],
        lineages=dict(A=["Alpha"], B=["Beta"], A_name="alpha", B_name="beta"),
        anchors=None,
    ),
    "murine_nc": dict(
        path="data/generalize/murine_nc.h5ad",
        cluster_key="celltype_update", species="mouse", base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="velocity", velocity_key="velocity_S",
        lineages=None, anchors=None,
    ),
    "human_limb": dict(
        path="data/generalize/human_limb.h5ad",
        cluster_key="leiden_R_celltype", species="human", base_grn=HUMAN_GRN,
        prepare=False, velocity_mode="velocity", velocity_key="velocity_S",
        lineages=None, anchors=None,
    ),
    "schwann": dict(
        path=f"{DYN}/schwann.h5ad",
        cluster_key="location", species="mouse", base_grn=MOUSE_GRN,
        prepare=True, velocity_mode="velocity", velocity_key="velocity_S",
        lineages=None, anchors=None,
    ),
}

N_GENES = 200
# Hill exponent ceiling for the CDF fit. Raised well above the package default (8)
# because some datasets (e.g. paul15) have sharply switching genes whose exponent was
# being clipped at 8, worsening the fit. Multi-start refinement (in fit_sigmoid) handles
# double-sigmoid genes.
HILL_N_MAX = 20.0
FIT_KWARGS = dict(
    n_epochs=600, batch_size=128, learning_rate=0.1,
    reconstruction_regularization=100, bias_regularization=1, bias_penalty="l1",
    refit_gamma=True, use_plateau_scheduler=True, plateau_patience=100,
    plateau_factor=0.1, drop_last=True, include_neighbors=True, neighbor_fraction=0.2,
)
