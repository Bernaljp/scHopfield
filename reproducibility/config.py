"""Per-dataset configuration for the comprehensive scHopfield reports.

This is the reproducibility configuration: the per-dataset settings and the fit
parameters behind every figure in the paper. Everything in it is also stated in Methods
or in a figure legend.

Each dataset is analyzed the same way and written to
``<SCHOPFIELD_REPORTS>/<dataset>/{data,plots}/`` plus ``RESULTS.md``. That tree is
regenerated output measured in gigabytes and is not distributed. See
``reproducibility/paths.py`` for the environment variables that locate it and the raw
data.

Neither the seven datasets nor the base GRN scaffolds are redistributed here. The
datasets are public, each under its own accession, and have to be fetched before a fit
can be run. The scaffolds are third-party tables under their own license terms, so each
dataset below names one by registry name rather than by path, and ``sch.fetch_base_grn``
downloads and caches it on first use. See ``DATA_SOURCES.md`` for the terms.

velocity_mode:
  'velocity'    use the RNA-velocity layer (velocity_key)
  'pseudotime'  infer dynamics from pseudotime (estimate_velocity_from_pseudotime) --
                the point we want to emphasize: scHopfield does not require velocity.

lineages: A vs B cell-type groups for the perturbation analyses (driver scoring, KO
bias, epistasis). Where the biology is clear we set them explicitly; otherwise None and
they are derived data-drivenly (the two most pseudotime-terminal, network-distinct
clusters).
"""

import paths

# Every dataset path below is relative to SCHOPFIELD_DATA. Two of the seven were read
# from a separate data directory on the machine where the analyses were run, which is
# SCHOPFIELD_DYNAMISC_DATA.
DATA = paths.DATASETS
DYN = paths.DYNAMISC
# The base GRN scaffolds are not paths. They name tables in scHopfield's registry,
# which sch.fetch_base_grn resolves to a pinned CellOracle commit, downloads once and
# caches; there is no data/ directory in a clean checkout for a path to point into.
# Five of the seven datasets are fit with the mouse atlas prior and two with the human
# promoter prior. The two mouse hematopoiesis entries below previously named a separate
# file, data/hematopoiesis/base_GRN.parquet, which is a byte-identical duplicate of the
# atlas, so one source is named here rather than two paths for the same table.
MOUSE_GRN = "mouse_atlas"
HUMAN_GRN = "human_promoter"

DATASETS = {
    # paul15 -- the pseudotime showcase (has Pseudotime, no reliance on RNA velocity).
    "paul15": dict(
        path=f"{DATA}/hematopoiesis/base_preprocessed.h5ad",
        cluster_key="paul15_clusters", species="mouse",
        base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="pseudotime", pseudotime_key="Pseudotime",
        # use ALL clusters 1..19 in numeric order (megakaryocyte 8Mk and 19Lymph were being
        # skipped); the two 1-cell clusters (11DC, 19Lymph) may still drop out of the fit.
        order=["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP", "8Mk", "9GMP",
               "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu",
               "18Eos", "19Lymph"],
        lineages=dict(A=["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP", "8Mk"],
                      B=["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo",
                         "16Neu", "17Neu", "18Eos"],
                      A_name="erythroid", B_name="myeloid"),
        anchors=["Gata1", "Stat3"],
        # feature these genes in the perturbation panels (Gata1, Stat3 + two others)
        perturb_genes=["Gata1", "Stat3", "Klf1", "Cebpa"],
    ),
    # paul15 with the COARSER cell-type annotation (7 types) -- same pipeline, run alongside.
    "paul15_coarse": dict(
        path=f"{DATA}/hematopoiesis/base_preprocessed.h5ad",
        cluster_key="cell_type", species="mouse",
        base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="pseudotime", pseudotime_key="Pseudotime",
        order=["MEP", "Erythroids", "Megakaryocytes", "GMP", "late_GMP",
               "Granulocytes", "Monocytes"],
        lineages=dict(A=["Erythroids", "Megakaryocytes", "MEP"],
                      B=["GMP", "late_GMP", "Granulocytes", "Monocytes"],
                      A_name="erythroid", B_name="myeloid"),
        anchors=["Gata1", "Stat3"],
        perturb_genes=["Gata1", "Stat3", "Klf1", "Cebpa"],
    ),
    # Dynamo hematopoiesis -- the genuinely missing one. Dynamo-processed object
    # (moment layers M_t/M_n..., no spliced/Ms), so map M_t -> Ms and use its velocity
    # layer directly rather than running scVelo prepare.
    "dynamo_hematopoiesis": dict(
        path=f"{DYN}/hematopoiesis.h5ad",
        # HUMAN: primary human CD34+ HSPCs (Qiu et al., Cell 2022), hence the uppercase HGNC symbols.
        # This was previously declared mouse and fit with the mouse scATAC atlas; build_scaffold matches
        # symbols case-insensitively so it ran silently, but the two priors agree on only 35% of edges.
        cluster_key="cell_type", species="human", base_grn=HUMAN_GRN,
        prepare=False, velocity_mode="velocity",
        velocity_key="velocity_alpha_minus_gamma_s", ms_layer="M_t",
        # dynamo ships the embedded velocity already; use it directly for the input-velocity
        # field instead of reprojecting velocity_S with scVelo.
        velocity_embedding_key="velocity_umap",
        # center-to-sides (HSC in the middle), matching the notebook order
        order=["Meg", "Ery", "MEP-like", "HSC", "GMP-like", "Mon", "Bas", "Neu"],
        lineage_pairs=[dict(A=["Ery", "Meg", "MEP-like"], B=["Neu", "Mon", "Bas", "GMP-like"],
                            A_name="erythroid", B_name="myeloid")],
        lineages=dict(A=["Ery", "Meg", "MEP-like"], B=["Neu", "Mon", "Bas", "GMP-like"],
                      A_name="erythroid", B_name="myeloid"),
        anchors=None,
    ),
    "pancreas": dict(
        path=f"{DATA}/Pancreas/pancreas_scvelo_ready.h5ad",
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
        path=f"{DATA}/generalize/murine_nc.h5ad",
        cluster_key="celltype_update", species="mouse", base_grn=MOUSE_GRN,
        prepare=False, velocity_mode="velocity", velocity_key="velocity_S",
        lineages=None, anchors=None,
    ),
    "human_limb": dict(
        path=f"{DATA}/generalize/human_limb.h5ad",
        cluster_key="leiden_R_celltype", species="human", base_grn=HUMAN_GRN,
        prepare=False, velocity_mode="velocity", velocity_key="velocity_S",
        lineages=None, anchors=None,
    ),
    # schwann -- use the CELL-TYPE annotation ('assignments'), not anatomical 'location'.
    # Location-based clusters (DRG/Incisor/Trunk/Limb) gave two arbitrary, dubious "lineage"
    # pairs (the confusing duplicate 5.1/5.2 sections); the neural-crest cell types give a
    # single meaningful glia-vs-neuron fate decision.
    "schwann": dict(
        path=f"{DYN}/schwann.h5ad",
        cluster_key="assignments", species="mouse", base_grn=MOUSE_GRN,
        prepare=True, velocity_mode="velocity", velocity_key="velocity_S",
        exclude_clusters=["none"],
        lineage_pairs=[dict(A=["SC", "SatGlia", "Gut_glia"],
                            B=["Sensory", "Symp", "Gut_neuron"],
                            A_name="glia", B_name="neuron")],
        lineages=dict(A=["SC", "SatGlia", "Gut_glia"],
                      B=["Sensory", "Symp", "Gut_neuron"],
                      A_name="glia", B_name="neuron"),
        anchors=None,
    ),
}

N_GENES = 2000
# Pseudotime-derived velocity has an ARBITRARY SCALE. estimate_velocity_from_pseudotime returns
# dx/d(pseudotime), and pseudotime is dimensionless, so the inferred rates (gamma, and with them W
# and I) are determined only up to a common factor. Left unfixed on paul15 this put the implied rate
# far above every splicing/labeling dataset and the fit answered by pinning ~62% of genes (70% on
# paul15_coarse) at the hard gamma_max = 10 cap, degenerating most of the model into a saturated
# -10x field.
#
# We fix the time unit by NONDIMENSIONALIZING: time is measured in units of the median gene's
# expression turnover time, i.e. the velocity is scaled so that
#     median_g [ rms_cells(v_g) / rms_cells(x_g) ] = PSEUDOTIME_RATE_TARGET = 1.
# The median over genes is used rather than a global rms because the latter is dominated by a few
# fast genes (the two differ by 0.26x to 1.02x across datasets). This is a change of UNITS only: it
# rescales every rate by one common factor and therefore cannot alter the inferred network
# structure, only the units the rates are reported in.
PSEUDOTIME_RATE_TARGET = 1.0
# Hill exponent ceiling for the CDF fit. Raised well above the package default (8)
# because some datasets (e.g. paul15) have sharply switching genes whose exponent was
# being clipped at 8, worsening the fit. Multi-start refinement (in fit_sigmoid) handles
# double-sigmoid genes.
HILL_N_MAX = 20.0

# Progenitor side of the energy-depth comparison: every annotated type NOT listed here is
# pooled into the terminal side. Single source of truth, imported by
# reproducibility/make_cross_dataset.py (Extended Data Fig. 3a) and by the analysis
# pipeline's multi-dataset report (the supplementary panel). Those two carried
# independent copies until 2026-08-16, and the second had drifted onto labels that mostly do
# not exist in the data: it matched nothing at all for paul15, murine_nc, human_limb and
# schwann, so those four were silently dropped from the panel, and on dynamo it matched Meg,
# a terminal fate with one of the deepest basins, which inverted the progenitor-to-terminal
# comparison for that dataset. Check membership against the data before editing.
PROGENITORS = {
    "pancreas": {"Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine"},
    "paul15": {"7MEP", "9GMP", "10GMP"},
    "paul15_coarse": {"MEP", "GMP", "late_GMP"},
    "dynamo_hematopoiesis": {"HSC", "MEP-like", "GMP-like"},
    "murine_nc": {"Neural crest (PNS glia)", "Neural crest (PNS neurons)"},
    "human_limb": {"PAX3+MyoProg", "PAX3+PAX7+MyoProg", "PAX7+MyoProg1", "PAX7+MyoProg2",
                   "MyoB1", "MyoB2"},   # myoblasts are pre-terminal; terminal = the MyoC myocytes
    "schwann": {"NCC"},
}
# CANONICAL: fit the two-component (bimodal) Hill activation by default for every dataset. A dataset can
# still override with an explicit "bimodal_hill": False in its DATASETS entry. (Promoted 2026-07-26; the
# single-Hill fits are backed up as adata_analyzed_singlehill.h5ad.)
BIMODAL_HILL = True
FIT_KWARGS = dict(
    n_epochs=600, batch_size=128, learning_rate=0.1,
    reconstruction_regularization=100, bias_regularization=1, bias_penalty="l1",
    refit_gamma=True, use_plateau_scheduler=True, plateau_patience=100,
    plateau_factor=0.1, drop_last=True, include_neighbors=True, neighbor_fraction=0.2,
    # only_TFs=True is the intended default: only TF->gene edges may be nonzero
    # (gene->gene hard-zeroed by the column mask), and off-scaffold TF->gene edges are
    # soft-penalized by scaffold_regularization. See scaffold-penalty-semantics.
    only_TFs=True,
    # ---- Boundedness (CANONICAL as of 2026-07-29; replaces boundedness_lambda=10 'saturated') ----
    #
    # The model is f(x) = W sigma(x) - Gamma x + I with a SATURATING Hill activation, so the drive
    # is globally bounded by some D and r(x) = x.f(x) <= -gamma_min ||x||^2 + D ||x||. That is the
    # "bounded drift plus leak" level of the Theorem-1 hierarchy: boundedness is certified BY
    # ARCHITECTURE with absorbing radius D/gamma_min, and no penalty is required. The only way it
    # fails is gamma_i -> 0, which does not unbound the system, it sends the radius to infinity.
    #
    # The previous default (mode='saturated', lambda=10) constrained the FIXED POINT per gene inside
    # the data range: an interior constraint far stronger than the far-field condition the theorem
    # asks for, applied everywhere rather than only far out. It cost roughly half the test velocity
    # reconstruction on every dataset (pancreas 0.958 -> 0.496, dynamo +0.687 -> -0.044, schwann
    # 0.742 -> 0.003, i.e. no better than noise) and on pancreas its own certificate was WORSE than
    # applying nothing, because it raises the median gamma while the bound depends on the minimum
    # (gamma_min stayed at 2.8e-07).
    #
    # Canonical is now the "C1" configuration, chosen from a 33-arm sweep over 3 datasets:
    #   gamma_min  = 0.01  -- hard per-gene floor, fixes the collapsed-gamma TAIL, no gradient
    #                         pressure anywhere in the data region. Distinct from the deprecated
    #                         gamma_min_frac (a fraction of the MEDIAN, which moved every gene).
    #   mode       = 'radial' + lambda = 0.1 -- the ONE term the theorem requires, a hinge on
    #                         r(x) + alpha||x||^2 - beta over far-field shell samples. Brings the
    #                         BULK drive down (pancreas median x* 5.79 -> 2.79). The other five
    #                         terms of the menu are duplicates, stronger than needed, or, for the
    #                         rotational term, actively harmful (it suppresses the limit cycles the
    #                         repressilator needs).
    # Test velocity cosine under C1 vs the old default: pancreas 0.943 vs 0.496, dynamo 0.664 vs
    # -0.044, schwann 0.732 vs 0.003. schwann does better still on radial l=1 + gamma_min=0.1
    # (cos 0.712, saturated x* p99 36270 -> 272, runaway 1633 -> 157) if a per-dataset override is
    # wanted later.
    boundedness_lambda=0.1,
    gamma_min=0.01,
)
