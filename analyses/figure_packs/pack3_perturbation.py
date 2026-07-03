"""Figure pack 3: perturbation analysis (pre/post scores, dose-response, simulations).

For datasets with a defined A-vs-B lineage choice: the structural (pre-perturbation)
driver scores, the well-defined pre/post lineage-commitment change from in-silico KO
(the improved score -- ``compute_perturbation_commitment_change``), fractional
dose-response curves, and per-cell-type KO impact. GPU/CPU (uses the fitted W arrays).

    figure_packs/pack3_perturbation/<dataset>/{plots,data}/ + FIGURE_GUIDE.md

Run:  PYTHONPATH=analyses/figure_packs .venv/bin/python analyses/figure_packs/pack3_perturbation.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch
import _common as C

OUT = "figure_packs/pack3_perturbation"
N_DRIVERS = 8


def _ko_commitment(adata, wt, gene, A, B, basis, ck):
    """Simulate a KO and return its per-lineage commitment change + cluster effects."""
    ko = sch.dyn.simulate_perturbation(adata, perturb_condition={gene: 0.0},
                                       cluster_key=ck, n_propagation=3, verbose=False)
    sch.tl.calculate_flow(ko, source="perturbed", basis=basis, cluster_key=ck, verbose=False)
    res = sch.tl.compute_perturbation_commitment_change(ko, wt, A, B, basis=basis, cluster_key=ck)
    eff = sch.tl.compute_cluster_effects(ko, C.present_clusters(adata, ck), cluster_key=ck)
    return res, eff, ko


def run_dataset(name, ck, lin):
    adata = C.load(name)
    A, B = lin["A"], lin["B"]
    An, Bn = lin["A_name"], lin["B_name"]
    basis = C.basis_of(adata)
    out = f"{OUT}/{name}"
    os.makedirs(f"{out}/plots", exist_ok=True); os.makedirs(f"{out}/data", exist_ok=True)
    print(f"[pack3] {name}: lineages {An} vs {Bn}, basis {basis}", flush=True)

    # ---- pre-perturbation structural scores ----
    tf = sch.tl.score_driver_tfs(adata, A, B, cluster_key=ck)
    drivers = list(tf.reindex(tf[["score_A", "score_B"]].max(1).sort_values(ascending=False).index)
                   .head(N_DRIVERS).index)

    # 1: pre-perturbation lineage bias of the top drivers
    top = tf.loc[drivers].sort_values("lineage_bias")
    fig, ax = plt.subplots(figsize=(6, 4.4))
    colors = ["#2a6f97" if v >= 0 else "#bc4749" for v in top["lineage_bias"]]
    ax.barh(range(len(top)), top["lineage_bias"], color=colors)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=f"structural lineage bias  (+{An} / -{Bn})",
           title=f"{name}: pre-perturbation driver bias")
    fig.tight_layout(); fig.savefig(f"{out}/plots/01_pre_structural_bias.png", dpi=140); plt.close(fig)

    # ---- WT flow (once) + per-driver KO commitment change ----
    sch.tl.calculate_flow(adata, source="original", basis=basis, cluster_key=ck, verbose=False)
    rows, effects = [], {}
    for g in drivers:
        try:
            res, eff, _ = _ko_commitment(adata, adata, g, A, B, basis, ck)
            rows.append(dict(gene=g, pre_A=res["pre_A"], post_A=res["post_A"],
                             delta_A=res["delta_A"], delta_B=res["delta_B"],
                             delta_mean=res["delta_mean"],
                             structural_bias=float(tf.loc[g, "lineage_bias"])))
            effects[g] = eff
            print(f"    {g}: delta_A={res['delta_A']:+.3f} delta_B={res['delta_B']:+.3f}", flush=True)
        except Exception as exc:
            print(f"    {g} skipped: {exc}", flush=True)
    dfk = pd.DataFrame(rows)

    # 2: post-perturbation commitment change per driver (delta_A vs delta_B)
    if len(dfk):
        d = dfk.sort_values("delta_A")
        fig, ax = plt.subplots(figsize=(7, 4.6))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["delta_A"], 0.4, label=f"{An} cells", color="#2a6f97")
        ax.bar(x + 0.2, d["delta_B"], 0.4, label=f"{Bn} cells", color="#e09f3e")
        ax.set_xticks(x); ax.set_xticklabels(d["gene"], rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(ylabel="commitment change (post - pre)",
               title=f"{name}: KO effect on lineage commitment")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/plots/02_post_commitment_change.png", dpi=140); plt.close(fig)

        # 3: pre (structural) vs post (commitment) -- do structural drivers move fate?
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.scatter(dfk["structural_bias"], dfk["delta_mean"], s=60, c="#5f0f40", zorder=3)
        for _, r in dfk.iterrows():
            ax.annotate(r["gene"], (r["structural_bias"], r["delta_mean"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set(xlabel="structural lineage bias (pre)", ylabel="mean commitment change (post)",
               title=f"{name}: structural prior vs KO effect")
        fig.tight_layout(); fig.savefig(f"{out}/plots/03_pre_vs_post.png", dpi=140); plt.close(fig)

    # 4: fractional dose-response for the strongest driver
    g0 = drivers[0]
    try:
        dr = sch.dyn.run_dose_response(
            adata, g0, lineage_A_clusters=A, lineage_B_clusters=B, basis=basis,
            wt_flow_key=f"original_velocity_flow_{basis}",
            fractions=[0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], cluster_key=ck, verbose=False)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(dr["level_frac"], dr["lineage_bias"], "-o", color="#0f4c5c")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(1.0, color="grey", ls="--", lw=0.8)
        ax.set(xlabel=f"{g0} dose (fraction of natural max)", ylabel=f"lineage bias (+{An}/-{Bn})",
               title=f"{name}: {g0} dose-response")
        fig.tight_layout(); fig.savefig(f"{out}/plots/04_dose_response.png", dpi=140); plt.close(fig)
        dr.to_csv(f"{out}/data/dose_response_{g0}.csv", index=False)
    except Exception as exc:
        print(f"    dose-response skipped: {exc}", flush=True)

    # 5: per-cell-type KO impact for the strongest driver
    if g0 in effects:
        eff = effects[g0].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.bar(range(len(eff)), eff.values, color="#c0605a")
        ax.set_xticks(range(len(eff))); ax.set_xticklabels(eff.index, rotation=40, ha="right", fontsize=7)
        ax.set(ylabel="mean |delta_X| after KO", title=f"{name}: {g0} KO impact per cell type")
        fig.tight_layout(); fig.savefig(f"{out}/plots/05_ko_impact_{g0}.png", dpi=140); plt.close(fig)

    # 6: commitment-change landscape over the embedding (strongest driver)
    try:
        res, _, ko = _ko_commitment(adata, adata, g0, A, B, basis, ck)
        emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
        dc = res["delta_commitment"]
        n = min(len(dc), emb.shape[0]); emb = emb[:n]; dc = dc[:n]
        fig, ax = plt.subplots(figsize=(5.8, 5))
        lim = np.nanpercentile(np.abs(dc), 95) or 1.0
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=dc, s=7, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set(xticks=[], yticks=[], title=f"{name}: {g0} KO commitment shift  (+{An}/-{Bn})")
        fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(f"{out}/plots/06_commitment_landscape_{g0}.png", dpi=140); plt.close(fig)
    except Exception as exc:
        print(f"    landscape skipped: {exc}", flush=True)

    if len(dfk):
        dfk.to_csv(f"{out}/data/ko_commitment_{name}.csv", index=False)
    tf.loc[drivers].to_csv(f"{out}/data/driver_scores_{name}.csv")
    return dict(name=name, drivers=drivers, top_driver=g0)


GUIDE = """# Figure pack 3: perturbation analysis

In-silico knockouts on the fitted GRNs, for datasets with a defined A-vs-B lineage
choice. Regenerated by `analyses/figure_packs/pack3_perturbation.py`. Targets paper
section R3 (perturbation phenotypes) and Fig 4.

**Scores used (the improved definitions).** The *pre-perturbation* score is
`score_driver_tfs` -- a structural, standardized composite of W-norm / out-degree /
energy-correlation. The *post-perturbation* score is
`compute_perturbation_commitment_change`: it builds a data-derived lineage axis
(centroid(A) - centroid(B) in the embedding) and measures the per-cell cosine alignment
of the developmental flow with that axis, **before** (WT) and **after** (KO), returning
the difference. Both terms are cosines against the same fixed axis, so the pre/post
comparison is normalized and directly interpretable (unlike the older `score_A/score_B`
fields that shared names but different definitions).

Datasets: {datasets}

Per dataset (`<dataset>/plots/`):
1. `01_pre_structural_bias` -- structural lineage bias of the top drivers (pre).
2. `02_post_commitment_change` -- KO effect on per-lineage commitment (post - pre).
3. `03_pre_vs_post` -- structural prior vs actual KO commitment shift (do the
   structurally biased genes actually move fate when knocked out?).
4. `04_dose_response` -- fractional dose-response of the strongest driver (KO -> OE).
5. `05_ko_impact_*` -- per-cell-type magnitude of the strongest-driver KO.
6. `06_commitment_landscape_*` -- per-cell commitment shift over the embedding.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = C.available()
    done = []
    for name, lin in C.LINEAGES.items():
        if name in avail:
            try:
                done.append(run_dataset(name, avail[name], lin)["name"])
            except Exception as exc:
                import traceback; print(f"FAILED {name}: {exc}"); traceback.print_exc()
    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE.replace("{datasets}", ", ".join(done)))
    print(f"wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
