"""Figure pack 4: double knockouts and epistasis.

For datasets with a defined A-vs-B lineage choice: single- and double-KO lineage-bias
screens over the top drivers, and the epistasis (synergy / cancellation) between pairs
relative to the additive expectation. Uses the fitted GRNs (ODE knockouts). CPU.

    figure_packs/pack4_double_ko/<dataset>/{plots,data}/ + FIGURE_GUIDE.md

Run:  PYTHONPATH=analyses/figure_packs .venv/bin/python analyses/figure_packs/pack4_double_ko.py
"""
import itertools
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

OUT = "figure_packs/pack4_double_ko"
N_DRIVERS = 5


def run_dataset(name, ck, lin):
    adata = C.load(name)
    A, B = lin["A"], lin["B"]
    basis = C.basis_of(adata)
    order = C.present_clusters(adata, ck)
    out = f"{OUT}/{name}"
    os.makedirs(f"{out}/plots", exist_ok=True); os.makedirs(f"{out}/data", exist_ok=True)
    print(f"[pack4] {name}: {lin['A_name']} vs {lin['B_name']}", flush=True)

    tf = sch.tl.score_driver_tfs(adata, A, B, cluster_key=ck)
    drivers = list(tf.reindex(tf[["score_A", "score_B"]].max(1).sort_values(ascending=False).index)
                   .head(N_DRIVERS).index)
    A_genes = [g for g in drivers if tf.loc[g, "lineage_bias"] > 0]
    B_genes = [g for g in drivers if tf.loc[g, "lineage_bias"] <= 0]

    sch.tl.calculate_flow(adata, source="original", basis=basis, cluster_key=ck, verbose=False)
    wtk = f"original_velocity_flow_{basis}"

    single_bias, _ = sch.dyn.run_ko_screen(
        adata, drivers, A, B, basis, wtk, cluster_key=ck, cluster_order=order, verbose=False)
    pairs = list(itertools.combinations(drivers, 2))
    pair_bias, _ = sch.dyn.run_pairwise_ko_screen(
        adata, pairs, A, B, basis, wtk, cluster_key=ck, cluster_order=order, verbose=False)
    epi = sch.dyn.compute_epistasis(pair_bias, single_bias,
                                    lineage_A_genes=A_genes, lineage_B_genes=B_genes)

    # 1: double-KO lineage-bias matrix (symmetric grid over drivers)
    n = len(drivers); idx = {g: i for i, g in enumerate(drivers)}
    Mbias = np.full((n, n), np.nan)
    for g, d in single_bias.items():
        Mbias[idx[g], idx[g]] = d["lineage_bias"]
    for (a, b), d in pair_bias.items():
        Mbias[idx[a], idx[b]] = d["lineage_bias"]; Mbias[idx[b], idx[a]] = d["lineage_bias"]
    lim = np.nanmax(np.abs(Mbias))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(Mbias, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n)); ax.set_xticklabels(drivers, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(drivers, fontsize=8)
    ax.set_title(f"{name}: single (diag) + double KO lineage bias")
    fig.colorbar(im, ax=ax, fraction=0.046, label="lineage bias")
    fig.tight_layout(); fig.savefig(f"{out}/plots/01_double_ko_bias_matrix.png", dpi=140); plt.close(fig)

    # 2: synergy (epistasis) matrix
    Msyn = np.full((n, n), np.nan)
    for _, r in epi.iterrows():
        i, j = idx[r["geneA"]], idx[r["geneB"]]
        Msyn[i, j] = r["synergy_score"]; Msyn[j, i] = r["synergy_score"]
    lim = np.nanmax(np.abs(Msyn)) or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(Msyn, cmap="PRGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n)); ax.set_xticklabels(drivers, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(drivers, fontsize=8)
    ax.set_title(f"{name}: epistasis (synergy score)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="synergy (>0 amplifies)")
    fig.tight_layout(); fig.savefig(f"{out}/plots/02_epistasis_matrix.png", dpi=140); plt.close(fig)

    # 3: expected (additive) vs actual double-KO bias
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(epi["expected_bias"], epi["lineage_bias"], s=55, c=epi["synergy_score"],
               cmap="PRGn", vmin=-lim, vmax=lim, zorder=3, edgecolor="k", linewidth=0.4)
    mn = float(np.nanmin([epi["expected_bias"].min(), epi["lineage_bias"].min()]))
    mx = float(np.nanmax([epi["expected_bias"].max(), epi["lineage_bias"].max()]))
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="additive")
    for _, r in epi.iterrows():
        ax.annotate(f"{r['geneA']}+{r['geneB']}", (r["expected_bias"], r["lineage_bias"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="expected bias (single_A + single_B)", ylabel="actual double-KO bias",
           title=f"{name}: epistasis vs additive expectation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/plots/03_expected_vs_actual.png", dpi=140); plt.close(fig)

    # 4: most synergistic / antagonistic pairs
    e = epi.reindex(epi["synergy_score"].sort_values().index)
    labels = [f"{r.geneA}+{r.geneB}" for r in e.itertuples()]
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(e) + 1.2))
    ax.barh(range(len(e)), e["synergy_score"],
            color=["#762a83" if v < 0 else "#1b7837" for v in e["synergy_score"]])
    ax.set_yticks(range(len(e))); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="synergy score", title=f"{name}: pair epistasis (green=synergy, purple=antagonism)")
    fig.tight_layout(); fig.savefig(f"{out}/plots/04_pair_ranking.png", dpi=140); plt.close(fig)

    epi.to_csv(f"{out}/data/epistasis_{name}.csv")
    json.dump({"drivers": drivers, "A_genes": A_genes, "B_genes": B_genes,
               "n_pairs": len(pairs)}, open(f"{out}/data/pack4_{name}.json", "w"), indent=2)
    return name


GUIDE = """# Figure pack 4: double knockouts and epistasis

Single- and double-gene in-silico knockouts over the top lineage drivers, and their
epistasis relative to the additive (Bliss-style) expectation on lineage bias.
Regenerated by `analyses/figure_packs/pack4_double_ko.py`. Targets paper section R3 /
Fig 4 (perturbation phenotypes), extending single KOs to genetic interactions.

Datasets: {datasets}

Per dataset (`<dataset>/plots/`):
1. `01_double_ko_bias_matrix` -- lineage bias of every single KO (diagonal) and double
   KO (off-diagonal).
2. `02_epistasis_matrix` -- synergy score per pair (>0 amplifies the bias, <0 cancels).
3. `03_expected_vs_actual` -- actual double-KO bias vs the additive expectation; points
   off the diagonal are epistatic.
4. `04_pair_ranking` -- the most synergistic and most antagonistic pairs.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = C.available()
    done = []
    for name, lin in C.LINEAGES.items():
        if name in avail:
            try:
                done.append(run_dataset(name, avail[name], lin))
            except Exception as exc:
                import traceback; print(f"FAILED {name}: {exc}"); traceback.print_exc()
    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE.replace("{datasets}", ", ".join(done)))
    print(f"wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
