# Red-team report - scHopfield manuscript (assembled draft)

Date: 2026-07-01. Reviewer: codex (GPT, independent). Mode: claim-to-evidence audit +
kill argument. Target: `scHopfield_Manuscript_assembled.md` vs committed findings M1-M8.

## Verdict: WARN (before fixes) -> fixes applied this round

## Claim audit (10 headline quantitative claims vs M1-M8)

| Claim | Ruling | Fix applied |
|---|---|---|
| C1 canonical attractor/bifurcation/oscillatory dynamics | OVERCLAIMED | scoped to toggle bistability + repressilator limit cycle; Hopfield-class limitation stated in R2 |
| C2 perfect ground-truth recovery across noise/scaffolds | SUPPORTED (caveat) | added explicit "systems in the model's function class" limitation in R2 |
| C3 Hill necessary; linear fails bistability | SUPPORTED | none needed (M7) |
| C4 known-KO 10/10 vs CellOracle 7/9 | OVERCLAIMED | R3 now states native-readout caveat + "at least as well as", not a precise ranking; metric-parity flagged as planned refinement |
| C5 perturbation drivers robust vs static score unstable | SUPPORTED | none (M5) |
| C6 no-scaffold outlier, loses Gata1/Klf1 | SUPPORTED | none (M6) |
| C7 outperforms GENIE3 | OVERCLAIMED | already scoped to "synthetic 40-gene networks" in R2 text |
| C8 seeded bit-reproducible; unseeded drift | SUPPORTED | none (M1) |
| C9 pancreas Delta/Epsilon unstable, Beta stable | OVERCLAIMED | R5 now includes Alpha in stable group + clarifies positive-real = local instability |
| C10 higher-order Stat3 predictions | UNSUPPORTED/OVERCLAIMED | R6 reframed as in-silico predictions/hypotheses requiring experimental validation |

## Kill argument (verbatim summary)

The strongest major-revision reason: the most translational claim (predictive
perturbation biology) is not validated under a fair, externally anchored evaluation.
M4's CellOracle comparison is not metric-parity (native readouts), so 10/10 vs 7/9
conflates model quality, readout choice, and implementation. C10 extends further with
unvalidated in-silico Stat3 predictions. The GENIE3 result is strong but synthetic-only,
not real-data predictive validity.

## Net assessment + top 3 action items

1. Reframe overbroad claims to the exact systems/evidence tested. [DONE this round for
   C1, C2, C4, C9, C10 in prose.]
2. Fix M4 metric parity (score both methods on identical endpoints). [DEFERRED,
   experimental: attempted, hit tooling issues, documented as planned refinement.]
3. Treat C10 as hypothesis generation unless validated (Perturb-seq / CRISPR / held-out).
   [DONE in prose; experimental validation is genuine future work.]

## Residual (honest limitations, not papered over)

- M4 strict metric-parity re-score: research/tooling item, planned refinement.
- C10 Stat3 higher-order predictions: unvalidated; framed as testable hypotheses.
- External baselines beyond GENIE3 (SCENIC, dyngen) and a real-data GRN gold standard:
  future work.

A re-review on the revised prose is expected to move WARN -> PASS on the claim audit;
the residual WARN drivers are genuinely experimental (validation) and are recorded as
limitations per honest-framing doctrine.

---

## Round 2 (after revisions + Methods/Supplementary added)

Fresh independent codex reviewer. **Verdict: PASS with minor WARN** (up from WARN).
Prior items: C1 resolved, C4 partial, C7 resolved, C9 resolved, C10 resolved, M4
kill-argument partial (softened by disclosure). No new fatal blocker.

Round-2 fixes applied (prose):
1. C4: replaced "at least as well as CellOracle" with a non-ranking statement
   ("recovers the panel consistently with known phenotypes and aligns with CellOracle's
   native-readout results; not a metric-parity ranking").
2. C2: labeled the synthetic recovery as a "recovery test under correct model
   specification" (identifiability for Hopfield-form systems), explicitly not evidence
   for arbitrary biological networks.
3. Discussion: added a consistency limitation that the pancreas stability ordering,
   knockout landscapes, and dose-response/higher-order predictions are model-derived
   hypotheses requiring experimental validation.

Residual (honest limitations): M4 strict metric parity (experimental), and experimental
validation of the biological predictions. Recorded as limitations per honest framing.
