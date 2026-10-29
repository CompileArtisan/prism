# RadioMapSeer study: do cross-map-trained graph models beat side-information regression-kriging?

**Status: PRE-FREEZE (draft). Nothing here is evidence until `results_rms/prereg_rms.json` is written, its sha256 is committed to git, and the confirmatory run has been done once.**

Dev-stage numbers in this repo (validation maps only) are exploratory. They are not results.

---

## 1. Question

Does a graph model trained **across** training maps beat `rk_side` (ridge on label-free map features + IDW of the residual) on **held-out** building maps, at 0.5% / 2% / 5% label density?

Earlier rounds (city study, dense indoor run, Rounds 7-8) trained each GNN per map from a few hundred labels. Every one lost to IDW or `rk_side`. Published GNN radio-map papers (Tania and Kim, Asilomar 2025; Bufort et al., arXiv 2501.06236) train across many maps and compare mostly against other neural or tabular models, not against IDW or regression-kriging on the same label-free features. This study builds that contrast with a Kim-style graph.

**Pre-declared positive GNN result.** `gnn_kim` or `rgnn_xmap` beats `rk_side` in the `primary` family with:

- `p_holm < 0.05`,
- relative RMSE gain >= 2%,
- bootstrap CI above 0,

at **at least 2 of the 3 densities**, and is not significantly worse at the third. If this is not met, the statement is "no", reported with the same weight as a yes.

What this study cannot fix: RadioMapSeer is simulated (no real-data evidence), it is 5.9 GHz (not 6G), and the forking paths in section 9.

---

## 2. Data

- **RadioMapSeer, DPM simulations.** 701 city maps x 80 transmitters, 256x256 px at 1 m, 5.9 GHz, building height 25 m, Tx/Rx height 1.5 m. Six cities (Ankara, Berlin, Glasgow, Ljubljana, London, Tel Aviv).
- **Split by building map, never by transmitter.** Standard RadioUNet-style split (550 train / 50 val / 100 test, ordering taken from the RadioUNet loader). The confirmatory run uses **99 test maps**.
- **Training pool.** 8 transmitters per training map (seeded draw).
- **Unit of inference = test building map.** Average over its test transmitters and label draws.
- **Sensitivity data.** High-fidelity IRT subset (train on DPM, evaluate on IRT), declared in advance; availability to be confirmed.
- **Optional replication.** BRATLab Radiomap-Data, only if it can be split by area without overlap.

## 3. Task

- **Labels.** Fraction alpha of outdoor (non-building) pixels sampled uniformly at random; alpha in {0.5%, 2%, 5%}; 5 independent label draws per (map, transmitter).
- **Test points.** All remaining outdoor pixels (random split, primary). Blocked split is a secondary analysis.
- **Target.** Pathloss / RSS in dB. All arms predict the same quantity after the same per-map trend, fitted on that map's labels only.
- **Label-free features** (building raster + Tx position only): log distance to Tx, LOS flag, blocked length along the Tx-Rx segment, number of building crossings, building cover within 30 m, distance to nearest building.

## 4. Arms

**Classical (fit per map)**

| Arm | Description |
|---|---|
| `idw` | Inverse-distance weighting |
| `krig_global` | Local K-nearest kriging (5% for cost) |
| `rk_side` | Ridge on the six features + IDW of the residual. **The baseline that matters.** |
| `rk_gbm` | Gradient-boosting variant |

**Learned across maps** (alpha drawn log-uniformly in [0.3%, 6%] during training)

| Arm | Description |
|---|---|
| `gnn_kim` | GATv2, 3 layers, heads (4,4,4), hidden (64,64,40), 64x64 tiles. Edge iff distance <= 3*sqrt(2) cells AND no building between AND same LOS/NLOS state. Gaussian edge weights. Input = normalised sparse labels (-1 if missing) + features. MSE loss. |
| `gnn_kim_oldedge` | Same, without the LOS/NLOS-state edge condition (topology ablation) |
| `rgnn_xmap` | `rk_side` prediction + correction from the `gnn_kim` graph model, trained on `rk_side` residuals across training maps |
| `radiounet` | Public RadioUNet first U, same training pool, alpha sampling and loss |
| `cnn_feat` | `radiounet` with the six features as extra input channels (10 in total) |

**Tuning budget (equal for all).** 30 random configurations per learned arm on the val maps; matching grid for classical arms. Test maps are never used for tuning, early stopping or normalisation. Report parameter count and train/inference time for every learned arm.

**Seeds.** 3 training seeds per learned arm; fall back to 1 seed for all learned arms only if declared before the freeze.

## 5. Metrics

- **Primary:** RMSE (dB) on held-out outdoor pixels. Relative gain of A vs B, positive = A better, always shown next to dB values.
- **Coverage:** `bacc_q25` (threshold from that map's training labels).
- **Secondary (descriptive):** MAE; RMSE on LOS vs NLOS pixels; RMSE within 10 m of a building edge.

## 6. Statistics and families

Paired across test maps. Exact sign-flip p, bootstrap CI over maps, verdict labels (`A better` / `A worse` / `equivalent` within +/-2% / `inconclusive`), **Holm within each family**. City-cluster bootstrap (6 clusters) as a descriptive sensitivity analysis.

| Family | Contrasts (each at 0.5 / 2 / 5%) | Tests | Status |
|---|---|---|---|
| `primary` | `gnn_kim` vs `rk_side`; `rgnn_xmap` vs `rk_side` | 6 | Inferential. Decides the positive-GNN rule |
| `vs_classical` | `gnn_kim` vs `idw`; `rgnn_xmap` vs `idw` | 6 | Inferential |
| `feat` | `rk_side` vs `idw` | 3 | Inferential |
| `topo` | `gnn_kim` vs `gnn_kim_oldedge` | 3 | Inferential |
| `deep` | `gnn_kim`, `rgnn_xmap` vs `radiounet` | 6 | **Descriptive** |
| `deep_feat` | `gnn_kim`, `rgnn_xmap` vs `cnn_feat` | 6 | Inferential, two-sided, 5% non-inferiority margin (see section 7) |

**Total: 30 tests in 6 families.** Every family can reach `p_holm < 0.05` with 99 maps (min `p_holm` about 1.5e-5 to 3.0e-5). Anything not in this table is exploratory and labelled so.

## 7. Pre-specified decision rules

Written before the runs they govern.

- **Inference stride (all GNN arms).** Adopt stride 32 only if `rgnn_xmap` at stride 32 beats stride 64 by more than 1% mean relative RMSE with bootstrap CI above 0 at every density on val maps. Otherwise keep stride 64. **Outcome: 0.27 / 0.54 / 0.44% gain, so stride 64 is kept.**
- **Status of `deep_feat`.** It stays inferential only if the re-tuned `cnn_feat` beats `idw` on val maps at every density (3.606 / 2.655 / 2.267 dB). **Outcome: it does, so `deep_feat` stays inferential.** Stated before the confirmatory run: on dev numbers the GNNs are 10-16% worse than `cnn_feat`, so `deep_feat` is expected to show them significantly worse.
- **No other changes.** Thresholds, densities, test maps, label draws, the 2% equivalence margin and the 5% non-inferiority margin are fixed.

## 8. Leak and sanity checks (must pass before the test run)

1. No test map or test transmitter in training, tuning or normalisation data (assert on map ids).
2. Features use no power value, only the building raster and Tx position.
3. Per-map trend, ridge and IDW parameters are fit on that map's labels only.
4. Shuffled-label control: `gnn_kim` trained on permuted labels must test at least as badly as `idw`. (Read the caveat on this check in the notebook first.)
5. Zero-label control: a model given no labels must not beat `rk_side` with n=0 labels by more than the features alone give.
6. Edge-rule unit test on a toy map: no edge crosses a building and no edge joins a LOS and an NLOS node.
7. See notebook section 8 (the addendum refers to checks 1-7; the plan draft listed 1-6).

Also: unit test (notebook section 4 (e)) asserting that a freshly built `radiounet` / `cnn_feat` can output negative values.

## 9. Disclosure of forking paths

- The plan was designed **after** seeing the negative city and dense-indoor results, the Round-7/8 results (including the unconfirmed `rattn` lead), and after reading Kim et al. and Bufort et al. Design choices that came from that: cross-map training, the Kim edge rule, the residual-on-`rk_side` arm.
- `rattn` is **not** included. If it is to be confirmed, it gets its own one-contrast plan on fresh data.
- `cnn_feat` and `deep_feat` were added on 2026-10-26 (decision B), after dev results.
- **Defect found 2026-10-27 (before the freeze).** The public `RadioWNet` ends with `convrelu(20+inputs, 1, 5, 2, 1)`, i.e. Conv -> ReLU, so its output is never negative. It cannot represent the signed residual used here. The earlier dev check searched only for `sigmoid`, `tanh` and `clamp` and missed it, so the README then wrongly said "no output squashing". Synthetic evidence for the mechanism: validation MSE 0.096 with the final ReLU vs 0.005 without (`relu_synth_test.py`).
  - Fix: `PublicRadioUNet` removes that one ReLU, nothing else.
  - All earlier dev numbers for `radiounet` / `cnn_feat` are **void** (old trial files archived as `tune_<arm>_trials_preReluFix.csv`).
  - Both arms were re-tuned with the same protocol. GNN and classical arms are unaffected.
  - The fix and re-tuning used val maps only. No test map has been read.
  - The module sha256 changed (`0957f09b13a0c54e` was the pre-fix hash); the frozen plan stores the final hash.
- The supplement table of rounds lists, for each round, the date, what had been seen before designing it, and the number of contrasts.

## 10. Dev-stage outcomes so far (val maps only; not evidence)

| Item | Result |
|---|---|
| Fix check, default hp, 1 500 steps | `cnn_feat` 5.585 -> 2.972 dB; `radiounet` 5.596 -> 2.950 dB. Overfit test on 4 pairs: 1.776 dB vs trend-only 6.747 dB (ratio 0.26, passes) |
| Re-tuned `radiounet` (trial 20) | 3.007 / 2.251 / 2.014 dB at 0.5 / 2 / 5% labels |
| Re-tuned `cnn_feat` (trial 1) | 2.184 / 1.818 / 1.700 dB |
| `rk_side` reference | 2.854 / 2.352 / 2.123 dB |
| `idw` reference | 3.606 / 2.655 / 2.267 dB |
| `rgnn_xmap`, 8 000 steps, 1 seed, 100 val maps, 2 draws | 2.594 / 2.084 / 1.887 dB vs `rk_side` 2.848 / 2.354 / 2.122; gains 8.7 / 11.1 / 10.6% (CI over maps 8.0-9.3 / 10.4-11.7 / 9.9-11.3; better on 99% of maps) |

Caveats: single 1 500-step runs are noisy (same default `cnn_feat` config gave 2.972 dB in the fix check and 3.386 dB as tuning trial 0). The `rgnn_xmap` numbers are optimistic (val maps were tuned on, one seed). The positive rule would be met on these numbers; this is not evidence.

---

## 11. Repository layout

| Path | Purpose |
|---|---|
| `PLAN_radiomapseer_DRAFT.md` | Study plan (merge the addendum in before the freeze; the notebook hashes this file at the freeze) |
| `rmbench_rms.py` | Benchmark code (committed with the freeze) |
| `rmbench_rms.ipynb` | Notebook driving dev, checks, freeze and confirmatory run |
| `relu_synth_test.py` | Synthetic CPU test of the final-ReLU mechanism |
| `results_rms/` | All outputs |
| `results_rms/relu_fix_status.json` | Records that both CNN arms were re-tuned; the freeze cell refuses to run without it |
| `results_rms/tune_<arm>_trials*.csv` | Tuning trials (`..._preReluFix.csv` = void, archived) |
| `results_rms/prereg_rms.json` | Frozen config (+ `.sha256`). Runner refuses to start without it and refuses to overwrite it |

## 12. Procedure and notebook flags

Run in this order. Set the flag, run its cell.

1. `RUN_CNN_FIXCHECK` - default-hp CNN arms after the fix + tiny-pool overfit test (about 10 min on a T4). **Done.**
2. `RETUNE_CNN_ARMS` - re-tune `radiounet` and `cnn_feat` (about 2 h per arm). **Done.**
3. `RUN_DEV_FINALCHECK` - power check of `rgnn_xmap` at final budget + stride decision (about 1.5 h). **Done.**
4. `RUN_CHECKS` - leak and sanity checks (section 8). **To do.**
5. Decide seeds (3, or 1 to save about 3.5 h) and confirm section 13. **To do.**
6. Merge the addendum into the plan file. **To do.**
7. `FREEZE` and `I_CONFIRM_SECTION_11` - write `prereg_rms.json` and its sha256; commit `prereg_rms.json`, `.sha256` and `rmbench_rms.py` to git **before** the test run.
8. `RUN_CONFIRM` - one run on the 99 test maps (about 27 GPU-hours). No arm, threshold or family may change afterwards.
9. Report all 30 contrasts, wins and losses, with dB values, CIs and Holm p. Sensitivity analyses (IRT subset, blocked split, city-cluster bootstrap) after the primary tables.

## 13. Decisions to confirm before freezing

- Positive-result threshold (2 of 3 densities, >= 2% gain) and the 5% non-inferiority margin.
- 8 transmitters per training map; 3 training seeds vs 1.
- Whether to include the BRATLab replication dataset.
- Whether the IRT subset is available in the form assumed.

## 14. Literature notes

Numbers from these papers are **not** comparable with ours (different data, units, label densities).

- **Tania and Kim (Asilomar 2025).** GATv2 on 41x41 blocks, edge rule as in `gnn_kim`, BRATLab/Feko data, 5-30% sampling. Compared only against an older edge rule and an SICNN; no IDW or kriging. Does not state whether same-region blocks are kept apart between train and test.
- **Bufort, Lebocq, Cathabard (arXiv 2501.06236).** Real 4G data, about 20,000 sites, 10 message-passing blocks. Test RMSE 9.8 dB (ray edges) vs 10.5 dB (without) vs 10.2 dB (GBM): a gain of about 0.3-0.7 dB (3-7%), untested for significance. This is the realistic size of a structural GNN advantage.

# RadioMapSeer study: addendum to `PLAN_radiomapseer_DRAFT.md` (2026-10-27)

Status: **draft addendum, not frozen.** Merge into the plan file before the freeze (the notebook hashes the plan file at the freeze). Items marked *proposed* are defaults I suggest; the author decides.

## 1. Arms and families (changes since the draft)

| Change | Detail |
|---|---|
| New arm `cnn_feat` | The public RadioUNet first U with the six label-free features as 6 extra input channels (10 in total). Added 2026-10-26 (decision B). |
| New family `deep_feat` | `gnn_kim`, `rgnn_xmap` vs `cnn_feat`; two-sided; 5% non-inferiority margin; 6 tests. |
| Family `deep` | Kept, **descriptive** (decision B). |
| Test count | 30 tests in 6 families (the draft had 24 in 5). Every family can reach `p_holm < 0.05` with 99 maps (min `p_holm` 1.5e-5 to 3.0e-5). |
| Positive-GNN rule | **Unchanged.** `gnn_kim` or `rgnn_xmap` beats `rk_side` with `p_holm < 0.05`, relative gain >= 2%, bootstrap CI above 0, at >= 2 of the 3 densities, and is not significantly worse at the third. It depends on the `primary` family only. |

## 2. Defect found and fixed before the freeze (2026-10-27)

- **Defect.** The public `RadioWNet` (`lib/modules.py`) ends with `self.conv_up000 = convrelu(20+inputs, 1, 5, 2, 1)`, i.e. Conv -> **ReLU** -> MaxPool(1). The output of the first U is therefore never negative. The network was designed for non-negative gain images. This study trains it on a **signed** residual (observed minus the per-map trend, mean about 0), which it cannot represent: it can output 0 where the target is negative, so a large part of the residual variance is unreachable.
- **Why it was missed.** The 2026-10-26 dev check searched `modules.py` for `sigmoid`, `tanh` and `clamp`. The squashing is an `nn.ReLU` inside `convrelu`, so the search found nothing and the README stated "no output squashing". That statement was wrong.
- **Evidence for the mechanism (synthetic, CPU; `relu_synth_test.py`).** Signed pointwise target on 64x64 maps (55% of target pixels negative, variance 0.174), 300 steps: validation MSE 0.096 with the final ReLU (lowest output exactly 0) against 0.005 without it (lowest output -1.36). This shows the mechanism only. It says nothing yet about RadioMapSeer.
- **Fix.** `PublicRadioUNet` removes that one ReLU and keeps everything else of the public network. A unit test (notebook section 4 (e)) asserts that a freshly built `radiounet` / `cnn_feat` can output negative values.
- **Consequences.**
  1. The dev-stage numbers of `radiounet` and `cnn_feat` (tuning, the 20 000-step check, the 5.4-5.6 dB plateau) came from the defective network and are **void**. The old trial files are archived as `tune_<arm>_trials_preReluFix.csv`.
  2. `radiounet` and `cnn_feat` must be re-tuned with the same protocol (30 configurations, 1 500 steps, val maps). The freeze cell refuses to run until `results_rms/relu_fix_status.json` says both are re-tuned.
  3. The GNN arms and the classical arms do not use this network. Their tuned hyper-parameters and dev numbers are unaffected.
  4. The module sha256 changes (`0957f09b13a0c54e` was the pre-fix hash). The frozen plan will store the final hash.
  5. Forking-path disclosure: the defect was found after seeing val results; the fix and the re-tuning use val maps only; no test map has been read.

## 3. Work order before the freeze (notebook flags)

1. `RUN_CNN_FIXCHECK = True` (about 10 min on a T4): default-hp `cnn_feat` and `radiounet` after the fix, plus a tiny-pool overfit test. Pre-fix trial-0 reference: 5.585 / 5.596 dB.
2. `RETUNE_CNN_ARMS = True` (about 2 h per arm): re-tune both CNN arms.
3. `RUN_DEV_FINALCHECK = True` (about 1.5 h): exploratory power check of `rgnn_xmap` at the final training budget, and the stride decision (section 4 below).
4. `RUN_CHECKS = True` (notebook section 8): leak and sanity checks 1-7 must pass. Read the caveat on check 4 first.
5. Decide the number of training seeds (3, or 1 to save about 3.5 h) and confirm the proposals below.
6. Update this plan file, then `FREEZE` and `I_CONFIRM_SECTION_11`; commit `prereg_rms.json`, its `.sha256` and `rmbench_rms.py` to git; only then `RUN_CONFIRM`.

## 4. Pre-specified decision rules (written before the runs they govern)

- **Inference stride (applies to all GNN arms).** Adopt stride 32 (overlapping tiles, about 4x the GNN test pass) only if `rgnn_xmap` at stride 32 beats `rgnn_xmap` at stride 64 by more than 1% mean relative RMSE with the bootstrap CI above 0 at every density on the val maps; otherwise keep stride 64. Stride 32 on 40 val maps, 2 label draws.
- **Status of `deep_feat`** (*proposed*). After the re-tuning, `deep_feat` stays an inferential non-inferiority family only if the re-tuned `cnn_feat` is a working baseline on the val maps, defined as: its val RMSE is below the `idw` val RMSE at every density (3.606 / 2.655 / 2.267 dB). Otherwise `deep_feat` is reported as descriptive, like `deep`, because a non-inferiority test against a baseline that does not use the labels is nearly guaranteed to pass. Decide and record this before the freeze; do not change it after the confirmatory run.
- **No other changes.** Thresholds, densities, test maps, label draws, the 2% equivalence margin and the 5% non-inferiority margin are as in the draft.

## 5. What is still not done

Leak and sanity checks (section 8), the seed decision, the freeze, the confirmatory run on the 99 test maps (about 27 GPU-hours), the sensitivity analyses. None of the dev-stage numbers is evidence. If the pre-declared rule is not met, the statement is "no".

## 6. Outcomes of the pre-freeze work (2026-10-28, val maps only)

| Item | Result |
|---|---|
| Fix check (default hp, 1 500 steps) | `cnn_feat` 5.585 -> 2.972 dB, `radiounet` 5.596 -> 2.950 dB. Overfit test on 4 training pairs: 1.776 dB vs trend-only 6.747 dB (ratio 0.26, passes) |
| Re-tuning (30 configurations each) | `radiounet` best trial 20, val RMSE 3.007 / 2.251 / 2.014 dB (0.5 / 2 / 5% labels); `cnn_feat` best trial 1, 2.184 / 1.818 / 1.700 dB. `rk_side` for reference: 2.854 / 2.352 / 2.123 dB. Single 1 500-step runs are noisy (same default `cnn_feat` configuration: 2.972 dB in the fix check, 3.386 dB as tuning trial 0) |
| Status of `deep_feat` (rule of section 4) | `cnn_feat` beats `idw` (3.606 / 2.655 / 2.267 dB) at every density, so the rule keeps `deep_feat` as a two-sided non-inferiority family. **Stated before the confirmatory run:** on the dev numbers the GNNs are 10-16% worse than `cnn_feat`, so `deep_feat` is expected to show them significantly worse. |
| Stride rule (section 4) | `rgnn_xmap` at stride 32 vs 64: 0.27 / 0.54 / 0.44% gain on 40 val maps, below the 1% threshold. **Keep stride 64.** |
| Dev power check, `rgnn_xmap` at 8 000 steps (1 seed, 100 val maps, 2 label draws) | 2.594 / 2.084 / 1.887 dB vs `rk_side` 2.848 / 2.354 / 2.122: gains 8.7 / 11.1 / 10.6% (CI over maps 8.0-9.3 / 10.4-11.7 / 9.9-11.3; better on 99% of maps). Optimistic: val maps were tuned on, one seed. The plan's positive rule would be met on these numbers; this is not evidence |

| Leak and sanity checks (section 8, 2026-10-29) | All seven gates passed. Shuffled-label `gnn_kim` 5.490 / 5.094 / 4.766 dB vs `idw` 3.606 / 2.655 / 2.267 (literal rule passes). Zero-label `gnn_kim` 3.676 dB vs feature-only GBM 4.040 dB; the 0.9 x 4.040 = 3.636 dB threshold is passed by 0.04 dB (narrow). Controls 4 and 5 cover `gnn_kim` only |

Still to do before the freeze: the seed decision (recommendation: keep 3, the default), merging this addendum into `PLAN_radiomapseer_DRAFT.md` and placing that file in the working directory (the freeze cell now warns if it is missing), then `FREEZE = True` and `I_CONFIRM_SECTION_11 = True`, commit `prereg_rms.json`, its `.sha256` and `rmbench_rms.py`, and only then `RUN_CONFIRM = True`.
