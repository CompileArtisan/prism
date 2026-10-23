# RadioMapSeer study: do cross-map-trained graph models beat side-information regression-kriging?

**Status: DRAFT. Nothing in this file is frozen until the hash step in section 10 is run.**
Items marked **[CHOOSE]** are defaults I picked; change them *before* freezing. Items marked **[VERIFY]** are facts about the dataset or code that must be checked against the downloaded files, because I have not inspected them.

## 0. Why this study exists (and what it can and cannot show)

The city study, the dense indoor run and Rounds 7-8 all trained each GNN **per map**, from at most a few hundred labels, for 150 steps. Every one of them lost to IDW or to `rk_side`. The published GNN radio-map papers train **across many maps** with dense ground truth and compare mostly against other neural models. This study tests that regime directly, against the classical baseline that matters (`rk_side`), on a public benchmark that RadioUNet-type models are also evaluated on.

It cannot fix: real-data evidence (RadioMapSeer is simulated), 6G (5.9 GHz), or the earlier forking paths (see section 9). It does not promise a positive result. The pre-declared positive result (section 2) may well come out NO, and that is a publishable outcome.

## 1. Literature positioning (what Kim and Orange Labs do, and what they leave open)

**Tania and Kim, "A GNN-Based Radio Map Construction With Uncertainty Prediction" (Asilomar 2025).**
- Graph: one node per grid point in a block. Edge (l, k) exists iff distance <= d_th (3*sqrt(2) grid cells), **no building between them**, and **both nodes share the same LOS/NLOS state relative to the base station**. Gaussian distance weights. GATv2, 3 layers (4,4,4 heads; hidden 64/64/40), input = normalised sparse RSS (-1 where unmeasured). Trained across blocks on dense ground truth, MSE then Gaussian NLL (uncertainty).
- Data: BRATLab Radiomap (Altair Feko, 1750 MHz band), three regions, 393 blocks of 41x41 (273 train / 60 val / 60 test), sampling rates 5-30% of block pixels.
- Compared against: the same GNN with the older edge rule, and an SICNN (CNN). **No IDW, no kriging, no regression-kriging with LOS features.** Reported gain: ~29% lower RMSE than the old edge rule on average.
- Open points for us: (a) the paper does not say whether blocks from the same region are kept apart between train and test, so spatial leakage cannot be ruled out from the text; (b) RMSE units are not stated in the tables, so **do not compare its numbers with ours**; (c) its sampling rates (5-30% of pixels) are far denser than our 30-300 labels per map.

**Bufort, Lebocq, Cathabard (Orange Labs), "Data-Driven Radio Propagation Modeling using GNNs" (arXiv 2501.06236).**
- Real 4G field measurements, about 300M points, about 20,000 sites in France; 400x400 images at 5 m; training and validation sites are disjoint. Two edge sets (grid neighbours; Tx-aligned "ray" edges that **ignore buildings**), polar edge attributes, FiLM conditioning on frequency / antenna height / EIRP, 10 message-passing blocks, 1.6M parameters, about 100 h on an A100, one image per batch.
- Reported test RMSE: 9.8 dB with ray edges vs 10.5 dB without vs 10.2 dB for a gradient-boosting tabular model (8.5 / 8.9 / 9.1 dB after an outdoor filter). They state GAT and graph-network blocks gave similar results. Compared with a legacy physical model; **no IDW / kriging baseline, no confidence intervals or paired tests.**
- Open points for us: the gain over a feature-only GBM is about 0.3-0.7 dB (roughly 3-7%) and untested for significance; this is the realistic size of a "GNN with structure" advantage at best.

**What this means for the plan.** Both papers support "graph structure helps a GNN relative to a structure-free neural or tabular model", and both train across hundreds to thousands of maps. Neither tests a GNN against IDW or against regression-kriging on the *same* label-free features, which is exactly the contrast your own results say decides the question. This plan builds that contrast with a Kim-style graph.

## 2. Question and pre-declared positive result

**Question.** Does a graph model trained across training maps beat `rk_side` (ridge on label-free map features + IDW of the residual) on held-out building maps, at 0.5% / 2% / 5% label density?

**Pre-declared positive GNN result [CHOOSE the thresholds].** At least one of `gnn_kim`, `rgnn_xmap` beats `rk_side` in the `primary` family with p_holm < 0.05, a relative RMSE gain of at least 2%, and a bootstrap CI above 0, at **at least 2 of the 3 densities**, and is not significantly worse than `rk_side` at the third. (Stricter than the city study, which needed one density.)

If this is not met, the paper's statement is "no", reported with the same weight as a yes.

## 3. Data

- **Primary: RadioMapSeer, DPM simulations [VERIFY layout].** 701 city maps x 80 transmitters, 256x256 px at 1 m, 5.9 GHz, building height fixed at 25 m, Tx/Rx height 1.5 m. Maps come from OpenStreetMap, six cities (Ankara, Berlin, Glasgow, Ljubljana, London, Tel Aviv).
- **Split by building map, never by transmitter.** Use the standard 550 train / 50 val / 100 test map split used in the RadioUNet literature **[VERIFY against the RadioUNet loader and use its ordering; if it differs, use the loader's]**. Test maps are touched once (section 10).
- **Training pool.** 8 transmitters per training map, chosen by a seeded draw **[CHOOSE; compute-driven]**. Optional scale check: 2 / 8 / 32 transmitters per map, reported as exploratory.
- **Secondary (sensitivity, declared now): the high-fidelity IRT subset [VERIFY counts and which version is available].** Same maps/features, evaluation only if models are trained on DPM (transfer from coarse to fine simulation).
- **Optional replication dataset: BRATLab Radiomap-Data** (Feko, 5 bands, about 2,000 areas, as described in Kim et al.). I have not inspected it; include only if it can be split by area without overlap.
- **Unit of inference = test building map.** Within a map, average over its test transmitters and over replicates. Because the 100 test maps come from six cities, report a city-cluster bootstrap as a sensitivity analysis (6 clusters, descriptive).

## 4. Task definition

- **Labels.** For each (map, transmitter), sample a fraction alpha of outdoor (non-building) pixels uniformly at random as labelled points. alpha in {0.5%, 2%, 5%} (about 300 / 1.2k / 3k labels; 0.5% matches the roughly-300-sample setting used with RadioUNet, 5% is the lowest rate in Kim et al.). 5 independent label draws per (map, transmitter).
- **Test points.** All remaining outdoor pixels (random split, primary). A blocked split (contiguous held-out region) is a declared secondary analysis.
- **Target.** Pathloss/RSS in dB as provided; all arms predict the same quantity after the same shared trend fitted on the training labels of that map only.
- **Features (label-free, from the building raster and Tx position only):** log distance to Tx, LOS flag, blocked length along the Tx-Rx segment, number of building crossings, building cover within 30 m, distance to nearest building. (The city study's "maximum obstruction height" is dropped because all building heights are 25 m here.)

## 5. Arms

**Classical (fit per map, no cross-map training)**
- `idw`, `krig_global` (local K-nearest kriging at 5% for cost), `rk_side` (ridge on the section-4 features + IDW of residual), `rk_gbm`.

**Learned across maps (trained on the training pool; one model per arm; alpha drawn log-uniformly in [0.3%, 6%] during training)**
- `gnn_kim`: GATv2, 3 layers, heads (4,4,4), hidden (64,64,40) as in Kim et al., on 64x64 tiles, edge rule = distance <= 3*sqrt(2) cells AND no building between AND same LOS/NLOS state; Gaussian edge weights; input = normalised sparse labels (-1 if missing) + section-4 features. MSE loss.
- `gnn_kim_oldedge`: same, edge rule without the LOS/NLOS-state condition (topology ablation).
- `rgnn_xmap`: `rk_side` prediction + a correction from the `gnn_kim` graph model, trained on rk_side residuals across the training maps (the cross-map version of the Round-7 residual model; no per-map gate needed because training is cross-map).
- `radiounet`: RadioUNet-style CNN (building map + Tx + sparse samples as input) trained from the public code on the same training pool, same alpha sampling, same loss. Report parameter count and training/inference time for every learned arm.

**Tuning budget (equal for everyone) [CHOOSE].** 30 random configurations per learned arm on the 50 val maps (learning rate, width, depth, edge radius, weight decay). Classical arms get a matching grid on val maps (ridge alpha, IDW power/K, kriging K). Defaults are trial 0. Test maps are never used for tuning, early stopping or normalisation statistics.

**Seeds.** 3 training seeds per learned arm; metrics averaged over seeds inside a unit; seed spread reported. If compute does not allow this, fall back to 1 seed for every learned arm and say so **before** freezing.

## 6. Metrics

- **Primary: RMSE (dB)** on held-out outdoor pixels. Gains are relative RMSE of A vs B, positive = A better, always shown next to dB values.
- **Coverage accuracy (M3 requirement): `bacc_q25`**, the same single coverage metric as the dense indoor run (threshold from the training labels of that map).
- Secondary, descriptive: MAE; RMSE by LOS vs NLOS pixels; RMSE within 10 m of a building edge.

## 7. Statistics and families

Paired across test maps. Exact sign-flip p, bootstrap CI over maps, verdict labels (`A better` / `A worse` / `equivalent` within +/-2% / `inconclusive`), **Holm within each family**. Same conventions as the city study.

| Family | Contrasts (each at 0.5% / 2% / 5%) | Tests |
|---|---|---|
| `primary` | `gnn_kim` vs `rk_side`; `rgnn_xmap` vs `rk_side` | 6 |
| `vs_classical` | `gnn_kim` vs `idw`; `rgnn_xmap` vs `idw` | 6 |
| `feat` | `rk_side` vs `idw` | 3 |
| `topo` | `gnn_kim` vs `gnn_kim_oldedge` | 3 |
| `deep` (two-sided; non-inferiority margin 5% RMSE) | `gnn_kim` vs `radiounet`; `rgnn_xmap` vs `radiounet` | 6 |

Total 24 tests. Anything not in this table is exploratory and labelled so in the paper. Before freezing, the freeze step checks that each family can reach p_holm < 0.05 with 100 maps.

## 8. Leak and sanity checks (must pass before the test run)

1. No test map or test transmitter appears in the training, tuning or normalisation data (assert on map ids).
2. Features use no power value, only the building raster and Tx position.
3. Per-map trend, ridge and IDW parameters are fit on that map's labels only.
4. Shuffled-label control: train `gnn_kim` with labels permuted across pixels; its test RMSE must be at least as bad as `idw`.
5. Zero-label control: a model given no labels must not beat the trend-only baseline by more than what the features alone give (`rk_side` with n=0 labels).
6. Edge-rule unit test on a toy map: no edge crosses a building and no edge joins a LOS and an NLOS node.

## 9. Disclosure of forking paths

This plan was designed **after** seeing: the negative city and dense-indoor results, the Round-7/8 results (including the unconfirmed `rattn` lead on dev and val cities), and after reading Kim et al. and Bufort et al. The design choices that came from that: cross-map training, the Kim edge rule, and the residual-on-`rk_side` arm. `rattn` is not included here; if it is to be confirmed, it gets its own one-contrast plan on fresh city data. The supplement table of rounds lists, for each round, the date, what had been seen before designing it, and the number of contrasts.

## 10. Procedure

1. **Dev stage (val maps only, exploratory).** Build and debug the arms; tune on the 50 val maps; nothing here is reported as evidence.
2. **Freeze.** Write the config (arms, hyper-parameters chosen on val, families, thresholds, test map ids, seeds, label-draw salts) to `results_rms/prereg_rms.json`; store its sha256; commit to git **before** the test run. The runner refuses to start without the file and refuses to overwrite it.
3. **Confirmatory run.** One run on the 100 test maps. No arm, threshold or family may change afterwards.
4. **Report.** All 24 contrasts printed, wins and losses, with dB values, CIs and Holm p. Sensitivity analyses (IRT subset, blocked split, city-cluster bootstrap) after the primary tables.

Proposed new files (no existing file is modified): `rmbench_rms.py`, `rmbench_rms.ipynb`, `results_rms/`.

## 11. Decisions to confirm before freezing

- Pre-declared threshold (2 of 3 densities, >= 2% gain) and the 5% non-inferiority margin versus RadioUNet.
- 8 transmitters per training map and 3 training seeds (compute).
- Whether to include the BRATLab replication dataset.
- Whether the IRT subset is available in the form assumed in section 3.
