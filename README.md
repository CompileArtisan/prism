# Sparse radio-map estimation: IDW vs kriging vs graph models, from simulation to real Wi-Fi data

A unit-level benchmark of classical interpolators (IDW, ordinary kriging) and learned models (GCN, GAT, a kriging-initialised graph model called KGNN) that predict received power from a handful of labeled measurements. One *unit* is one transmitter's radio map, and every statistical test is paired across independent parents (transmitters, building-floors or layouts, depending on the dataset).

The work ran in eight stages plus two partial runs, followed by three exploratory rounds on the city pool (Round 7, 7b and 8; val and dev cities only, no frozen plan) and a **dense indoor run** (2026-10-21; 50 / 100 / 200 / 300 labels, plan frozen before the run) and, since 2026-10-22, a **RadioMapSeer cross-map study** (dev stage run 2026-10-25/26; **plan not yet frozen, no confirmatory run**). The older stages are never overwritten:

| Stage                                        | Notebook                                                 | Data                                                                                                                 | Purpose                                                                                                                                                                                                                   |
|----------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Main sweep** (2026-10-11)                  | `radiomap_6g_sweep.ipynb`                                | DeepMIMO `i1_2p5`, `i2_28b`                                                                                          | Classical vs graph models, with sensitivity variants                                                                                                                                                                      |
| **Round 1** (2026-10-11)                     | `radiomap_ext_real_crosslayout.ipynb`                    | UJIIndoorLoc (real Wi-Fi), wall simulator, DeepMIMO                                                                  | More layouts, robust trend, side information, tuning, transfer, both splits, pre-registration                                                                                                                             |
| **Round 2** (2026-10-12)                     | `radiomap_ext_conf2.ipynb`                               | UJI (fresh APs), SODIndoorLoc (new real data), wall simulator (new layouts), DeepMIMO                                | Pre-specified replication of the two surviving questions on fresh data                                                                                                                                                    |
| **Round 3**                                  | `radiomap_ext_conf3_kriging_attention.ipynb`             | UJI, SODIndoorLoc (60-label density added), wall simulator                                                           | Meta-learned prior (`kgnn_prior*`), side-aware kernel, ensembles, gradient-boosting baseline (`gb_geo`), cost                                                                                                             |
| **Round 4**                                  | `radiomap_ext_conf4_ablation.ipynb`                      | Round-3 real Wi-Fi units (UJI, SOD), fresh test salt `abl4`                                                          | Ablation: where does KGNN's edge come from? Learned parameters vs the fixed local-kriging structure                                                                                                                       |
| **Round 5**                                  | `radiomap_ext5_regime_scale.ipynb`                       | Wall simulator (256 source, 64 development and 64 fresh evaluation units) with a **planted** side-information signal | When can learning across layouts help at all? Regime map, extra baselines, training-scale experiment, and (v4) a pre-trained pointwise prior with a cross-validated do-no-harm gate, dose-response over planted amplitude |
| **Round 6** (**run**)                        | `radiomap_ext6_learnability.ipynb`                       | Same wall simulator, new fresh layouts (pool seed 31 / simulator seed 9), 24 generated covariates                    | Repeat the round-5 control where a single map provably cannot learn the signal: nonlinear within-map comparators, globally scaled plant, learnability-gap diagnostic (D0)                                                 |
| **City study (M3)** (plan frozen 2026-10-19) | `rmbench_city.ipynb` (+ `rmbench_city.py`, `d0_city.py`) | DeepMIMO ray-traced **city** layouts (`city_*_3p5`, 53 eligible cities, 5 label-free map features)                   | Do GCN / GAT (with and without the map features) beat IDW, kriging and regression-kriging on independent city layouts? Pre-registered, confirmatory run on 10 held-out cities                                             |
| **Round 7** (exploratory) | `rmbench_r7.ipynb` sections 1-8 (+ `rmbench_r7.py`) | City pool: 7 dev cities (12 units) and 6 val cities (10 units), 30 / 100 / 300 labels; confirm cities not used | Can a GCN that learns a cross-fitted, CV-gated correction on top of `rk_side` (kNN or feature-space graph over every grid cell) improve on it? |
| **Round 7b** (exploratory) | `rmbench_r7.ipynb` section 10 | Same cities, plus building rasters for the 13 val + dev scenes | Obstruction-aware ("wall") graph: does it change the neighbours at all, and does it help? |
| **Round 8** (exploratory) | `rmbench_r7.ipynb` section 11 + `rmbench_r8.py` | Same cities and rasters | Label-centric, obstruction-aware attention interpolator (`rattn`): messages go from the K nearest *labelled* points to each query instead of through a graph over every grid cell |
| **Dense indoor run** (2026-10-21) | `dense_indoor_radiomap.ipynb` (+ `run_indoor_dense.py`) | Wall simulator (12 layouts x 6 transmitters, 1000 points each) and DeepMIMO `i1_2p5` (24 tx), `i2_28b` (1 tx), `i3_60` (2 tx) | Brief's dense setting (200+ labels): do GCN / GAT beat IDW and kriging at 50 / 100 / 200 / 300 labels on indoor layouts, with one common coverage metric (`bacc_q25`) |
| **RadioMapSeer cross-map study** (draft plan; dev stage run 2026-10-25/26; **not frozen**) | `rmbench_rms.ipynb` (+ `rmbench_rms.py`, written by the notebook; `PLAN_radiomapseer_DRAFT.md`, `KAGGLE_RUNBOOK_rms.md`, `kaggle_sync.py`) | RadioMapSeer DPM simulations: 701 building maps x 80 transmitters, 256x256 px at 1 m; 501 / 100 / 99 train / val / test maps (RadioUNet loader) | Do graph models trained **across many maps** beat `rk_side` on held-out maps at 0.5% / 2% / 5% label density? 30 contrasts in 6 families are planned (24 in 5 in the original draft; the `cnn_feat` arm and the `deep_feat` family were added 2026-10-26); only the dev stage (val maps) has been run |
| **UJI extension runs** (partial)             | `run_ext.py`                                             | UJIIndoorLoc, 15 / 30 / 60 labels, random and blocked, leave-one-building-out transfer                               | Real indoor cross-layout evidence for the M3 timeline. **Incomplete**: see the UJI section                                                                                                                                |

## Current interpretation (updated 2026-10-28)

**Bottom line: there is no established positive GNN result on real radio-map data, and the pre-declared positive GNN result of the city study is NO.** On the 10 frozen confirmation cities, neither a GCN nor a GAT with the five map features beats regression-kriging on the same features (`rk_side`) at any density; the GCN is significantly worse at 100 and 300 labels and the GAT at 100 labels. GNNs that see only geometry lose to IDW by 8-19% in every cell, significantly. What is positive and robust is the **side information**, not the learned interpolator: the five map features give a 15-28% RMSE reduction to the GNNs and about 15-22% to a plain ridge + IDW model over IDW. Topology (kNN vs Delaunay) and architecture (GCN vs GAT) make no detectable difference.

Earlier rounds point the same way. The KGNN's small edge over IDW and global kriging is reproducible, but Round 4's ablation shows it comes from the fixed local-kriging structure and nugget, not from learned parameters. The strongest positive learning result is semi-synthetic: in Round 6 a pre-trained **pointwise MLP prior** (not message passing) beats every comparator, including nonlinear within-map regression-kriging, once a shared signal of amplitude >= 0.2 is deliberately planted into simulated maps. The Round-6 false-positive control did **not** pass its written rule as stated (see Round 6). The city-pool transfer gate (D0 on real ray-traced cities) **failed**, so cross-city pre-training was not run at confirmatory scale.

**Rounds 7-8 (exploratory; val and dev cities only; no frozen plan; confirm cities untouched by construction of the notebook cells) do not change the bottom line.** A GCN that learns a gated correction on top of `rk_side` adds nothing (the gate stays closed in 0-5% of runs; ungated it is worse at 30 and 100 labels), and an obstruction-aware graph is 90-98% the same graph as kNN and does not help. The one lead is Round 8's `rattn`, an attention interpolator over the nearest labelled points with the 5 map features inside its kernel: +3.2% / +6.2% over `rk_side` at 100 / 300 labels on the 7 dev cities (positive in all 7), but +0.03% on the 6 val cities at 100 labels, and it is not a GCN/GAT (see Round 8). It is a hypothesis to confirm, not a result.

**Dense indoor run (2026-10-21) does not change the bottom line.** At 50 / 100 / 200 / 300 labels on 12 simulated indoor layouts the three frozen primary contrasts all fail: `gcn_knn` and `gat_del` are 8-10% worse than IDW and 9-12% worse than global kriging in RMSE, Holm-significant in every cell, and the gap does not close by 300 labels. The coverage metric (`bacc_q25`) points the same way (GNN below IDW in all 8 cells, Holm-significant in 7). Side information again carries the gain (`rk_side` vs IDW +20-23% RMSE), and GNNs given the same side information are 16-31% worse than `rk_side`. Delaunay vs kNN shows no benefit. On DeepMIMO `i1_2p5` the GNNs also lose to IDW, but that scenario is 100% line-of-sight with RMSE of 0.06-0.10 dB, so its effects are statistically clear and practically negligible, and its 24 parents are transmitters in one room. See [Dense indoor run](#dense-indoor-run-2026-10-21).

**RadioMapSeer cross-map study (2026-10-22 to 2026-10-26): dev stage only, no result yet.** This study tests the regime the earlier rounds could not: graph models trained across hundreds of building maps with dense ground truth, against `rk_side`, on a public benchmark. The plan is still a **draft**: `results_rms/prereg_rms.json` has not been written, the leak and sanity checks (plan section 8) have not been run, and the test maps have not been read. Nothing below is evidence for or against the pre-declared positive result. What the dev stage (100 val maps, the same maps that were used for tuning, so optimistic) shows: a residual graph model on top of `rk_side` (`rgnn_xmap`) is ahead of `rk_side` by +3.3% / +3.9% / +2.4% at 0.5% / 2% / 5% labels; the Kim-style graph model on its own (`gnn_kim`) beats IDW (+9.2% / +8.1% / +2.1%) but loses to `rk_side` (-14.8% / -3.7% / -4.6%); removing the LOS/NLOS condition from the edge rule changes nothing visible; and the RadioUNet arm saturates at 5.4-5.6 dB (worse than IDW) and barely uses the labels. Training length was not the cause (20,000 steps gained 0.01 dB over the last 8,000). **Root cause found 2026-10-27, confirmed on the GPU 2026-10-28:** the public RadioUNet ends in Conv -> ReLU, so its output is never negative, and this study trains it on a signed residual. The 5.4-5.6 dB plateau was an implementation defect, not a property of the setup, and the earlier statement that the network has no output squashing was wrong (the check looked for sigmoid / tanh / clamp, not ReLU). With the final ReLU removed and both CNN arms re-tuned, the feature-fed `cnn_feat` is the strongest arm on the val maps (2.18 / 1.82 / 1.70 dB, 20-24% below `rk_side`), and `rgnn_xmap` trained for the final 8 000 steps beats `rk_side` by 8.7 / 11.1 / 10.6% (item 24; exploratory, val maps tuned on, one seed). So the dev stage now carries a **positive signal for the GNN against the classical baseline, and a stronger one for a feature-fed CNN**. Neither is a result until the frozen confirmatory run on the 99 test maps. See [RadioMapSeer cross-map study](#radiomapseer-cross-map-study-dev-stage-only-no-confirmatory-run).

Use the saved raw outputs to verify every number before quoting it. Some long tables and notebook outputs were truncated in the review, and the `radiogat` and 6G sweep notebooks were only skimmed. Do not treat a result as confirmed merely because its experiment script defines the method.

## Summary of findings

1. **The KGNN advantage is real but small, and it replicates.** KGNN beats IDW in nearly every cell of every dataset and round (+1.3% to +7%, 0.1–0.4 dB on real Wi-Fi, 0.003–0.008 dB on DeepMIMO; Figure 1). In round 3 the sign holds in all 7 real-Wi-Fi random-split cells, but those gains are descriptive (no paired test was run). Against a fitted global kriging it wins in most cells, by 8% at 5 labels on real Wi-Fi but only 1–3% at 10–20 labels.
2. **Only one pre-registered claim passed its own strict rule.** `kgnn` vs `idw` on the wall simulator (6 of 6 cells). The primary claim `kgnn` vs `krig_global` was **NOT SUPPORTED** on real Wi-Fi and on the simulator: 5 of 6 cells were significantly better, and one cell stayed open.
3. **GCN and GAT never beat IDW.** In 3 families, 2 splits and all densities, the plain label-aware GCN/GAT lose to IDW (−2% to −14%), and tuning does not change that.
4. **Side information is what helps, not the learned interpolator.** On the wall simulator a plain ridge regression on wall-count and LOS covariates plus IDW (`rk_side`) beats IDW by 22–23%. Giving the same covariates to KGNN (`kgnn_side`) lands within about ±1% of it. Giving them to a GNN is far worse than `rk_side` (−19% to −34%).
5. **Pre-training on other layouts does not rescue the GNNs.** On real data, GNNs pre-trained on other buildings are not better than IDW (significantly worse at 20 labels). On the simulator they beat IDW but still lose to `rk_side`.
6. **A bad shared trend, not the interpolator, caused the biggest errors at very few labels on real data.** A robust trend selection fixed it (trend-only RMSE at 5 labels fell from 60.8 dB to 11.7 dB).

7. **A meta-learned prior helps only at the very fewest labels on real data.** `kgnn_prior` is the best method at 5 labels on UJI (+4.9% over IDW), but it falls behind from-scratch KGNN as labels grow (−12% on SOD at 20 and 60 labels). It is zero-shot and about 100x cheaper than KGNN (Figure 7).
8. **Gradient boosting on coordinates is a weak baseline at low density and competitive only on dense data.** `gb_geo` is 18–33% worse than IDW on UJI random and 30% worse on SOD at 5 labels, but 4.5% better than IDW at 60 labels on SOD.

9. **Where the edge comes from: the local-kriging structure, not training (round 4).** A local kriging of the K=12 nearest labels with a fixed default covariance (exponential in space, polar term, nugget; **no training at all**) already carries the gain. On UJI (random split) it beats IDW by +2.7% / +2.8% / −0.2% and global kriging by +5.8% / +2.2% / −1.1% at 5 / 10 / 20 labels (Holm-significant at 5 labels vs global kriging and at 10 labels vs IDW). Learning the six covariance parameters on top adds nothing measurable (pooled primary |gain| ≤ 1.1%, ≤ 0.1 dB; verdict NOT ESTABLISHED). The practical consequence is positive: a **training-free local kriging with a nugget** is the cheap, robust method to use at 5–10 labels, and it needs no GNN.

10. **Cross-layout learning can win, but only when a planted signal is strong (round 5, v4, semi-synthetic).** A pointwise prior pre-trained on other layouts, followed by `rk_side` on the remainder and gated by 5-fold cross-validation, beats all five comparators (`idw`, `rk_side`, `krig_global`, `kgnn`, `gp_ml`) on fresh layouts in 6 of 6 (K, density) cells at planted amplitude 0.5 (CI lower bounds 4.4–5.3%, mean gain 7.2–8.6%), and in none at amplitude ≤ 0.3 (mean gain 0.0–3.0%, CI lower bound below the 2% margin). With no planted signal (amplitude 0) the gate returns `rk_side` (gain 0.00–0.04%, equivalent in all 6 cells). This is a simulator with an artificial signal; it says when transfer *can* help, not that it does on real maps.
11. **The round-5 control was confounded; Round 6 repeated it with a cleaner plant.** Round 5's D0 diagnostic failed (median corr(prior, f) = -0.19; linear R² = 0.73). Round 6 passed its revised D0 (corr(prior, f) 0.548 vs 0.096 for the best within-map comparator, linear R² of `f` 0.168) and found 17 of 30 (K, density, amplitude) cells in which the gated prior beats every comparator, including nonlinear regression-kriging, by more than the 2% margin on fresh layouts (smallest winning amplitude 0.2). This is a semi-synthetic result for a pointwise MLP prior. The written no-harm control (equivalent to `rk_side` at amplitude 0) returned `False`: at 100 labels the gated prior was *better* than `rk_side` by 2.1-2.6%, outside the equivalence band; it was equivalent at 300 and 1000 labels.
12. **City study, confirmatory run (10 held-out cities): the pre-declared positive GNN result is NO.** `gcn_knn_side` vs `rk_side`: -7.5% / -11.8% / -9.8% at 30 / 100 / 300 labels; `gat_del_side` vs `rk_side`: -8.5% / -7.9% / -7.3% (0.18-0.36 dB). Holm-significant (within the 6-test primary family): GCN at 100 and 300 labels, GAT at 100 labels.
13. **The map features are what help.** Adding the 5 features to a GNN improves it by +15% to +28% (0.5-1.5 dB; all 6 contrasts Holm-significant at the smallest attainable p). `rk_side` beats IDW by +17% / +22% / +15% (Holm-significant at 100 and 300 labels, not at 30) and global kriging by +15% / +19% / +10% (significant at 100). A gradient-boosted mean (`rk_gbm`) adds nothing over `rk_side`.
14. **GNNs on geometry alone are worse than IDW in the city pool.** `gcn_knn` and `gat_del` lose to IDW by 8-19% at every density (6 of 6 cells Holm-significant). Topology and architecture contrasts are all inconclusive after Holm.
15. **Cross-city transfer gate failed (D0 on the 7 dev cities).** A prior pre-trained on 30 source cities, with no labels from the target city, plus IDW of the remainder reaches 3.31 dB median RMSE at 100 labels vs 2.77 dB for `rk_side` (median gain -7.7%, positive in 3 of 7 cities). The map features carry the prior (+28 points over the geometry-only prior), but not enough to beat a within-map fit. Stage 2 transfer was not run, as the pre-declared rule requires.

16. **A residual GCN on top of `rk_side` adds nothing (Round 7, exploratory, 7 dev cities).** With the do-no-harm gate (paired z > 2.5, relative gain > 1%) the gate opened in 0-5% of runs, so `rgnn_knn` and `rgnn_feat` equal `rk_side` to three decimals at 30 and 100 labels (RMSE 4.009 / 3.157 dB; IDW 5.311 / 3.739). Ungated (λ = 1) they lose 13.0% / 1.9% (kNN) and 10.5% / 3.0% (feature graph) at 30 / 100 labels and are inconclusive at 300 (+0.3% / +0.6%). The graph choice (kNN vs feature space) makes no difference.
17. **The wall (obstruction-aware) graph is almost the kNN graph and does not help (Round 7b).** On 6 dev units 1.3-16.2% of kNN edges cross a building cell and the wall graph shares 90-98% of its edges with kNN. On the 6 val cities all three β values (1, 3, 10) lose to `rk_side` ungated (mean -0.9% / -2.1% / -3.0%); β = 1 was selected as the least bad, not as a gain. Dev: ungated wall arm -7.8% / -2.8% / +2.0% vs `rk_side`, no difference from the kNN or feature graphs.
18. **One unconfirmed lead: `rattn_side`, an attention interpolator with map features in the kernel (Round 8, exploratory).** Ungated on the 7 dev cities it beats `rk_side` by +0.4% / +3.2% / +6.2% at 30 / 100 / 300 labels (all 7 cities positive at 100 and 300; CI above 0 at 100 and 300), and the gain is attributable to the map features (`rattn_side_ng` vs `rattn_geo_ng` +1.3% / +4.0% / +6.4%); a learned isotropic kernel without them is not better than `rk_side` (-0.9% / -0.9% / -0.2%), as in Round 4. It does **not** reproduce on the 6 val cities at 100 labels (+0.03%, 3 of 6 cities positive), one city (`city_53_sankt-peterburg`, where `rk_side` is itself worse than IDW) contributes a gain of 10.7% / 18.3%, the gate keeps the model closed in 85-98% of runs, and the model is a one-hop attention kernel, not message passing.

19. **RadioMapSeer, dev stage (val maps only, exploratory, not frozen): a residual graph model on `rk_side` is the only learned arm ahead of it.** On the 100 val maps (tuned on them, one label draw per density) `rgnn_xmap` gives +3.3% / +3.9% / +2.4% vs `rk_side` at 0.5% / 2% / 5% labels (2.761 / 2.261 / 2.073 dB vs 2.854 / 2.352 / 2.123 dB). This is the cross-map version of the Round-7 residual model, which added nothing per map. It is a dev-stage observation with a short tuning budget (1,500 steps per trial); the confirmatory run uses 8,000 steps, three seeds and 99 unseen test maps and may come out differently.
20. **RadioMapSeer, dev stage: `gnn_kim` alone does not beat `rk_side`, and the LOS/NLOS edge condition shows no benefit.** `gnn_kim` -14.8% / -3.7% / -4.6% vs `rk_side`, and +9.2% / +8.1% / +2.1% vs `idw`; `gnn_kim_oldedge` (no LOS/NLOS condition) is -16.1% / -1.9% / -3.2% vs `rk_side`, i.e. within about 2 points of `gnn_kim` at each density (slightly worse at 0.5%, slightly better at 2% and 5%). The same pattern as the city study (geometry-plus-features GNN below `rk_side`) appears on a different dataset, with cross-map training.
21. **RadioMapSeer, dev stage (pre-fix, void; see item 23): the RadioUNet arm plateaued at 5.55 dB.** Tuned (1,500 steps): 5.643 / 5.458 / 5.428 dB at 0.5% / 2% / 5% labels; trained 20,000 steps: 5.439 / 5.363 / 5.365 dB (100 val maps); a model that predicts the trend only scores 7.41 / 7.38 / 7.38 dB. The error is flat in the number of labels and the training loss equals the val error, so it is under-fitting, not over-fitting. The public network has no output squashing (no `sigmoid`, `tanh` or `clamp` in `RadioUNet/lib/modules.py`). The plain `radiounet` arm never receives the six label-free map features the other arms get; adding them did not change the plateau (item 22) because the cause was a different one (item 23).

22. **RadioMapSeer, dev stage: giving the CNN the six features (`cnn_feat`) did not change the plateau (pre-fix, void; see item 23), and part of `rgnn_xmap`'s dev gain is selection on the val maps.** `cnn_feat` (the same U-net with 10 input channels, 10 932 401 parameters; best of 30 configurations at 1 500 steps, trial 22, lr 4.2e-4, weight decay 1.3e-5) scores 5.479 / 5.404 / 5.402 dB at 0.5% / 2% / 5% labels, a mean of 5.428 dB against 5.510 for `radiounet` and 2.443 for `rk_side`; its error moves by 0.08 dB across a 10x change in labels, and across the 30 trials the mean val RMSE ranged only from 5.43 to 5.97 dB. Two different input sets gave the same plateau because both arms share the defect of item 23. Separately, about half of `rgnn_xmap`'s mean gain over `rk_side` appears only after choosing the configuration on the val maps: its default configuration (trial 0) averaged 2.407 dB over the three densities, 1.5% below `rk_side` (2.443 dB), and the selected one 2.365 dB, 3.2% below.

23. **RadioMapSeer, dev stage: the CNN plateau is an implementation defect (found 2026-10-27); the CNN dev numbers are void.** The public `RadioWNet` (`lib/modules.py`, layer `conv_up000`) ends with `convrelu(20 + inputs, 1, 5, 2, 1)` = Conv -> ReLU -> MaxPool(1), so the output of its first U is >= 0. The target here is a signed residual (observed minus the per-map trend, mean about 0), which a non-negative output cannot represent. That explains why `radiounet` and `cnn_feat` both stalled at 5.4-5.6 dB, were flat in the number of labels, and gained nothing from the six features. A synthetic check (`relu_synth_test.py`, CPU, 64x64 maps, a signed pointwise target of which 55% of pixels are negative, 300 steps) gave a validation MSE of 0.096 with the ReLU (target variance 0.174, lowest output exactly 0) against 0.005 without it (lowest output -1.36). That shows the mechanism only, not a RadioMapSeer result. **Fix:** `PublicRadioUNet` drops that ReLU and keeps the rest of the public network (notebook module cell; unit test section 4 (e) asserts that the output can be negative). **Confirmed on the GPU (2026-10-28, T4):** default hyper-parameters, 1 500 steps: `cnn_feat` 5.585 -> 2.972 dB and `radiounet` 5.596 -> 2.950 dB (val-map mean); overfit test on 4 training pairs 1.776 dB against 6.747 dB trend-only (ratio 0.26). Both arms were then re-tuned (30 configurations, 115 and 124 min); the pre-fix trials are archived as `tune_<arm>_trials_preReluFix.csv`. The GNN and classical arms do not use this network and were unaffected. The module hash changed (`0896363c77b45520` in the 2026-10-28 session). The freeze cell refuses to run unless both CNN arms are re-tuned.

24. **RadioMapSeer, dev stage after the fix (2026-10-28; exploratory, val maps that were also tuned on): `cnn_feat` is the strongest arm, and `rgnn_xmap` at the final budget is clearly above `rk_side`.** (a) CNN arms, best of 30 configurations at 1 500 steps: `cnn_feat` 2.184 / 1.818 / 1.700 dB at 0.5% / 2% / 5% labels (23.5 / 22.7 / 19.9% below `rk_side`), `radiounet` (no features) 3.007 / 2.251 / 2.014 dB (5.4% worse than `rk_side` at 0.5%, 4.3 / 5.1% better at 2% / 5%). The same default configuration of `cnn_feat` scored 2.972 dB in the fix check and 3.386 dB as tuning trial 0, so one 1 500-step run carries about 0.4 dB of run-to-run noise and the selected configuration is partly a lucky draw. (b) `rgnn_xmap` trained at the final budget (8 000 steps, 1 seed, 20.5 min) and scored on the 100 val maps with 2 label draws: 2.594 / 2.084 / 1.887 dB against `rk_side` 2.848 / 2.354 / 2.122, i.e. relative gains of **8.7 / 11.1 / 10.6%** (bootstrap CI over maps 8.0-9.3 / 10.4-11.7 / 9.9-11.3; better on 99% of maps) and 28.7 / 21.3 / 16.1% over `idw`. That is much larger than the 3.3 / 3.9 / 2.4% of the 1 500-step tuning table, so most of the gain comes from longer training, not from selection. The numbers remain optimistic (val maps were used for tuning; one training seed; the CI covers map-to-map variation only). On these dev numbers the plan's positive rule would be met for `rgnn_xmap` at 3 of 3 densities. (c) Stride: stride 32 improved `rgnn_xmap` by only 0.27 / 0.54 / 0.44% (40 maps), below the pre-specified 1% rule, so **stride 64 is kept**. (d) On the val maps `cnn_feat` is about 16 / 13 / 10% better than `rgnn_xmap`, although `cnn_feat` has only seen 1 500 training steps in these numbers and `rgnn_xmap` 8 000. The family `deep_feat` (GNN vs `cnn_feat`, non-inferiority margin 5%) is therefore expected to show the GNNs worse than the CNN. A final result of "the GNN beats the classical baseline but not a feature-fed CNN" must be reported as such.

Absolute differences are small compared with the noise of the targets, so always read percentages next to dB values. See [Limitations](#limitations).

![KGNN vs IDW across datasets and rounds](figures/fig1_kgnn_vs_idw_replication.png)

*Figure 1. KGNN vs IDW, random split, every dataset and round. Round-3 rows (†) are computed from mean-RMSE tables, not from a paired test.*

### What can and cannot be claimed

| Claim | Status |
|---|---|
| KGNN gives a small RMSE gain over IDW | Observed in several datasets and rounds, but Round-4 ablation attributes the edge primarily to fixed local-kriging structure; do not claim a demonstrated learning gain |
| KGNN beats fitted global kriging at 5–10 labels on real Wi-Fi | Consistent in 6 of 6 buildings (round 2); the frozen claim rule is **not met** (one cell open) |
| GCN/GAT beat IDW | **No**, in any family, split or density |
| Side information via a GNN beats ridge + IDW (`rk_side`) | **No** (practically equivalent at best) |
| A meta-learned prior helps | Only at 5 labels on real Wi-Fi and on the wall simulator; hurts at higher density on real data |
| Learning the KGNN kernel beats the same kernel untrained (round 4 primary) | **NOT ESTABLISHED.** Pooled real Wi-Fi, 6 cells: −1.1% to +1.0%; no Holm-significant result |
| Training-free local kriging (fixed default covariance + nugget) beats IDW and global kriging at 5–10 labels on real Wi-Fi | **Supported descriptively and in part by Holm-significant secondary contrasts** (UJI random: 2 of 6 cells significant, rest same sign or inconclusive). Not a pre-registered primary |
| The nugget matters | Yes: freezing it at ~0 costs +4.1% RMSE at 20 labels (UJI random, Holm-significant); learning it vs the 0.2 default adds nothing |
| A pre-trained cross-layout prior beats classical regression-kriging when a shared signal is planted | **Semi-synthetic only.** Round 6: gated pointwise-MLP prior beats all 11 comparators (incl. nonlinear within-map RK) by more than 2% in 17 of 30 cells on fresh layouts; smallest winning amplitude 0.2 (0.3 for K = 16 at 300 labels). Mean gain over the strongest comparator: 2.8-8.4% at amplitude 0.2, 11.1-18.3% at 0.3, 21.4-33.6% at 0.5 |
| The CV gate does no harm without shared structure | **Not established under the Round-6 written rule** (equivalent to `rk_side` at amplitude 0 in every cell). At 100 labels the gated prior was better than `rk_side` by 2.06% (K = 16, CI 0.98-3.23) and 2.56% (K = 64, CI 1.40-3.78), so the CI sat outside the +/-2% band; at 300 and 1000 labels it was equivalent. The *ungated* priors lost 3-10% at 300 and 1000 labels |
| A GCN/GAT with the 5 map features beats regression-kriging on the same features (city pool) | **No** (confirmatory, 10 cities): -7% to -12%, significantly worse in 3 of 6 cells |
| The 5 ray-traced map features improve interpolation on held-out cities | **Yes** for ridge + IDW (`rk_side` vs IDW +15-22%, Holm-significant at 100 and 300 labels; not at 30 labels after Holm) and for the GNNs (+15-28%) |
| GNNs on geometry alone beat IDW (city pool) | **No**: 8-19% worse, significant in all 6 cells |
| kNN vs Delaunay, GCN vs GAT (city pool) | No detectable difference (all Holm p = 0.77-1.0) |
| Cross-city pre-training helps on real ray-traced cities | **Not supported**: D0 failed (median gain -7.7%, 3 of 7 cities positive, dev cities only) |
| A GCN that learns a gated correction on top of `rk_side` beats `rk_side` (Round 7, city dev) | **No** (exploratory, 7 dev cities): gate open in 0-5% of runs, so the gated model equals `rk_side`; ungated -13% to +0.6%, never better with a CI above 0 |
| An obstruction-aware (wall) graph helps a GNN (Round 7b) | **No** (exploratory): 90-98% edge overlap with kNN; ungated -7.8% / -2.8% / +2.0% vs `rk_side`; no difference from kNN or feature graphs |
| An attention interpolator over the nearest labels with the 5 map features in its kernel beats `rk_side` (Round 8, `rattn_side_ng`) | **Lead, not a claim.** Dev (7 cities): +0.4% / +3.2% / +6.2% at 30 / 100 / 300 labels, positive in all 7 cities at 100 and 300; val (6 cities, 100 labels): +0.03%. No frozen plan, no Holm-significant cell possible with 7 cities, confirm cities untouched, not a GCN/GAT |
| The same kernel without map features (`rattn_geo_ng`) beats `rk_side` (Round 8) | **No**: -0.9% / -0.9% / -0.2% (equivalent), the same finding as Round 4 (a learned kernel adds nothing over a fixed one) |
| A cross-layout prior helps on real radio maps | **Not established.** No positive real-data result demonstrates this shared side-information structure |
| GCN/GAT beat IDW and global kriging at 50-300 labels on indoor layouts (dense run, wall simulator, 12 layouts) | **No** (frozen primary contrasts): -8% to -12% RMSE, Holm-significant in 12 of 12 cells; coverage `bacc_q25` below IDW in 8 of 8 cells (significant in 7) |
| Side information helps a dense indoor map (wall simulator) | **Yes**: `rk_side` vs IDW +19.8% to +23.2% RMSE, +18% to +26% coverage metric, all Holm-significant. The author's own simulator, so informative by construction |
| A GNN with side information beats `rk_side` (dense, wall simulator) | **No**: `gcn_knn_side` -30% to -31%, `gat_del_side` -16% to -23%, all significant |
| Delaunay beats kNN (dense run) | **No**: wall RMSE -0.4% to -1.6% for GCN (significant at 200 and 300 labels, in the wrong direction), -0.1% to -1.1% for GAT; coverage metric and DeepMIMO `i1_2p5` show no significant difference |
| GAT beats GCN (dense run) | **Marginally, only at high density**: wall `gat_del` vs `gcn_del` +1.3% / +2.5% at 200 / 300 labels (significant); `gat_knn` vs `gcn_knn` +1.1% at 300 only; DeepMIMO +3.5% at 300 (`gat_del` vs `gcn_del`). Both still lose to IDW |
| Real indoor data can support 200+ labels (UJI) | **No**: under the default UJI config only 2 of 13 eligible units have >= 100 + 40 points and none has >= 150 + 40; UJI is 13 units in 2 building-floors, so it is descriptive only |
| A graph model trained across maps beats `rk_side` on held-out RadioMapSeer maps (plan `primary` family) | **Not tested yet.** Plan not frozen; test maps unread. Dev stage only (val maps, tuned on them): `rgnn_xmap` +2.4% to +3.9%, `gnn_kim` -3.7% to -14.8% vs `rk_side` |
| The LOS/NLOS-state condition in the Kim edge rule helps (`topo` family) | **No evidence in the dev stage**: `gnn_kim` vs `gnn_kim_oldedge` within about 2 points either way (mixed sign). Confirmatory test pending |
| A RadioUNet-style CNN is a strong cross-map baseline here | **Dev stage: yes (val maps only, not evidence).** After the fix (item 23) `cnn_feat` reaches 2.18 / 1.82 / 1.70 dB at 0.5% / 2% / 5% labels, 20-24% below `rk_side`, and the feature-free `radiounet` beats `rk_side` at 2% and 5%. The `deep_feat` family is the confirmatory test of the GNNs against it |
| Anything about 6G / mmWave / sub-THz | **Not supported** by these data |

![Real Wi-Fi RMSE versus labels](figures/fig2_real_wifi_rmse_vs_labels.png)

## Files

| File | Purpose |
|---|---|
| `radiomap_6g_sweep.ipynb` | Main DeepMIMO experiment: setup, units, split, correctness checks, tuning, evaluation, paired contrasts, sensitivity variants |
| `rmbench_core.py` | Data loading and preprocessing, shared trend, train/tune/eval split, IDW and kriging, resumable executor, statistics (Wilcoxon, Holm) |
| `rmbench_nn.py` | Torch models: label-aware `GCN`, `gcn_krig` hybrid, `KGNN`, `krig_untrained` (no torch_geometric; the graph is three index tensors) |
| `rmbench_ext.py` | Layer 1: UJIIndoorLoc units, narrowband DeepMIMO target, GAT, Delaunay graphs, outage metrics, leave-one-layout-out transfer, extended executor |
| `run_ext.py` | Command-line runner for layer 1 (`smoke`, `uji`, `deepmimo`). **Exploratory**: its results are not reused in the confirmatory rounds |
| `rmbench_ext2.py` | Layer 2: robust trend, wall simulator, side information, `rk_side`, cross-fitted tuning, pre-registration, transfer, gated outage reporting |
| `rmbench_ext3.py` | Layer 3: SODIndoorLoc, `krig_side` / `kgnn_side`, building-level cluster bootstrap, equivalence margin, programmatic claim evaluation |
| `rmbench_ext4.py` | Layer 4 (round 3): meta-learned kernel prior (`kgnn_prior*`), side-aware kernel, ensembles, gradient-boosting baseline |
| `rmbench_ext5.py` | Layer 5 (round 5): covariate and regime statistics, `gp_ml` / `rbf_spline` baselines, planted-signal control, training-scale experiment, `rk_gcn2`, and (v4 block, appended by the round-5 notebook) the pointwise prior `rk_prior` with its CV gate |
| `rmbench_ext6.py` | Layer 6 (round 6, adds only): nonlinear within-map comparators, globally scaled plant with generated covariates, `rk_prior6`, learnability-gap D0, fresh-layout seed ledger, plan freeze, executor |
| `radiomap_ext_real_crosslayout.ipynb` | Round 1 (confirmatory design, run 2026-10-11) |
| `radiomap_ext_conf2.ipynb` | Round 2 (pre-specified replication, run 2026-10-12) |
| `radiomap_ext_conf3_kriging_attention.ipynb` | Round 3 (meta-learned prior, side-aware kernel, ensembles, gradient boosting) |
| `radiomap_ext_conf4_ablation.ipynb` | Round 4 (KGNN ablation). Section 2 writes `rmbench_ablate.py` (`%%writefile`); the notebook also contains the former `run_ablation.py` as cells |
| `radiomap_ext5_regime_scale.ipynb` | Round 5 (v4 shown here: positive control, frozen plan, dose-response on fresh layouts). It truncates and re-appends the v4 block of `rmbench_ext5.py`, so re-running is safe |
| `radiomap_ext6_learnability.ipynb` | Round 6 (run): D0 learnability gap, Stage 1 control, frozen plan, Stage 2 on fresh layouts |
| `rmbench_ablate.py` | Ablation layer: switchable KGNN fit (`polar`, `freeze`, `tau_zero`, `step0`) and the `kgnn_abl` dispatcher. Adds only; `rmbench_core/nn/ext/ext2/ext3` are untouched |
| `run_ablation.py` | Same ablation driver as the notebook, as a command line (`smoke`, `run`, `report`) |
| `readme_figures.py` | Regenerates every figure in this README into `figures/` from numbers transcribed from the saved notebook outputs (nothing is re-estimated) |
| `run_indoor_dense.py` | **Dense indoor runner** (adds only; imports `rmbench_core/ext/ext2/ext3`): `diagnose-uji`, `freeze`, `smoke`, `run wall|dm`, `report`. Defines the single coverage metric (`bacc_q25`), the frozen plan and a resumable executor that adds a `within5` column |
| `dense_indoor_radiomap.ipynb` | Notebook driver for the dense run (setup, UJI diagnosis, optional UJI reruns, freeze, smoke, wall, DeepMIMO, figures). Outputs are saved in the notebook; raw rows go to `results_dense/` |
| `rmbench_dmfix.py` | Carrier-frequency fix for DeepMIMO 4.0.5 (reads `ds.rt_params.frequency`; the name fallback requires `<int>p<int>`). `import` it once per process before any DeepMIMO load |
| `rmbench_city.py` | City pool builder: `build` (download, rasterise true building faces at 2 m with a 2 m roof cut, compute the 5 map features, write units + manifest) and `roles [--freeze]` (assign source / val / dev / confirm roles, hash the plan) |
| `city_manifest.csv` | Output of `rmbench_city.py`: 92 candidate cities, eligibility, units, building counts, alignment statistics, role |
| `rmbench_city.ipynb` | **Canonical city GNN runner**: setup, dev run, freeze, confirm (once), confirm2, report-only tables. States that it replaces `run_city_gnn.py`, `d0_city.py` and `rmbench_city.py` for the confirmatory part (those are still needed to *build* the pool and to run D0) |
| `d0_city.py` | D0 on the city pool: `pretrain` (two priors on source cities: full and geometry-only) and `d0` (scores them on dev cities only, prints and saves the verdict) |
| `run_city_gnn.py` | Older command-line version of the city runner (`dev`, `freeze`, `confirm`, `confirm2`). **Superseded**: its `FAMILIES` (`feat` / `gnn` / `topo`) and its Holm over all rows differ from the frozen plan (`primary`, `gnn_feat`, `gnn_vs_classical`, `feat`, `gnn_geo`, `topo`; Holm within family), so `confirm` would refuse with a plan-hash mismatch. Use the notebook |
| `rmbench_r7.ipynb` | Rounds 7, 7b and 8 (exploratory, city val and dev cities): residual GCN `rgnn`, wall graph and rasters, `rattn` runs. Section 9 (export) is the last section of the notebook but must be run **before** section 11, which imports `rmbench_r7` |
| `rmbench_r7.py` | Export of the `export`-tagged cells (sections 1-6) of `rmbench_r7.ipynb`: kNN / feature / wall graph builders, raster helpers (`build_rasters`, `blocked_length`, `load_raster`), cross-fitting, residual GCN, gated predictor `rgnn`, `r7_jobs`, executors, `leak_check7`. Needed for `n_jobs > 1`; imported by `rmbench_r8.py`. Checked identical to the notebook export on 2026-10-21 |
| `rmbench_r8.py` | Hand-written Round 8 module (not exported from a notebook): the `rattn` kernel, gated predictor, `r8_jobs`, `run_tasks8`, `leak_check8`. Adds only; needs `rmbench_r7.py` |
| `make_fake_city.py` | Synthetic city units (optionally with a planted nonlinear function of the 5 features) for plumbing and sensitivity tests of `d0_city.py` |
| `inspect_dm_cities.py` | Downloads a few candidate DeepMIMO scenarios and prints per-transmitter summary statistics (no model fitted); writes `dm_inspect.csv` |
| `PLAN_radiomapseer_DRAFT.md` | RadioMapSeer study: the plan (question, positive-result rule, data, task, arms, 24 contrasts in 5 families (the notebook now implements 30 in 6: **update this file for `cnn_feat` / `deep_feat` before the freeze**, the notebook hashes it), leak checks, forking-path disclosure, decisions to confirm). **Draft until frozen** |
| `rmbench_rms.ipynb` | RadioMapSeer study: implementation and runner (flags for dev stage, checks, freeze, confirmatory run, sensitivity analyses). Learned arms: `gnn_kim`, `gnn_kim_oldedge`, `rgnn_xmap`, `radiounet`, `cnn_feat` (added 2026-10-26). Since 2026-10-27: sections 6b (output-ReLU fix check and re-tuning of the CNN arms) and 6c (exploratory dev power check); flags `RUN_CNN_FIXCHECK`, `RETUNE_CNN_ARMS`, `RUN_DEV_FINALCHECK`. Standalone: its `%%writefile rmbench_rms.py` cell writes the module and it imports no other `rmbench_*` module. Do not edit that cell after the freeze (its sha256 is part of the plan) |
| `rmbench_rms.py` | Written by the notebook (arms, graph builder with the Kim edge rule, GATv2 net, training and tuning, resumable executors, statistics, freeze). Commit it to `main` at the freeze; until then it exists only in the notebook and on branch `rms-run` |
| `PLAN_radiomapseer_ADDENDUM_2026-10-27.md` | Draft addendum to the plan: 30 tests in 6 families, the ReLU defect and fix, the work order before the freeze, pre-specified rules (stride, status of `deep_feat`). Merge into the plan file before the freeze |
| `relu_synth_test.py` | CPU mechanism test of the final-ReLU defect on a synthetic signed target (needs `torch`, `torchvision`, a clone of `RonLevie/RadioUNet`) |
| `KAGGLE_RUNBOOK_rms.md` | How to run the study on Kaggle in resumable stages (account, token, dataset, bootstrap cell, stage table, troubleshooting) |
| `kaggle_sync.py` | Pushes the small result files (json / csv / jsonl / sha256 / md / png, models under 50 MB) from a Kaggle session to branch `rms-run` of the repo and restores them in the next session (`restore`, `push`, `start_background`). Needs a Kaggle Secret `GITHUB_TOKEN`. **Known bug**: see [Known issues](#known-issues-found-while-running-the-radiomapseer-study) |
| `diagnose_scene.py`, `diagnose_raster.py` | Geometry-only diagnostics that justified the true-face rasterisation and the 2 m roof cut (bounding-box share inside buildings; horizontal-face heights; ray-tracer settings). Write `scene_check_<slug>.png` and `raster_check_<slug>.png` |

`rmbench_rms.py` is the exception: it is standalone and imports none of the layers below. The layers import each other (`ext3` imports `ext2`, which imports `ext`; `ext6` imports `ext5`; `r8` imports `r7`, which imports `core` and `ext5`), so **all files must sit together** and none may be saved over another. The extension notebooks run a preflight check that stops with an explanation if a file was overwritten.

## Methods

All methods predict the **same residual** after a shared trend fitted on the training labels only, so no method gets a better trend than another.

| Name | Description |
|---|---|
| `trend` | Trend only, no spatial term |
| `idw` | Inverse-distance weighting; power and neighbourhood chosen by leave-one-out on the labeled points |
| `krig_global` | Ordinary kriging with a variogram fitted on the labeled points |
| `krig_local` | The same, using the K nearest labels |
| `krig_untrained` | KGNN's local kriging with its fixed default covariance and no learning |
| `kgnn` | Kriging-initialised message-passing model: each target is a kriging estimate over its K nearest labels, with a **learned** covariance (exponential in space plus a polar term around the transmitter, and a nugget). Step 0 (plain kriging) is a candidate for early stopping, so it can only move away from kriging if that helps on held-out labels |
| `gcn` | Label-aware GCN: geometric features, observed residual, mask and multi-hop label diffusion; trained by hiding random subsets of the labels |
| `gcn_krig` | The same GCN predicting a correction to a labeled-points-only IDW baseline. The output layer is zero-initialised and step 0 (pure IDW) is an early-stopping candidate |
| `gcn_knn`, `gcn_del`, `gat_knn`, `gat_del` | Extension GNNs. **Graph**: kNN (k=8) or Delaunay triangulation (edges longer than 4x the median are pruned). **Architecture**: GCN or edge-aware multi-head GAT. Same label-aware inputs and episodic masked-label training as `gcn`. `gat_del_idw` is the IDW-anchored hybrid |
| `*_tuned` | Same model after random-search tuning on units disjoint from the evaluation units |
| `*_side` | The same model with extra covariates (wall count along the transmitter-receiver segment, LOS flag) |
| `rk_side` | Regression-kriging comparator: ridge on **the same covariates the GNN sees**, then IDW of the remaining residual. Exists so that a side-information GNN is not compared with a method that cannot use side information |
| `krig_side`, `kgnn_side` | `rk_side` with the leftover residual interpolated by global kriging / KGNN instead of IDW. Each contrast differs from the next in exactly one factor |
| `kgnn_prior`, `kgnn_prior_sk`, `kgnn_priorft` | KGNN whose covariance parameters come from a **meta-learned prior** fitted on other layouts and applied zero-shot. `_sk` uses a side-aware kernel; `_priorft` fine-tunes the prior on the target labels |
| `ens_geo`, `ens_side`, `ens_sk` | Ensembles. `ens_side` = `kgnn_side` + `rk_side`; `ens_sk` = `kgnn_prior_sk` + `rk_side`. `ens_geo` combines the label-aware geometric models (composition: see the round-3 notebook) |
| `gb_geo` | Gradient boosting on coordinates and geometric features (no side information) |
| `gp_ml`, `rbf_spline` | Extra classical baselines added in round 5: Gaussian process with marginal-likelihood hyper-parameters, and an RBF-spline interpolator. (`coord_mlp`, `np_prior`, `rk_gcn` are also in `rmbench_ext5.py`; see the file) |
| `rk_gcn2`, `rk_gcn2_s0` | Round 5, v3: the exact `rk_side` prediction plus a learned correction; a trained checkpoint replaces step 0 only if paired validation loss improves. `_s0` returns `rk_side` itself |
| `rk_prior_K*`, `rk_prior_K*_ng` | Round 5, v4: a pointwise MLP prior pre-trained on fully observed **source** layouts (inputs = model features, no coordinates, no labels of the target map); `rk_side` is applied to `res − prior(X)` and the prior is added back. The gated arm picks prior vs plain `rk_side` by 5-fold CV inside the labelled set (accept only if paired z > 2.5 and relative MSE gain > 1%); `_ng` is ungated. `K` = number of source layouts |
| `rk_onehot`, `rk_spline`, `rk_gbm` (+ `_g`) | Round 6: regression-kriging with a **nonlinear within-map** mean (quantile-bin one-hot + ridge, additive cubic splines + ridge, gradient boosting with CV-chosen size), IDW on the leftover residual. `_g` = gated against `rk_side` by the same rule as the prior |
| `prior6_K*`, `prior6_K*_ng` | Round 6: the round-5 prior with inputs = geometry plus **globally scaled** covariates |
| `xfer_*`, `xferft_*`, `within_*` | GNN pre-trained on **other** layouts and applied zero-shot / with 30 fine-tune steps, versus pre-trained on the same layout (but different floors) as control |
| `rgnn_knn`, `rgnn_feat`, `rgnn_wall` (+ `_ungated`) | Round 7: `rk_side` + λ·c(x). c is a 2-layer GraphSAGE-style GCN (hidden 32, 150 steps, zero-initialised head, so step 0 is `rk_side`) on a graph over every grid cell of the unit: kNN (k = 8), kNN in [position/ℓ, β·z-scored map features], or kNN on distance + β·blocked length from a building raster. Trained on 5-fold **cross-fitted** `rk_side` residuals of the labelled points with masked-label episodes. λ ∈ {0, 0.25, 0.5, 1} by 5-fold CV of the whole pipeline; λ > 0 only if paired z > 2.5, relative gain > 1% and at least 30 labels. `_ungated` fixes λ = 1 |
| `rattn_geo`, `rattn_side`, `rattn_wall` (+ `_ng`) | Round 8: `trend + ridge + Σ w_qj r_j`, a softmax-weighted sum of the ridge residuals at the K = 16 nearest **labelled** points (the residuals `rk_side` feeds to IDW). Logits = −p₀·log(d/ℓ), with p₀ the IDW power from LOO, plus an MLP of edge features (last layer zero-initialised) and a learned null slot that shrinks to 0. Edge features: log distance and neighbour rank (`geo`); + query and query-minus-source map features (`side`); + blocked length and blocked fraction of the segment from the raster (`wall`). Output is linear in the residuals. Trained leave-one-out on the labelled set (Huber loss, 200 steps, 20% of queries held out for early stopping when n ≥ 30). Gate: `rk_side` + λ·(`rattn` − `rk_side`), λ ∈ {0, 0.5, 1}, same rule as `rgnn`. `_ng` = ungated (λ = 1) |
| `gnn_kim` | RadioMapSeer study. **Cross-map** graph model after Tania and Kim (Asilomar 2025): a node per outdoor pixel in a 64x64 tile; edge iff distance <= sqrt(r2) (r2 = 18, i.e. 3*sqrt(2) cells, tuned), **no building on the segment** and **both nodes share the same LOS/NLOS state** w.r.t. the transmitter; Gaussian distance weight, self loops. GATv2 layers (4 heads; width and depth tuned, default 64 / 3; 15 217 parameters at the default). Inputs per node: scaled residual label, label mask and the 6 label-free features (`log_dist_tx`, `los`, `blocked_len`, `n_crossings`, `cover_30m`, `dist_nearest_bld`). Predicts the residual after the per-map trend; trained with MSE on unlabelled outdoor pixels, with the label density drawn log-uniformly in [0.3%, 6%] per training example; prediction on non-overlapping 64x64 tiles. One model for all maps |
| `gnn_kim_oldedge` | Same as `gnn_kim` but the edge rule has **no LOS/NLOS-state condition** (topology ablation) |
| `rgnn_xmap` | `rk_side` prediction plus a correction from the `gnn_kim` graph model (one extra input channel = the `rk_side` residual; zero-initialised head, so step 0 equals `rk_side`), trained across maps on `rk_side` residuals. The cross-map version of the Round-7 residual model; **no per-map gate** is needed because the correction is trained across maps. 15 345 parameters at the default |
| `radiounet` (RadioMapSeer study) | The public `RadioWNet(phase="firstU")` from `RonLevie/RadioUNet` (the first U-net only; 10 928 711 parameters), wrapped to take 4 channels (building map, Tx position, sparse residual, label mask) and trained with the same pool, label-density sampling and loss as the GNNs. Receives **no** engineered features. Full 256x256 maps, batch 4. **Since 2026-10-27 the final ReLU of the public network is removed** (item 23); results from before that date are void |
| `cnn_feat` (RadioMapSeer study) | Added 2026-10-26 (decision B). The same `RadioWNet(phase="firstU")` with the six label-free features (`log_dist_tx`, `los`, `blocked_len`, `n_crossings`, `cover_30m`, `dist_nearest_bld`) as **6 extra input channels** (10 in total; 10 932 401 parameters). Same pool, label-density sampling, loss and batch size as `radiounet`. A unit test checks that gradients reach the feature channels, that the input channels match the batch builder and, since 2026-10-27, that the output can be negative (final ReLU removed, as for `radiounet`) |

### Mechanisms added in the extension rounds

- **Real Wi-Fi units (UJI, SOD).** One unit = one access point on one floor. AP positions are unknown, so the transmitter is **localised from the labeled points only** (soft-L1 fit of a log-distance model from four starting points). A point counts as detected if at least a set fraction of its captures saw the AP, and undetected points are dropped, so coverage statements are conditional on detection. Layout = building, parent = building-floor.
- **Robust trend (`trend_policy="robust"`).** Three candidate trends (constant, fixed exponent 2, fitted exponent) are compared by K-fold cross-validation on the labeled points, with the transmitter re-localised inside every training fold. The simplest candidate within one standard error of the best is used, and predictions are clipped to the label range. `"legacy"` reproduces the earlier behaviour for before/after tables.
- **Data-only feasibility rule.** How many floors survive is decided from counts (APs detected at enough reference points per floor), never from model results. This is what lifted UJI from 2 usable floors to 12 (setting: densities 5/10/20, 20 test points).
- **Wall simulator (`wall`).** Random buildings (size, interior walls with doors, wall loss, path-loss exponent and shadowing drawn per layout), 6 transmitters each. The only family where layouts are independent draws by construction. Side information is nearly deterministic here, so it says whether a regime exists where side information helps a GNN, not how real buildings behave.
- **Cross-fitted tuning.** Units are split into 2 folds by parent (by layout for `wall`). Hyper-parameters for fold *f* are tuned on the other fold with a separate test-set salt, then frozen. No evaluation unit is ever tuned on. Trial 0 is the library default.
- **Two splits.** `random` (test points interleaved with labels) and `blocked` (test set is one contiguous region, buffer 0). They are pre-registered as separate families and shown side by side. Blocked is the harder, more honest setting.
- **Gated outage metrics.** Outage F1 and accuracy are reported only where IDW has skill ≥ 0.2, classes need ≥ 8 test points, and an absolute dBm threshold is kept only if it splits every layout into 10–90%. Quantile thresholds (from the training labels) are always defined.
- **Pre-registration.** The contrast lists, thresholds, densities, replicate ids, test salt, tuning protocol and claim rule are written to a hashed JSON file (`prereg.json`, `prereg_conf2.json`) before any confirmatory run. Re-running with an identical plan is a no-op, and a different plan raises. Anything designed after seeing earlier output is listed in the file itself.
- **Round-2 statistics.** Parent-level paired tests with Holm correction and bootstrap CI, a **2% equivalence margin** (needs ≥ 8 parents), and a **building-level cluster bootstrap** (needs ≥ 5 buildings), because floors of one building are not independent. The claim rule: a contrast is SUPPORTED only if it is Holm-significant in the claimed direction at **every density in every split** and the building-level CI excludes 0.
- **City pool (M3).** DeepMIMO `city_*_3p5` scenarios; the map is the scene's true horizontal building faces (roof z >= 2 m) rasterised at 2 m; five label-free features per receiver (obstruction edges, blocked length, maximum obstruction height along the transmitter-receiver ray, building cover within 30 m, distance to the nearest building). Eligibility uses geometry and path *existence* only (never a power value or a model). Roles (source / val / dev / confirm / confirm2) are assigned by a fixed rule and frozen; the unit of inference is the city. Details in the pipeline section.
- **Cross-map training (RadioMapSeer study).** One model per learned arm is trained on a pool of (map, transmitter) pairs from the 501 training maps (8 transmitters per map, 4 008 pairs, chosen by a seeded draw) and applied to unseen maps; the label set of every training example is a fresh random draw. Splits are by **building map**, never by transmitter (`R.assert_disjoint`). Early selection uses 12 val maps, tuning uses all 100 val maps, the test maps are first read by the confirmatory run. All arms predict the same residual after a per-map trend fitted on that map's labels only. Label encoding differs from the plan's "-1 if missing": because the target is a signed residual with mean near 0, a -1 sentinel would collide with real values, so the learned arms get a (scaled residual, mask) channel pair instead.

## Design rules

- **Unit = one parent.** Replicates are averaged inside a unit, units inside a parent, and tests are paired across parents.
- **Tuning and evaluation units are disjoint.**
- **Label-free features only** (`geo`, `geo_los`, `geo_side`). Nothing is computed from the label's own channel.
- **Power** (DeepMIMO) is the incoherent sum over real paths. Points below the noise floor are dropped.
- **Pre-specified contrasts, a cap on the number of tests, Holm correction** within each (dataset, split, contrast list) family. Every contrast is printed, wins and losses.
- **Resumable JSONL executor.** A failing job never kills a task, and finished tasks are skipped on rerun.
- **Leak test** for every method (Section 3 of the main notebook). A FAIL invalidates that method's column.

## Setup

```bash
pip install numpy scipy pandas matplotlib torch deepmimo scikit-learn
```

The city study additionally needs the DeepMIMO city scenarios (50-220 MB each; the server rate-limits bursts, `rmbench_city.py build` retries with back-off) and was run with `uv run`; `import rmbench_dmfix` must come before any DeepMIMO load.

- DeepMIMO scenarios `i2_28b`, `i1_2p5` (and `i3_60`, unverified slug) must be available locally or downloadable.
- UJIIndoorLoc downloads from UCI into `data/ujiindoorloc`. SODIndoorLoc downloads from GitHub (`bijingxue/SODIndoorLoc`) into `data/sodindoorloc`. Both need network access on first run.
- RadioMapSeer study: Kaggle dataset `radiomapseer` (png/ and gain/ folders; link in the RadioUNet README), a GPU session (the code uses one GPU; Tesla T4 was used), Internet On (it clones `RonLevie/RadioUNet` and `kaggle_sync.py` pushes to GitHub), and `pip install numpy scipy pandas scikit-learn pillow matplotlib torch torchvision`. Everything else: `KAGGLE_RUNBOOK_rms.md`.
- Main notebook: `RUN_NEURAL = False` runs only the classical methods (no torch). `USE_SYNTHETIC = True` is an offline smoke test.
- Extension notebooks: `SMOKE = True` runs the plumbing on fake data. It says nothing about science.

## Running

1. Restart the kernel (modules are imported once, in Section 1).
2. Run a notebook top to bottom. In the main notebook run Section 3 (correctness checks) before trusting any number. In the extension notebooks, commit the pre-registration file to git **before** the confirmatory cells.
3. Timings on a 16-core machine (15 workers): main sweep about 9 min tuning plus about 9 min evaluation. Round 2 evaluates in minutes per family because it uses defaults only. Round 1 is the slow one (tuning plus pre-training); lower `pretrain_steps` for a first pass.

**Cache gotcha.** `run_tasks` skips any task id already present in the JSONL file, and a task id covers all of its jobs. If you add a method to the job list and reuse an old cache file, the new method silently never runs. Use a new file name and prefix whenever you change the job list or the code of an existing method (the main notebook uses `tune2.jsonl` / `eval2.jsonl`).

## Pipeline: exact run order, files and outputs

Everything runs from one folder that contains all `rmbench_*.py` modules. Restart the kernel before each notebook. `results_*` folders are created by the code; names below are exactly those written by the scripts and notebooks. A `{...}` is filled in at run time.

### A. Classical + graph benchmark (earlier rounds)

| Step | Run | Reads | Writes |
|---|---|---|---|
| Main sweep | `radiomap_6g_sweep.ipynb` | DeepMIMO `i1_2p5`, `i2_28b` | `tune2.jsonl`, `eval2.jsonl`, `eval2_{variant}.jsonl` (sensitivity variants); `tune.json` / `tune2.json` / `eval2.json` (tuned hyper-parameters, evaluation summaries) |
| Round 1 | `radiomap_ext_real_crosslayout.ipynb` | UJI, wall simulator, DeepMIMO | `results_ext2/prereg.json`, `dev_trend.jsonl`, `dev_trend.json`, `confirmatory_master.csv`, `confirmatory_master_transfer.csv`; per family and metric `{family}_{metric}.csv`, outage `{family}_{c}.csv`; tuning `tune_{tag}_f{f}_trial_scores.csv`, `tuned_{tag}.json` |
| Round 2 | `radiomap_ext_conf2.ipynb` | UJI (fresh APs), SOD, wall, DeepMIMO; reads `results_ext2/` (never writes it) | `results_conf2/prereg_conf2.json`, `confirmatory_master_conf2.csv`, `claims_conf2.csv`, `rmse_by_density.png` |
| Round 3 | `radiomap_ext_conf3_kriging_attention.ipynb` | UJI, SOD, wall | `results_conf3/prereg_conf3.json`, `confirmatory_master_conf3.csv`, `claims_conf3.csv`, `regime.png`, caches `cache_uji/`, `cache_sod/`; per family `{family}_rmse.csv`, `{fam}_rmse_table.csv`, `{fam}_layout_level.csv` |
| Round 4 | `radiomap_ext_conf4_ablation.ipynb` (or `run_ablation.py`; both use `rmbench_ablate.py`) | `results_conf3/` caches (read only) | `results_ablation/prereg_ablation.json`, `abl_{t}_{split}.jsonl`; `abl_primary_cells.csv`, `abl_primary_per_building_{s}.csv`, `abl_step0_share.csv`, `abl_{t}_{s}_rmse_table.csv`, `abl_{t}_{s}_layout_level.csv`, `abl_fig1_contrasts_{fam}.png`, `abl_fig2_rmse_vs_labels.png`, `abl_fig3_share_left_step0.png` |
| Round 5 | `radiomap_ext5_regime_scale.ipynb` | wall simulator (pool seeds 11 / 12 / 21) | `results_ext5/prereg_ext5_amend_v3.json`, `prereg_ext5_amend_v4.json`, `D_plant_claim_cells.csv`, `figures/v4_dose_response.png`; `results_ext5/v4/`: `dev_plant.jsonl`, `dev_plant.json`, `confirm_{tag}.jsonl`, `confirm_{tag}.json`, `confirm_contrasts.csv`, `confirm_claim_cells.csv`; priors `v4/dev/a0.3_K{K}.pkl`, `v4/confirm/{tag}_K{K}.pkl` |
| Round 6 | `radiomap_ext6_learnability.ipynb` | same caches (pool seed 31, simulator seed 9 for the fresh layouts) | `results_ext5/prereg_ext6.json`, `results_ext5/seed_ledger.json`; `results_ext5/v6/`: `D0.csv`, `dev_plant.jsonl`, `dev_plant.json`, `confirm_a{0.0,0.1,0.2,0.3,0.5}.jsonl` and `.json`, `confirm_contrasts.csv`, `confirm_claim_cells.csv`; priors `v6/priors/a{amp}_K{K}.pkl` |
| UJI / DeepMIMO exploratory layer | `python run_ext.py smoke` / `uji [--blocked]` / `deepmimo [--blocked]` | UJI (downloaded to `data/ujiindoorloc`), DeepMIMO | see section C. `smoke` writes only to `ext_smoke/` |

### B. City study (the M3 question on ray-traced city layouts)

Run in this order. Steps 1-3 build the pool; step 4 is the gate; steps 5-8 are the confirmatory path. Do not change the job list, the contrasts or the config between steps 6 and 7.

| # | Command | What it does | Writes |
|---|---|---|---|
| 1 | `uv run rmbench_city.py build` (`--first 24` for a test) | Downloads each `city_<n>_<name>_3p5`, builds one unit per transmitter (true horizontal building faces with roof z >= 2 m, rasterised at 2 m; 5 label-free features), checks that scene and receiver grid line up. Resumable; failed downloads are retried with back-off | `rmbench_cache_city/units/*.npz`, `city_manifest.csv` |
| 2 | `uv run rmbench_city.py roles` (then `roles --freeze`) | Roles by group position p mod 8: 0-3 source, 4 val, 5 dev, 6-7 confirm (same-city aliases stay together). After the freeze, roles are read back, never recomputed; cities that become eligible later get role `confirm2` | `city_manifest.csv` (role column), `results_city/prereg_city.json` |
| 3 | (optional) `diagnose_scene.py`, `diagnose_raster.py`, `inspect_dm_cities.py` | Geometry-only checks of the pool | `scene_check_<slug>.png`, `raster_check_<slug>.png`, `dm_inspect.csv` |
| 4 | `uv run d0_city.py pretrain` then `uv run d0_city.py d0 [--n-units 24]` | Trains the full (geometry + 5 features) and geometry-only priors on the 30 source cities (validation on the 6 val cities); scores them on the **dev** cities only against five within-map arms | `results_city/city_full_K30.pkl`, `city_geo_K30.pkl`, `d0_city.csv`, `d0_city_verdict.json` |
| 5 | `rmbench_city.ipynb` section 3 (`RUN_DEV = True`) | **Exploratory** dev run, 7 cities, 13 jobs; nothing here may change an arm | `results_city/city_gnn_dev.jsonl`, `city_gnn_dev_contrasts.csv` |
| 6 | `rmbench_city.ipynb` section 4 (`FREEZE = True`) | Hashes jobs, families, config and the confirm-city list; refuses if any family cannot reach p_holm < 0.05 with the available number of cities | `results_city/prereg_city_gnn.json` (sha256 `8e48c85156c3d677...`, frozen 2026-10-19 10:33:47) |
| 7 | `rmbench_city.ipynb` section 5 (`RUN_CONFIRM = True`) | The single confirmatory run: 18 units in 10 cities, 30 / 100 / 300 labels, 5 replicates, 13 jobs = 270 tasks, 3 510 job rows, 0 failures (1 528 s with 8 workers) | `results_city/city_gnn_confirm.jsonl`, `city_gnn_confirm_contrasts.csv` |
| 8 | `rmbench_city.ipynb` section 6 (`RUN_CONFIRM2 = True`) | Replication on `confirm2` cities. **The pool is empty** (0 cities / 0 units: five downloads failed and no later city became eligible) | `results_city/city_gnn_confirm2.jsonl`, `city_gnn_confirm2_contrasts.csv` (not produced) |
| 9 | `rmbench_city.ipynb` section 7 | Report-only tables and a per-city RMSE table from any of the `.jsonl` files (no new fits) | none |

Pool numbers (from `city_manifest.csv` and the notebook): 92 candidate scenarios, **53 eligible** (>= 1 transmitter with >= 1000 points above the noise floor, >= 10 building objects, <= 10% of signal-bearing receivers on a building cell and the path-less share exceeding it by >= 0.25). 39 are ineligible: 34 failed the alignment test or have too few buildings (e.g. cities 84, 85, 88 have 2-6 building objects) and 5 failed to download (`city_83_tempe`, `city_89_nairobi`, `city_90_nanjing`, `city_91_xiangyang`, `city_93_rangoon`: "Scenario folder not found"). Roles: source 30 cities / 51 units, val 6 / 10, dev 7 / 12, confirm 10 / 18, confirm2 0 / 0.

Jobs (13, all library defaults, 150 training steps for every neural arm, no tuning): `trend`, `idw`, `krig_global`, `kgnn`, `rk_spline`, `rk_gbm`, `rk_side`, `gcn_knn`, `gcn_del`, `gat_knn`, `gat_del`, `gcn_knn_side`, `gat_del_side`. Unit of inference = the city. Gains are relative RMSE, A vs B, positive means A is better; CI = bootstrap over cities; p = exact sign-flip test over cities; **Holm within each family**; verdict labels come from the CI (`A better` / `A worse` / `equivalent` within +/-2% / `inconclusive`), significance from `p_holm`.

Pre-declared families (from the frozen plan): `primary` (`gcn_knn_side` and `gat_del_side` vs `rk_side`), `gnn_feat` (side vs no-side GNN), `gnn_vs_classical` (side GNNs vs `idw` and `krig_global`), `feat` (`rk_side` vs `idw` / `krig_global`; `rk_gbm` vs `rk_side`), `gnn_geo` (`gcn_knn` and `gat_del` vs `idw`), `topo` (Delaunay vs kNN; GAT vs GCN). **Pre-declared positive GNN result:** a side-feature GNN beats `rk_side` with verdict `A better` and p_holm < 0.05 in the `primary` family at >= 1 density. With 10 cities the smallest attainable p_holm is 0.0117 (`primary`, `gnn_feat`, `gnn_geo`), 0.0176 (`feat`) and 0.0234 (`gnn_vs_classical`, `topo`).

### C. UJI exploratory layer (`run_ext.py`)

| Command | Writes (in `results_ext_uji/`) |
|---|---|
| `python run_ext.py uji [--blocked] [--n-jobs 8]` | `uji_grid[_blocked].jsonl` (job rows); `uji_grid[_blocked]_rmse.csv`, `_mae.csv`, `_contrasts_rmse.csv`; outage metrics `_f1_q10.csv`, `_f1_q25.csv`, `_f1_abs85.csv`, `_bacc_*`, `_acc_*`, `_area_*` (suffixes `q10`, `q25`, `abs85`) |
| (same command, unless `--no-lolo`) | `uji_lolo[_blocked].jsonl` and the same CSV set, plus `pretrained_uji_lolo[_blocked]/` (pre-trained GNN checkpoints) |
| `python run_ext.py deepmimo [--blocked]` | `results_ext_dm/dm_grid[_blocked].jsonl` and the same CSV set |

The runner is resumable (finished task ids are skipped). UJI configuration: densities 15 / 30 / 60, 40 test points, 5 replicates.

### D. Round 7 / 7b / 8 (city val and dev cities; exploratory)

All runs use `CITY_CFG` (densities 30 / 100 / 300, 5 replicates unless stated, random split), library-default settings, `n_jobs = 1`, and **only val and dev cities** (`d0_city.pool()["val"]`, `["dev"]`); no confirm city is read. No preregistration file was written for these rounds. Task prefixes are new for every step (`r7smoke`, `r7dev`, `r7tune`, `r7wall`, `r8val`, `r8dev`), so the cache gotcha above does not bite; a step that prints "N done, 0 to run" reused an earlier `.jsonl`.

| # | Run | Reads | Writes |
|---|---|---|---|
| 1 | `rmbench_r7.ipynb` sections 1-6, then **section 9** (export) | none | `rmbench_r7.py` |
| 2 | Sections 7-8: smoke test on 2 dev units (100 labels, 1 replicate) and `leak_check7` | city pool | `results_city/r7_smoke.jsonl` |
| 3 | Section 8: dev run, 12 units in 7 cities, 6 jobs (`rgnn_knn`, `rgnn_feat`, ungated twins, `rk_side`, `idw`), 180 tasks | city pool | `results_city/r7_dev.jsonl` |
| 4 | Section 10a: `build_rasters` for the 13 val + dev scenes (resumable, needs network and the original build code) | DeepMIMO | `rmbench_cache_city/rasters/<slug>.npz` |
| 5 | Section 10b: edge-overlap check, kNN vs wall graph, 6 dev units | rasters | none |
| 6 | Section 10c: choose β ∈ {1, 3, 10} on the 10 val units (100 labels, 2 replicates, ungated arms) | rasters | `results_city/r7_tune_val.jsonl` |
| 7 | Section 10d-e: dev run of the wall arms with the selected β, 180 tasks; report | rasters | `results_city/r7_dev_wall.jsonl` |
| 8 | Section 11: `leak_check8`, then 11a val sanity (10 val units, 100 labels, 2 replicates, 20 tasks) | `rmbench_r8.py`, rasters | `results_city/r8_val.jsonl` |
| 9 | Section 11b-c: Round 8 dev run (6 jobs, 180 tasks, about 7.7 min) and report | rasters | `results_city/r8_dev.jsonl` |

Hyper-parameters. Round 7 (`r7_jobs`): k = 8, hidden 32, 2 layers, dropout 0.1, lr 1e-2, 150 steps, gate z = 2.5 / rel = 1% / min_labels = 30. Round 8 (`r8_jobs`): K = 16, hidden 16, 200 steps, lr 1e-2, weight decay 1e-3, same gate; fixed a priori, nothing tuned on dev. β for the wall graph was chosen on the val cities only.

### E. Dense indoor run (`dense_indoor_radiomap.ipynb` / `run_indoor_dense.py`)

| Step | Command / cell | Writes |
|---|---|---|
| UJI diagnosis | `python run_indoor_dense.py diagnose-uji` | prints per-building row / failure counts of `results_ext_uji/*.jsonl` and the number of UJI units that could carry 60 / 100 / 150 / 200 labels |
| Freeze | `python run_indoor_dense.py freeze` | `results_dense/prereg_dense.json` and `.sha256` (refuses to overwrite; `run` refuses to start without it). Commit before running |
| Smoke | `python run_indoor_dense.py smoke` | `dense_smoke/` (4 layouts, 15 steps; plumbing only) |
| Wall run | `python run_indoor_dense.py run wall` | `results_dense/dense_wall.jsonl`; `dense_wall_{rmse,bacc_q25,acc_q25,within5,area_q25,...}.csv`; `dense_wall_contrasts_{rmse,bacc_q25}.csv` |
| DeepMIMO run | `python run_indoor_dense.py run dm` | `results_dense/dense_dm.jsonl` and the same CSV set |
| Figures, summary | last two notebook cells | `results_dense/figures/*.png`, `results_dense/summary_dense.csv` |

Configuration: densities 50 / 100 / 200 / 300, 5 replicates, 200 test points, random split, test salt `dense1`, robust trend, library-default neural settings (150 steps, no tuning). Wall simulator: 12 layouts x 6 transmitters x 1000 points (simulator seed 7). DeepMIMO: `i1_2p5` (24 units kept), `i2_28b` (1), `i3_60` (2). Methods: `trend`, `idw`, `krig_global`, `kgnn`, `gcn_knn`, `gcn_del`, `gat_knn`, `gat_del` (+ `rk_side`, `gcn_knn_side`, `gat_del_side` on the wall simulator only). Cache gotcha as above: change `TAG_SUFFIX` if the job list changes.

### F. RadioMapSeer cross-map study (`rmbench_rms.ipynb`, run on Kaggle in stages)

Status: stages A-C done (smoke test not recorded here), stage D (checks), E (freeze) and F (confirmatory run) **not done**. The notebook writes `rmbench_rms.py` and everything under `results_rms/` (`results_rms_smoke/` in smoke mode). Flags are in the config cell; every flag not listed for a stage is False. Long stages use *Save Version -> Save & Run All*; the geometry cache is rebuilt in every new session (scratch space, about 18-28 min on the T4 machine) and `results_rms/cache_real` must be a symlink to scratch so `/kaggle/working` stays small. Full instructions, bootstrap cell and troubleshooting: `KAGGLE_RUNBOOK_rms.md`.

| Stage | Flags set | Writes / done when | Status |
|---|---|---|---|
| A. Smoke | `SMOKE = True` | synthetic maps, tiny budgets, goes through freeze and confirm into `results_rms_smoke/`; plumbing only | not recorded here |
| B. Data check, geometry, compute projection | `RUN_VERIFY_DATA`, `RUN_GEOMETRY`, `RUN_BENCH` | png counts and loader asserts pass; section 4 prints `tune_h`, `final_train_h`, `eval_h` per arm | done |
| B2. Decide the budget | edit `R.make_cfg(...)` | any reduction is part of the frozen plan | **kept at the plan's defaults so far** |
| C. Tune | + `RUN_TUNE_CLASSICAL`, `RUN_TUNE_LEARNED` | `classical_hp.json`, `tune_classical_trials.csv`, `learned_hp.json`, `tune_<arm>_trials.csv` for all 5 learned arms (`cnn_feat` added 2026-10-26) | done (one pass of 4 arms, about 10 h on one T4, plus 74 min for the classical grid; `cnn_feat` added 120 min, about 12 h in total; the two CNN arms were redone after the ReLU fix, see C2) |
| C2. Output-ReLU fix (2026-10-27) | `RUN_CNN_FIXCHECK`, then `RETUNE_CNN_ARMS` | `relu_fix_status.json` says `retuned` for `radiounet` and `cnn_feat`; old trial files archived as `tune_<arm>_trials_preReluFix.csv`; the freeze cell checks this | **done 2026-10-28** (fix check 5.59 -> 2.97 dB; re-tuning 115 + 124 min; item 24) |
| C3. Dev power check (exploratory, optional) | `RUN_DEV_FINALCHECK` | `dev_finalcheck_rows.csv`, `dev_finalcheck_contrasts.csv`; prints the pre-specified stride 32 vs 64 decision | **done 2026-10-28** (item 24; stride 64 kept) |
| D. Checks | `RUN_GEOMETRY`, `RUN_TUNE_CLASSICAL`, `RUN_CHECKS` | `checks_rms.json` with `"passed": true` (checks 4 and 5 are an operationalisation of two ambiguous plan sentences; read notebook section 8 first) | **not run** |
| E. Freeze | + `FREEZE`, `I_CONFIRM_SECTION_11` | `prereg_rms.json` + `.sha256`; then `KS.push(...)` and commit the files, the sha256 and `rmbench_rms.py` to git **before** F | **not done** |
| F. Confirmatory run (once) | + `RUN_CONFIRM` | `confirm_classical.jsonl`, `confirm_learned.jsonl`, `confirm_models_info.csv`, `confirm_contrasts.csv`, `confirm_rmse_table.csv`, `models/*.pt`; repeat the commit until it prints `confirmatory run finished` | **not done** |
| G. Sensitivity (optional) | `RUN_SENS_BLOCKED`, `RUN_SENS_IRT`, `RUN_SCALE_CHECK` | `sens_*` files; IRT needs the IRT folders in the dataset; the city-cluster bootstrap needs a `city_of_map.csv` (not available) | not done |

Task definition used by the code (plan sections 3-6): labels are a fraction alpha in {0.5%, 2%, 5%} of outdoor (non-building) pixels drawn uniformly at random, 5 independent draws per (map, transmitter, alpha); test points are all remaining outdoor pixels; 4 test transmitters per test map; unit of inference = test map; gains are relative RMSE of A vs B (positive = A better), sign-flip p-value (Monte-Carlo, 200 000 flips), bootstrap CI over maps, `equivalent` = CI inside +/-2%, **Holm within each family**. Families (30 tests since 2026-10-26; 24 in the original draft): `primary` (`gnn_kim`, `rgnn_xmap` vs `rk_side`; 6), `vs_classical` (both vs `idw`; 6), `feat` (`rk_side` vs `idw`; 3), `topo` (`gnn_kim` vs `gnn_kim_oldedge`; 3), `deep` (`gnn_kim`, `rgnn_xmap` vs `radiounet`; two-sided, 5% non-inferiority margin; 6; **descriptive** since decision B), `deep_feat` (`gnn_kim`, `rgnn_xmap` vs `cnn_feat`; same rule; 6). **Pre-declared positive GNN result:** at least one of `gnn_kim`, `rgnn_xmap` beats `rk_side` with p_holm < 0.05, relative gain >= 2% and a bootstrap CI above 0 at **at least 2 of the 3 densities**, and is not significantly worse at the third. If it is not met, the statement is "no". (This rule depends on the `primary` family only, so it is unchanged by the 30-test update.) The coverage metric `bacc_q25` is reported as exploratory (it is not in the 30-test table).

#### Deviations from the plan (all also listed in the notebook and, after the freeze, in the frozen file)

1. Split **501 / 100 / 99** (RadioUNet loader, seed 42) instead of the plan's 550 / 50 / 100; the plan says to use the loader's.
2. Label encoding (scaled residual, mask) instead of -1 sentinel (see Mechanisms).
3. `radiounet` = the public first U-net with 4 input channels, trained with this pipeline. `cnn_feat` (added 2026-10-26 after the dev check) is the same network with the six label-free features as extra channels; the `deep` family is descriptive and the family `deep_feat` is new (30 tests in 6 families). Both CNN arms use the public network **without its final ReLU** (fix of 2026-10-27, item 23) and were re-tuned after it.
4. GNN inference on non-overlapping 64x64 tiles (`stride = 64`), so tile borders see truncated neighbourhoods; `stride = 32` would average overlapping tiles at 4x the cost.
5. Sign-flip p-values are Monte-Carlo (smallest attainable p about 5e-6); the freeze step checks that each family can reach p_holm < 0.05 (all six can, min p_holm 1.5e-5 to 3e-5 with 99 maps).
6. `krig_global` uses all labels when there are at most 1 500 and the 16 nearest otherwise.
7. Normalisation uses fixed constants (no statistic is computed from data). The pixel-to-dB mapping (`db_lo = -147`, `db_hi = -47.84`) is **not in the RadioUNet code** and is unverified; it rescales dB figures only, not gains, p-values or verdicts.
8. Classical tuning is coordinate-wise on the val maps; learned arms get 30 random configurations each (trial 0 = default).
9. 4 test transmitters per test map and 5 label draws are the author's choices.
10. Not implemented: the BRATLab replication dataset and the city-cluster bootstrap (no `city_of_map.csv`).

#### Known issues found while running the RadioMapSeer study

| Where | Problem | Fix |
|---|---|---|
| `kaggle_sync.py`, `_ensure_clone` | The clone is shallow and single-branch, so `git fetch origin` does not bring in `rms-run` after the first push. Every later `push` and the next session's `restore` fail with `'origin/rms-run' is not a commit`. | Replace `_git("fetch", "origin", check=False)` by `_git("fetch", "--depth", "1", "origin", f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", check=False)`. After this patch the background sync pushed normally (commit `5a0ef5cca2` carries `classical_hp.json` and `tune_classical_trials.csv`). **Commit the same patch to `main`**, because the bootstrap cell copies `kaggle_sync.py` from `main` in every new session. In the 2026-10-26 session the failure message still appeared during the geometry and classical-tuning cells (the session's copy was unpatched); after the patch and `importlib.reload(KS)` the sync worked (`rms-run @ 9fb9d51876`, end of session, 2 changed files). Verify that the patch is on `main`. |
| `rmbench_rms.ipynb`, section 2 (`verify_real_data`) | `d.shape.iloc[0][0]` returns the DataFrame's `(rows, cols)` tuple, not the `shape` column (`AttributeError: 'tuple' object has no attribute 'iloc'`). The check itself is unchanged by the fix. | Use `d["shape"].iloc[0][0]`. The working notebook has it; make sure the copy in the repo does too. |
| `KAGGLE_RUNBOOK_rms.md`, bootstrap cell | The dataset glob searches only two levels below `/kaggle/input`; newer Kaggle mounts are deeper, so the cell raises "RadioMapSeer not found". | `cands = glob.glob("/kaggle/input/**/png/buildings_complete", recursive=True)`. |
| Running the notebook with *Run All* | Every cell runs, so after section 5 the notebook moves straight on to section 6 if `RUN_TUNE_LEARNED = True`. Section 6 for four arms is hours long. | Set the flags first; use *Save & Run All* for long stages. With all later flags False the remaining cells are quick no-ops. |
| Kaggle session type | An interactive session with a GPU attached counts GPU quota while running CPU-only stages (geometry, classical tuning). | Use a CPU session for stage D-style CPU work where possible, and *Save & Run All* for GPU stages. |

## Results

### Main sweep: DeepMIMO `i1_2p5` (run 2026-10-11)

48 evaluation transmitter maps, 5 replicates per unit, 30 / 100 / 300 labels, 1000 test points. The trend alone has an RMSE of 0.15 dB, and all methods sit between 0.05 and 0.11 dB. Percentages are the relative RMSE gain of the first method over the second (positive means the first is better). ✔ = significant after Holm correction over 33 tests.

| Comparison | 30 labels | 100 labels | 300 labels |
|---|---|---|---|
| `kgnn` vs `idw` | +6.5% ✔ | +7.3% ✔ | +5.7% ✔ |
| `kgnn` vs `krig_global` | +1.3% | +1.4% | +3.2% ✔ |
| `kgnn` vs `krig_untrained` | +0.6% | +0.6% ✔ | +0.9% ✔ |
| `gcn_krig` vs `idw` | −2.5% ✔ | −0.5% | −0.2% |
| `gcn_krig` vs `krig_global` | −8.2% ✔ | −6.8% ✔ | −2.9% ✔ |
| `gcn` vs `idw` | −13.0% ✔ | −9.5% ✔ | −9.3% ✔ |
| `gcn` vs `krig_global` | −19.3% ✔ | −16.4% ✔ | −12.2% ✔ |
| `krig_global` vs `idw` | +5.3% ✔ | +5.9% ✔ | +2.6% ✔ |

The KGNN gain over `krig_global` is 0.0010–0.0018 dB, and over `idw` 0.0033–0.0063 dB. `i2_28b` has a single unit, so it is descriptive only (at 300 labels KGNN has the lowest RMSE: about 0.54 dB vs 0.59 for IDW and 0.64 for global kriging).

**Sensitivity variants** (automatic, Section 7c; declared in code before any variant result existed; `*` = significant after Holm within each variant):

| Contrast | primary | `trend_fspl` (exponent fixed at 2) | `blocked` (contiguous test region) | `channel` (channel-power target) |
|---|---|---|---|---|
| `kgnn` vs `idw` | +6.5* / +7.3* / +5.7* | +6.7* / +7.2* / +5.7* | +5.2 / +4.7* / +3.6* | −4.2* / +4.0* / +3.5* |
| `kgnn` vs `krig_global` | +1.3 / +1.4 / +3.2* | +1.3 / +1.2 / +3.1* | −0.5 / −3.9 / −9.6 | −4.9* / +2.6* / +4.1* |
| `gcn_krig` vs `idw` | −2.5* / −0.5 / −0.2 | −2.3* / −0.5 / −0.3 | −0.4 / −0.1 / +0.0 | −1.9* / −1.1* / −0.5* |
| `gcn_krig` vs `krig_global` | −8.2* / −6.8* / −2.9* | −8.3* / −7.1* / −3.1* | −6.4 / −9.2 / −13.6 | −2.5* / −2.5* / +0.0 |
| `gcn` vs `idw` | −13.0* / −9.5* / −9.3* | −16.3* / −11.3* / −10.1* | +0.7 / −0.2 / −1.5 | −5.7* / −3.9* / −3.5* |

`trend_fspl` reproduces the primary results almost exactly. KGNN beats IDW in 10 of 12 cells, and beats global kriging significantly in 4 of 12 (under `blocked` the sign reverses, none significant). No GCN variant beats IDW or global kriging in any cell. The 12 cells are not independent, so the counts describe consistency, not 12 confirmations.

### Round 1: more layouts, tuning, side information, transfer (run 2026-10-11)

**Data.** UJI: 96 units from 12 building-floors in 3 buildings (the old setting kept only 2 of 13 floors). Wall simulator: 72 units in 12 layouts. DeepMIMO path-sum: 27 units (24 from `i1_2p5`; `i2_28b` and `i3_60` are descriptive only). Relative RMSE gain (%) of A over B, `*` = Holm-significant, random vs blocked:

| Family | Contrast | random (low / mid / high density) | blocked |
|---|---|---|---|
| UJI (5/10/20 labels) | `kgnn_tuned` vs `idw` | +4.2* / +2.7 / +1.3 | +3.0* / +2.2 / +0.6 |
| | `gat_del_tuned` vs `idw` | −2.3 / −7.3* / −6.1* | −3.4 / −5.8 / −1.9 |
| | `gcn_knn_tuned` vs `idw` | −3.2 / −4.1 / −3.3 | −8.6 / −3.6 / +0.2 |
| | `krig_global` vs `idw` | −4.0 / −0.7 / −0.7 | −4.5 / −0.8 / −1.4 |
| Wall (20/50/100) | `kgnn_tuned` vs `idw` | +6.0* / +3.4* / +2.7* | +4.7 / +3.6 / +2.6 |
| | `gat_del_tuned` vs `idw` | −10.2* / −8.2* / −9.0* | −1.8 / −0.8 / +2.1 |
| | `gcn_knn_tuned` vs `idw` | −10.3* / −8.7* / −9.1* | −4.2 / −0.7 / +1.3 |
| DeepMIMO (30/100/300) | `kgnn_tuned` vs `idw` | +7.1* / +6.9* / +5.8* | +3.3* / +4.3* / +2.9* |
| | `gat_del_tuned` vs `idw` | −10.2* / −7.4* / −6.6* | −5.4 / +5.2 / −0.6 |
| | `gcn_knn_tuned` vs `idw` | −13.8* / −10.3* / −9.8* | −7.8 / −0.0 / −3.2 |

![GNNs vs IDW](figures/fig4_gnn_vs_idw.png)

*Figure 4. No GNN variant is significantly better than IDW in any cell.*

- **Tuning barely matters.** Tuned and default KGNN do not differ significantly anywhere (|difference| ≤ 0.3%). Tuned GCN/GAT gain at most about 2%, never significantly. The tuned GNNs still lose to IDW, so the earlier "GNNs may be under-tuned" worry does not explain their deficit.
- **Topology and architecture are second-order.** GAT vs GCN and Delaunay vs kNN differ by 0–2%, mostly not significant.
- **Side information (wall simulator, random split).** `rk_side` beats `idw` by 22–23% (RMSE 4.87 vs 6.27 dB at 20 labels). Side information also helps GNNs (`gat_del_side` vs `gat_del` +13–15%, `gcn_knn_side` vs `gcn_knn` +5–7%) and `gat_del_side` beats `idw` by 4–7%, but both side GNNs lose to `rk_side` by 19–34%, significantly. On DeepMIMO, `rk_side` is within ±1.5% of IDW at 100 and 300 labels, and side GNNs again lose to it by 10–18%.
- **Transfer.** On UJI (leave-one-building-out, within-building control trained on other floors of the same building), `xfer_*` is not better than IDW (−1% / −5% / −9%, significantly worse at 20 labels), and is worse than scratch-trained GNNs at 20 labels. On the wall simulator `xfer_*_side` beats IDW (+5% to +25%, Holm-significant) but loses to `rk_side` (random split: −8% to −30%; blocked: not significant). Fine-tuning for 30 steps gives lower RMSE descriptively; see the saved tables.
- **Trend repair (exploratory).** On 12 floors, 5 labels: trend-only RMSE 60.8 → 11.7 dB, IDW 13.8 → 10.9 dB, kriging 13.8 → 11.2 dB. None of these is significant after Holm at 12 floors, but the direction is consistent. At 5 labels the CV picks the constant trend in all 72 cases, so a fitted distance exponent was not trusted.
- **Predictability.** Noise fraction (nugget share of the residual variogram) is ≈0.03 for DeepMIMO path-sum, 0.28–0.52 for UJI, and 0.01–0.15 for the wall simulator. The `i1_2p5` target is almost deterministic, which is why dB effects there are in the thousandths. Absolute-dBm outage thresholds did not split `i1_2p5` at all (0% below every candidate) and were dropped; quantile thresholds were used.

### Round 2: pre-specified replication on fresh data (run 2026-10-12)

The hypotheses were chosen after round 1, but the **data** are fresh: new UJI APs (89 units, none overlapping round 1), **SODIndoorLoc** (50 units, 5 building-floors in 3 buildings), 16 new wall layouts (96 units), a new test salt and new replicate ids. DeepMIMO `dm_path` reuses the same transmitters and is never used for a claim. Methods are library defaults only (no tuning, no GCN/GAT arms). Real data are pooled (`real_wifi`: UJI + SOD, 17 building-floors, **6 buildings**, densities 5/10/20).

Relative RMSE gain (%) of A over B, `*` = Holm-significant at parent level:

| Family | Contrast | random | blocked |
|---|---|---|---|
| `real_wifi` (5/10/20) | `kgnn` vs `krig_global` | +8.7* / +3.1* / +2.2* | +8.3* / +2.4* / +1.7 |
| | `kgnn` vs `idw` | +3.6* / +2.7* / +2.3* | +3.3 / +1.7 / +1.6 |
| | `krig_global` vs `idw` | −5.6* / −0.4 / +0.1 | −5.4 / −0.7 / −0.1 |
| `wall` (20/50/100) | `kgnn` vs `krig_global` | +2.7* / +1.4* / +1.1* | +3.6* / +3.3* / +2.2 |
| | `kgnn` vs `idw` | +4.4* / +3.4* / +2.8* | +3.0* / +2.4* / +2.2* |
| | `kgnn_side` vs `rk_side` | −0.5* / +0.5 / +1.3* | −0.6 / −0.1 / +0.2 |
| | `kgnn_side` vs `krig_side` | −0.5* / −0.0 / +0.4 | −0.5 / −0.3 / +0.1 |
| | `kgnn_side` vs `kgnn` | +14.4* / +18.1* / +18.5* | +16.2 / +20.2* / +21.0* |
| `dm_path` (30/100/300; descriptive) | `kgnn` vs `idw` | +5.6* / +6.5* / +5.6* | +3.4* / +3.7* / +3.4* |
| | `kgnn` vs `krig_global` | +1.1 / +0.7 / +2.5 | +1.6 / −0.8 / −6.3 |

Absolute size on real data: UJI RMSE at 5 labels is 8.88 dB (`kgnn`), 9.25 (`idw`), 9.67 (`krig_global`), so the KGNN gain is about 0.4 dB over IDW and 0.8 dB over kriging, shrinking to about 0.1 dB at 20 labels.

**Building-level view.** KGNN beats global kriging in all 6 buildings at 5 labels and in 5–6 of 6 at 10 and 20 labels (per-building gain −0.1% to +11.6% for random, −0.5% to +19.1% for blocked). The building-level cluster-bootstrap CI excludes 0 at all three densities in both splits. With 6 buildings the smallest possible exact sign-test p-value is 0.031, so this supports consistency and effect size, not strong significance.

**Programmatic claim evaluation** (exact rule from the pre-registration):

| Family | Contrast | Primary | Status | Cells better / worse / equivalent / open (of 6) |
|---|---|---|---|---|
| `real_wifi` | `kgnn` vs `krig_global` | H4 | NOT SUPPORTED | 5 / 0 / 0 / 1 |
| `wall` | `kgnn` vs `krig_global` | H4 | NOT SUPPORTED | 5 / 0 / 0 / 1 |
| `wall` | `kgnn_side` vs `rk_side` | H5 | NOT SUPPORTED | 1 / 1 / 4 / 0 |
| `wall` | `kgnn` vs `idw` | no | **SUPPORTED: A better** | 6 / 0 / 0 / 0 |
| `real_wifi` | `kgnn` vs `idw` | no | NOT SUPPORTED | 3 / 0 / 0 / 3 |
| `wall` | `kgnn_side` vs `kgnn` | no | NOT SUPPORTED | 5 / 0 / 0 / 1 |

"NOT SUPPORTED" with 5 of 6 cells better means the direction is consistent but one cell (the blocked, highest-density cell) misses Holm significance; it is not evidence against the effect. The honest reading of H5 is that `kgnn_side`, `krig_side` and `rk_side` are practically equivalent: once the covariates are regressed out, the choice of interpolator for the leftover residual does not matter.

![Per-building KGNN vs kriging](figures/fig3_per_building_kgnn_vs_kriging.png)

*Figure 3. Per-building gain over global kriging, round 2, random split.*

**Do round-1 effect sizes replicate?** (exploratory) `kgnn` vs `idw` is stable (UJI +4.5 / +2.2 / +1.5 → +4.0 / +2.1 / +1.5 random; wall +5.8 / +3.3 / +2.7 → +4.4 / +3.4 / +2.8; DeepMIMO within ±1.5 points). `krig_global` vs `idw` is **not** stable (wall blocked +2.8 / +3.1 / +3.1 → −0.7 / −1.0 / −0.0, UJI at 20 labels flips sign), so kriging's edge over IDW should not be relied on.

### Round 3: meta-learned prior, side-aware kernel, gradient boosting (real Wi-Fi and wall simulator)

Round 3 was designed after rounds 1 and 2. Under your frozen claim rule all primary contrasts (H6 and H7) came out **NOT SUPPORTED**, and they must be reported as such. <!-- TODO: add the H6/H7 contrast definitions, the cells better/worse/equivalent/open counts, the unit counts per family, seed, test salt and prereg hash from the round-3 notebook; they were not available when this README was updated. -->

**Mean RMSE (dB), parent-level, real Wi-Fi.** Bold = lowest in the column.

| Family | Labels | `trend` | `idw` | `krig_global` | `kgnn` | `kgnn_prior` | `gb_geo` |
|---|---|---|---|---|---|---|---|
| UJI, random | 5 | 10.478 | 9.622 | 9.730 | 9.310 | **9.150** | 12.753 |
| | 10 | 8.265 | 7.610 | 7.789 | **7.454** | 7.511 | 9.009 |
| | 20 | 7.137 | 6.718 | 6.761 | **6.710** | 6.733 | 8.016 |
| UJI, blocked | 5 | 10.835 | 10.428 | 10.479 | 10.124 | **9.957** | 12.362 |
| | 10 | 9.955 | 9.449 | 9.585 | **9.330** | 9.331 | 9.894 |
| | 20 | 9.238 | 8.694 | 8.805 | **8.388** | 8.701 | 8.891 |
| SOD, random | 5 | 11.880 | 9.570 | 10.402 | **9.300** | 9.489 | 12.468 |
| | 10 | 8.686 | 7.326 | 7.436 | **6.989** | 7.543 | 7.705 |
| | 20 | 7.367 | 6.061 | 6.051 | **5.810** | 6.483 | 6.177 |
| | 60 | 6.643 | 4.986 | 4.959 | 4.825 | 5.423 | **4.762** |

Relative gain (%) computed from these means (descriptive, **no paired test**):

| Family | Labels | `kgnn` vs `idw` | `kgnn` vs `krig_global` | `kgnn_prior` vs `idw` | `kgnn_prior` vs `kgnn` | `gb_geo` vs `idw` |
|---|---|---|---|---|---|---|
| UJI, random | 5 / 10 / 20 | +3.2 / +2.0 / +0.1 | +4.3 / +4.3 / +0.8 | +4.9 / +1.3 / −0.2 | +1.7 / −0.8 / −0.3 | −32.5 / −18.4 / −19.3 |
| UJI, blocked | 5 / 10 / 20 | +2.9 / +1.3 / +3.5 | +3.4 / +2.7 / +4.7 | +4.5 / +1.2 / −0.1 | +1.6 / −0.0 / −3.7 | −18.5 / −4.7 / −2.3 |
| SOD, random | 5 / 10 / 20 / 60 | +2.8 / +4.6 / +4.1 / +3.2 | +10.6 / +6.0 / +4.0 / +2.7 | +0.8 / −3.0 / −7.0 / −8.8 | −2.0 / −7.9 / −11.6 / −12.4 | −30.3 / −5.2 / −1.9 / +4.5 |

- **KGNN holds up on fresh real data.** It is ahead of IDW and of global kriging in every real-Wi-Fi cell above, but the UJI random gain at 20 labels is 0.1%, i.e. nothing.
- **The prior is a low-label tool.** It is the lowest-RMSE method at 5 labels on UJI (random and blocked), then loses to from-scratch KGNN. On SOD it is worse than IDW from 10 labels on. Per building (Figure 5) it is worse than scratch KGNN in most cells, with SOD b1 and b3 the worst (−11% to −16% at 20 labels).
- **Gradient boosting** is far worse at low density. It only wins at 60 labels on SOD (+4.5% over IDW, +1.3% over KGNN). Its tuning status is not documented here; see [Limitations](#limitations).
- **Wall simulator.** Nothing reliably beats `rk_side` (Figure 6). `ens_side` is within +1.5% and `kgnn_prior_sk` is up to 2.5% worse; all differences sit inside or at the edge of the ±2% equivalence margin.
- **Cost.** `kgnn_prior` costs 0.006 s per prediction job against 0.697 s for `kgnn` (Figure 7), which is its practical argument.

![Prior vs scratch KGNN per building](figures/fig5_prior_vs_scratch_per_building.png)

![Wall simulator side information](figures/fig6_wall_side_information.png)

![Cost per prediction](figures/fig7_cost.png)

### Round 4: how much of KGNN is learning? (real Wi-Fi ablation, run 2026-10-13)

**Design.** Seven arms share Kmax = 12, `s2_init` = 0.3 and common random numbers (the seed comes from the labelled indices, not the arm name, so paired differences carry no seed noise): `abl_full` (stock KGNN), `abl_untr_polar` (0 steps, polar term on), `abl_untr_nopolar` (0 steps, polar off; identical to `krig_untrained`), `abl_nopolar`, `abl_nonug_fixed` (nugget frozen at 0.2), `abl_nonug_zero` (nugget ~0) and `abl_nostep0` (untrained checkpoint not a validation candidate). Data: the round-3 real Wi-Fi units (UJI 66 units / 11 building-floors / 3 buildings; SOD 47 units / 5 building-floors / 3 buildings; pooled 16 parents, 6 layouts), densities 5/10/20 (SOD also 60, not pooled), random and blocked splits, test salt `abl4`, replicate ids 700+, 5 replicates, defaults only, no failed jobs. The plan is frozen in `results_ablation/prereg_ablation.json`. **Primary:** `abl_full` vs `abl_untr_polar`, pooled. Holm per (family, split) call over 36 tests.

**Checks.** `krig_untrained` and `abl_untr_nopolar` agree exactly (max |ΔRMSE| = 0.00 dB in all 6 family-split tables), and `kgnn` vs `abl_full` differ by at most 0.005 dB on average (seed only).

**Primary (pooled real Wi-Fi), relative RMSE gain of learned over untrained polar kriging:**

| Split | Labels | gain | ΔRMSE (dB) | Holm p | layout CI > 0 |
|---|---|---|---|---|---|
| random | 5 / 10 / 20 | −0.9% / −0.2% / +1.0% | −0.08 / −0.01 / +0.07 | 1.00 / 1.00 / 0.19 | no / no / yes |
| blocked | 5 / 10 / 20 | −0.4% / −1.1% / −0.2% | −0.05 / −0.11 / −0.02 | 1.00 / 0.81 / 1.00 | no / no / no |

Frozen rule outcome: **NOT ESTABLISHED**. Five of six cells have a parent-level CI inside ±2%; blocked at 10 labels does not, so EQUIVALENT is not declared either. No cell is significantly worse.

**What does carry the gain (UJI, random split, relative RMSE gain of A over B, `*` = Holm-significant; from `figures/abl_uji_random_*`; the pooled and SOD tables are in `figures/` and have not been transcribed here):**

| Contrast | 5 labels | 10 labels | 20 labels |
|---|---|---|---|
| untrained polar kriging vs `krig_global` | +5.8%* | +2.2% | −1.1% |
| untrained polar kriging vs `idw` | +2.7% | +2.8%* | −0.2% |
| polar term, no learning (`abl_untr_polar` vs `abl_untr_nopolar`) | +0.1% | −0.1% | −0.2% |
| nugget at all (`abl_full` vs `abl_nonug_zero`) | −0.4% | +2.1% | +4.1%* |
| learned nugget vs fixed 0.2 (`abl_full` vs `abl_nonug_fixed`) | −1.1% | −0.4% | +0.4% |
| learned polar term (`abl_full` vs `abl_nopolar`) | +1.5%* | +0.2% | −0.1% |
| step-0 candidate (`abl_full` vs `abl_nostep0`) | 0.0% | 0.0% | −0.1% |

RMSE (dB, UJI random): IDW 9.070 / 7.606 / 6.825, untrained polar kriging 8.826 / 7.389 / 6.837, `abl_full` 8.864 / 7.385 / 6.750.

**Reading.** The advantage over IDW and global kriging at 5–10 labels is already there with **no training**. The covariance structure (local neighbourhood, nugget) is what matters; the polar term is neutral untrained and gives a small gain only when learned at 5 labels. The honest name for the method is therefore *local kriging with a fixed, sensible covariance and a nugget*. KGNN's learned parameters are optional.

**Diagnostic caveat.** In `abl_fig3_share_left_step0.png` about 70% of `abl_full` fits at 20 and 60 labels left step 0 (the validation-selected checkpoint was a trained one), so learning did move the parameters; the held-out gain was still about zero. Bars at 5 and 10 labels are empty, which is probably an artifact of an empty validation split at so few labels (`best_ep` stays 0); do not cite that figure for 5 and 10 labels until `_split_val` is checked.

### Round 5: when can learning across layouts help? Planted-signal control (v4, wall simulator)

**Why.** Rounds 1–4 found no case where a learned model beat the classical comparators by more than a few percent. Before concluding that learning cannot help, the pipeline needs a control in which it provably could. Round 5 plants `y += amp · std(y) · f(side covariates)` with the same nonlinear `f` on every map (`amp` = 0, 0.1, 0.2, 0.3, 0.5), and asks whether a learner pre-trained on other layouts recovers it. v3 (residual-GCN arms `rk_gcn2`) **failed** this control: the best CI lower bound over arms and densities was 0.009 (%, against a 2% margin), and `rk_gcn2` accepted its trained checkpoint in 0–2.5% of fits. v4 replaced the message-passing correction by the pointwise prior described under Methods. <!-- TODO: add the v1/v2 results (regime map, extra baselines, strawman diagnostics, training-scale experiment) and the v3 amendment details from the round-5 notebook outputs and prereg_ext5_amend_v3.json; they were not available when this README was updated. -->

**Data and design.** 256 source layouts (K = 16 or 64 used), 8 validation layouts, 64 development evaluation units (seed 11) and **64 fresh evaluation units** (seed 21, cache `cache_scale_fresh`, salt `scale5fresh`), 3 000 points per map, densities 100 / 300 / 1000 labels, one replicate, 300 test points. Comparators: `idw`, `rk_side`, `krig_global`, `kgnn`, `gp_ml`. The v4 plan was frozen before Stage 2 (sha256 `c47de96e…`, `results_ext5/prereg_ext5_amend_v4.json`); margin 2%, layout-cluster bootstrap, gate z > 2.5 and relative gain > 1%.

**Stage 1 (development layouts, amplitude 0.3).** The positive control is reported as PASSED, but only through the **ungated** arms: `prior_K16_ng` and `prior_K64_ng` cleared the margin against all five comparators at 100 and 300 labels (worst CI lower bound 2.0–2.6%, mean gain over the strongest comparator 3.3–4.3%). No gated arm cleared it (CI lower bounds 0.2–1.3%). The gate accepted the prior in 11–52% of fits, more often at higher density. Pre-training validation R² on held-out source layouts was +0.49 (K = 16, amplitude 0.3) and +0.60 / +0.62 at amplitude 0 (K = 16 / 64); i.e. the prior learns pointwise structure of the simulator even when nothing is planted, and the gain over `rk_side` in that case is still 0.0–0.04%.

**D0 failed.** The pre-registered recoverability diagnostic gave a median corr(prior, f) of **−0.19** (per-unit range −0.59 to +0.29) and a linear R² of `f` on the model inputs of **0.73**. `f` was meant to have ~0 linear part. The planted covariates were z-scored per map, so on every map `f` was a different function of the model inputs, and the pipeline had still passed Stage 1. Treat the positive control as informative about the gate, not as clean evidence that the prior recovers `f`. This is the reason for round 6.

**Stage 2 (fresh layouts, run once).** Mean gain (%) of the gated prior over the *strongest* of the five comparators, by planted amplitude:

| K | labels | amp 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|
| 16 | 100 | 0.03 | 0.02 | 0.42 | 2.22 | 6.16 |
| 16 | 300 | 0.00 | −0.02 | 0.47 | 2.71 | 8.09 |
| 16 | 1000 | 0.00 | 0.01 | 0.77 | 2.45 | 7.21 |
| 64 | 100 | 0.04 | 0.09 | 0.23 | 1.60 | 7.96 |
| 64 | 300 | 0.00 | −0.04 | 0.68 | 2.83 | 8.61 |
| 64 | 1000 | 0.00 | 0.04 | 0.93 | 2.98 | 7.64 |

The gated prior beats every comparator by more than the 2% margin (CI lower bound above it) in **6 of 6 cells at amplitude 0.5** (lower bounds 4.4–5.3%) and in no cell at amplitude ≤ 0.3, although the mean gain at 0.3 is 1.6–3.0%. The smallest winning amplitude is therefore the top of the grid; the crossover lies somewhere between 0.3 and 0.5 and the grid is too coarse to locate it. The gate accepted the prior in 0–3% of fits at amplitude 0 and in 38–69% at amplitude 0.5. **False-positive control (amplitude 0):** the gated prior is equivalent to `rk_side` in all 6 cells (mean gain 0.00–0.04%); the ungated arms lose 0.7–0.9% at 300 and 1000 labels, still inside the margin. K = 16 vs K = 64 makes little difference.

**How to read this.** It is a positive result for the *idea* in a simulator with a strong planted signal, with a gate that does no harm when the signal is absent. It is **not** evidence for real radio maps, and not evidence for GNNs: the winning learner is a pointwise MLP, and the v3 message-passing arms failed the same control. The comparator set contained no nonlinear within-map regressor, so part of the gain may be available within a single map. Round 6 tests exactly that.

### Round 6: learnability gap (run, `radiomap_ext6_learnability.ipynb`)

Round 6 extends the planted-signal experiment of Round 5: the planted `f` is a function of 26 covariates (the simulator's side information plus 24 generated ones) scaled by **fixed global constants** fitted on source maps only; `f` is 32 pairwise products of ridge functions with no linear part (linear R² of `f` on the covariates 0.142 on an evaluation map). Comparators: `idw`, `krig_global`, `kgnn`, `gp_ml`, `rk_side` and the nonlinear within-map regression-kriging arms `rk_onehot`, `rk_spline`, `rk_gbm`, each also CV-gated against `rk_side` (`_g`). 256 source layouts (K = 16 or 64 used), 8 validation layouts, 64 development units, and **64 fresh evaluation units** (pool seed 31, simulator seed 9; the ledger `results_ext5/seed_ledger.json` now lists pool seeds 11 / 12 / 21 / 31 and simulator seeds 5 / 7 / 9 as burned). Densities 100 / 300 / 1000, one replicate. The plan was frozen before the fresh layouts were built (`results_ext5/prereg_ext6.json`, sha256 `0141caf74b0bf8a4...`). Prior pre-training: validation R² on held-out source layouts +0.477 (K = 64, amplitude 0.3).

**Stage 1, D0 (development layouts): PASSED.** Median corr(prior, f) on held-out maps 0.548 vs 0.096 for the best within-map comparator (gap 0.452; threshold 0.25); linear R² of `f` on the model inputs 0.168. With 1000 labels the best within-map comparator reaches only 0.148, so the gap is not a label-budget artifact. **Stage 1 positive control: passed** (printed rows: worst CI lower bound 10-19% against all 11 comparators, gated and ungated arms).

**Stage 2 (fresh layouts, run once).** Mean gain (%) of the *gated* prior over the **strongest** comparator, by planted amplitude (`confirm_contrasts.csv`, `confirm_claim_cells.csv`; D0 itself is in `D0.csv`):

| K | labels | amp 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|
| 16 | 100 | 2.06 | 1.89 | 6.56 | 14.07 | 23.94 |
| 16 | 300 | 0.00 | -0.08 | 2.78 | 11.12 | 24.83 |
| 16 | 1000 | -0.08 | 0.00 | 3.95 | 11.30 | 21.41 |
| 64 | 100 | 2.56 | 4.06 | 8.42 | 17.67 | 32.26 |
| 64 | 300 | 0.20 | 0.33 | 5.85 | 18.27 | 33.64 |
| 64 | 1000 | -0.08 | 0.09 | 7.92 | 17.22 | 31.86 |

The gated prior beats **every** comparator by more than the 2% margin (cluster-CI lower bound) in **17 of 30 cells**: all 6 cells at amplitude 0.5, all 6 at 0.3, five of six at 0.2 (all but K = 16 / 300 labels); none at amplitude <= 0.1. The smallest winning amplitude is 0.2 for every (K, density) except K = 16 at 300 labels (0.3). At amplitude 0.3 (K = 64) the strongest comparator is a within-map regression-kriging arm (`rk_spline_g`, `rk_side`, `rk_gbm`), which the gated prior beats by 17.2-18.3%; IDW, global kriging, `kgnn` and `gp_ml` are beaten by 18-26%. The gate accepted the prior in 0-3% of fits at 300 / 1000 labels with no signal, 28-30% at 100 labels, and 72-100% at amplitude 0.3-0.5.

**False-positive control (amplitude 0, vs `rk_side`): the written rule failed (`gate does no harm at amp = 0: False`).**

| Arm | 100 labels | 300 labels | 1000 labels |
|---|---|---|---|
| gated, K = 16 | +2.06% [0.98, 3.23], A better | 0.00, equivalent | 0.00, equivalent |
| gated, K = 64 | +2.56% [1.40, 3.78], A better | +0.20, equivalent | 0.00, equivalent |
| ungated, K = 16 | +1.49% [-0.62, 3.44], inconclusive | -6.91%, A worse | -9.77%, A worse |
| ungated, K = 64 | +3.28% [1.09, 5.31], A better | -3.20%, A worse | -4.91%, A worse |

The frozen rule required equivalence in all cells. It failed because the gated prior beat `rk_side` at 100 labels by more than the band, not because it did harm; at amplitude 0 it never lost to `rk_side` in a gated cell. Report the rule as written. The ungated arms do lose at 300 and 1000 labels, which is what the gate is for.

**How to read this.** Semi-synthetic: the covariates beyond the simulator's own side information are generated and `f` is planted, and the winning model is a pointwise MLP prior, not message passing. It shows when cross-layout learning *can* help (shared high-dimensional structure that one map of ~100 labels cannot learn, D0), not that such structure exists in real radio maps. Do not select the best amplitude or K after the fact.

### City study (M3): GCN / GAT on ray-traced city layouts, confirmatory run (plan frozen 2026-10-19)

**Data.** DeepMIMO `city_*_3p5` scenarios at 3.5 GHz. Map = true horizontal building faces (roof >= 2 m) rasterised at 2 m; five label-free features per receiver (log1p number of obstruction edges, log1p blocked length, log1p maximum obstruction height, building cover within 30 m, log1p distance to the nearest building). No path counts or delays enter any model. 10 confirmation cities / 18 units, 30 / 100 / 300 labels (the brief's sparse and dense settings; the dense setting is 300 rather than 200+), 5 replicates, 500 random test points, library-default models.

**Mean RMSE (dB) over the 10 confirmation cities** (`city_gnn_confirm.jsonl`; units and replicates averaged within a city). Bold = lowest.

| Method | 30 | 100 | 300 |
|---|---|---|---|
| `trend` | 6.606 | 6.305 | 6.251 |
| `idw` | 5.400 | 4.143 | 3.197 |
| `krig_global` | 5.267 | 3.974 | 3.009 |
| `kgnn` | 5.223 | 3.907 | 2.903 |
| `gcn_knn` | 6.130 | 4.667 | 3.465 |
| `gcn_del` | 6.253 | 4.743 | 3.492 |
| `gat_knn` | 6.190 | 4.789 | 3.417 |
| `gat_del` | 6.113 | 4.867 | 3.438 |
| `gcn_knn_side` | 4.637 | 3.528 | 2.927 |
| `gat_del_side` | 4.572 | 3.414 | 2.846 |
| `rk_side` | 4.334 | 3.166 | 2.664 |
| `rk_spline` | **4.218** | 3.228 | 2.694 |
| `rk_gbm` | 4.434 | **3.127** | **2.635** |

**Paired contrasts** (`city_gnn_confirm_contrasts.csv`; relative RMSE gain %, positive = A better; n = 10 cities; p_holm within family; `*` = p_holm < 0.05):

| Family | A vs B | 30 | 100 | 300 |
|---|---|---|---|---|
| primary | `gcn_knn_side` vs `rk_side` | -7.5 (p 0.168) | -11.8* (0.023) | -9.8* (0.039) |
| primary | `gat_del_side` vs `rk_side` | -8.5 (0.168) | -7.9* (0.029) | -7.3 (0.082) |
| gnn_feat | `gat_del_side` vs `gat_del` | +21.6* | +27.9* | +15.9* |
| gnn_feat | `gcn_knn_side` vs `gcn_knn` | +22.8* | +23.6* | +15.0* |
| gnn_vs_classical | `gat_del_side` vs `idw` | +11.9 (0.219) | +15.6 (0.117) | +8.7 (0.223) |
| gnn_vs_classical | `gcn_knn_side` vs `idw` | +11.9 (0.223) | +13.5 (0.117) | +7.2 (0.219) |
| gnn_vs_classical | `gat_del_side` vs `krig_global` | +9.9 (0.223) | +12.5 (0.123) | +2.5 (0.223) |
| gnn_vs_classical | `gcn_knn_side` vs `krig_global` | +9.8 (0.223) | +10.4 (0.117) | +1.2 (0.375) |
| feat | `rk_side` vs `idw` | +16.8 (0.186) | +22.2* (0.035) | +15.4* (0.035) |
| feat | `rk_side` vs `krig_global` | +14.9 (0.186) | +19.4* (0.035) | +9.7 (0.082) |
| feat | `rk_gbm` vs `rk_side` | -6.3 (1.0) | +0.1 (1.0) | +0.0 (1.0) |
| gnn_geo | `gcn_knn` vs `idw` | -14.4* | -13.6* | -10.2* |
| gnn_geo | `gat_del` vs `idw` | -12.8* | -18.7* | -8.3* |
| topo | Delaunay vs kNN (GCN), (GAT); GAT vs GCN (kNN), (Delaunay) | all inconclusive, p_holm 0.77-1.0 | | |

For `gnn_feat` and `gnn_geo` the p_holm values are 0.0117 in every cell (the smallest attainable with 10 cities). The side-feature GNN advantage over IDW and kriging is not significant after Holm within its 12-test family and is in any case explained by the features: `rk_side` achieves the same benefit with a ridge fit.

**Verdict printed by the run: PRE-DECLARED POSITIVE GNN RESULT = NO.**

**Dev run (7 cities, exploratory only, `city_gnn_dev.jsonl`).** Same direction on the primary family (`gcn_knn_side` vs `rk_side` -12.1% / -4.6% / -6.5%; `gat_del_side` -11.5% / -3.7% / -1.9%), but with 7 cities the smallest attainable p_holm in that family is 0.094, so nothing there could be significant. Two dev results did **not** carry over: `rk_side` vs IDW at 100 and 300 labels (+0.7% and -5.4% with very wide CIs, vs +22% and +15% on the confirmation cities) and `rk_gbm` vs `rk_side` (+8.0% / +7.4% at 100 / 300 labels, vs about 0 on the confirmation cities). The per-city table shows why: in very low-error cities (e.g. `city_53_sankt-peterburg`, IDW 0.58-0.87 dB at 100-300 labels) `rk_side` is much worse than IDW (1.88 / 1.29 dB), so the sign of the feature gain depends on the city.

**Cross-city transfer gate, D0 (7 dev cities, 12 units; `d0_city.csv`, `d0_city_verdict.json`).** At 100 labels, median RMSE over cities: `idw` 4.410, `rk_side` 2.773, `rk_onehot` 3.163, `rk_spline` 3.103, `rk_gbm` 2.869, prior (full) + IDW of the remainder 3.307, prior (geometry only) + IDW 4.266. Against the best within-map arm per city (an oracle choice): median gain of the full prior -7.7%, positive in 3 of 7 cities (43%), full minus geometry-only +28.0 points. Pre-declared pass rule (median gain >= 2%, positive in >= 60% of cities, full minus geo >= 1 point): `pass = false`. The map features clearly matter for the prior; cross-city transfer still does not beat a within-map fit. Stage 2 must not be run. The prior beating IDW by about 25% with no target labels is an unplanned dev-only observation; do not cite it as a result.

**What this supports.** On independent ray-traced city layouts, with library-default models, (i) the five map features are worth about 1 dB at 100 labels, (ii) a regression on those features followed by IDW captures that gain, (iii) GCN / GAT do not improve on it, and (iv) GNNs that only see geometry are worse than IDW. The claim is about *untuned library-default* GNNs on these features, not about GNNs in general.

### Round 7: residual GCN on top of `rk_side` (exploratory; city dev cities)

**Design.** ŷ = `rk_side` + λ·c(x); c is a GCN trained on cross-fitted `rk_side` residuals, λ chosen by the 5-fold do-no-harm gate (see Methods, `rgnn_*`). Data: 12 dev units in 7 cities, 30 / 100 / 300 labels, 5 replicates, 500 random test points, defaults only. `leak_check7` (`rgnn_feat`, one dev unit, 100 labels) returned `True`. There is no frozen plan; the Holm correction runs over the 15 tests of one call, so with 7 cities the smallest attainable p_holm is 0.234 and no cell can be significant. Verdict labels are descriptive. Numbers are transcribed from the saved cell outputs of `rmbench_r7.ipynb`; the `.jsonl` files were not re-read.

**Mean RMSE (dB) over the 7 dev cities** (units and replicates averaged within a city):

| Method | 30 | 100 | 300 |
|---|---|---|---|
| `idw` | 5.311 | 3.739 | 2.831 |
| `rk_side` | 4.009 | 3.157 | 2.523 |
| `rgnn_knn` (gated) | 4.009 | 3.157 | 2.516 |
| `rgnn_feat` (gated) | 4.009 | 3.157 | 2.519 |
| `rgnn_knn_ungated` | 4.526 | 3.201 | 2.543 |
| `rgnn_feat_ungated` | 4.419 | 3.239 | 2.541 |

**Relative RMSE gain vs `rk_side`** (%, positive = A better; CI over cities):

| A | 30 | 100 | 300 |
|---|---|---|---|
| `rgnn_knn` (gated) | 0.00, equivalent | 0.00, equivalent | +0.33, equivalent |
| `rgnn_feat` (gated) | 0.00, equivalent | 0.00, equivalent | +0.24, equivalent |
| `rgnn_knn_ungated` | -13.0 [-19.1, -7.3], A worse | -1.9 [-5.4, 1.3], inconclusive | +0.3 [-5.0, 7.3], inconclusive |
| `rgnn_feat_ungated` | -10.5 [-16.6, -4.8], A worse | -3.0 [-5.3, -0.4], A worse | +0.6 [-4.3, 7.4], inconclusive |

`rgnn_feat` vs `rgnn_knn` (gated): 0.00 / 0.00 / -0.09, equivalent. Gate open fraction: `rgnn_feat` 0.00 / 0.00 / 0.02, `rgnn_knn` 0.00 / 0.00 / 0.05 at 30 / 100 / 300 labels.

**Reading.** The gate does its job and the GCN has nothing to offer: wherever the gate is closed the output is `rk_side`, and where it is forced open (ungated) the correction hurts at low density and is a coin flip at 300 labels. Possible reasons, **not tested here**: the IDW inside `rk_side` already extracts the spatial structure from the residuals, so the cross-fitted residuals left for the GCN are mostly noise at 30-300 labels; and on a 2 m grid a 2-layer GCN sees about 4 m, so most nodes see no labelled neighbour (the motivation for Round 8).

### Round 7b: obstruction-aware (wall) graph (exploratory)

**Does the wall graph differ from kNN?** Section 10b (6 dev units; kNN k = 8 vs `d + β·blocked_length` with β = 3 over 24 candidates):

| Unit | n | median kNN edge (m) | share of kNN edges crossing a building | mean blocked length (m) | edge overlap (kNN vs wall) |
|---|---|---|---|---|---|
| `city_5_philadelphia` #0 | 1041 | 2.24 | 0.089 | 0.125 | 0.928 |
| `city_5_philadelphia` #2 | 1253 | 2.24 | 0.162 | 0.267 | 0.903 |
| `city_13_columbus` #2 | 2126 | 2.24 | 0.043 | 0.070 | 0.943 |
| `city_53_sankt-peterburg` #0 | 2873 | 2.24 | 0.013 | 0.015 | 0.962 |
| `city_80_philadelphia` #0 | 3885 | 2.06 | 0.030 | 0.041 | 0.975 |
| `city_87_jinan` #1 | 2266 | 3.16 | 0.032 | 0.092 | 0.966 |

Almost no kNN edge crosses a building and the two edge sets overlap 90-98%. The notebook's own stopping rule for this check ("if almost no kNN edge crosses a building and the two edge sets overlap ~100%, the wall graph cannot help: stop here") is close to met; the run continued anyway.

**β on the 6 val cities** (ungated arms, 100 labels, 2 replicates; mean gain % vs `rk_side`): kNN graph -3.01, wall β = 1 -0.91, β = 3 -2.13, β = 10 -3.03. Per-city values range from -10.5 to +1.3. Every candidate loses on average; β = 1 was selected as the least bad.

**Dev run with β = 1** (7 cities, % gain vs `rk_side` at 30 / 100 / 300 labels): gated `rgnn_wall_b1` 0.00 / 0.00 / +0.58, all equivalent; ungated -7.8 [-12.7, -3.4] A worse / -2.8 [-4.9, -0.7] A worse / +2.0 [-2.5, 8.3] inconclusive. Ungated wall vs ungated kNN graph: +3.6 / -1.1 / +1.6, all inconclusive; vs ungated feature graph: +1.5 / +0.1 / +1.3, all inconclusive. Gate open fraction 0.00 / 0.00 / 0.05.

### Round 8: label-centric attention interpolator `rattn` (exploratory; city val and dev cities)

**Design.** Round 7's GNNs live on a graph over every grid cell, so with 30-300 labels most nodes see no label. `rattn` sends messages directly from the K = 16 nearest labelled points to each query: a softmax-weighted sum of ridge residuals whose logits start at the IDW kernel (power from LOO) plus a zero-initialised MLP of edge features and a null slot (see Methods, `rattn_*`). Arms: `rattn_geo` (log distance, rank: a learned isotropic kernel, the same question as Round 4's learned vs fixed kernel), `rattn_side` (+ the 5 map features of the query and query-minus-source), `rattn_wall` (+ blocked length and fraction from the raster). Hyper-parameters are fixed a priori and nothing was tuned on dev; the round was **designed after seeing Round 7 dev and val output**, and the decision to read the ungated `_ng` arms as the diagnostic came after the gated arms turned out to equal `rk_side`. `leak_check8` (`rattn_wall`, one dev unit, 100 labels) returned `True`.

**Val sanity run** (6 val cities, 100 labels, 2 replicates; gain % vs `rk_side`; gated arms are 0.00 in every city except `rattn_wall` in marrakesh, +0.11):

| City | `rattn_geo_ng` | `rattn_side_ng` | `rattn_wall_ng` |
|---|---|---|---|
| `city_12_fortworth` | +0.78 | +1.74 | +1.13 |
| `city_25_instanbul` | -2.24 | -0.27 | -0.60 |
| `city_4_phoenix` | +1.51 | +1.46 | +4.75 |
| `city_52_marrakesh` | -2.25 | -1.69 | +3.10 |
| `city_67_fujiyoshida` | +0.02 | +0.43 | +1.98 |
| `city_86_ankara` | -2.58 | -1.49 | -0.33 |
| **mean** | -0.79 | +0.03 | +1.67 |

**Dev run** (7 cities, 12 units, 30 / 100 / 300 labels, 5 replicates; gain % vs the named B, [CI over cities], verdict; p_holm in parentheses). One Holm call covers 27 tests, so with 7 cities **0.42 is the smallest attainable p_holm**, reached by every cell in which all 7 cities have the same sign; nothing here can be Holm-significant.

| A vs B | 30 | 100 | 300 |
|---|---|---|---|
| `rattn_side_ng` vs `rk_side` | +0.4 [-0.8, 1.7] equivalent (1.0) | +3.2 [0.8, 6.2] A better (0.42) | +6.2 [2.5, 10.8] A better (0.42) |
| `rattn_wall_ng` vs `rk_side` | +1.0 [0.2, 2.3] A better (1.0) | +1.6 [-1.0, 4.8] inconclusive (1.0) | +6.2 [2.0, 12.8] A better (0.42) |
| `rattn_geo_ng` vs `rk_side` | -0.9 [-1.5, -0.4] equivalent (0.59) | -0.9 [-1.6, 0.1] equivalent (1.0) | -0.2 [-0.9, 0.5] equivalent (1.0) |
| `rattn_side` (gated) vs `rk_side` | +0.07 [0.0, 0.2] equivalent | +0.39 [0.0, 1.1] equivalent | +2.9 [0.1, 7.4] A better (1.0) |
| `rattn_wall` (gated) vs `rk_side` | -0.005 equivalent | -0.05 equivalent | +1.7 [0.2, 4.0] A better (1.0) |
| `rattn_geo` (gated) vs `rk_side` | 0.00 equivalent | 0.00 equivalent | 0.00 equivalent |
| `rattn_side_ng` vs `rattn_geo_ng` | +1.3 [-0.2, 2.9] inconclusive (1.0) | +4.0 [2.1, 6.6] A better (0.42) | +6.4 [2.6, 11.1] A better (0.42) |
| `rattn_wall_ng` vs `rattn_side_ng` | +0.7 [-0.2, 1.3] equivalent (1.0) | -1.7 [-2.1, -1.3] A worse (0.42) | +0.1 [-2.5, 3.4] inconclusive (1.0) |
| `rattn_wall_ng` vs `rgnn_knn_ungated` | +11.7 [6.9, 16.3] A better (0.42) | +3.3 [0.1, 6.2] A better (1.0) | +5.8 [2.9, 9.1] A better (0.42) |

Gate open fraction (dev): `rattn_geo` 0.00 / 0.00 / 0.00, `rattn_side` 0.02 / 0.07 / 0.15, `rattn_wall` 0.02 / 0.03 / 0.07 at 30 / 100 / 300 labels (val, 100 labels: 0.00 / 0.00 / 0.05).

**Per-city dev results** (RMSE in dB for `rk_side` and `idw`; ungated gain % vs `rk_side` for the three arms; the 30-label rows are in the notebook output):

| City | labels | `rk_side` | `idw` | `rattn_geo_ng` | `rattn_side_ng` | `rattn_wall_ng` |
|---|---|---|---|---|---|---|
| `city_13_columbus` | 100 | 4.26 | 4.40 | +1.52 | +7.72 | +6.61 |
| | 300 | 3.20 | 3.24 | -1.41 | +8.02 | +3.83 |
| `city_26_dubai` | 100 | 2.44 | 3.24 | -1.44 | +0.91 | -0.73 |
| | 300 | 1.78 | 2.24 | +1.05 | +3.77 | +5.30 |
| `city_53_sankt-peterburg` | 100 | 1.88 | 0.87 | -0.45 | +10.65 | +9.44 |
| | 300 | 1.29 | 0.58 | -0.66 | +18.27 | +24.97 |
| `city_5_philadelphia` | 100 | 3.46 | 4.24 | -1.27 | +1.49 | +0.17 |
| | 300 | 2.76 | 3.09 | -1.14 | +0.27 | +0.73 |
| `city_68_reykjavik` | 100 | 4.48 | 5.56 | -0.81 | +0.14 | -2.69 |
| | 300 | 3.94 | 4.67 | -0.45 | +1.53 | +0.31 |
| `city_80_philadelphia` | 100 | 2.57 | 3.98 | -1.94 | +2.27 | +0.18 |
| | 300 | 2.12 | 2.94 | +0.73 | +7.96 | +5.37 |
| `city_87_jinan` | 100 | 3.01 | 3.88 | -1.87 | +0.53 | -0.87 |
| | 300 | 2.58 | 3.06 | +0.75 | +3.00 | +2.43 |

**Reading.**

- **What carries the gain.** A learned isotropic kernel is no better than `rk_side` (`rattn_geo_ng` -0.9 / -0.9 / -0.2%), the same finding as Round 4. Adding the 5 map features to the kernel is what moves it (`rattn_side_ng` vs `rattn_geo_ng` +4.0 / +6.4% at 100 / 300 labels); the obstruction features add nothing beyond them (`rattn_wall_ng` vs `rattn_side_ng` +0.7 / -1.7 / +0.1%). One hypothesis, untested: the map features make the smoothing non-stationary (less weight across shadow boundaries), which a stationary IDW inside `rk_side` cannot do.
- **Consistency.** `rattn_side_ng` beats `rk_side` in all 7 dev cities at 100 and 300 labels (median city gain +1.5% at 100, +3.8% at 300), but the val cities do not reproduce it at 100 labels (+0.03% mean, 3 of 6 cities positive). Two exploratory samples of 6-7 cities disagree; the true effect at 100 labels may be near zero.
- **Concentration.** `city_53_sankt-peterburg` has by far the largest gains (10.7% / 18.3%), and there `rk_side` is itself worse than IDW (1.88 vs 0.87 dB at 100 labels). After the gain `rattn_side_ng` is about 1.7 dB at 100 and 1.05 dB at 300 labels, still above IDW. Part of that gain is recovery from a bad `rk_side` fit, not a better interpolator.
- **Gate.** The gated models are `rk_side` almost everywhere (open 0-15% of runs); only the 300-label cells are positive with a CI above 0 (+2.9% side, +1.7% wall). The gate rule (z > 2.5, relative gain > 1%) is too strict for effects of this size at this label count; it was not loosened, because any change after seeing dev would be post hoc.
- **Not a GCN/GAT.** `rattn` is a one-hop attention kernel with output linear in the residuals, closer to a learned kernel smoother (like KGNN) than to message passing between layers. Do not cite it as evidence for GCN/GAT; if it is reported, name it for what it is.

**Not done yet (needed before any confirmatory claim).** The `rattn` arms have not been run on the val cities at 30 and 300 labels; no plan has been frozen; R8 has been compared only against `rk_side` and IDW (the stronger `rk_gbm` / `rk_spline`, `krig_global` and `kgnn` were run only in the city study's confirm run, not on dev here); the confirm cities have not been used for any Round 7/8 arm; rasters exist only for the 13 val + dev scenes (needed for `rattn_wall`, not for `rattn_side`); and the `confirm2` pool is still empty, so there is no replication set.

### Dense indoor run (2026-10-21)

Plan `results_dense/prereg_dense.json` (sha256 `06ca69ef...a790184`, written 2026-10-21) fixes the grid, the metrics, the contrast lists and three primary contrasts (`gcn_knn` vs `idw`, `gat_del` vs `idw`, `gat_del` vs `krig_global`). The hypotheses themselves are not new: earlier rounds already showed GNN <= IDW, so this run is a confirmation of an existing negative on a denser grid, not a fresh discovery. Smoke test: 528 rows, 0 failed. Wall run: 15 840 rows, 0 failed. DeepMIMO run: 4 320 rows, 0 failed.

**Single coverage metric.** Outage threshold = 25th percentile of the unit's TRAIN labels; primary = balanced accuracy of covered vs outage on the test points (`bacc_q25`). Plain accuracy is reported as `acc_q25` (a constant predictor already scores about 0.75), `within5` = fraction of test points within 5 dB, `area_q25` = |predicted - true outage fraction|. Cells with fewer than 8 test points in either class are NaN. In the contrast CSV for this metric the column `mean_delta_db` is on the 1 - `bacc` scale, not dB.

**Wall simulator (inference over 12 layouts; 72 units).** Parent-level mean RMSE (dB):

| Method | 50 | 100 | 200 | 300 |
|---|---|---|---|---|
| `trend` | 7.689 | 7.584 | 7.525 | 7.519 |
| `idw` | 5.392 | 4.757 | 4.253 | 3.978 |
| `krig_global` | 5.302 | 4.699 | 4.211 | 3.944 |
| `kgnn` | 5.223 | 4.629 | 4.132 | 3.850 |
| `gcn_knn` | 5.886 | 5.142 | 4.626 | 4.354 |
| `gcn_del` | 5.910 | 5.226 | 4.696 | 4.420 |
| `gat_knn` | 5.854 | 5.140 | 4.583 | 4.308 |
| `gat_del` | 5.918 | 5.196 | 4.636 | 4.312 |
| `rk_side` | **4.143** | **3.709** | **3.367** | **3.189** |
| `gcn_knn_side` | 5.412 | 4.820 | 4.367 | 4.186 |
| `gat_del_side` | 5.076 | 4.448 | 3.921 | 3.689 |

Coverage metric `bacc_q25` (higher is better): `idw` 0.838 / 0.859 / 0.873 / 0.882, `kgnn` 0.843 / 0.861 / 0.877 / 0.885, `gcn_knn` 0.819 / 0.842 / 0.863 / 0.871, `gat_del` 0.822 / 0.845 / 0.864 / 0.876, `rk_side` 0.881 / 0.893 / 0.901 / 0.903, `trend` 0.686 / 0.694 / 0.698 / 0.700.

Paired contrasts (RMSE; positive = A better; Holm within the call, 14 contrasts x 4 densities = 56 tests; 45 of 56 significant on RMSE and 29 of 56 on the coverage metric):

| A vs B | 50 | 100 | 200 | 300 | Holm-significant |
|---|---|---|---|---|---|
| `gcn_knn` vs `idw` (primary) | -9.2% | -8.1% | -8.8% | -9.5% | 4 / 4 |
| `gat_del` vs `idw` (primary) | -9.8% | -9.2% | -9.0% | -8.4% | 4 / 4 |
| `gat_del` vs `krig_global` (primary) | -11.6% | -10.6% | -10.1% | -9.3% | 4 / 4 |
| `kgnn` vs `idw` | +3.1% | +2.7% | +2.8% | +3.2% | 4 / 4 |
| `krig_global` vs `idw` | +1.7% | +1.2% | +1.0% | +0.8% | 3 / 4 (not at 300) |
| `rk_side` vs `idw` | +23.2% | +22.0% | +20.8% | +19.8% | 4 / 4 |
| `gcn_knn_side` vs `rk_side` | -30.6% | -30.0% | -29.7% | -31.3% | 4 / 4 |
| `gat_del_side` vs `rk_side` | -22.5% | -19.9% | -16.4% | -15.7% | 4 / 4 |
| `gcn_del` vs `gcn_knn` | -0.4% | -1.6% | -1.5% | -1.5% | 2 / 4 (wrong direction) |
| `gat_del` vs `gat_knn` | -1.1% | -1.1% | -1.1% | -0.1% | 1 / 4 (200; wrong direction) |
| `gat_del` vs `gcn_del` | -0.1% | +0.6% | +1.3% | +2.5% | 2 / 4 (200, 300) |
| `gat_knn` vs `gcn_knn` | +0.6% | +0.05% | +0.9% | +1.1% | 1 / 4 (300) |

Absolute gaps: `gcn_knn` vs `idw` -0.49 / -0.39 / -0.37 / -0.38 dB; `gat_del` vs `idw` -0.53 / -0.44 / -0.38 / -0.33 dB. The GAT-vs-IDW gap in dB narrows with density but the relative gap stays at -8% to -10%. On the coverage metric `gcn_knn` is below `idw` in all four cells (-0.020 to -0.010, significant) and `gat_del` in all four, significant at 50 / 100 / 200 labels but not at 300 (Holm p = 0.16). `kgnn` and `krig_global` vs `idw` on the coverage metric: no significant difference. The repeated `p_holm = 0.0273` is the floor of the exact sign-flip test with 12 parents times the 56-test cap, not a coincidence in the data.

**DeepMIMO (descriptive except `i1_2p5`).** `i1_2p5` (2.5 GHz, 100% LOS, 24 transmitters of one layout): RMSE (dB) at 50 / 100 / 200 / 300 labels: `trend` 0.160 / 0.155 / 0.152 / 0.152, `idw` 0.089 / 0.075 / 0.063 / 0.058, `krig_global` 0.083 / 0.071 / 0.061 / 0.056, `kgnn` 0.082 / 0.071 / 0.060 / 0.054, `gcn_knn` 0.101 / 0.083 / 0.069 / 0.063, `gat_del` 0.099 / 0.083 / 0.068 / 0.062. Paired over the 24 transmitters (44 tests; 28 significant on RMSE, 18 on the coverage metric): `gcn_knn` vs `idw` -14% / -11% / -10% / -10%, `gat_del` vs `idw` -12% / -10% / -8% / -8%, `gat_del` vs `krig_global` -19% / -16% / -12% / -11% (all significant); `kgnn` vs `idw` +7% / +6% / +6% / +6% (significant); no significant Delaunay-vs-kNN difference in any cell; `gat_del` vs `gcn_del` +3.5% at 300 labels only. The coverage metric sits at 0.97-0.99, near its ceiling.

`i2_28b` (28 GHz, 0% LOS, **1 transmitter**) and `i3_60` (60 GHz, 99% LOS, **2 transmitters**) have too few parents for any test. Descriptive RMSE (dB) at 50 / 300 labels: `i2_28b` `idw` 0.832 / 0.546, `kgnn` 0.847 / 0.490, `krig_global` 0.836 / 0.618, `gat_del` 0.943 / 0.596, `gcn_knn` 0.983 / 0.660, `trend` 1.413 / 1.404; `i3_60` `idw` 1.023 / 0.734, `kgnn` 0.971 / 0.670, `krig_global` 1.084 / 0.814, `gat_del` 1.204 / 0.902, `gcn_knn` 1.345 / 0.967, `trend` 1.259 / 1.261. The only thing that can be said is that the same ordering (kgnn <= idw < GNNs) appears at 28 and 60 GHz in these single-/two-transmitter cases. This is the only part of the project that touches mmWave, and it is anecdotal.

**What this run adds to the brief.** Dense (200+) measurement scenarios are now covered on the wall simulator and DeepMIMO; the coverage metric is defined once and used on both; topology (kNN vs Delaunay) and architecture (GCN vs GAT) effects are small and not in favour of the GNN. It does not add real-data evidence: real indoor data (UJI) cannot support 200+ labels (see below), and cross-layout transfer was **not** re-tested in this run.

### RadioMapSeer cross-map study (dev stage only; no confirmatory run)

**Status (2026-10-26).** Draft plan. The dev stage (sections 1-6 of `rmbench_rms.ipynb`: data verification, geometry features, unit tests, compute projection, classical tuning, learned-arm tuning, including the `cnn_feat` arm added 2026-10-26) has been run on Kaggle (Tesla T4, Python 3.13, torch 2.11, numpy 2.1) against the train and val maps only. **Not done**: leak and sanity checks (plan section 8), the freeze, the confirmatory run on the 99 test maps, the sensitivity analyses. Everything below is exploratory dev-stage output; it chooses hyper-parameters and finds bugs and is **not** evidence. Nothing in the confirmatory path may change after the freeze.

**Data verification** (training maps only; `RUN_VERIFY_DATA`):

| Item | Result |
|---|---|
| RadioUNet loader split (re-parsed from `lib/loaders.py`) | map ids 0-699 shuffled with seed 42; train / val / test = 0-500 / 501-600 / 601-699 (inclusive) = **501 / 100 / 99** maps; the plan said 550 / 50 / 100 |
| File counts | `png/buildings_complete` 701; `png/antennas` 56 080; `gain/DPM` 56 080; `gain/IRT2` 56 080; `gain/IRT4` 1 402 (701 x 80 = 56 080) |
| Images | all 256x256; building png values {0, 255}; building fraction mean 0.228 (min 0.039, max 0.399); antenna png has exactly 1 non-zero pixel; Tx pixel inside a building in 2% of sampled (map, tx); gain pixel range (0, 255); 16.8% of outdoor pixels at or below the noise floor; mean gain pixel 88.8 outdoors vs 3.2 on buildings |
| IRT files for the sampled (map, tx) | IRT2 100%, **IRT4 0%** (so the IRT4 sensitivity run cannot be done on those pairs; IRT2 can) |
| `city_of_map.csv` | not available; the city-cluster bootstrap is skipped |

**Features and tests.** Six label-free features per pixel computed from the building raster and the Tx position only, with no gain image read (`log_dist_tx`, `los`, `blocked_len`, `n_crossings`, `cover_30m`, `dist_nearest_bld`). Training pool: 4 008 (map, tx) pairs from 501 maps; geometry cache for train + val maps built in 1 110-1 655 s per session (three sessions). Edge-rule unit test passed on a toy map (with the LOS/NLOS condition: 20 378 edges, 0 LOS-NLOS edges, 0 through a building; without it: 22 806 edges, 2 428 LOS-NLOS edges, 0 through a building). The LOS flag agrees with brute force on 275 random pixels. Label sets are disjoint from test pixels and outdoor-only (random split: 1 180 labels = 2.00% of outdoor pixels, 57 833 test pixels). Family feasibility with 99 maps: all six families can reach p_holm < 0.05 (smallest attainable p_holm 1.5e-5 to 3.0e-5).

**Configuration as run in the dev stage** (`R.make_cfg`; these are the plan's defaults and become the frozen configuration unless changed before the freeze): densities 0.5% / 2% / 5%; training density log-uniform in [0.3%, 6%]; 8 training transmitters per map; 4 test transmitters per test map; 5 label draws; 3 training seeds; 30 tuning configurations per learned arm at 1 500 steps; final training 8 000 steps; checkpoint every 500 steps on 12 val maps; tuning on 100 val maps; 64x64 tiles, batch 4 (GNNs) and batch 4 (RadioUNet); `floor_mode = "clip"`.

**Compute projection** (section 4; seconds per step and per inference measured on the T4, projected to the full plan):

| arm | parameters | tune_h | final_train_h (3 seeds) | eval_h (one test pass) |
|---|---|---|---|---|
| `gnn_kim` | 15 217 | 3.2 | 1.4 | 1.2 |
| `gnn_kim_oldedge` | 15 217 | 3.2 | 1.4 | 1.3 |
| `rgnn_xmap` | 15 345 | 3.7 | 1.5 | 1.7 |
| `radiounet` | 10 928 711 | 2.1 | 1.1 | 0.1 |
| **total** | | **12.2** | **5.3** | **4.4** (21.8 h) |

Two of three timing runs gave 22.0 h and 21.8 h. A third run gave 38.2 h because the first arm's step time was an outlier (0.738 s per step for `gnn_kim` against 0.19-0.21 s for the identical-size `gnn_kim_oldedge`); treat the 22 h as the estimate and add margin for evaluation checkpoints, data loading and the 20-28 min geometry rebuild at the start of each session (realistic total about 27 h). The measured tuning time was shorter than projected: about 2.5-3 h per GNN arm and 1.9 h for `radiounet`, about **10 h in total** for the four original arms, plus 2.0 h (120.5 min) for `cnn_feat`, about 12 h. The CNN arms were tuned again after the ReLU fix (115 and 124 min), so the valid tuning total is about 12 h; the void pre-fix CNN tuning cost about 4 h more. The final-training and test-pass cost of `cnn_feat` is not benchmarked; if it is close to `radiounet`'s (1.1 h and 0.1 h) the confirmatory run grows by about 1.2 h, to roughly 23 h projected, about 28 h realistic. The projection table above predates `cnn_feat`: its final-training (3 seeds) and test-pass cost has not been benchmarked (`RUN_BENCH = False` in the last session; the expectation is that it is close to `radiounet`'s 1.1 h and 0.1 h), so the 22 h / 27 h totals are a slight underestimate until re-measured. The classical grid (49 settings x 100 val maps x 3 densities, 4 CPU workers) took 4 420 s.

**Classical arms, tuned on the 100 val maps** (RMSE in dB, one label draw per density; chosen settings: `idw` p = 2.0, K = 8; `krig_global` K = 32; `rk_side` ridge = 1.0, p = 1.5, K = 8; `rk_gbm` max_iter = 50, lr = 0.05, p = 2.0, K = 8; several sit at the edge of their grid, notably K = 8 and the smallest GBM settings):

| Arm | 0.5% | 2% | 5% |
|---|---|---|---|
| `idw` | 3.606 | 2.655 | 2.267 |
| `krig_global` | 3.529 | 2.574 | 2.191 |
| `rk_side` | **2.854** | **2.352** | 2.123 |
| `rk_gbm` | 2.999 | 2.368 | **2.085** |

**Learned arms, best of 30 configurations on the same val maps** (short tuning budget, 1 500 steps per trial; "val RMSE" is the mean over the three densities; relative gain columns are vs `rk_side` and are descriptive):

| Arm | best trial | trial-0 val RMSE | best val RMSE | 0.5% | 2% | 5% | gain vs `rk_side` (0.5 / 2 / 5%) | parameters |
|---|---|---|---|---|---|---|---|---|
| `gnn_kim` | 10 | 3.174 | 2.645 | 3.275 | 2.440 | 2.220 | -14.8% / -3.7% / -4.6% | 51 721 |
| `gnn_kim_oldedge` | 5 | 3.291 | 2.634 | 3.313 | 2.396 | 2.191 | -16.1% / -1.9% / -3.2% | 51 721 |
| `rgnn_xmap` | 23 | 2.407 | 2.365 | 2.761 | 2.261 | 2.073 | +3.3% / +3.9% / +2.4% | 14 089 |
| `radiounet` (re-tuned after the fix) | 20 | 2.991 | 2.424 | 3.007 | 2.251 | 2.014 | -5.4% / +4.3% / +5.1% | 10 928 711 |
| `cnn_feat` (re-tuned after the fix) | 1 | 3.386 | 1.901 | 2.184 | 1.818 | 1.700 | +23.5% / +22.7% / +19.9% | 10 932 401 |

(For reference vs IDW: `gnn_kim` +9.2% / +8.1% / +2.1%, `gnn_kim_oldedge` +8.1% / +9.8% / +3.4%, `rgnn_xmap` +23.4% / +14.8% / +8.6%; `rk_gbm` vs `rk_side` -5.1% / -0.7% / +1.8%.) Tuning moved `gnn_kim` from 3.174 to 2.645 dB, so its default configuration was a poor one; `rgnn_xmap` barely moved (2.407 to 2.365), which fits a model whose output is `rk_side` plus a small correction. Even so, the move is worth about half of its mean gain over `rk_side` (trial 0: 1.5% below `rk_side`'s 2.443 dB mean over the three densities; selected: 3.2%), so part of the dev-stage gain is selection on the val maps. The `radiounet` and `cnn_feat` rows are the values after the output-ReLU fix and re-tuning (item 23); the pre-fix numbers (5.4-5.6 dB, selected trials 26 and 22) are void and archived.

**RadioUNet dev check (2026-10-26; 100 val maps; not part of the plan's confirmatory path). Pre-fix and void: the network could not output negative values (item 23); the reading below is superseded.** The tuned RadioUNet scored 5.4-5.6 dB at every density, worse than IDW and barely dependent on the number of labels. Before deciding whether to keep it as specified, three things were checked:

| Check | Result |
|---|---|
| Does the public network squash its output (which would stop it predicting negative residuals)? | `grep -i 'sigmoid\|tanh\|clamp' RadioUNet/lib/modules.py` found nothing |
| Trend-only baseline (zero residual) on the val maps | 7.412 / 7.384 / 7.380 dB at 0.5% / 2% / 5% |
| Same configuration trained for 20 000 steps (51 min) | val RMSE on 12 checkpoint maps: 6.155 (500 steps), 5.668 (2 000), 5.628 (5 000), 5.569 (10 000), 5.557 (12 000), **5.549 (20 000)**; on the 100 val maps: **5.439 / 5.363 / 5.365 dB** at 0.5% / 2% / 5% (tuned 1 500-step run: 5.643 / 5.458 / 5.428) |

Training loss at the end was about 0.24-0.44 in the code's scaled units (residual / 10), i.e. about 5.5 dB, equal to the val error. **Reading.** The arm learns part of the building shadowing (7.4 to 5.5 dB vs the trend) but not to use the sparse labels (10x more labels move the error by 0.07 dB; `rk_side` moves by 0.73 dB), and more steps do not help. Not tested, so only hypotheses: it does not receive the six map features; only 4 008 training pairs used (8 of the 80 transmitters of each of the 501 training maps; 40 080 pairs exist); a single-pixel Tx and 0.5-5% labels are weak inputs for a 10.9 M-parameter U-net; the residual target is relative to a per-map trend.

**Open decisions before the freeze** (anything changed must be done *before* `FREEZE` and then is part of the frozen plan):

| Decision | Options |
|---|---|
| What to do with `radiounet` | **Decided 2026-10-26: option B** (`cnn_feat`, family `deep_feat`, 30 tests in 6 families; `radiounet` and `deep` descriptive). **2026-10-27/28:** the CNN plateau was a final-ReLU defect (item 23), fixed; both CNN arms re-tuned. `cnn_feat` beats `idw` and `rk_side` on the val maps at every density, so under the proposed rule in `PLAN_radiomapseer_ADDENDUM_2026-10-27.md` `deep_feat` **stays a two-sided non-inferiority family**. Expect it to show the GNNs worse than `cnn_feat` (item 24). Confirm or change before the freeze |
| Plan file | `PLAN_radiomapseer_DRAFT.md` still describes 24 tests in 5 families; update it from `PLAN_radiomapseer_ADDENDUM_2026-10-27.md` (`cnn_feat` / `deep_feat`, the ReLU fix, the stride rule; the notebook hashes the plan file at the freeze) |
| Compute re-projection | The section-4 projection predates `cnn_feat`; benchmark it (`RUN_BENCH`) or accept a slightly longer confirmatory run |
| Inference stride | **Decided 2026-10-28: keep 64.** The pre-specified rule asked for more than 1% gain at every density; stride 32 gave 0.27 / 0.54 / 0.44% on 40 val maps |
| Training seeds | 3 (plan) or 1 for every learned arm (saves about 3.5 h; the plan allows it if stated before the freeze) |
| Positive-result threshold, non-inferiority margin | 2 of 3 densities, >= 2% gain; 5% margin vs RadioUNet (plan section 11) |
| Checks 4 and 5 (shuffled-label, zero-label controls) | operationalised by the author; read notebook section 8 and edit before freezing if you disagree |
| BRATLab replication | not implemented |

**Forking paths so far.** The plan was designed after the negative city and dense-indoor results, Rounds 7-8 (including the unconfirmed `rattn` lead) and after reading Kim et al. and Bufort et al.; `rattn` is deliberately not included. The RadioUNet dev check was run after seeing the val numbers above, and the A / B choice was made on val output (B, 2026-10-26); `cnn_feat` was added after seeing the RadioUNet val numbers. All of this belongs in the supplement table of rounds. The val maps were used for tuning and checkpoint selection, so all val numbers above are optimistic for the tuned arms.

### UJI cross-layout runs (partial; descriptive only)

Only part of this layer was available when this README was updated, and **none of it supports an inferential claim**:

- `uji_grid.jsonl` (random split) holds 702 job rows, all `ok`, from **one building-floor** (`uji|b0f1`, 6 access-point units, layout `uji_b0`), densities 15 / 30 / 60. Mean RMSE (dB) at 15 / 30 / 60 labels: `trend` 7.25 / 6.78 / 6.56, `idw` 7.27 / 6.66 / 6.21, `krig_global` 7.26 / 6.78 / 6.40, `kgnn` 7.44 / 6.73 / 6.34, `gcn_knn` 7.43 / 7.01 / 6.57, `gat_del` 7.97 / 7.01 / 6.62. The run is resumable: rerun `python run_ext.py uji` to complete it. Its contrast, outage and MAE tables were not available.
- `uji_lolo_blocked_*` (leave-one-building-out, blocked split) covers **2 evaluation buildings** (`uji_b0`, `uji_b2`); the pooled `ALL` rows have n = 2 clusters, so the smallest attainable exact p-value is 0.5; all Holm-adjusted p-values are 1.0 and no contrast is significant (`uji_lolo_blocked_contrasts_rmse.csv`). Building `b1` does not appear in the blocked output. `python run_indoor_dense.py diagnose-uji` (2026-10-21) shows why: under the default `UJI_CFG`, building `b1` produces **no eligible unit at all** (it never reaches the experiment), so this is a data-eligibility effect, not a failed run. Only `uji_b0` (7 units, 114-153 points) and `uji_b2` (6 units, 115-137 points) yield units, one building-floor each, i.e. **13 units in 2 building-floors**. Any statement that UJI has n ~ 13 independent parents counts units, not floors.
- Pooled mean RMSE (dB) at 15 / 30 / 60 labels (`uji_lolo_blocked_rmse.csv`): `idw` 12.67 / 11.77 / 10.27, `krig_global` 12.67 / 11.84 / 10.30, `kgnn` 12.71 / 12.03 / 10.43, `gcn_knn` 15.90 / 12.10 / 10.58, `gat_del` 16.58 / 12.24 / 10.69, `trend` 27.37 / 11.77 / 10.20, pre-trained `xfer_gat_del` 31.67 / 11.95 / 10.44, `xfer_gcn_knn` 38.42 / 11.89 / 10.23, fine-tuned `xferft_gat_del` 17.41 / 11.74 / 10.28. At 30 and 60 labels every method is within about 0.5 dB of the trend alone: there is no spatial structure to exploit in this setting. Pre-trained arms are 0.2-0.35 dB (2-3%) better than scratch GNNs of the same architecture at 30 and 60 labels, descriptively (all pairs underpowered).
- **The 15-label blocked rows are probably a trend artifact, not a model result.** The trend-only RMSE in building b0 is 43.6 dB at 15 labels (IDW 14.2 dB), i.e. the fitted trend extrapolates wildly into the held-out block, and every pre-trained arm adds that trend back (51.6 and 65.6 dB). Check the trend fit at very few labels before reporting any 15-label blocked number.
- Outage metrics (`uji_lolo_blocked_f1_*`, `bacc_*`, `acc_*`, `area_*`) exist for `q10`, `q25` and `abs85` thresholds; pooled outage F1 at the 10% quantile is at most 0.107 for any method (at 15 labels), so they do not separate the methods either.
- The UJI densities (15 / 30 / 60) do not reach the brief's dense setting (200+), and they cannot: with 40 test points, only 13 units can carry 60 labels, 2 units (both `uji_b0`) can carry 100, and **none** can carry 150 or 200. The dense claim therefore rests on the wall simulator and DeepMIMO (see the dense indoor run).
- **Status of the UJI files (2026-10-21 diagnosis).** `uji_grid_blocked.jsonl` (b0: 945 rows, b2: 810 rows) and `uji_lolo_blocked.jsonl` (b0: 1050, b2: 900) are complete for the 13 units and all `ok`. `uji_grid.jsonl` (random split) covers only `uji_b0` (702 rows, 6 of 7 units) and there is no `uji_lolo.jsonl`; `python run_ext.py uji` (random split) completes it and is resumable, but with 2 building-floors it can only be descriptive.

## Limitations

- **Hypotheses were chosen after seeing round 1.** Round 2 has fresh data, not fresh hypotheses, and the plan says so. The trend policy, outage gating and dataset choices were also designed after exploratory output (listed in `prereg.json`).
- **Few independent layouts.** The main sweep and the DeepMIMO families have one building layout. Real data have 6 buildings (UJI 3, SOD 3; SOD has 3 floors in one building and one floor in each of the others). Building-level inference cannot reach strong significance, and the cluster bootstrap with 6 clusters is itself approximate.
- **Tiny effects on DeepMIMO.** The `i1_2p5` scenario is 100% line-of-sight and the path-power target is nearly deterministic (skill 0.99). KGNN's win over kriging is about 0.002 dB, and the larger channel-power residuals come from the target definition, not from its trend. On real Wi-Fi the effect is larger in dB but shrinks to 1–2% by 20 labels.
- **The kriging comparison is the strong one, and it is only partly established.** IDW is the weaker baseline. In the main sweep only the 300-label win over global kriging survived correction. In the round-1 and round-2 DeepMIMO subsets (24–27 units) it does not, and under the blocked split the sign reverses on DeepMIMO.
- **Wall simulator is the author's own.** Wall counts and LOS are almost deterministic drivers of the loss, so side information is highly informative by construction. It says nothing about real buildings. UJI and SOD have no side information.
- **Wi-Fi RSS is not 6G.** UJI and SOD are 2.4/5 GHz with censored detections and **estimated** AP positions. Nothing here says anything about mmWave or sub-THz propagation. Layout and band are also confounded across DeepMIMO scenarios (`i2_28b` at 28 GHz, `i1_2p5` at 2.5 GHz), and single-/two-transmitter scenarios (`i2_28b`, `i3_60`) stay descriptive.
- **Tuning is parent-disjoint, not layout-disjoint,** for UJI and DeepMIMO (sibling floors / transmitters of the same building can sit in the tuning fold). It is layout-disjoint for `wall`. The tuning budget is small (8 trials per neural method), though tuning made no difference.
- **Blocked split uses buffer 0:** labeled points may sit right next to the test block.
- **`gcn_krig` at 30 labels** uses only 6 labeled points for early stopping; its loss to IDW there may reflect noisy selection (not tested).
- **Not tested:** kriging with the true AP position known (SOD ships AP coordinates) and, outside the RadioMapSeer study, any deep radio-map baseline (a RadioUNet-style model; in that study the public RadioUNet arms (`radiounet`, `cnn_feat`) exist; their first dev-stage plateau of 5.4-5.5 dB came from a final-ReLU defect (item 23, fixed), and after the fix `cnn_feat` is the strongest dev-stage arm, on val maps only). Gradient boosting is covered only by `gb_geo` in round 3 (geometric features, real Wi-Fi). State its tuning budget next to any claim about it, because a 18–33% loss at low density will look under-tuned to a reviewer. The narrowband channel shape is unverified; the `i3_60` slug loads in DeepMIMO 4.0.5 (2 transmitters, 99% LOS, path-power target) as of the dense run.
- **Round-3 hypotheses were also chosen after earlier output,** and all its primaries are NOT SUPPORTED. Round-3 gains over IDW and kriging in Figure 1 and the tables above are descriptive. Do not relabel the primaries by restricting to densities or splits that look best.
- **Six buildings** remain the weakest point for real-data inference; adding buildings (Tampere, UTSIndoorLoc, MallIndoorLoc) is the most valuable next step.
- **Round-4 limits.** The data are the round-3 units (fresh test salt and replicate ids, not fresh buildings). The training-free-kriging claim comes from secondary contrasts; only the learning-vs-untrained contrast was the primary, and it was not established. Do not describe round 4 as showing that learning helps, or relabel the primary using the cells that look best (random split at 20 labels, +1.0%, Holm p = 0.19).
- **Round-5 limits.** The planted signal is artificial and its strength (amplitude) is a free knob; the only winning amplitude is the top of the grid. The D0 diagnostic failed (corr −0.19, linear R² 0.73), Stage 1 passed through the ungated arms only, and the comparator set had no nonlinear within-map model. The round-5 fresh layouts (seed 21) are no longer untouched. The v3 failure and the v4 design were chosen after seeing earlier output (listed in the prereg files).
- **Round 6 is semi-synthetic.** The generated covariates are not real side information, `f` is planted, and the D0 thresholds are a priori choices. The frozen no-harm rule failed (see Round 6). The fresh layouts of Round 6 (seeds 31 / 9) are now burned.
- **City study.** (i) The pool has one layout per city and 10 confirmation cities: the smallest attainable family-wise p-value is 0.0117, per-city results are heterogeneous (the sign of the feature gain differs across cities), and the dev cities (7) cannot reach significance at all. (ii) The `confirm2` pool is empty, so there is **no replication** of the confirmatory run. (iii) Every neural arm is a library default with 150 steps and no tuning; this was pre-declared, but the negative GNN result is therefore about default-configured GNNs. (iv) Receiver grids are dense simulated maps, random test points favour interpolators (no blocked split was run on the city pool), and the target is ray-traced path power at 3.5 GHz, not 6G bands or measurements. (v) The map rasteriser and its eligibility filter were validated on a subset of cities by geometry and path existence; 34 of the 87 scenarios that downloaded were rejected by that filter. (vi) The 2% equivalence margin is a convention of this project. (vii) Dev-run numbers are exploratory and two of them did not carry over to the confirmation cities.
- **Script vs notebook.** `run_city_gnn.py` is out of date relative to the frozen plan (different families, Holm over all rows). The confirmatory contrasts in `results_city/` come from `rmbench_city.ipynb` (Holm within each family). Do not rerun `confirm` from the script.
- **Rounds 7-8 are exploratory.** No frozen plan; 7 dev and 6 val cities, so no Holm p-value below 0.234 (Round 7 / 7b, 15 tests per call) or 0.42 (Round 8, 27 tests per call) is attainable and the verdict labels are descriptive. Round 8 was designed after seeing Round 7 dev and val output. Its hyper-parameters were fixed a priori and not tuned on dev, but promoting the ungated `_ng` arms as the diagnostic was decided after the gated arms equalled `rk_side`. Dev and val disagree for `rattn_side_ng` at 100 labels (+3.2% vs +0.03%), and per-city effects are heterogeneous (+0.1% to +10.7% at 100 labels on dev).
- **`rattn` is not exactly `rk_side` at initialisation.** Its softmax weights run over the 16 nearest labels with a null slot at logit -3, whereas `rk_side` uses IDW with a LOO-chosen neighbourhood, so `rattn_geo_ng` differs from `rk_side` by -0.2% to -0.9% before any learning. Only λ = 0 is exactly `rk_side`.
- **Gate fold size at 30 labels (Round 8).** `fit_kernel` enables the early-stopping split only when n ≥ 30. At 30 labels the 5-fold gate fits on 24 labels, so gate CV evaluates a different training procedure (200 steps, no early stopping) than the final fit (6 queries held out). The gate is closed at 30 labels in 98% of runs anyway; whether the mismatch contributes is not tested.
- **Round 7 / 8 leak checks** ran on one dev unit at 100 labels each (`rgnn_feat`, `rattn_wall`); the other arms share the code path but were not checked separately.
- **Round 7b/8 rasters** exist for the 13 val + dev scenes only; `build_rasters` is marked untested in the dev sandbox in its docstring, and its output was not validated beyond the edge-overlap check.
- **Dense run: scope.** The wall simulator is the author's own (wall count and LOS are near-deterministic drivers, so the side-information result says nothing about real buildings), DeepMIMO `i1_2p5` is one 100%-LOS room (24 transmitters, not 24 layouts; effects of 0.003-0.01 dB are statistically clear and practically negligible), and `i2_28b` / `i3_60` have 1 and 2 transmitters. There is no real dense indoor data. Inference over layouts is for the 12 wall layouts only (smallest attainable Holm p = 0.0273 with this cap).
- **Dense run: fresh layouts not verified.** `WALL_SPEC` uses simulator seed 7, the same number as the Round-5 fresh layouts (pool seed 21 / simulator seed 7, now burned). Whether the dense layouts coincide with them was **not checked**; do not call them untouched. The test salt (`dense1`) differs, so the test points differ.
- **Dense run: no tuning, no cross-layout transfer.** Every neural arm is a library default with 150 steps (as pre-declared), so the negative GNN result is about default GNNs. Transfer to unseen indoor layouts was not re-run on the dense grid.
- **Dense run: freeze order.** The plan was frozen by `run_indoor_dense.py freeze` before the runs and the runner checks its hash, but whether the file was committed to git before the run was not verified.
- **UJI.** The available grid file covers one building-floor and the blocked leave-one-building-out file covers two buildings; neither supports a significance claim. The 15-label blocked results are dominated by an unstable trend fit.
- **No real-data evidence for cross-layout learning of side-information structure.** All positive transfer results are simulated; on real Wi-Fi, pre-training helped only at 5 labels (rounds 1 and 3).
- **RadioMapSeer study (dev stage).** (i) All numbers are on the val maps that were also used for tuning and checkpoint selection, so they are optimistic; nothing has been run on the 99 test maps. (ii) Tuning used 1 500 steps per trial while the final run uses 8 000; the best configuration at 1 500 steps need not be the best at 8 000. (iii) The data are **simulated** (DPM, 5.9 GHz, fixed building height 25 m, Tx/Rx at 1.5 m), the maps come from six cities with no per-map city label available here, and the pixel-to-dB mapping is unverified (relative gains do not depend on it). (iv) 2% of sampled transmitter pixels fall inside a building. (v) The `radiounet` arm under-performed because of a final-ReLU defect found on 2026-10-27 (item 23); a result for the `deep` family would say little about RadioUNet in general. After the fix the CNN arms do use the labels (item 24), but they were tuned on the val maps with a 30-configuration search at 1 500 steps and carry about 0.4 dB of single-run noise, so the `deep` and `deep_feat` families still say little about CNNs in general. (vi) The classical grid chose edge-of-grid values for several arms (K = 8, smallest GBM settings), so it may be slightly narrow; all classical arms got the same kind of grid. (vii) GNN inference uses non-overlapping tiles, so tile borders see truncated neighbourhoods. (viii) Checks 4 and 5 of the plan are the author's operationalisation of ambiguous sentences and have not yet been run. (ix) The compute projection varies by up to 1.8x between timing runs (21.8 h vs 38.2 h), the latter from one outlying arm timing. (x) The plan's literature figures (Kim et al. RMSE units not stated; Bufort et al. 3-7% gain over a feature-only GBM) must not be compared with the numbers here.
- **Exploratory outputs.** `run_ext.py` runs and the trend-repair table are exploratory. Only contrasts frozen in a pre-registration file are labelled confirmatory.

## Reproducibility

- Seeds are fixed in the config; per-task seeds derive from a stable hash of the task id. Round 1 uses `seed=1` (AP sample), test salt `conf1`, replicate ids 100+. Round 2 uses `seed=2`, test salt `conf2`, replicate ids 200+. The exploratory runs used `seed=0`.
- Pre-registration files (`results_ext2/prereg.json`, written 2026-10-11, sha256 `97162105…`; `results_conf2/prereg_conf2.json`, written 2026-10-12, sha256 `e8a2305d…`, which also embeds the round-1 hash) should be committed to git before the confirmatory cells are run.
- Figures: `python readme_figures.py` writes `figures/*.png`. All numbers in it are transcribed from the saved notebook outputs, and the round-3 relative gains are computed from the parent-level mean-RMSE tables, not from paired tests.
- Round 4 uses test salt `abl4`, replicate ids 700+, plan in `results_ablation/prereg_ablation.json`; smoke runs write to `ablation_smoke/` and never to `figures/`.
- Round 5 uses development layouts from pool seed 11 (simulator seed 5, salt `scale5`), extra source layouts from seed 12 (salt `scale5x`), and fresh layouts from pool seed 21 (simulator seed 7, salt `scale5fresh`); plan `results_ext5/prereg_ext5_amend_v4.json`, sha256 `c47de96e…`, committed before Stage 2.
- Round 6 uses pool seed 31 and simulator seed 9 (salt `scale6fresh`), ledger `results_ext5/seed_ledger.json`, plan `results_ext5/prereg_ext6.json`; unit caches under `results_ext5` are shared with round 5. Priors are stored under `results_ext5/v6/priors` and see source maps only.
- Dense indoor run: `run_indoor_dense.py` / `dense_indoor_radiomap.ipynb`; test salt `dense1`, `seed` 0, wall simulator seed 7 (12 layouts x 6 transmitters x 1000 points), caches `cache_dense_wall` and `cache_dense_dm`, plan `results_dense/prereg_dense.json` (sha256 `06ca69ef5095b09827bf3aeb12257ec1e4cde86ab1420579c0955a965a790184`, written 2026-10-21), outputs `results_dense/dense_wall.jsonl` and `dense_dm.jsonl`. 16 CPUs, 15 workers: wall evaluation about 9 min for the last 450 tasks, DeepMIMO about 25 min for 540 tasks.
- Results come from the saved notebook outputs and the `confirmatory_master*.csv` / `claims_conf2.csv` files they write.
- Tuned hyper-parameters in the main sweep: `gcn`: hidden 32, 2 layers, lr 0.0095, `geo` features. `gcn_krig`: hidden 16, 3 layers, lr 0.0059, dropout 0.3, `geo_los` features. `kgnn`: Kmax 12, lr 0.011, `s2_init` 0.3. Round 2 defaults: Kmax 12, `s2_init` 0.3, 150 steps.
- City study: configuration `CITY_CFG` (`max_points` 12 000, `min_unit_points` 1 000, `n_test` 500, densities 30 / 100 / 300, 5 replicates, `seed` 0, random split, `trend_mode` "fit"); feature version `raster-v2`; pool plan `results_city/prereg_city.json` (written by `rmbench_city.py roles --freeze`); confirmatory plan `results_city/prereg_city_gnn.json`, sha256 `8e48c85156c3d677c6a4cb4d4f4b5e...` (full hash in the file), frozen 2026-10-19 10:33:47 before the confirm run; the notebook asserts the hash before `confirm`. The dev run was done before the freeze and informed nothing in the plan (jobs and families are the pre-declared ones). Per-task seeds derive from a stable hash of the task id.
- D0 on the city pool: priors `results_city/city_full_K30.pkl` and `city_geo_K30.pkl`, trained on the 30 source cities with the 6 validation cities held out, scored on the 7 dev cities only (`n_units` 24 per density, 12 available). The confirm cities were not touched by D0.
- Round 7 / 7b / 8: no preregistration file; val and dev cities only (`d0_city.pool()["val"]`, `["dev"]`), `CITY_CFG` unchanged (densities 30 / 100 / 300, 5 replicates, `seed` 0, random split), task prefixes `r7smoke`, `r7dev`, `r7tune`, `r7wall`, `r8val`, `r8dev`, outputs in `results_city/` (`r7_smoke.jsonl`, `r7_dev.jsonl`, `r7_tune_val.jsonl`, `r7_dev_wall.jsonl`, `r8_val.jsonl`, `r8_dev.jsonl`), rasters in `rmbench_cache_city/rasters/`. Per-task seeds derive from a stable hash of the task id. `rmbench_r7.py` is the export of the notebook's `export`-tagged cells (identical when checked 2026-10-21); re-run section 9 after any edit to sections 1-6 and restart the kernel before importing it. The tables above were transcribed from the saved notebook cell outputs, not re-estimated.
- RadioMapSeer study (dev stage): split from the RadioUNet loader (shuffle seed 42, 501 / 100 / 99); config salt `rms1`; per-task seeds derive from a stable hash; tuning trials use seeds 1000 + trial index; the RadioUNet 20 000-step dev check used seed 777; classical tuning is deterministic given the val label draws. Environment of the dev run: Kaggle, Tesla T4, Python 3.13.15, numpy 2.1.3, torch 2.11.0+cu128; module sha256 prefix at the time of the first dev run `52496edbaa49de47`, and `0957f09b13a0c54e` after `cnn_feat` was added on 2026-10-26 (the ReLU fix of 2026-10-27 changes it again; `0896363c77b45520` in the 2026-10-28 session) (it changes with any edit to the module cell, and the frozen plan will store the final hash). Results so far live on branch `rms-run` of `CompileArtisan/prism` (`results_rms/classical_hp.json`, `tune_classical_trials.csv`, `learned_hp.json`, `tune_<arm>_trials.csv`; commit `5a0ef5cca2` for the classical tuning; `9fb9d51876` is the end-of-session push of 2026-10-26, 2 changed files). The plan file is `PLAN_radiomapseer_DRAFT.md`; `results_rms/prereg_rms.json` does **not** exist yet. Selected hyper-parameters (`learned_hp.json`): `gnn_kim` lr 2.7e-3, wd 2.0e-5, width 96, depth 4, r2 32; `gnn_kim_oldedge` lr 2.7e-3, wd 6.9e-4, width 96, depth 4, r2 18; `rgnn_xmap` lr 2.4e-3, wd 1.9e-6, width 96, depth 2, r2 18; `radiounet` lr 4.35e-4, wd 9.4e-5, base 32, depth 4 and `cnn_feat` lr 4.87e-4, wd 1.26e-5, base 32, depth 4 (re-tuned after the ReLU fix; the pre-fix values, lr 3.0e-4 and 4.2e-4, are void). In the 2026-10-26 session only `cnn_feat` was tuned; the other four were loaded from `learned_hp.json`. Data root in that session: `/kaggle/input/datasets/compileartisan/radiomapseer` (a `polygon/buildings_complete` folder also exists there and is not used). The dev-stage tables above were transcribed from the saved notebook outputs (`rmbench_rms_*.ipynb` runs of 2026-10-25/26), not re-estimated.
