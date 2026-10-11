# Sparse indoor radio-map estimation on DeepMIMO: IDW vs kriging vs graph models

A unit-level benchmark of classical interpolators (IDW, ordinary kriging) and graph neural models for predicting received power from a handful of labeled measurements. One *unit* is one transmitter's radio map, and every statistical test is paired across units.

**Headline (from the run on 2026-10-11):** on 48 transmitter maps of the `i1_2p5` scenario, a kriging-initialised graph model (KGNN) beats IDW at 30, 100 and 300 labels (5.7–7.3%) and beats global kriging at 300 labels (3.2%, about 0.002 dB). A plain label-aware GCN and an IDW-initialised GCN (`gcn_krig`) do not beat the classical baselines. The absolute differences are hundredths to thousandths of a dB. See [Results](#results) and [Limitations](#limitations).

## Files

| File | Purpose |
|---|---|
| `radiomap_6g_sweep.ipynb` | The experiment: setup, units, split, correctness checks, tuning, evaluation, paired contrasts |
| `rmbench_core.py` | Data loading and preprocessing, shared trend, train/tune/eval split, IDW and kriging, resumable executor, statistics (Wilcoxon, Holm) |
| `rmbench_nn.py` | Torch models: label-aware `GCN`, `gcn_krig` hybrid, `KGNN`, and `krig_untrained` (no torch_geometric; the graph is three index tensors) |

The notebook imports the two modules from its own directory, so all three files must sit together.

## Methods

All methods predict the **same residual** after a shared log-distance trend fitted on the training labels only, so no method gets a better trend than another.

| Name | Description |
|---|---|
| `trend` | Trend only, no spatial term |
| `idw` | Inverse-distance weighting, power and neighbourhood chosen by leave-one-out on the labeled points |
| `krig_global` | Ordinary kriging with a variogram fitted on the labeled points |
| `krig_local` | The same, using the K nearest labels |
| `krig_untrained` | KGNN's local kriging with its fixed default covariance and no learning |
| `gcn` | Label-aware GCN: geometric features, observed residual, mask and multi-hop label diffusion; trained by hiding random subsets of the labels |
| `gcn_krig` | The same GCN predicting a correction to a labeled-points-only IDW baseline. The output layer is zero-initialised, and step 0 (pure IDW) is an early-stopping candidate |
| `kgnn` | Kriging-initialised message-passing model with a learned covariance |

## Design rules

- **Unit = one transmitter.** Replicates are averaged inside a unit, then tests are paired across units.
- **Tuning and evaluation units are disjoint** (16 tuning units, 49 evaluation units).
- **Label-free features only** (`geo`, `geo_los`). Nothing is computed from the label's own channel.
- **Power** is the incoherent sum over real paths. Points below the noise floor are dropped.
- **Pre-specified contrasts, a cap on the number of tests and Holm correction** over all of them (33 tests in the current run).
- **Resumable JSONL executor.** A failing job never kills a task, and finished tasks are skipped on rerun.
- **Leak test** for every method (Section 3 of the notebook). A FAIL invalidates that method's column.

## Setup

```bash
pip install numpy scipy pandas matplotlib torch deepmimo
```

DeepMIMO scenarios `i2_28b` and `i1_2p5` must be available locally or downloadable. In the notebook:

- `RUN_NEURAL = False` runs only the classical methods, with no torch needed.
- `USE_SYNTHETIC = True` runs a fast offline smoke test on synthetic units.

## Running

1. Restart the kernel (the modules are imported once, in Section 1).
2. Run the notebook top to bottom. Run Section 3 (correctness checks) before trusting any number.
3. On a 16-core machine (15 workers), the last run took about 9 min for tuning (96 tasks) and about 9 min for evaluation (735 tasks, 8 methods, 5880 job rows).

**Cache gotcha.** `run_tasks` skips any task id already present in the JSONL file, and a task id covers all of its jobs. If you add a method to the job list and reuse an old cache file, the new method silently never runs. The notebook therefore uses `tune2.jsonl` / `eval2.jsonl` with prefixes `tune2` / `eval2`. Use a new file name and prefix whenever you change the job list or the code of an existing method.

## Results

`i1_2p5`, 48 evaluation units, 5 replicates per unit, densities of 30 / 100 / 300 labels, 1000 test points. The shared trend alone has an RMSE of 0.15 dB. All methods sit between 0.05 and 0.11 dB.

Percentages are the relative RMSE gain of the first method over the second (positive means the first is better). ✔ means significant after Holm correction over 33 tests.

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

The KGNN gain over `krig_global` is 0.0010–0.0018 dB, and over `idw` 0.0033–0.0063 dB.

`i2_28b` has a single unit, so nothing can be tested there. Descriptively, at 300 labels KGNN has the lowest RMSE (about 0.54 dB vs 0.59 for IDW and 0.64 for global kriging).

## Sensitivity variants (automatic, Section 7c)

After the primary analysis the notebook runs a fixed list of variants without any manual step. The list is declared in code before any variant result exists, and every variant is reported:

| Variant | What changes | Purpose |
|---|---|---|
| `trend_fspl` | Exponent fixed at 2, only the offset fitted | How methods behave when the shared trend is misspecified |
| `blocked` | Test set is one contiguous region | Labels are no longer interleaved with test points |
| `channel` | `compute_channels()` power, calibrated to the path sum | Added **only** if the built-in diagnostic finds it differs from `path_sum` by more than `CHANNEL_GATE_DB` (0.05 dB std); otherwise it would repeat the primary run |

Hyper-parameters are those tuned once on the tuning units under the primary settings, so each variant costs about one evaluation pass (roughly 9 min on 15 workers). Holm correction is applied within each variant's own contrast family. The output is one table of relative gains (`*` = significant after Holm) for every variant, plus a count of how many (variant, density) cells show each key method significantly better, significantly worse or neither. `replicate_table` gives descriptive tables for single-unit scenarios such as `i2_28b`. Set `RUN_VARIANTS = False` to skip the section.

These code paths were tested offline on synthetic units and a mocked DeepMIMO dataset only (classical methods; no torch in the test environment). Run them on real data before relying on them.

## Sensitivity results (run 2026-10-11, `i1_2p5`, 48 units; `*` = significant after Holm within each variant)

Relative gain of the first method over the second at 30 / 100 / 300 labels.

| Contrast | primary | `trend_fspl` | `blocked` | `channel` |
|---|---|---|---|---|
| `kgnn` vs `idw` | +6.5* / +7.3* / +5.7* | +6.7* / +7.2* / +5.7* | +5.2 / +4.7* / +3.6* | −4.2* / +4.0* / +3.5* |
| `kgnn` vs `krig_global` | +1.3 / +1.4 / +3.2* | +1.3 / +1.2 / +3.1* | −0.5 / −3.9 / −9.6 | −4.9* / +2.6* / +4.1* |
| `gcn_krig` vs `idw` | −2.5* / −0.5 / −0.2 | −2.3* / −0.5 / −0.3 | −0.4 / −0.1 / +0.0 | −1.9* / −1.1* / −0.5* |
| `gcn_krig` vs `krig_global` | −8.2* / −6.8* / −2.9* | −8.3* / −7.1* / −3.1* | −6.4 / −9.2 / −13.6 | −2.5* / −2.5* / +0.0 |
| `gcn` vs `idw` | −13.0* / −9.5* / −9.3* | −16.3* / −11.3* / −10.1* | +0.7 / −0.2 / −1.5 | −5.7* / −3.9* / −3.5* |

- `trend_fspl` reproduces the primary results almost exactly, so trend misspecification does not drive them.
- `kgnn` beats `idw` in 10 of 12 (variant, density) cells and is worse in one (`channel`, 30 labels).
- `kgnn` beats `krig_global` significantly in 4 of 12 cells (primary, `trend_fspl` and `channel` at 300 labels, and `channel` at 100), is significantly worse in 1 (`channel`, 30 labels), and under `blocked` the sign reverses but none of the differences is significant.
- No GCN variant beats `idw` or `krig_global` in any cell.
- The 12 cells are not independent (same units, and `trend_fspl` is nearly a copy of the primary run), so the counts describe consistency, not 12 separate confirmations.

## Limitations

- **One building layout.** The 48 units share a floor plan. The tests support "across transmitter positions in this layout", not "across buildings".
- **Tiny effect sizes.** The KGNN win over kriging is about 0.002 dB on a 0.15 dB trend residual. This scenario is 100% line-of-sight, so the path-power target is close to a deterministic function of distance. Report dB values next to percentages.
- **The kriging comparison is the strong one.** IDW is the weaker baseline. Only the 300-label win over global kriging survives correction.
- **Single-unit scenario.** `i2_28b` is descriptive only.
- **Layout and band are confounded** across `i2_28b` (28 GHz) and `i1_2p5` (2.5 GHz). Everything here is in-domain, and no cross-scenario claim is made.
- **Shared trend** is fitted on the labels, which is noisy at 30 labels and benefits every method equally.
- **Small tuning budget** (6 random-search trials per neural method, 2 replicates). The neural methods may be under-tuned relative to the classical ones.
- **`gcn_krig` at 30 labels** uses only 6 labeled points for early stopping. Its loss to IDW there may reflect noisy selection, not the method. This was not tested.
- **Target definition matters.** The diagnostic (transmitter 0 of `i1_2p5`) shows path-power-sum residual of 0.35 dB after the fitted trend and 0.45 dB with a fixed free-space exponent, but 1.13 dB (fitted) and 1.18 dB (fixed) for channel power, which differs from the path sum by 1.10 dB std after removing a constant. So the larger residuals in the other pipeline come from the channel-power target, not from its trend.

## Reproducibility

Results come from the saved notebook outputs. Seeds are fixed in the config (`seed = 0`), and per-task seeds are derived from a stable hash of the task id. Tuned hyper-parameters in the last run:

- `gcn`: hidden 32, 2 layers, lr 0.0095, `geo` features
- `gcn_krig`: hidden 16, 3 layers, lr 0.0059, dropout 0.3, `geo_los` features
- `kgnn`: Kmax 12, lr 0.011, `s2_init` 0.3
