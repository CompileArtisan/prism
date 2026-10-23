# Running `rmbench_rms.ipynb` (RadioMapSeer study) on Kaggle

Everything here is based on reading `rmbench_rms.ipynb` and `PLAN_radiomapseer_DRAFT.md`. I did not run anything. Items marked **[CHECK]** are Kaggle facts or numbers I could not verify from here.

## 0. What matters before you start

| Fact | Consequence |
|---|---|
| Kaggle GPU quota is about 30 h/week and a session is capped at about 12 h (Kaggle docs; one source says 9 h, so trust the counter in the notebook's session panel). **[CHECK]** | The study will not fit in one session. It must be split into stages that resume. |
| `/kaggle/working` is the only folder saved as output, capped at about 20 GB (and a file-count limit). **[CHECK]** | The notebook's geometry cache (`results_rms/cache_real`, thousands of small `.npz`) must not live there. Step 5 redirects it. |
| `/kaggle/input/...` is read-only. | The dataset goes there. The notebook only reads `DATA_ROOT`, so that is fine. |
| Resume granularity in the notebook: classical tuning resumes (`classical_hp.json`), learned tuning resumes **per arm** (`learned_hp.json`), confirmatory evaluation resumes **per task** (JSONL), final models are saved **after each complete training** (`models/*.pt`). | If a session dies in the middle of one arm's 30-trial tuning, that arm restarts from trial 0. Keep each arm's tuning inside one session (see the stage table). |
| `prereg_rms.json` stores `data_root`, `radiounet_repo` and `out_dir` as **relative or absolute strings**, and the confirmatory run re-initialises from them. | Keep the dataset slug and the working directory (`/kaggle/working`) identical in every session after the freeze. |
| Sessions lose `/kaggle/working` unless you commit the notebook version. | `kaggle_sync.py` pushes the small files to a GitHub branch every 30 min, so you can restore them in the next session. |

## 1. One-time Kaggle account setup

1. Sign in at kaggle.com and verify your phone number (Settings -> Phone verification). Internet access and GPUs in notebooks need it.
2. You will need **Internet = On** in the notebook (it clones `RonLevie/RadioUNet` in section 2 and `kaggle_sync.py` pushes to GitHub).

## 2. Create the GitHub token for syncing (one time)

1. GitHub -> Settings -> Developer settings -> Personal access tokens -> **Fine-grained tokens** -> Generate.
2. Repository access: only `CompileArtisan/prism`. Permissions: **Contents: Read and write**. Short expiry (30 days) is fine.
3. Copy the token. You will paste it into Kaggle Secrets in step 5. Never paste it into a notebook cell.

## 3. Put the new files in your GitHub repo (from your laptop)

Copy these four files into your local clone of `prism` (they are all new; no existing file is modified):

`rmbench_rms.ipynb`, `PLAN_radiomapseer_DRAFT.md`, `KAGGLE_RUNBOOK_rms.md`, `kaggle_sync.py`

```bash
cd prism
git checkout main && git pull
git add rmbench_rms.ipynb PLAN_radiomapseer_DRAFT.md KAGGLE_RUNBOOK_rms.md kaggle_sync.py
git commit -m "add RadioMapSeer cross-map study (notebook, draft plan, Kaggle runbook, sync helper)"
git push origin main
```

Then paste the README section from the end of this file into `README.md` and commit it as a second commit.
`rmbench_rms.py` is **not** committed now: the notebook writes it (`%%writefile`), and `kaggle_sync.py` pushes it to the run branch. Commit it to `main` after the freeze (step 9).

## 4. Create the Kaggle dataset with RadioMapSeer

Do not download inside the notebook every session: it eats the 20 GB output space and time.

1. On your laptop, download the dataset from the link in the RadioUNet README (https://drive.google.com/file/d/1PTaPpLOKraVCRZU_Tzev4D5ZO32tpqMO/view) and unzip it.
2. Confirm the layout the notebook expects: `png/buildings_complete/{id}.png`, `png/antennas/{id}_{tx}.png`, `gain/DPM/{id}_{tx}.png`. `gain/IRT2` and `gain/IRT4` are only needed for the optional IRT sensitivity run (`RUN_SENS_IRT`). If the total is too large for one Kaggle dataset, leave them out first and add them as a second dataset later. **[CHECK size]** I have not seen the unzipped size.
3. Zip the folders you keep into one archive, e.g. `RadioMapSeer.zip` containing a top-level folder `RadioMapSeer/` with `png/` and `gain/` inside.
4. kaggle.com -> Datasets -> New Dataset -> upload `RadioMapSeer.zip`, title `radiomapseer`, visibility **Private**. Kaggle normally extracts an uploaded zip; open the dataset's Data tab afterwards and confirm you see the folder tree rather than one zip file. **[CHECK]**
   - CLI alternative: `kaggle datasets create -p RadioMapSeer_folder --dir-mode zip` (needs your `kaggle.json`).
5. Optional: add `city_of_map.csv` (columns `map,city`) to the same dataset if you have it; section 12 of the notebook uses it for the city-cluster bootstrap and skips that analysis otherwise.

## 5. Create the notebook

1. kaggle.com/code -> New Notebook -> **File -> Import Notebook** -> upload `rmbench_rms.ipynb`.
2. Right sidebar -> **Add Input** -> Your datasets -> `radiomapseer`.
3. **Add-ons -> Secrets** -> add `GITHUB_TOKEN` = the token from step 2, and tick "attached" for this notebook.
4. **Session options**: Accelerator = **GPU P100** (or T4 x2: the code uses one GPU, `torch.device("cuda")`, so the second T4 does nothing). Internet = **On**. If a "Persistence" option is shown, choose "Files only" (keeps `/kaggle/working` files when you stop a session). **[CHECK]**
5. Insert **one new code cell above the first cell** with the bootstrap below, and edit the config cell (the `SMOKE = False ...` cell) as shown.

### Bootstrap cell (new first cell)

```python
import os, glob, shutil, subprocess, sys
os.chdir("/kaggle/working")                       # must be identical in every session (the frozen plan stores relative paths)

# 1) code from your repo (public): kaggle_sync.py and the plan file (its sha256 goes into the frozen file)
if not os.path.exists("kaggle_sync.py"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CompileArtisan/prism.git", "/kaggle/temp/prism_main"], check=True)
    for f in ["kaggle_sync.py", "PLAN_radiomapseer_DRAFT.md"]:
        shutil.copy(f"/kaggle/temp/prism_main/{f}", f)
sys.path.insert(0, "/kaggle/working")
import kaggle_sync as KS

# 2) bring back results from the previous session (no-op the first time)
KS.restore()

# 3) geometry cache outside /kaggle/working (thousands of small files, regenerable)
SCRATCH = "/kaggle/temp" if os.access("/kaggle/temp", os.W_OK) or not os.path.exists("/kaggle/temp") else "/tmp"
os.makedirs(f"{SCRATCH}/cache_real", exist_ok=True); os.makedirs("results_rms", exist_ok=True)
if not os.path.lexists("results_rms/cache_real"):
    os.symlink(f"{SCRATCH}/cache_real", "results_rms/cache_real")

# 4) find the dataset (the slug must never change after the freeze)
cands = glob.glob("/kaggle/input/*/png/buildings_complete") + glob.glob("/kaggle/input/*/*/png/buildings_complete")
assert cands, "RadioMapSeer not found under /kaggle/input: attach the dataset (Add Input)"
KAGGLE_DATA_ROOT = os.path.dirname(os.path.dirname(cands[0]))
print("DATA_ROOT =", KAGGLE_DATA_ROOT, "| cpu cores:", os.cpu_count())

KS.start_background(30)                           # push small files to branch rms-run every 30 min
```

### Config cell edits (the cell that starts with `SMOKE = False`)

```python
DATA_ROOT = KAGGLE_DATA_ROOT        # was "RadioMapSeer"
RADIOUNET_REPO = "RadioUNet"        # leave: cloned into /kaggle/working/RadioUNet (needs Internet = On)
N_JOBS = min(4, os.cpu_count() or 1)  # classical arms use worker processes; if you see spawn/import errors set N_JOBS = 1
```

Leave the other flags as in the stage table below. Nothing else in the notebook needs to change; do **not** edit the `%%writefile rmbench_rms.py` cell after the freeze (the module hash is part of the frozen plan).

### Last cell (new, at the very end): bundle the results for download

```python
import shutil
KS.push("end of session")
shutil.make_archive("/kaggle/working/results_rms_bundle", "zip", ".", "results_rms")   # cache_real is a symlink to /kaggle/temp; check the size printed below
print(os.path.getsize("/kaggle/working/results_rms_bundle.zip") / 2**20, "MiB")
```

If the zip contains the cache (it may follow the symlink), delete the bundle cell's output and instead zip only `results_rms/*.json|csv|jsonl|png` and `results_rms/models`.

## 6. Interactive session or "Save & Run All"?

- **Interactive** (press Run): use it for the smoke test and the first real checks. It stops if you close the tab for a long time or the machine idles, so do not rely on it for hours-long work.
- **Save Version -> Save & Run All (Commit)**: runs the whole notebook top to bottom on Kaggle's servers without your browser. Use it for every long stage. The flags are read from the saved notebook, so set them **before** saving. In the save dialog, if there is an option to save output even when the run fails, turn it on. **[CHECK]** Results appear under the notebook -> Versions -> the version -> **Output**.
- A commit that hits the session limit fails as a whole. That is why every long stage below is sized to fit, and why the background sync matters: after a failed commit the next session starts with `KS.restore()` and continues.

## 7. Stages (one stage per session)

Flags are in the config cell. A stage ends when its "done when" line is true; then push, then move on.

| # | Stage | Flags to set (everything not listed = False) | How | Done when |
|---|---|---|---|---|
| A | Smoke test | `SMOKE = True` | Interactive, CPU is fine, a few minutes | Runs to the end through freeze and confirm in `results_rms_smoke/`. It says nothing about science. |
| B | Verify data + geometry + compute projection | `RUN_VERIFY_DATA`, `RUN_GEOMETRY`, `RUN_BENCH` = True | Interactive on GPU | Section 3 shows the png counts, loader asserts pass; section 4 prints `TOTAL projected ... h` and per-arm `tune_h`, `final_train_h`, `eval_h`. |
| B2 | **Decide the budget** | edit `tune_n_cfg`, `tune_steps`, `final_steps`, `n_tx_test`, `n_draws`, `n_seeds` in the `R.make_cfg(...)` call | | Every learned arm's `tune_h` is about 8 h or less (fits one commit), the total is something your weekly quota can pay for. Any reduction is part of the frozen plan and must be disclosed (plan section 5 says to decide "before freezing"). |
| C | Tune classical + learned arms | B's flags plus `RUN_TUNE_CLASSICAL`, `RUN_TUNE_LEARNED` = True | Commit. If it is too long for one commit, run it again: finished arms are skipped (`learned_hp.json`). | `classical_hp.json`, `learned_hp.json` and `tune_*_trials.csv` exist for all 4 arms (check the branch or the Output tab). |
| D | Checks | `RUN_GEOMETRY`, `RUN_TUNE_CLASSICAL` = True (they load saved files), `RUN_CHECKS = True` | Commit or interactive | `results_rms/checks_rms.json` has `"passed": true`. Read the check 4 and 5 caveats in section 8 first. |
| E | **Freeze** | `RUN_GEOMETRY`, `RUN_TUNE_CLASSICAL` = True, `RUN_CHECKS = False`, `FREEZE = True`, `I_CONFIRM_SECTION_11 = True` | Interactive (short). Set `I_CONFIRM_SECTION_11` only after you decided the plan's section 11 items. | `prereg_rms.json` and `prereg_rms.json.sha256` exist. Then **step 9 (git commit of the frozen files)** before anything else. |
| F | Confirmatory run (once) | E's flags (with `FREEZE = False`) plus `RUN_CONFIRM = True` | Commit, repeat until it prints `confirmatory run finished`. Each repeat skips finished tasks. | `confirm_classical.jsonl`, `confirm_learned.jsonl`, `confirm_models_info.csv`, `confirm_contrasts.csv`, `confirm_rmse_table.csv`, the report figure. |
| G | Sensitivity (optional, after F) | `RUN_SENS_BLOCKED` and/or `RUN_SENS_IRT` = True (and `RUN_SCALE_CHECK` for the exploratory scale check) | Commit | `sens_*` JSONL files and the printed reports. |

Notes on the table:
- Section 3's geometry step rebuilds the cache in every new session (the cache is scratch). Its cost is shown in B; it is the price of keeping `/kaggle/working` small.
- The `models/*.pt` files are pushed by the sync only if they are under 50 MB each; larger ones stay in the Kaggle output. If stage F is interrupted between two model trainings, the finished models survive in the same session's Output (commit) or on the branch (small ones). A model in the middle of training is lost (8000 steps), nothing else.
- In stages E, F and G the notebook re-reads `learned_hp.json` and `classical_hp.json` from `results_rms/`. `KS.restore()` puts them there from the branch. Check that `results_rms/prereg_rms.json` exists before setting `RUN_CONFIRM = True`.

## 8. Keeping a run alive and monitoring it

- Commit runs continue after you close the browser. Check progress at kaggle.com/code -> your notebook -> Versions (shows "Running"); the log is visible while it runs.
- The notebook prints `tune ... trial i/29` lines and `classical[...]: N tasks to run, M already done`; use them to see how far a stage is.
- `KS.start_background(30)` pushes small files every 30 minutes. In GitHub: `prism` -> branch `rms-run` -> `results_rms/`.
- Weekly quota: Kaggle shows GPU hours used on your profile/notebook settings. Plan stages so that C + F do not exceed it; the B projection is the number to compare.
- If a stage keeps dying at the same point (out of memory, spawn errors): lower `gnn_batch` / `radiounet_batch` before the freeze, or set `N_JOBS = 1`.

## 9. Freeze and git (this is the pre-registration step)

The plan requires the frozen file to be committed **before** the confirmatory run. On Kaggle the proof is the commit on the sync branch. Right after stage E:

1. In the notebook run `KS.push("FREEZE: prereg_rms.json")`. It prints the commit hash. Open the branch `rms-run` on GitHub and check that `results_rms/prereg_rms.json`, `results_rms/prereg_rms.json.sha256` and `rmbench_rms.py` are there.
2. On your laptop (optional but recommended, so `main` carries the same files):

```bash
git fetch origin rms-run
git checkout main
git checkout origin/rms-run -- results_rms/prereg_rms.json results_rms/prereg_rms.json.sha256 results_rms/classical_hp.json results_rms/learned_hp.json results_rms/splits.json rmbench_rms.py
git add results_rms rmbench_rms.py
git commit -m "freeze RadioMapSeer plan (sha256 from prereg_rms.json.sha256)"
git tag rms-freeze && git push origin main --tags
```

3. Only then set `RUN_CONFIRM = True`.

## 10. Getting the results

- **Small files** (contrasts, tables, JSONL, hp files, small models): branch `rms-run`, folder `results_rms/` (merge into `main` when finished: `git checkout main && git merge origin/rms-run`, or pull the files you need).
- **Everything else**: notebook -> Versions -> the version -> **Output** tab -> download, or `kaggle kernels output <user>/<notebook-slug> -p out/` with the Kaggle CLI.
- The files you quote in the paper: `confirm_contrasts.csv` (all 24 contrasts with dB values, CI, Holm p, verdict), `confirm_rmse_table.csv`, `confirm_models_info.csv` (params, training time per learned arm), the report figure, `prereg_rms.json`.
- Check each number against these files before quoting it. The notebook's printed positive-result verdict (`pre-declared positive GNN result`) is computed from the `primary` family only.

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RadioMapSeer not found under /kaggle/input` | Dataset not attached, or extracted with an extra folder level: look at the Data tab; the bootstrap searches two levels. |
| `git clone ... RadioUNet` fails | Internet is Off. Turn it on in Session options. |
| Section 3 assert on loader ranges | The upstream RadioUNet repo changed. Do not edit the assert to pass; tell me what it printed. |
| `kaggle_secrets` / token errors, `sync FAILED` | The secret is not attached to this notebook, or the token lacks Contents: write on `prism`. The run continues; fix and call `KS.push()`. |
| Worker pool errors in the classical arms | Set `N_JOBS = 1` (slower, but the same results). |
| CUDA out of memory | Lower `gnn_batch` or `radiounet_batch` in the `R.make_cfg(...)` call before the freeze. |
| `confirmatory run refuses to start` | The frozen file is missing (run `KS.restore()`), or `rmbench_rms.py` / the plan hash changed. Do not edit the module cell. |
| Output folder too big / too many files | The cache symlink is missing: `results_rms/cache_real` must point to scratch. |

---

## README section to paste into `README.md` (suggested location: after the "Files" table and before "Methods")

```markdown
## RadioMapSeer cross-map study (draft plan, not yet frozen)

Tests the regime the earlier rounds could not: graph models trained **across many building maps** (RadioMapSeer, DPM simulations, 501 / 100 / 99 train / val / test maps from the RadioUNet loader) against `rk_side`, at 0.5% / 2% / 5% label density, with a Kim-style graph (`gnn_kim`), its old-edge ablation, a residual model on `rk_side` (`rgnn_xmap`) and RadioUNet. 24 pre-declared contrasts in five families, Holm within family. Status: **draft until `results_rms/prereg_rms.json` is frozen and committed**; nothing in it is evidence before that.

| File | Purpose |
|---|---|
| `PLAN_radiomapseer_DRAFT.md` | The plan (design, arms, families, thresholds, forking-path disclosure) |
| `rmbench_rms.ipynb` | Implementation and runner. Writes `rmbench_rms.py` and `results_rms/`; imports no other `rmbench_*` module |
| `KAGGLE_RUNBOOK_rms.md` | How to run it on Kaggle in resumable stages |
| `kaggle_sync.py` | Pushes small result files from a Kaggle session to branch `rms-run` |

Run order: smoke test -> data check and compute projection -> budget decision -> tuning -> checks -> **freeze and commit** -> confirmatory run (once) -> sensitivity analyses. Details and flags: `KAGGLE_RUNBOOK_rms.md`.
```
