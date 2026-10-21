"""run_indoor_dense.py -- NEW runner (adds only; rmbench_*.py and run_ext.py are untouched).

Put it next to rmbench_core.py, rmbench_nn.py, rmbench_ext.py, rmbench_ext2.py, rmbench_ext3.py, rmbench_dmfix.py.

  python run_indoor_dense.py diagnose-uji          # why is building b1 missing? which UJI units could even carry 100/200 labels?
  python run_indoor_dense.py freeze                # write results_dense/prereg_dense.json + sha256 (do this BEFORE `run`)
  python run_indoor_dense.py smoke                 # 2-minute plumbing check on the wall simulator (says nothing about science)
  python run_indoor_dense.py run wall              # dense indoor, wall simulator (author's own simulator, with side information)
  python run_indoor_dense.py run dm                # dense indoor, DeepMIMO i1_2p5 / i2_28b / i3_60 (ray-traced; i3_60 slug unverified)
  python run_indoor_dense.py report wall|dm        # tables only (no model fitting)

SINGLE COVERAGE METRIC (used for every dataset, written into the prereg file before any result exists)
  threshold  : 25th percentile of the TRAIN labels of that unit (leak-free, adapts to each unit's dynamic range)
  outage     : true or predicted power below the threshold
  PRIMARY    : balanced accuracy of covered-vs-outage on the held-out test points, column `bacc_q25`
               (plain accuracy is reported as `acc_q25`; with 25% outage a constant predictor already scores 0.75, which is why
               balanced accuracy is the primary). Cells where the test set has < 8 points in either class are NaN (rmbench_ext2 rule).
  SECONDARY  : `within5` = fraction of test points with |error| <= 5 dB; `area_q25` = |predicted - true outage fraction|.
Inference: paired across PARENTS (wall: layouts, DeepMIMO: transmitters), Holm over the contrast list, cap fixed in the prereg.
"""
import os, sys, json, hashlib, argparse, time, traceback, datetime
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import rmbench_core as rc, rmbench_ext as rx, rmbench_ext2 as r2, rmbench_ext3 as r3

OUT = "results_dense"
PREREG = f"{OUT}/prereg_dense.json"
DENSITIES = (50, 100, 200, 300)          # 50 = sparse end of the brief (30-50), 200 and 300 = dense (200+)
N_REP = 5
COV_PRIMARY = "bacc_q25"

DENSE_CFG = dict(rc.DEFAULT_CFG, densities=DENSITIES, n_rep=N_REP, n_test=200, min_unit_points=600, max_points=3000,
                 outage_q=(0.1, 0.25), outage_abs_dbm=(), outage_min_class=8, test_salt="dense1", trend_policy="robust", seed=0)
WALL_CFG = dict(DENSE_CFG, cache_dir="cache_dense_wall")
DM_CFG = dict(DENSE_CFG, cache_dir="cache_dense_dm")
WALL_SPEC = dict(n_layouts=12, n_tx=6, n_pts=1000, seed=7)         # CHECK the seed against your earlier wall rounds so these layouts are fresh
DM_SCENARIOS = ("i1_2p5", "i2_28b", "i3_60")

# --- pre-specified contrasts (A vs B, delta > 0 => A better) --------------------------------------------------------------------
C_COMMON = [("idw", "trend"), ("krig_global", "idw"), ("kgnn", "idw"),
            ("gcn_knn", "idw"), ("gat_del", "idw"), ("gcn_knn", "krig_global"), ("gat_del", "krig_global"),
            ("gcn_del", "gcn_knn"), ("gat_del", "gat_knn"),            # topology: Delaunay vs kNN
            ("gat_knn", "gcn_knn"), ("gat_del", "gcn_del")]           # architecture: GAT vs GCN
C_SIDE = [("rk_side", "idw"), ("gcn_knn_side", "rk_side"), ("gat_del_side", "rk_side")]          # wall only (side information exists)

def jobs_for(kind, steps=150):
    J = rx.base_jobs() + rx.graph_jobs(("gcn_knn", "gcn_del", "gat_knn", "gat_del"), steps=steps)
    if kind == "wall":
        S = r2.side_jobs()
        for j in S:
            if j["method"] == "gnn": j["hp"] = {**j["hp"], "steps": steps}
        J += S
    return J

def contrasts_for(kind): return C_COMMON + (C_SIDE if kind == "wall" else [])

# --- worker side: extra coverage columns ----------------------------------------------------------------------------------------
def _patch_cov():
    """run_task_ext2 looks cov_metrics2 up as a module global at call time, so replacing it inside the worker adds `within5`."""
    if getattr(r2.cov_metrics2, "_dense", False): return
    orig = r2.cov_metrics2
    def cov_dense(pred, true, y_train, cfg):
        out = orig(pred, true, y_train, cfg)
        out["within5"] = float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)) <= 5.0))
        return out
    cov_dense._dense = True; r2.cov_metrics2 = cov_dense

def run_task_dense(task):
    r3.install(); _patch_cov(); return r2.run_task_ext2(task)

def run_tasks_dense(tasks, out_path, n_jobs=1, log_every=25):
    """Resumable JSONL executor (same protocol as rmbench_ext3.run_tasks3) dispatching run_task_dense."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks_dense] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
    t0, nf, shown = time.time(), 0, False
    enc = lambda o: None if o != o else float(o) if isinstance(o, np.floating) else str(o)
    def fail(t): return [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"],
                              status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
    def write(rows, fh):
        nonlocal nf, shown
        bad = [r for r in rows if r["status"] != "ok"]; nf += len(bad)
        if bad and not shown: shown = True; print(f"  first failure: {bad[0].get('name')} {bad[0].get('unit')}\n{bad[0].get('err', '')}")
        fh.write(json.dumps(rows, default=enc) + "\n"); fh.flush()
    with open(out_path, "a") as fh:
        if n_jobs <= 1:
            for i, t in enumerate(todo, 1):
                write(run_task_dense(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task_dense, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception: rows = fail(futs[f])
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

# --- units -----------------------------------------------------------------------------------------------------------------------
def build_wall(cfg, spec=WALL_SPEC):
    return r2.build_wall_units(cfg, n_layouts=spec["n_layouts"], n_tx=spec["n_tx"], seed=spec["seed"], n_pts=spec["n_pts"])

def build_dm(cfg, scenarios=DM_SCENARIOS, max_units=24):
    import rmbench_dmfix  # noqa: F401  (carrier-frequency fix, must come before any DeepMIMO load)
    ok, dropped = rx.build_dm_units(list(scenarios), cfg)
    for n, why in dropped: print("  dropped:", n, "-", why)
    by = rx.layouts_of(ok, cfg["cache_dir"]); ok = [n for v in by.values() for n in v[:max_units]]
    U = [rx.get_unit_ext(n, cfg["cache_dir"]) for n in ok]
    print(pd.DataFrame([dict(scenario=u["scenario"], f_ghz=u["freq_ghz"], los=u["los"].mean(), y_std=u["y"].std(), n=len(u["y"])) for u in U])
          .groupby("scenario").agg(["mean", "size"]).round(3).to_string())
    return ok

def feasible(names, cfg):
    """Drop units that cannot carry the largest density plus the test set (rc.split_for would raise)."""
    need = max(cfg["densities"]) + cfg["n_test"] + 1
    keep = [n for n in names if len(rx.get_unit_ext(n, cfg["cache_dir"])["y"]) >= need]
    if len(keep) < len(names): print(f"  {len(names) - len(keep)} units have < {need} points and are excluded (max density {max(cfg['densities'])} + n_test {cfg['n_test']})")
    return keep

# --- reporting -------------------------------------------------------------------------------------------------------------------
def report(df, kind, tag, cap):
    os.makedirs(OUT, exist_ok=True); ok = df[df.status == "ok"]
    print(f"\n##### {tag}: {len(df)} job rows, {len(ok)} ok, {len(df) - len(ok)} failed")
    if len(df) - len(ok): print(df[df.status != "ok"].groupby("name").size().to_string())
    n_par = ok.groupby("scenario").parent.nunique(); infer = list(n_par[n_par >= 6].index)
    print("parents per scenario (inference needs >= 6):", n_par.to_dict(), "-> inferential:", infer)
    for col, hib in [("rmse", False), (COV_PRIMARY, True), ("acc_q25", True), ("within5", True), ("area_q25", False)]:
        if col in ok:
            t = rx.agg_metric(df, col); t.to_csv(f"{OUT}/{tag}_{col}.csv"); print(f"\n=== {col} ({'higher' if hib else 'lower'} is better), parent-level mean"); print(t.round(3).to_string())
    di = df[df.scenario.isin(infer)]
    if not len(di): print("\nNo scenario has >= 6 parents: all tables are descriptive."); return
    for col, hib, nm in [("rmse", False, "RMSE"), (COV_PRIMARY, True, "1 - " + COV_PRIMARY)]:
        pc = rx.paired_metric(di, col, contrasts_for(kind), higher_is_better=hib, max_tests=cap); pc.to_csv(f"{OUT}/{tag}_contrasts_{col}.csv")
        if len(pc):
            print(f"\n=== paired contrasts on {nm}  (delta > 0 => A better; Holm over this call)")
            print(pc[["scenario", "density", "A", "B", "n_units", "mean_delta_db", "ci_lo", "ci_hi", "rel_gain_pct", "p_holm", "significant", "underpowered"]].round(4).to_string())
            print(f"{int(pc.significant.sum())}/{len(pc)} significant after Holm; {int(pc.underpowered.sum())} underpowered")

def cap_for(kind, n_scen): return len(contrasts_for(kind)) * len(DENSITIES) * max(1, n_scen)

# --- prereg ----------------------------------------------------------------------------------------------------------------------
def freeze():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(PREREG): sys.exit(f"{PREREG} already exists; refusing to overwrite a frozen plan.")
    spec = dict(written=datetime.datetime.now().isoformat(timespec="seconds"), question="Do GCN/GAT beat IDW / kriging (and rk_side where side information exists) "
                "on DENSE (200, 300) and sparse (50, 100) labels on indoor layouts?", densities=DENSITIES, n_rep=N_REP, n_test=DENSE_CFG["n_test"],
                test_salt=DENSE_CFG["test_salt"], wall=dict(WALL_SPEC, cfg_cache=WALL_CFG["cache_dir"]), dm_scenarios=DM_SCENARIOS,
                coverage_metric=dict(primary=COV_PRIMARY, threshold="25th percentile of train labels", secondary=["acc_q25", "within5", "area_q25"]),
                jobs_wall=[j["name"] for j in jobs_for("wall")], jobs_dm=[j["name"] for j in jobs_for("dm")],
                contrasts_wall=contrasts_for("wall"), contrasts_dm=contrasts_for("dm"), holm="over each call, cap = n_contrasts x n_densities x n_inferential_scenarios",
                hyper_parameters="library defaults, 150 steps, no tuning", primary_contrasts=[["gcn_knn", "idw"], ["gat_del", "idw"], ["gat_del", "krig_global"]],
                notes="Exploratory-in-origin hypotheses (earlier rounds already show GNN <= IDW); this plan only fixes the grid, metrics and tests before the data are scored.")
    with open(PREREG, "w") as f: json.dump(spec, f, indent=2, default=str)
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest(); open(PREREG + ".sha256", "w").write(h)
    print("wrote", PREREG, "sha256", h, "\nCOMMIT IT TO GIT before running `run`.")

def require_prereg():
    if not os.path.exists(PREREG) or not os.path.exists(PREREG + ".sha256"): sys.exit("Run `python run_indoor_dense.py freeze` (and commit the file) first.")
    if hashlib.sha256(open(PREREG, "rb").read()).hexdigest() != open(PREREG + ".sha256").read().strip(): sys.exit("prereg file changed after freezing.")

# --- UJI diagnostic --------------------------------------------------------------------------------------------------------------
def diagnose_uji(uji_dir="data/ujiindoorloc", results_dir="results_ext_uji"):
    print("=== 1. what the saved UJI result files contain")
    import glob
    for p in sorted(glob.glob(f"{results_dir}/*.jsonl")):
        d = rc.read_results(p)
        if not len(d): print(" ", p, "EMPTY"); continue
        g = d.groupby("layout").agg(rows=("name", "size"), ok=("status", lambda s: int((s == "ok").sum())), units=("unit", "nunique"), parents=("parent", "nunique"))
        print(f"\n {p}\n{g.to_string()}\n  densities: {sorted(d.density.unique())}  methods: {d.name.nunique()}")
        bad = d[d.status != "ok"]
        if len(bad): print("  failures by layout/name:\n", bad.groupby(["layout", "name"]).size().to_string(), "\n  first error:\n", str(bad.iloc[0].get("err", ""))[:600])
    print("\n=== 2. what the raw UJI csv could support (needs data/ujiindoorloc)")
    try: d = rx.fetch_uji(uji_dir)
    except Exception as e: print("  csv not available:", repr(e)[:150]); return
    cfg = dict(rx.UJI_CFG, cache_dir="diag_cache_uji"); ok, dropped = rx.build_uji_units(cfg, d)
    meta = pd.DataFrame([dict(layout=u["layout"], parent=u["parent"], n=len(u["y"])) for u in (rx.get_unit_ext(n, cfg["cache_dir"]) for n in ok)])
    print(meta.groupby("layout").agg(units=("n", "size"), floors=("parent", "nunique"), pts_min=("n", "min"), pts_median=("n", "median"), pts_max=("n", "max")).to_string())
    nt = cfg["n_test"]
    for dens in (60, 100, 150, 200):
        f = meta[meta.n >= dens + nt + 1]
        print(f"  density {dens} (+ {nt} test): {len(f)} units in {f.parent.nunique()} floors, buildings: {sorted(f.layout.unique())}")
    print("  -> if no building has >= 200 + n_test points, 200+ labels CANNOT be done on UJI; use the wall / DeepMIMO dense runs instead.")

# --- main ------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["diagnose-uji", "freeze", "smoke", "run", "report"]); ap.add_argument("kind", nargs="?", choices=["wall", "dm"])
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--max-units", type=int, default=24)
    a = ap.parse_args(); global OUT
    if a.cmd == "diagnose-uji": return diagnose_uji()
    if a.cmd == "freeze": return freeze()
    if a.cmd == "smoke":
        import shutil; shutil.rmtree("dense_smoke", ignore_errors=True)
        cfg = dict(WALL_CFG, cache_dir="dense_smoke/cache", densities=(30, 60, 120), n_rep=2, n_test=60, min_unit_points=300)
        ok = r2.build_wall_units(cfg, n_layouts=4, n_tx=2, seed=0, n_pts=400)
        tasks = r2.make_tasks2(ok, cfg["densities"], range(cfg["n_rep"]), jobs_for("wall", steps=15), cfg, prefix="smoke")
        run_tasks_dense(tasks, "dense_smoke/smoke.jsonl", n_jobs=1)
        df = rc.read_results("dense_smoke/smoke.jsonl"); OUT = "dense_smoke"
        return report(df, "wall", "smoke", cap=cap_for("wall", 1))
    if a.cmd == "run": require_prereg()
    if a.kind is None: sys.exit("say which: wall | dm")
    cfg = WALL_CFG if a.kind == "wall" else DM_CFG; tag = f"dense_{a.kind}"; path = f"{OUT}/{tag}.jsonl"
    if a.cmd == "run":
        os.makedirs(OUT, exist_ok=True)
        ok = build_wall(cfg) if a.kind == "wall" else build_dm(cfg, max_units=a.max_units)
        ok = feasible(ok, cfg)
        tasks = r2.make_tasks2(ok, cfg["densities"], range(cfg["n_rep"]), jobs_for(a.kind), cfg, prefix=tag)
        run_tasks_dense(tasks, path, n_jobs=a.n_jobs)
    df = rc.read_results(path); n_scen = df[df.status == "ok"].scenario.nunique()
    report(df, a.kind, tag, cap=cap_for(a.kind, n_scen))

if __name__ == "__main__":
    main()
