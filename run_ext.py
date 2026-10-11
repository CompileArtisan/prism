"""run_ext.py -- experiments for the extension (put next to rmbench_core.py, rmbench_nn.py, rmbench_ext.py).

  python run_ext.py smoke                         # 2-minute dry run on a FAKE UJI-schema dataset (checks plumbing, says nothing about science)
  python run_ext.py uji [--blocked] [--n-jobs 8]  # REAL data: UJIIndoorLoc, 3 buildings. topology/arch grid + outage + leave-one-building-out
  python run_ext.py deepmimo [--blocked]          # simulated, harder target: narrowband channel power, NLOS-heavy scenarios

Pre-specified (fixed here, before any result exists): the contrast lists below, the outage thresholds (train-label 10th/25th percentile and
-85 dBm for UJI / -77 dBm for DeepMIMO), no hyper-parameter tuning (library defaults for every neural model), Holm over each printed family.
"""
import os, sys, argparse, json, shutil
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import rmbench_core as rc, rmbench_ext as rx

PRE_CFGS = {"gat_del": dict(arch="gat", graph="delaunay", lr=5e-3), "gcn_knn": dict(arch="gcn", graph="knn", k=8, lr=5e-3)}

# --- pre-specified contrast families (A vs B, delta>0 => A better) ------------------------------------------------------------
TOPO = [("idw", "trend"), ("krig_global", "idw"), ("kgnn", "idw"),
        ("gcn_del", "gcn_knn"), ("gat_del", "gat_knn"),                 # topology: Delaunay vs kNN
        ("gat_knn", "gcn_knn"), ("gat_del", "gcn_del"),                 # architecture: GAT vs GCN
        ("gcn_del", "idw"), ("gat_del", "idw"), ("gat_del_idw", "idw")]  # learned vs classical     -> 10 x 3 densities = 30 tests (cap 36)
XFER = [("xfer_gat_del", "idw"), ("xfer_gcn_knn", "idw"),                # pre-trained on OTHER layouts vs classical
        ("xfer_gat_del", "gat_del"), ("xfer_gcn_knn", "gcn_knn"),        # value of pre-training (vs scratch, same arch)
        ("xfer_gat_del", "within_gat_del"), ("xfer_gcn_knn", "within_gcn_knn"),   # transfer gap (cross- vs within-layout pre-training)
        ("xferft_gat_del", "xfer_gat_del"), ("xferft_gcn_knn", "xfer_gcn_knn"),   # does 30-step fine-tuning help
        ("kgnn", "idw"), ("idw", "trend")]                               # -> 10 x 3 = 30 tests (cap 36)

def show(df, title):
    print(f"\n=== {title}"); print(df.round(3).to_string())

def report(df, out, contrasts, tag, pooled_only=False, cap=36, infer=None):
    os.makedirs(out, exist_ok=True)
    ok = df[df.status == "ok"]
    print(f"\n##### {tag}: {len(df)} job rows, {len(ok)} ok, {len(df) - len(ok)} failed")
    if len(df) - len(ok): print(df[df.status != "ok"].groupby("name").size().to_string())
    d = df
    # inference set: scenarios with >= 2 independent parents. Others (e.g. DeepMIMO i2_28b: 1 tx, i3_60: 2 tx) stay in the descriptive tables.
    di = d if infer is None else d[d.scenario.isin(infer)]
    for col, lab in [("rmse", "RMSE (dB), parent-level mean"), ("mae", "MAE (dB)")]:
        t = rx.agg_metric(d, col); t.to_csv(f"{out}/{tag}_{col}.csv"); show(t, lab)
    pc = rx.paired_metric(di if not pooled_only else di[di.scenario == "ALL"], "rmse", contrasts, max_tests=cap)
    pc.to_csv(f"{out}/{tag}_contrasts_rmse.csv")
    if len(pc):
        show(pc[["scenario", "density", "A", "B", "n_units", "mean_delta_db", "ci_lo", "ci_hi", "rel_gain_pct", "p_holm", "significant", "underpowered"]], f"paired contrasts on RMSE ({tag})")
        print(f"{int(pc.significant.sum())}/{len(pc)} significant after Holm; {int(pc.underpowered.sum())} underpowered")
    # coverage / outage ------------------------------------------------------------------------------------------------------
    cols = [c for c in d.columns if c.startswith(("f1_", "bacc_", "acc_", "area_"))]
    for c in cols:
        t = rx.agg_metric(d, c); t.to_csv(f"{out}/{tag}_{c}.csv")
    for c in [c for c in cols if c.startswith(("f1_q10", "bacc_q25", "area_q10"))] + [c for c in cols if c.startswith("f1_abs")]:
        show(rx.agg_metric(d, c), f"outage metric {c} (higher is better except area_*)")
    f1c = [c for c in cols if c.startswith("f1_q10")]
    if f1c:
        pcf = rx.paired_metric(di if not pooled_only else di[di.scenario == "ALL"], f1c[0], [c for c in contrasts if c[1] in ("idw", "gcn_knn", "trend")][:6], higher_is_better=True, max_tests=cap)
        if len(pcf): show(pcf[["scenario", "density", "A", "B", "n_units", "rel_gain_pct", "p_holm", "significant"]], f"paired contrasts on (1 - outage-F1), threshold {f1c[0]}")

def run_grid(names, cfg, out, tag, n_jobs, jobs=None, n_rep=None):
    tasks = rc.make_tasks(names, cfg["densities"], n_rep or cfg["n_rep"], jobs or (rx.base_jobs() + rx.graph_jobs()), cfg, prefix=tag)
    path = os.path.join(out, f"{tag}.jsonl"); rx.run_tasks_ext(tasks, path, n_jobs=n_jobs); return rc.read_results(path)

def run_lolo(names, cfg, out, tag, n_jobs, steps, n_eval, n_src):
    plan = rx.lolo_plan(names, cfg["cache_dir"], n_eval=n_eval, n_src=n_src)
    print("leave-one-layout-out plan (layout, n_eval, n_cross_src, n_within_src):", [(p["layout"], len(p["eval"]), len(p["cross"]), len(p["within"])) for p in plan])
    paths = rx.pretrain_for_plan(plan, cfg, PRE_CFGS, os.path.join(out, "pretrained_" + tag), steps=steps)
    tasks = rx.lolo_tasks(plan, paths, cfg, PRE_CFGS, prefix=tag)
    path = os.path.join(out, f"{tag}.jsonl"); rx.run_tasks_ext(tasks, path, n_jobs=n_jobs)
    return rc.read_results(path)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["smoke", "uji", "deepmimo"])
    ap.add_argument("--blocked", action="store_true", help="contiguous-block test region instead of random test points")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--uji-dir", default="data/ujiindoorloc")
    ap.add_argument("--pretrain-steps", type=int, default=1500); ap.add_argument("--no-lolo", action="store_true")
    ap.add_argument("--scenarios", default="i1_2p5,i2_28b,i3_60"); ap.add_argument("--max-units", type=int, default=24)
    a = ap.parse_args()
    split = dict(split_mode="blocked", block_buffer=0.0) if a.blocked else {}
    sfx = "_blocked" if a.blocked else ""

    if a.what in ("smoke", "uji"):
        if a.what == "smoke":
            cfg = dict(rx.UJI_CFG, cache_dir="ext_smoke/uji", densities=(10, 20, 30), n_rep=2, n_test=25, min_unit_points=50, uji_max_waps_per_floor=2, **split)
            shutil.rmtree("ext_smoke", ignore_errors=True); d = rx.make_fake_uji("ext_smoke/fake_uji", floors=(2, 2, 2), n_waps=12); out = "ext_smoke/results"
            steps, n_eval, n_src = 60, 3, 8
        else:
            cfg = dict(rx.UJI_CFG, **split); d = rx.fetch_uji(a.uji_dir); out = "results_ext_uji"; steps, n_eval, n_src = a.pretrain_steps, 20, 40
        ok, dropped = rx.build_uji_units(cfg, d)
        meta = pd.DataFrame([dict(unit=n, **{k: rx.get_unit_ext(n, cfg["cache_dir"])[k] for k in ("layout", "parent")}, n_pts=len(rx.get_unit_ext(n, cfg["cache_dir"])["y"])) for n in ok])
        print(f"{len(ok)} units | {meta.parent.nunique()} building-floors (= statistical units) | {meta.layout.nunique()} buildings"); print(meta.groupby("layout").agg(units=("unit", "size"), floors=("parent", "nunique"), pts=("n_pts", "median")).to_string())
        os.makedirs(out, exist_ok=True)
        df = run_grid(ok, cfg, out, "uji_grid" + sfx, a.n_jobs)
        report(df.assign(scenario="uji"), out, TOPO, "uji_grid" + sfx)
        if not a.no_lolo:
            dl = run_lolo(ok, cfg, out, "uji_lolo" + sfx, a.n_jobs, steps, n_eval, n_src)
            report(rx.by_layout(dl), out, XFER, "uji_lolo" + sfx, pooled_only=True)
        print("\nREAD THIS: parents are building-floors (n ~ 13); buildings are n = 3, so a building-level significance test is impossible "
              "(min attainable p with 3 clusters is 0.25). Per-building rows are descriptive only.")

    if a.what == "deepmimo":
        cfg = dict(rx.DM_HARD_CFG, cache_dir="rmbench_cache_dm_nb" + sfx.replace("_blocked", ""), **split); out = "results_ext_dm"; os.makedirs(out, exist_ok=True)
        scen = [s for s in a.scenarios.split(",") if s]
        ok, dropped = rx.build_dm_units(scen, cfg)
        for n, why in dropped: print("  dropped:", n, "-", why)
        by = rx.layouts_of(ok, cfg["cache_dir"]); ok = [n for v in by.values() for n in v[:a.max_units]]
        U = [rx.get_unit_ext(n, cfg["cache_dir"]) for n in ok]
        print(pd.DataFrame([dict(scenario=u["scenario"], f_ghz=u["freq_ghz"], los=u["los"].mean(), y_std=u["y"].std(), n=len(u["y"])) for u in U]).groupby("scenario").agg(["mean", "size"]).round(3).to_string())
        df = run_grid(ok, cfg, out, "dm_grid" + sfx, a.n_jobs)
        report(df, out, TOPO, "dm_grid" + sfx, infer=["i1_2p5"])
        print("\nNOTE: i2_28b has one transmitter and i3_60 two, so only i1_2p5 supports paired tests across units; layout and carrier frequency are "
              "confounded across DeepMIMO scenarios. Use the UJI run for the cross-layout claim.")

if __name__ == "__main__":
    main()
