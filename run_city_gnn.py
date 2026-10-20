"""run_city_gnn.py -- the M3 question on ray-traced city layouts: do GCN / GAT (with and without the 5 map features) beat IDW / kriging / regression-kriging?

  uv run run_city_gnn.py dev [--no-gnn] [--n-jobs 8]    # DEV cities. Free to look at, free to debug; NOTHING here is a confirmatory result.
  uv run run_city_gnn.py freeze                         # writes the hashed plan (jobs, contrasts, densities, reps, the CONFIRM city list)
  uv run run_city_gnn.py confirm [--n-jobs 8]           # CONFIRM cities, once; refuses unless the plan is frozen and unchanged
  uv run run_city_gnn.py confirm2 [--n-jobs 8]          # replication on the cities that became eligible after the freeze (same frozen jobs / contrasts)

Everything that is trained is a library default (no tuning), 150 steps for every neural arm. Statistical unit = the city (parent = layout); paired relative-RMSE gains,
cluster bootstrap CI over cities, Holm over ALL contrasts of the run (x5.contrast_table). Densities 30 / 100 / 300 labels = the brief's sparse (30-50) and dense (200+).
Pre-declared families (A vs B, gain > 0 => A better):
  feat : rk_side vs idw, rk_side vs krig_global, rk_gbm vs rk_side          (do the map features help classical interpolation?)
  gnn  : gcn_knn / gat_del vs idw; gcn_knn_side / gat_del_side vs rk_side; gat_del_side vs idw; side vs no-side GNNs
  topo : Delaunay vs kNN (GCN, GAT) and GAT vs GCN (kNN, Delaunay)
PRE-DECLARED 'positive GNN result': gcn_knn_side OR gat_del_side beats rk_side with verdict 'A better' (CI above 0) and p_holm < 0.05 at >= 1 density.
Anything else is reported as it comes out, including 'A worse'.
"""
import os, sys, json, time, hashlib, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import rmbench_core as rc, rmbench_dmfix                      # noqa: F401
import rmbench_city as cc, rmbench_ext as rx, rmbench_ext2 as r2, rmbench_ext5 as x5, rmbench_ext6 as x6, d0_city as d0

OUT = "results_city"; PLAN = os.path.join(OUT, "prereg_city_gnn.json")
CFG = dict(cc.CITY_CFG)                                       # densities (30, 100, 300), n_rep 5, n_test 500, random test points
FAMILIES = dict(
    feat=[("rk_side", "idw"), ("rk_side", "krig_global"), ("rk_gbm", "rk_side")],
    gnn=[("gcn_knn", "idw"), ("gat_del", "idw"), ("gcn_knn_side", "rk_side"), ("gat_del_side", "rk_side"), ("gat_del_side", "idw"), ("gat_del_side", "gat_del"), ("gcn_knn_side", "gcn_knn")],
    topo=[("gcn_del", "gcn_knn"), ("gat_del", "gat_knn"), ("gat_knn", "gcn_knn"), ("gat_del", "gcn_del")])
POSITIVE = [("gcn_knn_side", "rk_side"), ("gat_del_side", "rk_side")]

def make_jobs(gnn=True):
    within = [j for j in x6.within_jobs(gated=False) if j["name"] in ("rk_spline", "rk_gbm")]
    if not gnn: return rx.base_jobs(with_kgnn=False) + within + [j for j in r2.side_jobs() if j["name"] == "rk_side"]
    return rx.base_jobs() + within + r2.side_jobs() + rx.graph_jobs(("gcn_knn", "gcn_del", "gat_knn", "gat_del"))

def units_of(role):
    P = d0.pool(); return [n for L in P[role] for n, _ in L]

def run(role, tag, jobs, n_jobs, densities=None, n_rep=None, out=OUT):
    cfg = dict(CFG); dens = densities or cfg["densities"]; reps = range(n_rep or cfg["n_rep"]); units = units_of(role)
    if not units: raise RuntimeError(f"no units with role {role!r}: build the pool and run `rmbench_city.py roles` first")
    print(f"[{tag}] role={role}: {len(units)} units in {len(set(u.split('|')[1] for u in units))} cities, densities {tuple(dens)}, reps {len(reps)}, {len(jobs)} jobs: {[j['name'] for j in jobs]}")
    tasks = x5.make_tasks5(units, dens, reps, jobs, cfg, prefix=tag); path = os.path.join(out, f"{tag}.jsonl"); x6.run_tasks6(tasks, path, n_jobs=n_jobs)
    return path

def report(path, strict=True):
    x5.check_run(path, max_fail_frac=0.01, strict=strict); df = rc.read_results(path); df = df[df.status == "ok"]
    u = x5.unit_rmse(df); t = u.groupby(["parent", "density", "name"]).rmse.mean().groupby(["density", "name"]).mean().unstack("density")
    print("\nRMSE (dB), mean over cities (units and replicates averaged within a city):"); print(t.round(3).to_string())
    cts = []
    for fam, cs in FAMILIES.items():
        ct = x5.contrast_table(df.assign(family=fam, split="random", scenario="city"), cs, by=("family", "density"))
        if len(ct): cts.append(ct)
    if not cts: print("no contrasts could be computed"); return None
    ct = pd.concat(cts, ignore_index=True); ct["p_holm"] = rc.holm(ct.p.values); pd.set_option("display.width", 250)
    print("\npaired contrasts (relative RMSE gain %, A better if > 0; CI = bootstrap over cities; Holm over all rows):")
    print(ct[["family", "density", "A", "B", "n_clusters", "mean_gain_pct", "ci_lo", "ci_hi", "mean_gain_db", "p_holm", "verdict"]].round(3).to_string(index=False))
    pos = ct[[(a, b) in POSITIVE for a, b in zip(ct.A, ct.B)]]; hit = pos[(pos.verdict == "A better") & (pos.p_holm < 0.05)]
    print(f"\nPRE-DECLARED POSITIVE GNN RESULT (side-feature GNN beats rk_side, CI > 0, p_holm < 0.05 at >= 1 density): {'YES' if len(hit) else 'NO'}")
    if len(hit): print(hit[["A", "B", "density", "mean_gain_pct", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))
    ct.to_csv(path.replace(".jsonl", "_contrasts.csv"), index=False); return ct

def plan_hash(jobs):
    plan = dict(jobs=jobs, families=FAMILIES, positive=POSITIVE, cfg={k: v for k, v in CFG.items() if k != "cache_dir"}, confirm=sorted(units_of("confirm")))
    return plan, hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dev", "freeze", "confirm", "confirm2"]); ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2)); ap.add_argument("--densities", type=int, nargs="*"); ap.add_argument("--n-rep", type=int)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); jobs = make_jobs(not a.no_gnn)
    if a.mode == "dev":
        p = run("dev", "city_gnn_dev" + ("_nognn" if a.no_gnn else ""), jobs, a.n_jobs, a.densities, a.n_rep); report(p, strict=False); print("\nDEV run: exploratory only.")
    elif a.mode == "freeze":
        plan, h = plan_hash(make_jobs(True))
        if os.path.exists(PLAN) and json.load(open(PLAN))["sha256"] != h: raise AssertionError("plan differs from the frozen one")
        json.dump(dict(sha256=h, plan=plan, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(PLAN, "w"), indent=1, default=str); print("frozen", h[:16], "->", PLAN)
    else:
        if a.no_gnn or a.densities or a.n_rep: raise SystemExit("--no-gnn / --densities / --n-rep are for `dev` only")
        if not os.path.exists(PLAN): raise SystemExit("freeze the plan first: run_city_gnn.py freeze")
        _, h = plan_hash(make_jobs(True))
        if json.load(open(PLAN))["sha256"] != h: raise SystemExit("the plan or the confirm cities changed after freezing")
        p = run(a.mode, f"city_gnn_{a.mode}", make_jobs(True), a.n_jobs); report(p, strict=True)
