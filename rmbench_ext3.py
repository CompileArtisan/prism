"""rmbench_ext3 -- third layer (round 2: a pre-specified replication on fresh data).
rmbench_core.py, rmbench_nn.py, rmbench_ext.py, rmbench_ext2.py and run_ext.py stay UNTOUCHED (round-1 numbers remain reproducible);
this file only adds.

What it adds
  A. SODIndoorLoc (3 buildings in different cities, Wi-Fi RSS, UJI-like schema): downloader, reader, a fake of the same schema for
     the SMOKE run, and a unit builder that can EXCLUDE units already used in round 1 (so UJI units in round 2 are new APs).
  B. Two comparators that use the same side information as `rk_side`, so that every question is a single-factor contrast:
        rk_side   = ridge(side covariates)  + IDW      of the remaining residual      (round 1, rmbench_ext2)
        krig_side = ridge(side covariates)  + global ordinary kriging of the residual (new)
        kgnn_side = ridge(side covariates)  + KGNN                of the residual     (new)
        kgnn_side vs rk_side   -> does the learned kernel beat IDW on the same residual?
        kgnn_side vs krig_side -> does the learned kernel beat a fitted variogram on the same residual?
        kgnn_side vs kgnn      -> does side information help KGNN?
     They are made visible to rmbench_ext2's executor through `install()` (idempotent; it is called inside every worker).
  C. Pre-specified contrast lists, an equivalence margin, a layout-level (building-level) cluster bootstrap that respects the fact that
     floors of one building are not independent, per-building tables, pooling of real datasets, and a programmatic claim evaluation.
"""
from __future__ import annotations
import os, re, json, glob, time, traceback, urllib.request
import numpy as np, pandas as pd
from scipy.stats import binomtest
import rmbench_core as rc, rmbench_ext as rx, rmbench_ext2 as r2

# =============================================================================== A. SODIndoorLoc (verified schema, see notebook section 3)
SOD_RAW = "https://raw.githubusercontent.com/bijingxue/SODIndoorLoc/main"
SOD_FILES = {1: "CETC331/Training_CETC331.csv", 2: "HCXY/Training_HCXY_AP_30.csv", 3: "SYL/Training_SYL_AP_30.csv"}
SOD_NAMES = {1: "CETC331", 2: "HCXY", 3: "SYL"}
SOD_COLS = dict(bcol="BuildingID", fcol="FloorID", xcol="ECoord", ycol="NCoord")
SOD_PREFIX = "AP"            # MAC<k> of building b is renamed AP<b><k:04d>: MAC numbers repeat across buildings; the suffix after the prefix must stay an int

def fetch_sod(dest="data/sodindoorloc"):
    """Download the three training sheets (1 sample per RP for CETC331, 30 raw samples per RP for HCXY and SYL, pre-installed APs only).
    If the download fails, put the three files from https://github.com/bijingxue/SODIndoorLoc into `dest` by hand (names = basenames of SOD_FILES)."""
    os.makedirs(dest, exist_ok=True)
    for rel in SOD_FILES.values():
        p = os.path.join(dest, os.path.basename(rel))
        if os.path.exists(p) and os.path.getsize(p) > 1000: continue
        url = f"{SOD_RAW}/{rel}"; print("downloading", url)
        with urllib.request.urlopen(url, timeout=120) as r: data = r.read()
        with open(p, "wb") as fh: fh.write(data)
    return dest

def read_sod(dest):
    """Three sheets -> one table in the UJI-like schema build_wifi_units expects. Not detected = 100 (kept as is, nodet=100).
    The `_Avg` sheets are NOT used: they code not-detected as -105, which mixes level and detection."""
    frames = []
    for b, rel in SOD_FILES.items():
        p = os.path.join(dest, os.path.basename(rel))
        if not os.path.exists(p): raise FileNotFoundError(f"{p} missing; run r3.fetch_sod() or copy the file there")
        d = pd.read_csv(p); mac = {c: f"{SOD_PREFIX}{b}{int(c[3:]):04d}" for c in d.columns if re.fullmatch(r"MAC\d+", c)}
        if not mac: raise ValueError(f"{p}: no MAC<k> columns; header was {list(d.columns)[:6]}")
        miss = [c for c in ("ECoord", "NCoord", "FloorID") if c not in d.columns]
        if miss: raise ValueError(f"{p}: columns {miss} missing")
        frames.append(d.rename(columns=mac)[list(mac.values()) + ["ECoord", "NCoord", "FloorID"]].assign(BuildingID=b))      # BuildingID forced: never trust the file's own
    return pd.concat(frames, ignore_index=True, sort=False)

def make_fake_sod(dest, seed=0, n_waps=12):
    """Schema-faithful FAKE of the three SOD sheets (3 floors in building 1, one floor in buildings 2 and 3). Plumbing only, not data."""
    os.makedirs(dest, exist_ok=True); tmp = os.path.join(dest, "_fake_uji")
    rx.make_fake_uji(tmp, seed=seed, floors=(3, 1, 1), n_waps=n_waps)
    d = pd.read_csv(os.path.join(tmp, "trainingData.csv"))
    for b, rel in SOD_FILES.items():
        g = d[d.BUILDINGID == b - 1]
        out = pd.DataFrame({f"MAC{k + 1}": g[f"WAP{k + 1:03d}"].to_numpy() for k in range(n_waps)})
        out["ECoord"] = g.LONGITUDE.to_numpy() - g.LONGITUDE.min(); out["NCoord"] = g.LATITUDE.to_numpy() - g.LATITUDE.min()
        out["FloorID"] = g.FLOOR.to_numpy() + (1 if b == 1 else 4); out["BuildingID"] = b
        out.to_csv(os.path.join(dest, os.path.basename(rel)), index=False)
    return dest

def units_in_cache(cache_dir):
    """Names of the units stored in a cache folder (empty set if it does not exist)."""
    return {os.path.basename(p)[:-4].replace("__", "|") for p in glob.glob(os.path.join(cache_dir, "units", "*.npz"))}

def build_wifi_units3(df, cfg, dataset, bcol, fcol, xcol, ycol, wprefix="WAP", nodet=100, decimals=1, exclude=()):
    """r2.build_wifi_units with one addition: unit names in `exclude` (e.g. the round-1 units) are never chosen, so the per-floor cap is
    filled with APs that round 1 did not use. Everything else (median over captures, detection rule, hashing) is identical.
    Returns (ok_names, record) with the reason for every AP that was seen on a floor."""
    waps = [c for c in df.columns if c.startswith(wprefix)]; ok, rec = [], []; exclude = set(exclude)
    for (b, f), g in df.groupby([bcol, fcol]):
        key, R, frac = r2._floor_arrays(g, waps, xcol, ycol, nodet, decimals)
        med = R.groupby(key).median().where(frac >= cfg["min_det_frac"])
        xy = g[[xcol, ycol]].groupby(key).mean().to_numpy(float); xy = xy - xy.mean(0)
        cnt = med.notna().sum(); par = f"{dataset}|b{b}f{f}"
        elig = [w for w in waps if cnt[w] >= cfg["min_unit_points"]]
        fresh = [w for w in elig if f"{par}|{w}" not in exclude]
        chosen = sorted(fresh, key=lambda w: rc.stable_hash(cfg["seed"], b, f, w))[:cfg["uji_max_waps_per_floor"]]
        for w in waps:
            if cnt[w] == 0: continue
            why = ("kept" if w in chosen else "excluded (used in round 1)" if (w in elig and w not in fresh) else
                   "cut by per-floor cap" if w in elig else "too few detected points")
            rec.append(dict(unit=f"{par}|{w}", building=b, floor=f, wap=w, n_det_points=int(cnt[w]), reason=why))
        for w in chosen:
            m = med[w].notna().to_numpy(); pos, y = xy[m], med[w].to_numpy(float)[m]
            t = pos[int(np.argmax(y))]; rel = pos - t
            u = dict(name=f"{par}|{w}", scenario=dataset, layout=f"{dataset}_b{b}", parent=par, tx=int(str(w)[len(wprefix):] or 0), freq_ghz=2.4,
                     pos=pos, tx_xy=t, d=np.hypot(np.linalg.norm(rel, axis=1), r2.H0), az=np.arctan2(rel[:, 1], rel[:, 0]), los=np.zeros(len(y)), y=y,
                     offset_db=0.0, frac_kept=float(m.mean()), tx_known=0)
            rc.save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
    if hasattr(rc.get_unit, "cache_clear"): rc.get_unit.cache_clear()
    return ok, pd.DataFrame(rec)

# =============================================================================== B. side-information comparators sharing one residual
def _ridge_side(prep, tr):
    """The ridge step of r2.predict_rk_side, verbatim (same covariates prep['X'], same lambda grid, same leave-one-out choice). Returns
    (fit at ALL points, residual at the labeled points after the ridge, lambda). Labels at tr only."""
    X, res = prep["X"], prep["res"]; A = np.column_stack([np.ones(len(X)), X]); At, yt = A[tr], res[tr]
    pen = np.eye(A.shape[1]); pen[0, 0] = 0.0; best = (np.inf, None, None)
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        Minv = np.linalg.solve(At.T @ At + lam * pen, At.T); H = At @ Minv; e = yt - H @ yt
        score = float(np.mean((e / np.clip(1 - np.diag(H), 1e-6, None)) ** 2))
        if score < best[0]: best = (score, lam, Minv @ yt)
    _, lam, beta = best; fit = A @ beta
    return fit, yt - fit[tr], float(lam)

def predict_krig_side(u, tr, te, prep, hp):
    """ridge(side) + global ordinary kriging (fitted exponential variogram) of the remaining residual."""
    fit, r_tr, lam = _ridge_side(prep, tr); model = hp.get("model", "exp")
    par = rc.fit_variogram(u["pos"][tr], r_tr, model)
    pred, nfb = rc.ok_predict(u["pos"][te], u["pos"][tr], r_tr, par, model, None)
    return prep["trend"][te] + fit[te] + pred, dict(rk_lambda=lam, nugget=par[0], psill=par[1], vrange=par[2], n_fallback=nfb)

def predict_kgnn_side(u, tr, te, prep, hp, seed):
    """ridge(side) + KGNN of the remaining residual. KGNN is called exactly as rc.predict_method calls it, on a prep whose residual is
    the post-ridge residual (and whose sigma is recomputed from the labeled post-ridge residual). Only res[tr] is ever indexed."""
    from rmbench_nn import fit_predict_kgnn
    fit, r_tr, lam = _ridge_side(prep, tr); res2 = prep["res"] - fit
    prep2 = dict(prep, res=res2, sigma=float(res2[tr].std() + 1e-6))
    p, info = fit_predict_kgnn(u, tr, te, prep2, hp, seed); info = dict(info); info["rk_lambda"] = lam
    return prep["trend"][te] + fit[te] + p, info

def install():
    """Teach rc.predict_method the methods 'krig_side' and 'kgnn_side' (idempotent). rmbench_ext2.run_task_ext2 looks `rc.predict_method` up at
    call time, so this is all it needs; run_task_ext3 calls it inside every worker process (spawned workers do not inherit patches)."""
    if getattr(rc.predict_method, "_ext3", False): return
    orig = rc.predict_method
    def predict_method3(method, hp, u, tr, te, prep, seed=0, graph_cache=None):
        if method == "krig_side": return predict_krig_side(u, tr, te, prep, hp)
        if method == "kgnn_side": return predict_kgnn_side(u, tr, te, prep, hp, seed)
        return orig(method, hp, u, tr, te, prep, seed, graph_cache)
    predict_method3._ext3 = True; rc.predict_method = predict_method3

def run_task_ext3(task):
    install(); return r2.run_task_ext2(task)

def run_tasks3(tasks, out_path, n_jobs=1, log_every=25):
    """r2.run_tasks2 (same resumable JSONL protocol) dispatching run_task_ext3."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks3] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
    t0, nf, shown = time.time(), 0, False
    enc = lambda o: None if o != o else float(o) if isinstance(o, np.floating) else str(o)
    def write(rows, fh):
        nonlocal nf, shown
        bad = [r for r in rows if r["status"] != "ok"]; nf += len(bad)
        if bad and not shown: shown = True; print(f"  first failure: {bad[0].get('name')} {bad[0].get('unit')}\n{bad[0].get('err', '')}")
        fh.write(json.dumps(rows, default=enc) + "\n"); fh.flush()
    with open(out_path, "a") as fh:
        if n_jobs <= 1:
            for i, t in enumerate(todo, 1):
                write(run_task_ext3(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task_ext3, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"],
                                                  status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

# =============================================================================== C. round-2 design: jobs, contrasts, margins
def conf2_jobs(side):
    """Round-2 methods. Defaults only (no tuning: in round 1 tuned and default KGNN did not differ significantly). No GCN / GAT arms."""
    J = rx.base_jobs()                                                     # trend, idw, krig_global, kgnn(Kmax=12, s2_init=0.3)
    if side:
        J += [dict(name="rk_side", method="rk_side", hp=dict(fset="geo_side")),
              dict(name="krig_side", method="krig_side", hp=dict(model="exp", fset="geo_side")),
              dict(name="kgnn_side", method="kgnn_side", hp=dict(Kmax=12, s2_init=0.3, fset="geo_side"))]
    return J

C2_KRIG = [("kgnn", "krig_global"), ("kgnn", "idw"), ("krig_global", "idw")]                                        # x densities
C2_SIDE = [("kgnn_side", "rk_side"), ("kgnn_side", "krig_side"), ("krig_side", "rk_side"), ("kgnn_side", "kgnn")]   # x densities
PRIMARY = [("kgnn", "krig_global"), ("kgnn_side", "rk_side")]
EQUIV_MARGIN_PCT = 2.0
MIN_PARENTS_EQUIV = 8        # fewer parents: a bootstrap CI is not trustworthy enough to declare equivalence
MIN_LAYOUTS_CLAIM = 5        # fewer layouts (buildings): no claim, whatever the parent-level test says

def family_spec(fd, claim_eligible):
    cache = fd["cfg"]["cache_dir"]; U = [rx.get_unit_ext(n, cache) for n in fd["names"]]
    return dict(n_units=len(U), n_parents=len({u["parent"] for u in U}), n_layouts=len({u["layout"] for u in U}), densities=[int(x) for x in fd["cfg"]["densities"]],
                n_test=int(fd["cfg"]["n_test"]), side=bool(fd["side"]), infer=fd.get("infer"), claim_eligible=bool(claim_eligible))

def pooled_spec(specs, members, dens):
    return dict(n_units=sum(specs[m]["n_units"] for m in members), n_parents=sum(specs[m]["n_parents"] for m in members),
                n_layouts=sum(specs[m]["n_layouts"] for m in members), densities=[int(x) for x in dens], n_test="per member", side=False, infer=None,
                claim_eligible=True, pooled_from=list(members))

def run_conf2(tag, fd, split, root, PRE, n_jobs):
    """One (family, split) cell: defaults-only evaluation on the fresh test salt and fresh replicate ids. Resumable."""
    cfg = r2.family_cfg(fd, split, PRE); out = os.path.join(root, tag); os.makedirs(out, exist_ok=True)
    tasks = r2.make_tasks2(fd["names"], cfg["densities"], PRE["reps"], conf2_jobs(fd["side"]), cfg, prefix=f"{tag}|{split}")
    path = os.path.join(out, f"{tag}_{split}.jsonl"); run_tasks3(tasks, path, n_jobs)
    return rc.read_results(path)

def pool_real(dfs, densities, name="real_wifi"):
    """Stack result tables of several real datasets into one family (scenario := name) restricted to the densities they share.
    parent (building-floor) and layout (building) stay as they are, so floor-level and building-level inference both remain possible."""
    parts = []
    for tag, df in dfs.items():
        p = df[df.density.isin([int(x) for x in densities])].copy(); p["scenario"] = name; p["source"] = tag; parts.append(p)
    return pd.concat(parts, ignore_index=True)

# =============================================================================== D. statistics
def _verdict(r):
    tail = ", practically equivalent" if r.equivalent else ""
    if r.significant: return ("A better (Holm)" if r.mean_delta_db > 0 else "A WORSE (Holm)") + tail
    if r.n_units < MIN_PARENTS_EQUIV: return f"inconclusive (only {int(r.n_units)} parents)"
    if r.equivalent: return f"equivalent (95% CI within +/-{r.margin_pct:g}%)"
    return "inconclusive"

def contrast_report3(d, contrasts, family, out, margin_pct, infer=None, quiet=False):
    """r2.contrast_report (parent-level paired test, bootstrap CI over parents, Holm over the whole call, wins AND losses) plus the CI as a
    percentage of B's RMSE and an equivalence verdict: 'equivalent' means the whole 95% CI of the relative difference lies inside +/- margin."""
    pc = r2.contrast_report(d, [tuple(c) for c in contrasts], "rmse", family, out, infer=infer, quiet=True)
    if pc is None or len(pc) == 0: return pc
    di = d if infer is None else d[d.scenario.isin(infer)]
    mB = rc.unit_level(di).groupby(["scenario", "density", "name"]).rmse.mean(); pc = pc.copy()
    pc["mean_B"] = [float(mB.get((s, k, b), np.nan)) for s, k, b in zip(pc.scenario, pc.density, pc.B)]
    pc["ci_lo_pct"] = 100 * pc.ci_lo / pc.mean_B; pc["ci_hi_pct"] = 100 * pc.ci_hi / pc.mean_B; pc["margin_pct"] = float(margin_pct)
    pc["equivalent"] = (pc.ci_lo_pct >= -margin_pct) & (pc.ci_hi_pct <= margin_pct); pc["verdict"] = pc.apply(_verdict, axis=1)
    os.makedirs(out, exist_ok=True); pc.to_csv(os.path.join(out, f"{family}_rmse.csv"), index=False)
    if not quiet:
        r2._show(pc[["scenario", "density", "A", "B", "n_units", "rel_gain_pct", "ci_lo_pct", "ci_hi_pct", "p_holm", "verdict", "underpowered"]],
                 f"{family} | rmse | CONFIRMATORY | {len(pc)} tests, Holm over all (rel_gain>0: A better; CI in % of RMSE_B)")
        v = pc.verdict.str
        print(f"  {int(v.startswith('A better').sum())} A better, {int(v.startswith('A WORSE').sum())} A worse, {int(v.startswith('equivalent').sum())} equivalent (not significant), "
              f"{int(pc.verdict.str.startswith('inconclusive').sum())} inconclusive; {int(pc.underpowered.sum())} underpowered")
    return pc

def layout_level_table(d, contrasts, n_boot=2000, seed=0):
    """Building-level view. Floors of one building share walls, so the independent unit is the layout. For every (scenario, density, contrast):
    delta per parent -> mean per layout -> mean over layouts (each building weighs the same), a two-stage cluster bootstrap CI (resample layouts,
    then parents inside each drawn layout), the number of layouts where A is better and an exact sign test over layouts (descriptive, not Holm-adjusted)."""
    d = d[d.status == "ok"]
    u = d.groupby(["scenario", "layout", "parent", "unit", "density", "name"], as_index=False).rmse.mean()
    p = u.groupby(["scenario", "layout", "parent", "density", "name"], as_index=False).rmse.mean(); rng = np.random.default_rng(seed); rows = []
    for (scen, dens), g in p.groupby(["scenario", "density"]):
        w = g.pivot_table(index=["layout", "parent"], columns="name", values="rmse")
        for A, B in contrasts:
            if A not in w or B not in w: continue
            x = (w[B] - w[A]).dropna()
            if x.empty: continue
            by = {L: v.to_numpy(float) for L, v in x.groupby(level="layout")}; Ls = list(by); lm = np.array([by[L].mean() for L in Ls]); bs = np.empty(n_boot)
            for i in range(n_boot):
                pick = rng.integers(len(Ls), size=len(Ls))
                bs[i] = np.mean([by[Ls[j]][0] if len(by[Ls[j]]) == 1 else rng.choice(by[Ls[j]], len(by[Ls[j]])).mean() for j in pick])
            k = int((lm > 0).sum()); nz = int((lm != 0).sum())
            mb = float(w[B].groupby(level="layout").mean().mean())
            rows.append(dict(scenario=scen, density=int(dens), A=A, B=B, n_layouts=len(Ls), n_parents=len(x), layouts_A_better=k, mean_delta_layout=float(lm.mean()),
                             rel_gain_pct_layout=100 * float(lm.mean()) / mb, ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                             sign_p=float(binomtest(k, nz, 0.5).pvalue) if nz else 1.0))
    out = pd.DataFrame(rows)
    if len(out): out["ci_excludes_0"] = (out.ci_lo > 0) | (out.ci_hi < 0)
    return out

def per_layout_gain(d, A, B, infer=None):
    """Relative RMSE gain (%) of A over B for every layout (building) and density. Positive = A better. Descriptive."""
    d = d[d.status == "ok"]; d = d if infer is None else d[d.scenario.isin(infer)]
    u = d.groupby(["scenario", "layout", "parent", "unit", "density", "name"], as_index=False).rmse.mean()
    p = u.groupby(["scenario", "layout", "parent", "density", "name"], as_index=False).rmse.mean()
    w = p.pivot_table(index=["scenario", "layout", "parent", "density"], columns="name", values="rmse")
    if A not in w or B not in w: return pd.DataFrame()
    g = w[[A, B]].dropna().groupby(level=["scenario", "layout", "density"]).mean(); g["rel_gain_pct"] = 100 * (g[B] - g[A]) / g[B]
    return g["rel_gain_pct"].unstack("density").round(2)

def report_conf2(df, tag, split, fd, out, PRE):
    """Tables for one (family, split): RMSE of every method, the pre-specified contrasts (krig list; side list where the family has side
    information), and the layout-level table. Returns ({'krig': pc, 'side': pc}, layout_level_table)."""
    fam = f"{tag}_{split}"; d = df[df.status == "ok"]; os.makedirs(out, exist_ok=True)
    print(f"\n##### {fam}: {len(df)} job rows, {len(d)} ok, {len(df) - len(d)} failed")
    if len(df) - len(d): print(df[df.status != "ok"].groupby("name").size().to_string())
    t = rx.agg_metric(d, "rmse"); t.to_csv(os.path.join(out, f"{fam}_rmse_table.csv")); r2._show(t, "rmse, parent-level mean (all methods)")
    m, infer = PRE["equivalence_margin_pct"], fd.get("infer"); lst = [tuple(c) for c in PRE["contrasts_krig"]]
    pcs = {"krig": contrast_report3(d, lst, f"{fam}_krig", out, m, infer)}
    if fd.get("side"): side = [tuple(c) for c in PRE["contrasts_side2"]]; pcs["side"] = contrast_report3(d, side, f"{fam}_side2", out, m, infer); lst += side
    di = d if infer is None else d[d.scenario.isin(infer)]
    cl = layout_level_table(di, lst); cl.to_csv(os.path.join(out, f"{fam}_layout_level.csv"), index=False)
    return pcs, cl

def evaluate_claims(PCS, CLS, PRE):
    """Programmatic claim check, exactly the rule written in the pre-registration. A contrast is SUPPORTED for a family only if, in EVERY
    split and at EVERY density of the family: Holm-significant at parent level in the claimed direction AND the layout-level cluster CI excludes 0
    in the same direction. 'EQUIVALENT' if the 95% CI lies inside the margin in every cell. Only claim-eligible families are evaluated."""
    rows = []
    for tag, spec in PRE["families"].items():
        if not spec.get("claim_eligible"): continue
        for kind, lst in (("krig", PRE["contrasts_krig"]), ("side", PRE["contrasts_side2"])):
            if kind == "side" and not spec["side"]: continue
            for A, B in map(tuple, lst):
                cells = []
                for sp in PRE["splits"]:
                    pc = (PCS.get((tag, sp)) or {}).get(kind); cl = CLS.get((tag, sp))
                    for dens in spec["densities"]:
                        if pc is None or len(pc) == 0 or cl is None or len(cl) == 0: cells.append("missing"); continue
                        r = pc[(pc.A == A) & (pc.B == B) & (pc.density == dens)]; c = cl[(cl.A == A) & (cl.B == B) & (cl.density == dens)]
                        if r.empty or c.empty: cells.append("missing"); continue
                        r, c = r.iloc[0], c.iloc[0]
                        if c.n_layouts < MIN_LAYOUTS_CLAIM: cells.append("open"); continue
                        cells.append("better" if (r.significant and r.mean_delta_db > 0 and c.ci_lo > 0) else "worse" if (r.significant and r.mean_delta_db < 0 and c.ci_hi < 0)
                                     else "equiv" if (r.equivalent and r.n_units >= MIN_PARENTS_EQUIV) else "open")
                n = len(cells); cnt = {k: cells.count(k) for k in ("better", "worse", "equiv", "open", "missing")}
                st = ("not run" if cnt["missing"] == n else "SUPPORTED: A better" if cnt["better"] == n else "SUPPORTED: A worse" if cnt["worse"] == n else
                      f"EQUIVALENT (+/-{PRE['equivalence_margin_pct']:g}%)" if cnt["equiv"] == n else "NOT SUPPORTED")
                rows.append(dict(family=tag, A=A, B=B, primary=(A, B) in [tuple(x) for x in PRE["primary"]], status=st, cells=n, **cnt))
    return pd.DataFrame(rows)
