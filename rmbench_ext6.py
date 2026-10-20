"""rmbench_ext6 -- round 6. ADDS ONLY: rmbench_core / nn / ext / ext2 / ext3 / ext4 / ext5 stay untouched. Put next to them.

What changes relative to v4/v5 (numbers = your five points)
  1. WITHIN-MAP NONLINEAR COMPARATORS  rk_onehot (quantile-bin one-hot + ridge), rk_spline (additive cubic splines + ridge), rk_gbm (gradient boosting,
     n_estimators by 5-fold CV, out-of-fold residuals) -- each = regression-kriging with IDW of the leftover residual, exactly like rk_side. Every one comes
     ungated and CV-gated against rk_side (`_g`), the same do-no-harm gate the cross-layout prior uses, so no arm has a gating advantage.
  2. GLOBAL RAW PLANT  f is a function of the RAW covariates scaled by FIXED constants (fit once on SOURCE maps and stored in the plant dict), never
     per-map z-scores, so it is literally the same function on every map. The pre-trained prior now sees those same globally scaled covariates.
  3. HIGH-DIMENSIONAL SIDE INFORMATION  d_extra synthetic covariates per map (smooth random field + iid mix, `rho`) are appended to u['side'];
     f = sum of 32 pairwise products of ridge functions of 3 random coordinates each (zero linear part, interactions only). With ~30 covariates and
     ~100 labels a single map cannot learn it; many layouts can. This is SEMI-SYNTHETIC and must be labelled as such.
  4. D0 (`d0_table`, `d0_verdict`) now measures the LEARNABILITY GAP: corr(prior_K(X), f) on held-out maps next to the same correlation for every
     within-map comparator at n labels (and at a large n, to show the gap is a label-budget gap, not a capacity gap). Stage 2 is allowed only if D0 passes.
  5. FRESH LAYOUTS  `fresh_pool` refuses seeds in the ledger of burned seeds (11, 12, 21 pool; 5, 7 simulator), `spend_seeds` burns the new ones once Stage 2 has
     COMPLETED (the frozen plan already pins the seeds, so a partial run cannot be re-seeded), `freeze_plan` writes a sha256 of the plan and refuses a changed plan later.

Every predictor: f(u, tr, te, prep, hp, seed) -> (prediction in dBm at te, info dict); labels are read at tr only (leak_check applies).
"""
from __future__ import annotations
import os, json, time, math, copy, pickle, hashlib, warnings, traceback
import numpy as np, pandas as pd
import rmbench_core as rc, rmbench_ext5 as x5

BURNED = dict(pool=[11, 12, 21], cfg=[5, 7])
RIDGE_LAMS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

# =============================================================================== 1. plant: global raw covariates, high-dim side information
def _side2d(u):
    S = np.asarray(u["side"], float); return S.reshape(len(S), -1)

def _aug_fields(u0, d_extra, rho, seed, n_freq=24):
    """d_extra per-map covariates, each ~N(0,1) marginally: sqrt(rho)*smooth random Fourier field + sqrt(1-rho)*iid. Deterministic in (unit name, seed)."""
    N = len(u0["y"]); P = np.asarray(u0["pos"], float); rng = np.random.default_rng(int(rc.stable_hash("aug6", u0["name"], int(seed))))
    ell = 0.25 * (np.ptp(P, 0).max() + 1e-9)
    W = rng.normal(size=(d_extra, n_freq, 2)) / ell; ph = rng.uniform(0, 2 * np.pi, (d_extra, n_freq))
    sm = math.sqrt(2.0 / n_freq) * np.cos(np.einsum("nc,dfc->ndf", P, W) + ph[None]).sum(2)
    return math.sqrt(rho) * sm + math.sqrt(1 - rho) * rng.normal(size=(N, d_extra))

def side_full(u0, pl):
    S0 = _side2d(u0)
    return np.column_stack([S0, _aug_fields(u0, pl["d_extra"], pl["rho"], pl["seed"])]) if pl["d_extra"] > 0 else S0

def global_Z(S, pl):
    """FIXED scaling constants stored in the plant dict (fit once on source maps). Never per-map."""
    return (np.asarray(S, float) - np.asarray(pl["mu"])) / (np.asarray(pl["sd"]) + 1e-9)

def _f_raw(Z, pl):
    d = Z.shape[1]; rng = np.random.default_rng(int(pl["seed"])); out = np.zeros(len(Z))
    for _ in range(int(pl["n_terms"])):
        ia, ib = rng.choice(d, 3, replace=False), rng.choice(d, 3, replace=False); wa, wb, a = rng.normal(size=3), rng.normal(size=3), rng.normal()
        out += a * np.tanh(Z[:, ia] @ wa) * np.tanh(Z[:, ib] @ wb)           # product of two ridge functions: even in the joint sign, ~0 linear projection
    return out

def f_true(u0, pl):
    return (_f_raw(global_Z(side_full(u0, pl), pl), pl) - pl["f_mu"]) / (pl["f_sd"] + 1e-12)

def make_plant(source_pairs, amp=0.3, d_extra=24, rho=0.3, seed=0, n_terms=32, n_maps=24):
    """Fit the global constants on SOURCE maps only (evaluation maps never touch them). source_pairs = [(unit name, cache dir)]."""
    import rmbench_ext as rx
    pl = dict(amp=float(amp), d_extra=int(d_extra), rho=float(rho), seed=int(seed), n_terms=int(n_terms), mu=None, sd=None, f_mu=0.0, f_sd=1.0)
    pick = [source_pairs[i] for i in np.unique(np.linspace(0, len(source_pairs) - 1, min(n_maps, len(source_pairs))).astype(int))]
    S = np.concatenate([side_full(rx.get_unit_ext(n, c), pl) for n, c in pick]); pl["mu"] = S.mean(0).tolist(); pl["sd"] = (S.std(0) + 1e-9).tolist()
    f = _f_raw(global_Z(S, pl), pl); pl["f_mu"] = float(f.mean()); pl["f_sd"] = float(f.std() + 1e-9); return pl

def with_amp(pl, amp): return {**pl, "amp": float(amp)}

def plant6(u0, cfg):
    """No-op unless cfg['plant6']. amp=0 still appends the covariates (the false-positive control sees the same inputs). Uses side information only, never labels."""
    pl = (cfg or {}).get("plant6")
    if not pl: return u0
    if "side" not in u0: raise KeyError("plant6 needs u['side']")
    u1 = dict(u0); u1["side"] = side_full(u0, pl); f = f_true(u0, pl); y = np.asarray(u0["y"], float)
    u1["y"] = y + pl["amp"] * float(np.std(y)) * f; u1["f_true"] = f; return u1

# =============================================================================== 2. within-map nonlinear comparators (regression-kriging with a nonlinear mean)
def _split_X(u, prep):
    X = np.asarray(prep["X"], float); ns = _side2d(u).shape[1]; return X[:, :X.shape[1] - ns], X[:, X.shape[1] - ns:]      # geo | side (per-map standardised)

def _idw_on(u, tr, te, prep, fit_te, resid_tr):
    p, k = rc.tune_idw_loo(u["pos"][tr], resid_tr)
    return prep["trend"][te] + fit_te + rc.idw_predict(u["pos"][te], u["pos"][tr], resid_tr, p, k), dict(idw_power=p, idw_k=k)

def _ridge_fit(A, tr, res):
    At, yt = A[tr], res[tr]; pen = np.eye(A.shape[1]); pen[0, 0] = 0.0; best = (np.inf, None)
    for lam in RIDGE_LAMS:
        Minv = np.linalg.solve(At.T @ At + lam * pen, At.T); H = At @ Minv; e = yt - H @ yt
        sc = float(np.mean((e / np.clip(1 - np.diag(H), 1e-6, None)) ** 2))
        if sc < best[0]: best = (sc, lam, Minv @ yt)
    return best[1], best[2]

def _design(kind, u, tr, prep):
    G, S = _split_X(u, prep); one = np.ones((len(G), 1))
    if kind == "onehot":
        cols = []
        for j in range(S.shape[1]):
            e = np.quantile(S[tr, j], (0.25, 0.5, 0.75)); b = np.searchsorted(e, S[:, j]); cols.append(np.eye(4)[b][:, 1:])
        return np.column_stack([one, G] + cols)
    if kind == "spline":
        from sklearn.preprocessing import SplineTransformer
        T = SplineTransformer(n_knots=4, degree=3, knots="quantile", extrapolation="constant").fit(S[tr]); return np.column_stack([one, G, T.transform(S)])
    raise ValueError(kind)

def _rk_ridge(kind, u, tr, te, prep):
    A = _design(kind, u, tr, prep); lam, beta = _ridge_fit(A, tr, prep["res"]); fit = A @ beta
    pred, info = _idw_on(u, tr, te, prep, fit[te], prep["res"][tr] - fit[tr]); info["lam"] = float(lam); return pred, info

def _rk_gbm(u, tr, te, prep, seed, stages=(25, 50, 100, 200, 300)):
    from sklearn.ensemble import GradientBoostingRegressor
    X = np.asarray(prep["X"], float); y = prep["res"]; n = len(tr); rng = np.random.default_rng(seed); perm = rng.permutation(n); folds = np.array_split(perm, 5)
    mk = lambda: GradientBoostingRegressor(n_estimators=max(stages), learning_rate=0.05, max_depth=3, subsample=0.7, min_samples_leaf=3, random_state=int(seed))
    err = np.zeros(len(stages)); oof = np.zeros((len(stages), n))
    for h in folds:
        keep = np.ones(n, bool); keep[h] = False; m = mk().fit(X[tr[keep]], y[tr[keep]]); st = list(m.staged_predict(X[tr[h]]))
        for i, s in enumerate(stages): oof[i, h] = st[s - 1]
    err = ((oof - y[tr][None]) ** 2).mean(1); i = int(np.argmin(err)); s = stages[i]
    m = mk().fit(X[tr], y[tr]); fit_te = list(m.staged_predict(X[te]))[s - 1]
    pred, info = _idw_on(u, tr, te, prep, fit_te, y[tr] - oof[i]); info["gbm_n_est"] = float(s); return pred, info

def _gate(u, tr, base_fn, cand_fn, hp, seed):
    """5-fold CV inside the labelled set; accept the candidate only if paired z > hp.z and relative MSE gain > hp.rel (same rule as rk_prior)."""
    n = len(tr); rng = np.random.default_rng(seed); y = np.asarray(u["y"], float); perm = rng.permutation(n); e0 = np.zeros(n); e1 = np.zeros(n)
    for f in np.array_split(np.arange(n), int(hp.get("folds", 5))):
        hold = perm[f]; keep = np.ones(n, bool); keep[hold] = False; trf, tef = tr[keep], tr[hold]
        e0[hold] = (np.asarray(base_fn(trf, tef), float) - y[tef]) ** 2; e1[hold] = (np.asarray(cand_fn(trf, tef), float) - y[tef]) ** 2
    d = e0 - e1; z = float(d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12)); gain = float(d.mean() / (e0.mean() + 1e-12))
    return bool(z > float(hp.get("z", 2.5)) and gain > float(hp.get("rel", 0.01))), dict(cv_z=z, cv_gain_rel=gain)

def _predict_within(kind, u, tr, te, prep, hp, seed=0):
    import rmbench_ext2 as r2
    tr = np.asarray(tr); te = np.asarray(te); rk_hp = dict(fset="geo_side")
    cand = (lambda a, b, p=prep: _rk_gbm(u, a, b, p, seed)[0]) if kind == "gbm" else (lambda a, b, p=prep: _rk_ridge(kind, u, a, b, p)[0])
    pred, info = (_rk_gbm(u, tr, te, prep, seed) if kind == "gbm" else _rk_ridge(kind, u, tr, te, prep)); info["accepted"] = 1.0
    if not hp.get("gate", False): return pred, info
    base_pred, _ = r2.predict_rk_side(u, tr, te, prep, rk_hp); info["accepted"] = 0.0
    if len(tr) < int(hp.get("min_labels", 30)): return base_pred, info
    ok, g = _gate(u, tr, lambda a, b: r2.predict_rk_side(u, a, b, prep, rk_hp)[0], cand, hp, seed); info.update(g)
    if ok: info["accepted"] = 1.0; return pred, info
    return base_pred, info

def within_jobs(gated=True):
    J = []
    for k in ("onehot", "spline", "gbm"):
        J.append(dict(name=f"rk_{k}", method=f"rk_{k}", hp=dict(fset="geo_side", gate=False)))
        if gated: J.append(dict(name=f"rk_{k}_g", method=f"rk_{k}", hp=dict(fset="geo_side", gate=True, z=2.5, rel=0.01)))
    return J

# =============================================================================== 3. cross-layout prior on the GLOBALLY scaled covariates
def _X6(u, prep, pl):
    G, _ = _split_X(u, prep); return np.column_stack([G, global_Z(_side2d(u), pl)]).astype(np.float32)

def episode_arrays6(name, cache_dir, d, r, cfg, hp, plant=None, max_pts=600):
    """Same contract as x5.episode_arrays (patched in for the duration of pre-training), but inputs = [geometry | GLOBALLY scaled covariates]."""
    import rmbench_ext as rx, rmbench_ext2 as r2
    u0 = plant6(rx.get_unit_ext(name, cache_dir), dict(plant6=plant))
    tr, te = r2.split_for2(u0, d, 700 + r, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
    u, spec = r2.view_for_split2(u0, tr); prep = r2.prepare2(u, tr, "geo_side", cfg["clip"], spec)
    X = _X6(u, prep, plant); y = np.clip(np.asarray(prep["res"], float) / float(prep["sigma"]), -6, 6).astype(np.float32)
    idx = np.sort(np.random.default_rng(1000 * int(r) + int(d)).choice(len(y), min(len(y), max_pts), replace=False))
    return dict(X=X[idx], r=y[idx], f=np.asarray(u0["f_true"])[idx].astype(np.float32))

def pretrain_prior_scale6(SRC, VAL, Ks, cfg, out_dir, tag, plant, **kw):
    """x5.pretrain_prior_scale with x5.episode_arrays temporarily replaced by episode_arrays6. Source layouts only."""
    old = x5.episode_arrays; x5.episode_arrays = episode_arrays6
    try: return x5.pretrain_prior_scale(SRC, VAL, Ks, cfg, out_dir, tag, plant=plant, **kw)
    finally: x5.episode_arrays = old

def predict_rk_prior6(u, tr, te, prep, hp, seed=0):
    import rmbench_ext2 as r2
    tr = np.asarray(tr); te = np.asarray(te); rk_hp = dict(fset="geo_side"); ck = x5._load_prior(hp["init_path"]); pl = ck["plant"]
    X = _X6(u, prep, pl)
    if X.shape[1] != ck["din"]: raise ValueError(f"prior expects {ck['din']} features, this map gives {X.shape[1]}")
    sig = float(prep["sigma"]); p = np.asarray(ck["net"].predict(X), float) * sig
    prep1 = copy.copy(prep); prep1["res"] = np.asarray(prep["res"], float) - p
    base_pred, _ = r2.predict_rk_side(u, tr, te, prep, rk_hp); cand = np.asarray(r2.predict_rk_side(u, tr, te, prep1, rk_hp)[0], float) + p[te]
    info = dict(accepted=0.0, cv_z=0.0, cv_gain_rel=0.0, prior_std_sigma=float(p.std() / (sig + 1e-12)))
    if not hp.get("gate", True): info["accepted"] = 1.0; return cand, info
    if len(tr) < int(hp.get("min_labels", 30)): return base_pred, info
    ok, g = _gate(u, tr, lambda a, b: r2.predict_rk_side(u, a, b, prep, rk_hp)[0], lambda a, b: np.asarray(r2.predict_rk_side(u, a, b, prep1, rk_hp)[0], float) + p[b], hp, seed)
    info.update(g)
    if ok: info["accepted"] = 1.0; return cand, info
    return base_pred, info

def prior_jobs6(paths, z=2.5, rel=0.01, tag="prior6"):
    J = []
    for K, pth in paths.items():
        J.append(dict(name=f"{tag}_K{K}", method="rk_prior6", hp=dict(fset="geo_side", init_path=pth, gate=True, z=z, rel=rel)))
        J.append(dict(name=f"{tag}_K{K}_ng", method="rk_prior6", hp=dict(fset="geo_side", init_path=pth, gate=False)))
    return J

# =============================================================================== 4. D0: learnability gap
def _corr(a, b): a = np.asarray(a, float); b = np.asarray(b, float); return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0

def d0_table(plant, prior_path, eval_pairs, cfg, n_labels=(100, 1000), n_units=8, hp_cmp=None):
    """For each held-out EVALUATION map and label budget n: correlation with the true planted f (on test points) of
         prior   = pre-trained prior(X), uses NO labels of the map;
         within-map comparators fitted on n labels of that map (rk_side, rk_onehot, rk_spline, rk_gbm), prediction minus trend.
       Linear R2 of f on the global covariates is reported as a check that f is not linear by construction."""
    import rmbench_ext as rx, rmbench_ext2 as r2
    ck = x5._load_prior(prior_path); rows = []
    for n, c in eval_pairs[:n_units]:
        u0 = plant6(rx.get_unit_ext(n, c), dict(plant6=plant)); f = np.asarray(u0["f_true"])
        for nl in n_labels:
            if nl > len(f) - cfg["n_test"] - 5: continue
            tr, te = r2.split_for2(u0, nl, 900, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
            u, spec = r2.view_for_split2(u0, tr); prep = r2.prepare2(u, tr, "geo_side", cfg["clip"], spec)
            X6 = _X6(u, prep, plant); p = np.asarray(ck["net"].predict(X6), float); A = np.column_stack([np.ones(len(X6)), X6]); lin = A @ np.linalg.lstsq(A, f, rcond=None)[0]
            row = dict(unit=n, n_labels=nl, corr_prior=_corr(p[te], f[te]), linear_r2_f_on_X=float(1 - np.mean((lin - f) ** 2) / np.var(f)))
            tre = prep["trend"][te]
            row["corr_rk_side"] = _corr(r2.predict_rk_side(u, tr, te, prep, dict(fset="geo_side"))[0] - tre, f[te])
            for k in ("onehot", "spline", "gbm"): row[f"corr_rk_{k}"] = _corr(_predict_within(k, u, tr, te, prep, dict(gate=False), 0)[0] - tre, f[te])
            rows.append(row)
    return pd.DataFrame(rows)

def d0_verdict(D0, n_ref=100, min_prior=0.5, min_gap=0.25):
    """PASS needs (a) the prior recovers f on held-out maps and (b) the best within-map comparator at n_ref labels is clearly worse. Otherwise cross-layout
    transfer has nothing to add and Stage 2 must not be run. Also returns the within-map corr at the largest n (capacity check)."""
    w = [c for c in D0.columns if c.startswith("corr_rk_")]; s = D0[D0.n_labels == n_ref]
    if not len(s): raise ValueError(f"no rows at n_labels={n_ref}")
    prior, best = float(s.corr_prior.median()), float(s[w].max(axis=1).median()); big = D0[D0.n_labels == D0.n_labels.max()]
    out = dict(n_ref=n_ref, median_corr_prior=prior, median_best_within=best, gap=prior - best, median_linear_r2=float(s.linear_r2_f_on_X.median()),
               median_best_within_at_max_n=float(big[w].max(axis=1).median()), n_max=int(D0.n_labels.max()))
    out["pass"] = bool(prior >= min_prior and out["gap"] >= min_gap); return out

# =============================================================================== 5. fresh layouts, plan freeze, executor
def _ledger(path):
    if os.path.exists(path):
        with open(path) as fh: return json.load(fh)
    return {k: list(v) for k, v in BURNED.items()}

def fresh_pool(cfg_base, root, tag, pool_seed, sim_seed, n_eval, n_tx, n_pts, used_pairs, ledger_path):
    """Build NEVER-USED evaluation layouts. Refuses seeds that were already spent, checks content fingerprints against every layout used so far."""
    led = _ledger(ledger_path)
    if pool_seed in led["pool"] or sim_seed in led["cfg"]: raise AssertionError(f"seed already burned: pool {pool_seed} in {led['pool']} or sim {sim_seed} in {led['cfg']}")
    cfg = {**cfg_base, "cache_dir": f"{root}/cache_{tag}", "test_salt": tag, "seed": int(sim_seed)}
    names, _ = x5.build_scale_pool(cfg, n_eval, 0, n_tx, n_pts, seed=int(pool_seed)); pairs = [(n, cfg["cache_dir"]) for n in names]
    x5.assert_disjoint(pairs, used_pairs, "fresh evaluation", "previously used"); return cfg, names, pairs

def spend_seeds(ledger_path, pool_seed, sim_seed):
    """Call ONCE, after Stage 2 has completed and check_run passed: from now on these layouts are development layouts."""
    led = _ledger(ledger_path); led["pool"].append(int(pool_seed)); led["cfg"].append(int(sim_seed)); os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    with open(ledger_path, "w") as fh: json.dump(led, fh)

def freeze_plan(plan, path):
    h = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
    if os.path.exists(path):
        with open(path) as fh: old = json.load(fh)["sha256"]
        if old != h: raise AssertionError(f"plan changed after freezing ({old[:12]} -> {h[:12]}). Stage 2 results would not be confirmatory.")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh: json.dump(dict(sha256=h, plan=plan, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1, default=str)
    return h

def install():
    """Idempotent; spawned workers do not inherit patches, so every worker calls this."""
    x5.NEW_METHODS.update(rk_onehot=lambda u, tr, te, prep, hp, seed=0: _predict_within("onehot", u, tr, te, prep, hp, seed),
                          rk_spline=lambda u, tr, te, prep, hp, seed=0: _predict_within("spline", u, tr, te, prep, hp, seed),
                          rk_gbm=lambda u, tr, te, prep, hp, seed=0: _predict_within("gbm", u, tr, te, prep, hp, seed), rk_prior6=predict_rk_prior6)
    if getattr(x5._maybe_plant, "_ext6", False): return
    orig = x5._maybe_plant
    def mp(u0, cfg): return plant6(u0, cfg) if (cfg or {}).get("plant6") else orig(u0, cfg)
    mp._ext6 = True; x5._maybe_plant = mp
install()

def run_task6(task): install(); return x5.run_task5(task)
def leak_check6(jobs, unit, cfg, density=100, rep=0): install(); return x5.leak_check(jobs, unit, cfg, density=density, rep=rep)

def run_tasks6(tasks, out_path, n_jobs=1, log_every=25):
    """Resumable JSONL executor (same protocol as x5.run_tasks5) dispatching run_task6."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks6] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
    t0, nf, shown = time.time(), 0, False
    enc = lambda o: None if o != o else float(o) if isinstance(o, np.floating) else str(o)
    def write(rows, fh):
        nonlocal nf, shown
        bad = [r for r in rows if r["status"] != "ok"]; nf += len(bad)
        if bad and not shown: shown = True; print(f"  first failure: {bad[0].get('name')} {bad[0].get('unit')}\n{bad[0].get('err', '')}")
        fh.write(json.dumps(rows, default=enc) + "\n"); fh.flush()
    fail = lambda t: [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"], status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
    with open(out_path, "a") as fh:
        if n_jobs <= 1:
            for i, t in enumerate(todo, 1):
                write(run_task6(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task6, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception: rows = fail(futs[f])
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
