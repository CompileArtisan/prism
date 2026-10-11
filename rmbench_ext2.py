"""rmbench_ext2 -- second extension layer. rmbench_core.py, rmbench_nn.py, rmbench_ext.py and run_ext.py stay UNTOUCHED
(their exploratory numbers remain reproducible); this file only adds.

What it adds, by section
  A. WiFi (UJI-schema) data: a per-floor feasibility table, a builder with an honest `dropped` table (the old one mislabelled its
     reasons), and a generic adapter for any other fingerprint CSV (building / floor / x / y / WAP columns).
  B. Robust trend + transmitter localisation (labels only): multi-start localisation, three candidate trends (constant, fixed
     exponent n=2, fitted exponent) chosen by K-fold CV on the labeled set with the one-standard-error rule, predictions clipped to the
     labeled range. Every method shares it (it is the common trend). `trend_policy="legacy"` reproduces the old behaviour.
  C. A second simulator with many layouts and building geometry (random walls with doors, per-layout loss / exponent / shadowing).
     Side information = wall count along the TX-RX segment and a LOS flag; known to the models, NOT to IDW.
  D. Side-information features ("geo_side") and the fair classical comparator `rk_side` (ridge regression on the same covariates + IDW
     of the remaining residual), so a GNN that uses side information is not compared with a method that cannot.
  E. Executor `run_task_ext2` (fresh test salt, rep offsets, trend policies, gated outage metrics, skill score), tuning on units
     that are disjoint from the evaluation units (cross-fitting by parent or by layout), leave-layout-group-out transfer.
  F. Statistics: written pre-registration with a hash, full contrast tables with CI + Holm p (wins AND losses), master table,
     outage reporting only where the target is predictable.
"""
from __future__ import annotations
import os, json, time, math, hashlib, datetime, traceback
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import torch
import rmbench_core as rc, rmbench_ext as rx

H0 = rx.H0

def _show(df, title):
    print(f"\n=== {title}")
    print(df.round(3).to_string())

# =============================================================================== A. WiFi (UJI-schema) data
UJI_COLS = dict(bcol="BUILDINGID", fcol="FLOOR", xcol="LONGITUDE", ycol="LATITUDE")
SETTINGS = [(60, 40), (45, 40), (45, 30), (30, 30), (30, 20), (20, 20)]       # (max density, n_test), preference order
DENS = {60: (15, 30, 60), 45: (15, 30, 45), 30: (10, 20, 30), 20: (5, 10, 20)}

def read_uji(csv_dir):
    fr = [pd.read_csv(os.path.join(csv_dir, f)) for f in ("trainingData.csv", "validationData.csv") if os.path.exists(os.path.join(csv_dir, f))]
    return pd.concat(fr, ignore_index=True)

def _floor_arrays(g, waps, xcol, ycol, nodet, decimals):
    key = pd.MultiIndex.from_arrays([g[xcol].round(decimals), g[ycol].round(decimals)]).factorize()[0]
    R = g[waps].where(g[waps] != nodet)
    return key, R, R.notna().groupby(key).mean()

def wifi_floor_table(df, bcol, fcol, xcol, ycol, wprefix="WAP", nodet=100, decimals=1, det_fracs=(0.5, 0.3),
                     settings=SETTINGS, extra_min_points=(110,)):
    """Counts only (no model, no RMSE): for every building-floor, the number of reference points and how many APs are detected at >= T
    reference points, for each detection-fraction rule and each T in {max_density + n_test + 1 per candidate setting} + extra_min_points
    (110 = the old setting). T must be at least max density + n_test + 1, otherwise rc.split_for cannot cut a full test set."""
    waps = [c for c in df.columns if c.startswith(wprefix)]
    Ts = sorted({d + n + 1 for d, n in settings} | set(extra_min_points)); rows = []
    for (b, f), g in df.groupby([bcol, fcol]):
        key, R, frac = _floor_arrays(g, waps, xcol, ycol, nodet, decimals)
        for fr in det_fracs:
            cnt = (frac >= fr).sum()
            for T in Ts:
                rows.append(dict(building=b, floor=f, n_rows=len(g), n_ref_points=len(frac), det_frac=fr, min_points=T, n_aps=int((cnt >= T).sum())))
    return pd.DataFrame(rows)

def suggest_wifi_setting(tab, min_floors=13, min_aps=3, settings=SETTINGS, det_fracs=(0.5, 0.3)):
    """First setting (preference order: stricter detection rule first, larger density / test set first) under which at least
    `min_floors` building-floors keep >= `min_aps` APs. If none does, the setting with the most floors is returned with reached=False."""
    rows = []
    for fr in det_fracs:
        for d, n in settings:
            T = d + n + 1; t = tab[(tab.det_frac == fr) & (tab.min_points == T)]
            rows.append(dict(dmax=d, n_test=n, det_frac=fr, min_points=T, floors_with_aps=int((t.n_aps >= min_aps).sum()),
                             floors_total=len(t), aps_total=int(t.n_aps.sum())))
    S = pd.DataFrame(rows); good = S[S.floors_with_aps >= min_floors]
    pick = good.iloc[0] if len(good) else S.sort_values(["floors_with_aps", "aps_total"], ascending=False).iloc[0]
    return S, dict(densities=DENS[int(pick.dmax)], n_test=int(pick.n_test), min_unit_points=int(pick.min_points),
                   min_det_frac=float(pick.det_frac), floors_ok=int(pick.floors_with_aps), reached=bool(len(good)))

def build_wifi_units(df, cfg, dataset, bcol, fcol, xcol, ycol, wprefix="WAP", nodet=100, decimals=1):
    """Fingerprint table -> units. One unit = one AP on one floor of one building. layout = building, parent = building-floor.
    Repeated captures at one reference point -> median RSSI over detections; the point counts only if >= cfg['min_det_frac'] of its captures
    saw the AP. Returns (ok_names, record) where record lists EVERY AP that was seen on a floor with the reason it was kept or not."""
    waps = [c for c in df.columns if c.startswith(wprefix)]; ok, rec = [], []
    for (b, f), g in df.groupby([bcol, fcol]):
        key, R, frac = _floor_arrays(g, waps, xcol, ycol, nodet, decimals)
        med = R.groupby(key).median().where(frac >= cfg["min_det_frac"])
        xy = g[[xcol, ycol]].groupby(key).mean().to_numpy(float); xy = xy - xy.mean(0)
        cnt = med.notna().sum()
        elig = [w for w in waps if cnt[w] >= cfg["min_unit_points"]]
        chosen = sorted(elig, key=lambda w: rc.stable_hash(cfg["seed"], b, f, w))[:cfg["uji_max_waps_per_floor"]]
        for w in waps:
            if cnt[w] == 0: continue
            why = "kept" if w in chosen else ("cut by per-floor cap" if w in elig else f"too few detected points (< {cfg['min_unit_points']})")
            rec.append(dict(unit=f"{dataset}|b{b}f{f}|{w}", building=b, floor=f, wap=w, n_det_points=int(cnt[w]), reason=why))
        for w in chosen:
            m = med[w].notna().to_numpy(); pos, y = xy[m], med[w].to_numpy(float)[m]
            t = pos[int(np.argmax(y))]; rel = pos - t
            u = dict(name=f"{dataset}|b{b}f{f}|{w}", scenario=dataset, layout=f"{dataset}_b{b}", parent=f"{dataset}|b{b}f{f}", tx=int(str(w)[len(wprefix):] or 0),
                     freq_ghz=2.4, pos=pos, tx_xy=t, d=np.hypot(np.linalg.norm(rel, axis=1), H0), az=np.arctan2(rel[:, 1], rel[:, 0]),
                     los=np.zeros(len(y)), y=y, offset_db=0.0, frac_kept=float(m.mean()), tx_known=0)
            rc.save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
    rc.get_unit.cache_clear()
    return ok, pd.DataFrame(rec)

# =============================================================================== B. robust trend + localisation (labels only)
def localize_robust(pos, y, h0=H0, n_max=6.0):
    """Fit y = a - 10 n log10(sqrt(|p - t|^2 + h0^2)) for (t, a, n) by soft-L1 least squares from FOUR starts (strongest point, mean of the
    top 3, power-weighted centroid, plain centroid) and keep the lowest-cost solution. No acceptance rule here: whether the fitted trend is
    used at all is decided by cross-validation in select_trend()."""
    lo, hi = pos.min(0), pos.max(0); pad = 0.3 * (hi - lo + 1.0)
    order = np.argsort(y); w = 10 ** ((y - y.max()) / 10.0); w = w / w.sum()
    starts = [pos[order[-1]], pos[order[-3:]].mean(0), (w[:, None] * pos).sum(0), pos.mean(0)]
    best_cost, best = np.inf, None
    fun = lambda th: th[2] - 10 * th[3] * np.log10(np.sqrt(((pos - th[:2]) ** 2).sum(1) + h0 ** 2)) - y
    for s in starts:
        try:
            r = least_squares(fun, [s[0], s[1], y.max() + 25 * np.log10(h0), 2.5],
                              bounds=([lo[0] - pad[0], lo[1] - pad[1], -300, 0.0], [hi[0] + pad[0], hi[1] + pad[1], 100, n_max]),
                              loss="soft_l1", f_scale=3.0, max_nfev=200)
            if r.cost < best_cost: best_cost, best = r.cost, r.x[:2].copy()
        except Exception:
            pass
    return best if best is not None else starts[2].copy()

def _fit_all(pos, y, d_known, h0=H0):
    """Three candidate trends on one labeled set. d_known None -> transmitter unknown, localised here (one localisation shared by all)."""
    if d_known is None:
        tx = localize_robust(pos, y, h0); d = np.hypot(np.linalg.norm(pos - tx, axis=1), h0)
    else:
        tx, d = None, d_known
    lo, hi = float(y.min()), float(y.max()); a_f, n_f = rc.fit_trend(d, y); a_2, n_2 = rc.fit_trend_fixed(d, y, 2.0)
    return {"const": dict(kind="const", tx=tx, a=float(y.mean()), n=0.0, lo=lo, hi=hi),
            "fixed": dict(kind="fixed", tx=tx, a=a_2, n=n_2, lo=lo, hi=hi),
            "fit": dict(kind="fit", tx=tx, a=a_f, n=n_f, lo=lo, hi=hi)}

def _eval_spec(sp, pos_q, d_q, h0=H0):
    if sp["kind"] == "const": out = np.full(len(pos_q), sp["a"])
    else:
        d = d_q if sp["tx"] is None else np.hypot(np.linalg.norm(pos_q - sp["tx"], axis=1), h0)
        out = rc.trend_eval(d, sp["a"], sp["n"])
    return np.clip(out, sp["lo"], sp["hi"])          # a trend must not leave the range of the labels it was fitted on

def select_trend(u0, tr, h0=H0, force=None):
    """Choose the trend by K-fold CV on the labeled set ONLY (K = min(5, n_labeled // 3)); transmitter re-localised inside every training fold.
    One-standard-error rule: the simplest candidate (const < fixed n=2 < fitted n) whose CV RMSE is within one SE of the best."""
    y = u0["y"][tr]; pos = u0["pos"][tr]; known = bool(u0["tx_known"]); d_all = u0["d"][tr] if known else None
    final = _fit_all(pos, y, d_all, h0); order = ("const", "fixed", "fit"); cvm = {}
    if force is not None: kind = force
    elif len(tr) < 8: kind = "const"
    else:
        K = int(min(5, max(2, len(tr) // 3))); folds = np.array_split(np.random.default_rng(0).permutation(len(tr)), K)
        cv = {k: [] for k in order}
        for f in folds:
            trn = np.setdiff1d(np.arange(len(tr)), f)
            sp = _fit_all(pos[trn], y[trn], d_all[trn] if known else None, h0)
            for k in order:
                pr = _eval_spec(sp[k], pos[f], d_all[f] if known else None, h0); cv[k].append(float(np.sqrt(np.mean((pr - y[f]) ** 2))))
        cvm = {k: float(np.mean(v)) for k, v in cv.items()}; se = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in cv.items()}
        best = min(order, key=lambda k: cvm[k]); kind = next(k for k in order if cvm[k] <= cvm[best] + se[best])
    spec = dict(final[kind]); spec["cv_rmse"] = cvm.get(kind, float("nan")); spec["cv_const"] = cvm.get("const", float("nan")); spec["tx_loc"] = final["fit"]["tx"]
    return spec

def view_for_split2(u0, tr, h0=H0, force=None):
    """(unit as seen by the models, trend spec) for labeled set tr. Known transmitter: unit unchanged. Unknown: tx / d / az from the robust localisation."""
    spec = select_trend(u0, tr, h0, force)
    if u0["tx_known"]: return u0, spec
    t = spec["tx_loc"]; rel = u0["pos"] - t
    return {**u0, "tx_xy": t, "d": np.hypot(np.linalg.norm(rel, axis=1), h0), "az": np.arctan2(rel[:, 1], rel[:, 0])}, spec

def feature_matrix2(u, fset):
    """geo / geo_los as in rmbench_core. geo_side = geo + the unit's side information (u['side'] columns if present, else the LOS flag)."""
    if fset in ("geo", "geo_los"): return rc.feature_matrix(u, fset)
    if fset == "geo_side":
        cols = [rc.feature_matrix(u, "geo")]
        if "side" in u and np.asarray(u["side"]).size: S = np.asarray(u["side"], float); cols.append(S if S.ndim == 2 else S[:, None])
        else: cols.append(np.asarray(u["los"], float)[:, None])
        return np.column_stack(cols)
    raise ValueError(fset)

def prepare2(u, tr, fset, clip, spec):
    d = u["d"]
    trend = np.full(len(d), spec["a"]) if spec["kind"] == "const" else rc.trend_eval(d, spec["a"], spec["n"])
    trend = np.clip(trend, spec["lo"], spec["hi"]); res = u["y"] - trend
    Z, fr = rc.standardize(feature_matrix2(u, fset), tr, clip)
    return dict(a=spec["a"], n=spec["n"], res=res, sigma=float(res[tr].std() + 1e-6), X=Z, clipped=fr, trend=trend)

# =============================================================================== C. second simulator: many layouts, walls, side information
def _pieces(p0, p1, rl, n_doors, door=1.2):
    L = float(np.linalg.norm(p1 - p0)); dirn = (p1 - p0) / L; cur, segs = 0.0, []
    for c in np.sort(rl.uniform(0.1 * L, 0.9 * L, n_doors)):
        a, b = cur, c - door / 2
        if b - a > 0.3: segs.append((p0 + dirn * a, p0 + dirn * b))
        cur = max(cur, c + door / 2)
    if L - cur > 0.3: segs.append((p0 + dirn * cur, p0 + dirn * L))
    return segs

def _walls(rl, W, H):
    A, B, loss = [], [], []
    for x in np.sort(rl.uniform(0.15 * W, 0.85 * W, int(rl.integers(2, 5)))):
        l = rl.uniform(4, 10)
        for a, b in _pieces(np.array([x, 0.0]), np.array([x, H]), rl, int(rl.integers(1, 3))): A.append(a); B.append(b); loss.append(l)
    for y in np.sort(rl.uniform(0.2 * H, 0.8 * H, int(rl.integers(1, 3)))):
        l = rl.uniform(4, 10)
        for a, b in _pieces(np.array([0.0, y]), np.array([W, y]), rl, int(rl.integers(1, 3))): A.append(a); B.append(b); loss.append(l)
    return np.array(A), np.array(B), np.array(loss)

def _crossings(tx, pos, A, B):
    """(N, S) bool: does the segment tx -> pos[i] cross wall segment j (A[j] -> B[j])."""
    r = pos - tx; s = B - A; aq = A - tx
    den = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    nt = aq[None, :, 0] * s[None, :, 1] - aq[None, :, 1] * s[None, :, 0]
    nu = aq[None, :, 0] * r[:, None, 1] - aq[None, :, 1] * r[:, None, 0]
    ok = np.abs(den) > 1e-12; dd = np.where(ok, den, 1.0)
    t = np.where(ok, nt / dd, -1.0); uu = np.where(ok, nu / dd, -1.0)
    return ok & (t >= 0) & (t <= 1) & (uu >= 0) & (uu <= 1)

def make_wall_unit(layout_id, tx_id, seed=0, n_pts=500):
    """One transmitter in one random building. Layout-level draws: size, walls with doors, wall loss per wall, path-loss exponent, shadowing.
    y = -20 - 10 n log10(d) - sum(wall losses on the TX-RX segment) + correlated shadowing + 2 dB white noise. Layouts differ in geometry,
    so `layout` is a real random factor; transmitters of one layout share walls, so parent = layout (the statistical unit)."""
    rl = np.random.default_rng(rc.stable_hash("wall_layout", seed, layout_id))
    W, H = rl.uniform(30, 55), rl.uniform(20, 35); A, B, loss = _walls(rl, W, H)
    n_exp, sig, ell = rl.uniform(1.8, 2.6), rl.uniform(2.5, 4.5), rl.uniform(2.0, 4.0)
    rt = np.random.default_rng(rc.stable_hash("wall_tx", seed, layout_id, tx_id))
    tx = np.array([rt.uniform(0.1 * W, 0.9 * W), rt.uniform(0.1 * H, 0.9 * H)])
    pos = np.column_stack([rt.uniform(0, W, n_pts), rt.uniform(0, H, n_pts)]); rel = pos - tx
    d = np.hypot(np.linalg.norm(rel, axis=1), 2.0); cross = _crossings(tx, pos, A, B)
    nw = cross.sum(1).astype(float); wl = (cross * loss[None, :]).sum(1)
    y = -20 - 10 * n_exp * np.log10(d) - wl + rc._shadow_field(rt, pos, ell, sig) + rt.normal(0, 2.0, n_pts)
    lay = f"wall_L{layout_id:02d}"; los = (nw == 0).astype(float)
    return dict(name=f"wall|L{layout_id:02d}|tx{tx_id}", scenario="wall", layout=lay, parent=lay, tx=tx_id, freq_ghz=2.4, pos=pos, tx_xy=tx, d=d,
                az=np.arctan2(rel[:, 1], rel[:, 0]), los=los, y=y, offset_db=0.0, frac_kept=1.0, tx_known=1, side=np.column_stack([nw, los]))

def build_wall_units(cfg, n_layouts=12, n_tx=6, seed=0, n_pts=500):
    ok = []
    for L in range(n_layouts):
        for t in range(n_tx):
            u = make_wall_unit(L, t, seed, n_pts); rc.save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
    rc.get_unit.cache_clear(); return ok

# =============================================================================== D. side-information comparator
def predict_rk_side(u, tr, te, prep, hp):
    """Regression-kriging style: ridge on the SAME standardised covariates the GNN sees (log d, azimuth, dx, dy, + side information),
    ridge strength by leave-one-out on the labeled set, then IDW (power, k by LOO) of the remaining residual. Labels at tr only."""
    X, res = prep["X"], prep["res"]; A = np.column_stack([np.ones(len(X)), X]); At, yt = A[tr], res[tr]
    pen = np.eye(A.shape[1]); pen[0, 0] = 0.0; best = (np.inf, None, None)
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        Minv = np.linalg.solve(At.T @ At + lam * pen, At.T); H = At @ Minv; e = yt - H @ yt
        score = float(np.mean((e / np.clip(1 - np.diag(H), 1e-6, None)) ** 2))
        if score < best[0]: best = (score, lam, Minv @ yt)
    _, lam, beta = best; fit = A @ beta; r2 = yt - fit[tr]
    p, k = rc.tune_idw_loo(u["pos"][tr], r2)
    pred = prep["trend"][te] + fit[te] + rc.idw_predict(u["pos"][te], u["pos"][tr], r2, p, k)
    return pred, dict(rk_lambda=float(lam), idw_power=p, idw_k=k)

# =============================================================================== E. splits, metrics, executor
def split_for2(u, density, rep, n_test, mode="random", buffer=0.0, salt=""):
    """rc.split_for with a salt in the TEST-set seed: salt='' is bit-identical to rc.split_for; a new salt gives a new hold-out for the same unit.
    (rc.split_for seeds the test set by unit name only, so new replicates alone would re-use the old test points.)"""
    N = len(u["y"]); nt = min(n_test, N - density - 1)
    rng_t = np.random.default_rng(rc.stable_hash(*(("test", u["name"]) + ((salt,) if salt else ()))))
    if mode == "random": test = np.sort(rng_t.choice(N, nt, replace=False))
    elif mode == "blocked":
        centre = u["pos"][int(rng_t.integers(N))]
        test = np.sort(np.argsort(np.linalg.norm(u["pos"] - centre, axis=1), kind="stable")[:nt])
    else: raise ValueError(mode)
    pool = np.setdiff1d(np.arange(N), test)
    if mode == "blocked" and buffer > 0:
        from scipy.spatial import cKDTree
        pool = pool[cKDTree(u["pos"][test]).query(u["pos"][pool])[0] > buffer]
    if len(pool) < density: raise ValueError(f"only {len(pool)} labelable points left for density {density}")
    return np.random.default_rng([rc.stable_hash("rep", u["name"]), rep]).permutation(pool)[:density], test

def outage_metrics2(pred, true, thr, min_class=8):
    """Like rx.outage_metrics, but NaN when the test set has fewer than `min_class` outage or non-outage points (the metric would be constant or chance)."""
    to = true < thr
    if to.sum() < min_class or (~to).sum() < min_class: return dict(acc=np.nan, f1=np.nan, bacc=np.nan, area=np.nan)
    return rx.outage_metrics(pred, true, thr)

def cov_metrics2(pred, true, y_train, cfg):
    th = {f"q{int(round(100 * q))}": float(np.quantile(y_train, q)) for q in cfg.get("outage_q", ())}
    th.update({f"abs{int(round(-a))}": float(a) for a in cfg.get("outage_abs_dbm", ())}); out = {}
    for tag, t in th.items():
        for k, v in outage_metrics2(pred, true, t, int(cfg.get("outage_min_class", 8))).items(): out[f"{k}_{tag}"] = float(v)
    return out

_GRAPHS = {}

def run_task_ext2(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1"); torch.set_num_threads(1)
    cfg, rows = task["cfg"], []
    def fail_all(): return [dict(task_id=task["id"], unit=task["unit"], density=task["density"], rep=task["rep"], name=j["name"], method=j["method"],
                                 status="fail", err=traceback.format_exc(limit=3)) for j in task["jobs"]]
    try:
        u0 = rx.get_unit_ext(task["unit"], cfg["cache_dir"])
        tr, te = split_for2(u0, task["density"], task["rep"], cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
    except Exception:
        return fail_all()
    views, preps = {}, {}; yte = u0["y"][te]; vte = float(yte.var() + 1e-9)
    for j in task["jobs"]:
        hp = j.get("hp", {}); pol = hp.get("trend_policy", cfg.get("trend_policy", "robust")); fs = hp.get("fset", "geo")
        row = dict(task_id=task["id"], unit=u0["name"], scenario=u0["scenario"], layout=u0["layout"], parent=u0["parent"], density=task["density"],
                   rep=task["rep"], name=j["name"], method=j["method"], status="ok", err="", hp=json.dumps(hp, sort_keys=True, default=str), trend_policy=pol)
        t1 = time.time()
        try:
            if pol not in views:
                views[pol] = (rx.view_for_split(u0, tr), None) if pol == "legacy" else view_for_split2(u0, tr, force="const" if pol == "const" else None)
            u, spec = views[pol]
            if (pol, fs) not in preps:
                preps[(pol, fs)] = rc.prepare(u, tr, fs, cfg["clip"], cfg.get("trend_mode", "fit")) if pol == "legacy" else prepare2(u, tr, fs, cfg["clip"], spec)
            prep = preps[(pol, fs)]; seed = rc.stable_hash(u["name"], task["rep"], j["name"]) % 100000
            if j["method"] == "gnn": p, info = rx.fit_predict_graph(u, tr, te, prep, hp, seed, _GRAPHS); pred = prep["trend"][te] + p
            elif j["method"] == "rk_side": pred, info = predict_rk_side(u, tr, te, prep, hp)
            else: pred, info = rc.predict_method(j["method"], hp, u, tr, te, prep, seed, _GRAPHS)
            e = pred - yte
            if not np.all(np.isfinite(e)): raise FloatingPointError("non-finite prediction")
            mse = float(np.mean(e ** 2))
            row.update(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(e))), n_test=len(te), skill=1.0 - mse / vte,
                       trend_kind=(spec["kind"] if spec else "legacy"), trend_n=float(prep["n"]), tx_known=u0["tx_known"],
                       **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in info.items()},
                       **cov_metrics2(pred, yte, u["y"][tr], cfg))
        except Exception:
            row.update(status="fail", err=traceback.format_exc(limit=4), rmse=np.nan, mae=np.nan)
        row["seconds"] = time.time() - t1; rows.append(row)
    return rows

def make_tasks2(units, densities, reps, jobs, cfg, prefix):
    return [dict(id=f"{prefix}|{un}|{d}|{r}", unit=un, density=int(d), rep=int(r), jobs=jobs, cfg=cfg) for un in units for d in densities for r in reps]

def run_tasks2(tasks, out_path, n_jobs=1, log_every=25):
    """Resumable JSONL executor (same protocol as rx.run_tasks_ext) dispatching run_task_ext2."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks2] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
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
                write(run_task_ext2(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task_ext2, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"],
                                                  status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

# ----------------------------------------------------------------------------- job lists
def side_jobs(fset="geo_side"):
    G = rx.GRAPH_JOBS
    return [dict(name="gcn_knn_side", method="gnn", hp={**G["gcn_knn"], "steps": 150, "fset": fset}),
            dict(name="gat_del_side", method="gnn", hp={**G["gat_del"], "steps": 150, "fset": fset}),
            dict(name="rk_side", method="rk_side", hp=dict(fset=fset))]

def default_jobs(side):
    J = rx.base_jobs() + rx.graph_jobs(("gcn_knn", "gat_del"))
    return J + side_jobs() if side else J

def trend_dev_jobs():
    G = rx.GRAPH_JOBS; J = []
    for pol in ("legacy", "robust"):
        J += [dict(name=f"trend_{pol}", method="trend", hp=dict(trend_policy=pol)), dict(name=f"idw_{pol}", method="idw", hp=dict(trend_policy=pol)),
              dict(name=f"krig_global_{pol}", method="krig_global", hp=dict(model="exp", trend_policy=pol)),
              dict(name=f"gat_del_{pol}", method="gnn", hp={**G["gat_del"], "steps": 150, "trend_policy": pol})]
    return J + [dict(name="trend_const", method="trend", hp=dict(trend_policy="const"))]

# ----------------------------------------------------------------------------- pre-specified contrast lists (delta > 0  =>  A better)
CONTRASTS_COMMON = [("idw", "trend"), ("krig_global", "idw"), ("kgnn", "idw"), ("gcn_knn", "idw"), ("gat_del", "idw"), ("gat_del", "gcn_knn"),
                    ("kgnn_tuned", "kgnn"), ("gcn_knn_tuned", "gcn_knn"), ("gat_del_tuned", "gat_del"),
                    ("kgnn_tuned", "idw"), ("gcn_knn_tuned", "idw"), ("gat_del_tuned", "idw")]                      # 12 x 3 densities = 36
CONTRASTS_SIDE = [("rk_side", "idw"), ("gcn_knn_side", "rk_side"), ("gat_del_side", "rk_side"), ("gcn_knn_side", "gcn_knn"), ("gat_del_side", "gat_del"),
                  ("gat_del_side", "idw"), ("gcn_knn_side_tuned", "rk_side"), ("gat_del_side_tuned", "rk_side")]    # 8 x 3 = 24
CONTRASTS_OUTAGE = [("krig_global", "idw"), ("kgnn_tuned", "idw"), ("gcn_knn_tuned", "idw"), ("gat_del_tuned", "idw")]
CONTRASTS_XFER_GEO = [("xfer_gat_del", "idw"), ("xfer_gcn_knn", "idw"), ("xfer_gat_del", "gat_del"), ("xfer_gcn_knn", "gcn_knn"),
                      ("xfer_gat_del", "within_gat_del"), ("xfer_gcn_knn", "within_gcn_knn"), ("xferft_gat_del", "xfer_gat_del"), ("xferft_gcn_knn", "xfer_gcn_knn")]
CONTRASTS_XFER_SIDE = [("xfer_gat_del_side", "rk_side"), ("xfer_gcn_knn_side", "rk_side"), ("xfer_gat_del_side", "idw"), ("xfer_gcn_knn_side", "idw"),
                       ("xfer_gat_del_side", "gat_del_side"), ("xfer_gcn_knn_side", "gcn_knn_side"), ("xferft_gat_del_side", "xfer_gat_del_side"),
                       ("xferft_gcn_knn_side", "xfer_gcn_knn_side")]
TREND_DEV_CONTRASTS = [("trend_robust", "trend_legacy"), ("idw_robust", "idw_legacy"), ("krig_global_robust", "krig_global_legacy"),
                       ("gat_del_robust", "gat_del_legacy"), ("trend_robust", "trend_const")]
PRE_GEO = {"gat_del": dict(arch="gat", graph="delaunay", lr=5e-3), "gcn_knn": dict(arch="gcn", graph="knn", k=8, lr=5e-3)}
PRE_SIDE = {"gat_del_side": dict(arch="gat", graph="delaunay", lr=5e-3, fset="geo_side"), "gcn_knn_side": dict(arch="gcn", graph="knn", k=8, lr=5e-3, fset="geo_side")}

# =============================================================================== E2. tuning on units disjoint from the evaluation units
TUNE_BASES = {
    "gcn_knn": dict(method="gnn", fixed=dict(arch="gcn", graph="knn"), space="gcn"),
    "gat_del": dict(method="gnn", fixed=dict(arch="gat", graph="delaunay"), space="gat"),
    "kgnn": dict(method="kgnn", fixed={}, space="kgnn"),
    "gcn_knn_side": dict(method="gnn", fixed=dict(arch="gcn", graph="knn", fset="geo_side"), space="gcn"),
    "gat_del_side": dict(method="gnn", fixed=dict(arch="gat", graph="delaunay", fset="geo_side"), space="gat")}

def _space(kind):
    if kind == "kgnn": return dict(rc.KGNN_SPACE)
    sp = {k: v for k, v in rc.GCN_SPACE.items() if k != "fset"}             # fset is a model variant (geo vs side), not a tuned value
    if kind == "gat": sp.pop("k", None); sp["heads"] = ("choice", [4, 2, 8])  # k is irrelevant on a Delaunay graph; hidden in {16,32,64} is divisible by every head count
    return sp

def parent_folds(names, cache_dir, n_folds=2, by="parent", seed=0):
    """Fold of a unit = fold of its parent (or layout). Units of one parent never straddle folds."""
    key = {n: rx.get_unit_ext(n, cache_dir)[by] for n in names}
    groups = sorted(set(key.values()), key=lambda g: rc.stable_hash(seed, "fold", g)); fo = {g: i % n_folds for i, g in enumerate(groups)}
    return {f: [n for n in names if fo[key[n]] == f] for f in range(n_folds)}

def tune_crossfit(names, cfg, out, tag, bases, n_jobs, n_folds=2, by="parent", n_trials=8, tune_rep=2, max_tune_units=40, salt="tune", seed=0):
    """For every evaluation fold f: random search (trial 0 = library defaults) on units of the OTHER folds, new test salt, then the best trial
    per base is frozen for fold f. Evaluation units are never tuned on. Returns {fold: {base: dict(hp, trial, score, default_score)}}."""
    os.makedirs(out, exist_ok=True); folds = parent_folds(names, cfg["cache_dir"], n_folds, by, seed); tcfg = {**cfg, "test_salt": salt}; best = {}
    for f in folds:
        pool = sorted([n for g, v in folds.items() if g != f for n in v], key=lambda s: rc.stable_hash(seed, "tu", s))[:max_tune_units]
        jobs, trials = [], {}
        for b in bases:
            sp = TUNE_BASES[b]; hps = rc.random_search_space(_space(sp["space"]), n_trials, seed=rc.stable_hash(seed, tag, b) % 10 ** 6); trials[b] = hps
            jobs += [dict(name=f"{b}|t{t}", method=sp["method"], hp={**sp["fixed"], **hp}) for t, hp in enumerate(hps)]
        path = os.path.join(out, f"tune_{tag}_f{f}.jsonl")
        run_tasks2(make_tasks2(pool, cfg["densities"], range(tune_rep), jobs, tcfg, prefix=f"tune|{tag}|f{f}"), path, n_jobs)
        ok = rc.read_results(path); ok = ok[ok.status == "ok"]; best[f] = {}; allsc = []
        for b in bases:
            sub = ok[ok.name.str.startswith(b + "|t")]
            if sub.empty: continue
            ul = sub.groupby(["name", "unit", "density"], as_index=False).rmse.mean().groupby(["name", "density"], as_index=False).rmse.mean()
            score = ul.groupby("name").rmse.mean(); allsc.append(score.rename("mean_rmse").reset_index())
            if f"{b}|t0" not in score.index: continue
            win = score.idxmin(); t = int(win.split("|t")[1])
            best[f][b] = dict(hp={**TUNE_BASES[b]["fixed"], **trials[b][t]}, trial=t, score=float(score[win]), default_score=float(score[f"{b}|t0"]))
        if allsc: pd.concat(allsc, ignore_index=True).to_csv(os.path.join(out, f"tune_{tag}_f{f}_trial_scores.csv"), index=False)
    json.dump(best, open(os.path.join(out, f"tuned_{tag}.json"), "w"), indent=1, default=str)
    return best

def tuned_jobs(best_f):
    return [dict(name=f"{b}_tuned", method=TUNE_BASES[b]["method"], hp=dict(v["hp"])) for b, v in best_f.items()]

def family_cfg(fd, split, PRE):
    return {**fd["cfg"], "split_mode": split, "block_buffer": 0.0, "test_salt": PRE["test_salt"]}

def run_confirmatory(tag, fd, split, root, PRE, n_jobs):
    """One (family, split) cell of the pre-registered design: cross-fitted tuning, then evaluation of defaults, tuned models, side-information
    models and baselines on fresh units/test-salt/replicates."""
    cfg = family_cfg(fd, split, PRE); out = os.path.join(root, tag); os.makedirs(out, exist_ok=True); T = PRE["tuning"]
    folds = parent_folds(fd["names"], cfg["cache_dir"], T["n_folds"], fd.get("by", "parent"))
    bases = ["gcn_knn", "gat_del", "kgnn"] + (["gcn_knn_side", "gat_del_side"] if fd["side"] else [])
    best = tune_crossfit(fd["names"], cfg, out, f"{tag}_{split}", bases, n_jobs, T["n_folds"], fd.get("by", "parent"), T["n_trials"], T["tune_rep"], T["max_tune_units"], T["salt"]) \
        if T["enabled"] else {f: {} for f in folds}
    tasks = []
    for f, units in folds.items():
        tasks += make_tasks2(units, cfg["densities"], PRE["reps"], default_jobs(fd["side"]) + tuned_jobs(best[f]), cfg, prefix=f"{tag}|{split}|f{f}")
    path = os.path.join(out, f"{tag}_{split}.jsonl"); run_tasks2(tasks, path, n_jobs)
    return rc.read_results(path), best

# =============================================================================== E3. transfer to held-out layouts (groups of layouts)
def lolo_plan2(names, cache_dir, group_size=1, n_src=40, seed=0):
    """Held-out groups of layouts. group_size=1 is leave-one-layout-out; when that layout has >= 2 parents (floors) it is run as two halves,
    each evaluated on one half of the floors with the OTHER half as the 'within' source (parent-disjoint, unlike rx.lolo_plan)."""
    by = rx.layouts_of(names, cache_dir); layouts = sorted(by, key=lambda L: rc.stable_hash(seed, "lay", L)); plan = []
    for gi in range(0, len(layouts), group_size):
        G = layouts[gi:gi + group_size]; cid = G[0] if group_size == 1 else f"g{gi // group_size}"
        others = [by[L] for L in layouts if L not in G]; cross = []
        for i in range(max((len(v) for v in others), default=0)):
            for v in others:
                if i < len(v) and len(cross) < n_src: cross.append(v[i])
        ev_all = [n for L in G for n in by[L]]
        par = sorted({rx.get_unit_ext(n, cache_dir)["parent"] for n in ev_all}, key=lambda p: rc.stable_hash(seed, "par", p))
        if group_size == 1 and len(par) >= 2:
            for half, ps in (("A", set(par[0::2])), ("B", set(par[1::2]))):
                ev = [n for n in ev_all if rx.get_unit_ext(n, cache_dir)["parent"] in ps]; wi = [n for n in ev_all if n not in ev][:n_src]
                plan.append(dict(layout=f"{cid}#{half}", cross_id=cid, eval=ev, cross=cross, within=wi))
        else:
            plan.append(dict(layout=cid, cross_id=cid, eval=ev_all, cross=cross, within=[]))
    return plan

def build_episodes2(names, cfg, hp, reps=2):
    eps, cache = [], {}; pol = hp.get("trend_policy", cfg.get("trend_policy", "robust")); fs = hp.get("fset", "geo")
    for n in names:
        u0 = rx.get_unit_ext(n, cfg["cache_dir"])
        for d in cfg["densities"]:
            for r in range(reps):
                try:
                    tr, _ = split_for2(u0, d, 500 + r, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
                    u, spec = view_for_split2(u0, tr)
                    eps.append(rx.make_episode(u, tr, prepare2(u, tr, fs, cfg["clip"], spec), hp, cache))
                except Exception:
                    continue
    return eps

def pretrain_for_plan2(plan, cfg, pre_cfgs, out_dir, steps=1500, seed=0):
    paths, done = {}, set()
    for item in plan:
        for scope in ("cross", "within"):
            src = item[scope]; ident = item["cross_id"] if scope == "cross" else item["layout"]
            if not src: continue
            srcs = sorted(src, key=lambda s: rc.stable_hash(seed, "val", s)); nv = max(1, len(srcs) // 7)
            for tag, hp in pre_cfgs.items():
                path = os.path.join(out_dir, f"{ident.replace('#', '_')}__{scope}__{tag}.pt"); paths[(ident, scope, tag)] = path
                if os.path.exists(path) or path in done: continue
                print(f"  pretraining {ident} / {scope} / {tag} on {len(src)} source units"); done.add(path)
                tr_eps = build_episodes2(srcs[nv:], cfg, hp); va_eps = build_episodes2(srcs[:nv], cfg, hp, reps=1)
                if not tr_eps: print("    no usable source episodes, skipped"); paths.pop((ident, scope, tag)); continue
                rx.pretrain_graph(tr_eps, va_eps, hp, steps, seed, path)
    return paths

def lolo_tasks2(plan, paths, cfg, pre_cfgs, reps, side, finetune_steps=30, prefix="xfer"):
    tasks = []
    for item in plan:
        jobs = default_jobs(side)
        for tag, hp in pre_cfgs.items():
            p = paths.get((item["cross_id"], "cross", tag))
            if p:
                jobs.append(dict(name=f"xfer_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": 0}))
                jobs.append(dict(name=f"xferft_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": finetune_steps}))
            p = paths.get((item["layout"], "within", tag))
            if p: jobs.append(dict(name=f"within_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": 0}))
        tasks += make_tasks2(item["eval"], cfg["densities"], reps, jobs, cfg, prefix=f"{prefix}|{item['layout']}")
    return tasks

def run_transfer(tag, fd, split, root, PRE, n_jobs, group_size, pre_cfgs):
    cfg = family_cfg(fd, split, PRE); out = os.path.join(root, tag); os.makedirs(out, exist_ok=True); P = PRE["transfer"]
    plan = lolo_plan2(fd["names"], cfg["cache_dir"], group_size, P["n_src"])
    print("transfer plan (item, n_eval, n_cross_src, n_within_src):", [(p["layout"], len(p["eval"]), len(p["cross"]), len(p["within"])) for p in plan])
    paths = pretrain_for_plan2(plan, cfg, pre_cfgs, os.path.join(out, f"pretrained_{split}"), P["steps"])
    path = os.path.join(out, f"{tag}_xfer_{split}.jsonl")
    run_tasks2(lolo_tasks2(plan, paths, cfg, pre_cfgs, PRE["reps"], fd["side"], P["finetune_steps"], prefix=f"xfer|{tag}|{split}"), path, n_jobs)
    return rc.read_results(path)

# =============================================================================== diagnostics that need no model
def predictability_table(names, cache_dir, max_pts=400, seed=0):
    """DESCRIPTIVE (uses the unit's own labels; nothing here feeds any method or any test): noise fraction = nugget / (nugget + partial sill)
    of an exponential variogram of the trend residual. Near 1 = mostly unpredictable (fading-like) target; near 0 = smooth, spatially structured."""
    rows = []
    for n in names:
        u0 = rx.get_unit_ext(n, cache_dir); N = len(u0["y"]); idx = np.sort(np.random.default_rng(seed).choice(N, min(N, max_pts), replace=False))
        u, spec = view_for_split2(u0, idx); prep = prepare2(u, idx, "geo", 4.0, spec)
        par = rc.fit_variogram(u["pos"][idx], prep["res"][idx], "exp")
        rows.append(dict(unit=n, scenario=u0["scenario"], layout=u0["layout"], parent=u0["parent"], y_std=float(u0["y"].std()), resid_std=float(prep["res"][idx].std()),
                         noise_frac=float(par[0] / (par[0] + par[1] + 1e-9)), trend=spec["kind"]))
    return pd.DataFrame(rows)

def abs_threshold_table(names, cache_dir, cands):
    by = {}
    for n in names:
        u = rx.get_unit_ext(n, cache_dir); by.setdefault(u["layout"], []).append(u["y"])
    return pd.DataFrame([dict(layout=L, **{f"<{c:g}": float((np.concatenate(v) < c).mean()) for c in cands}) for L, v in by.items()]).set_index("layout")

def pick_abs_thresholds(tab, cands, lo=0.1, hi=0.9):
    """Keep an absolute threshold only if EVERY layout of the family has between lo and hi of its points below it (the metric is then defined everywhere)."""
    return tuple(c for c in cands if ((tab[f"<{c:g}"] >= lo) & (tab[f"<{c:g}"] <= hi)).all())

# =============================================================================== F. statistics
def write_prereg(spec, path):
    """Freeze the analysis plan. The file stores the spec, a UTC timestamp and its sha256. Re-running with an identical spec is a no-op;
    a different spec raises, so the plan cannot be edited quietly after results exist. Commit the file to git BEFORE running section 7."""
    blob = json.dumps(spec, sort_keys=True, indent=1, default=list); h = hashlib.sha256(blob.encode()).hexdigest()
    if os.path.exists(path):
        old = json.load(open(path))
        if old["sha256"] != h: raise RuntimeError(f"{path} already exists with a DIFFERENT plan (sha256 {old['sha256'][:12]} vs {h[:12]}). Delete it only if no confirmatory result exists.")
        return old
    rec = dict(created_utc=datetime.datetime.utcnow().isoformat() + "Z", sha256=h, spec=json.loads(blob)); json.dump(rec, open(path, "w"), indent=1); return rec

def load_prereg(path):
    rec = json.load(open(path)); blob = json.dumps(rec["spec"], sort_keys=True, indent=1, default=list)
    if hashlib.sha256(blob.encode()).hexdigest() != rec["sha256"]: raise RuntimeError("prereg file was modified after it was written")
    return rec["spec"]

def _direction(r):
    if r.significant: return "A better (Holm)" if r.mean_delta_db > 0 else "A WORSE (Holm)"
    return "n.s., CI includes 0" if r.ci_lo <= 0 <= r.ci_hi else "n.s., CI excludes 0 but Holm p >= .05"

def contrast_report(df, contrasts, metric, family, out, infer=None, pooled_only=False, higher_is_better=False, confirmatory=True, quiet=False):
    """EVERY contrast of the list is printed and saved (wins, losses, ties), with bootstrap CI over parents and Holm-adjusted p over the
    whole call. The cap on the number of tests is computed from the list itself (contrasts x densities x inference scenarios)."""
    d = df[df.status == "ok"] if "status" in df else df
    di = d if infer is None else d[d.scenario.isin(infer)]
    if pooled_only: di = di[di.scenario == "ALL"]
    if di.empty: print(f"[{family}] no rows"); return pd.DataFrame()
    cap = len(contrasts) * max(1, di.density.nunique()) * max(1, di.scenario.nunique())
    pc = rx.paired_metric(di, metric, [tuple(c) for c in contrasts], higher_is_better=higher_is_better, max_tests=cap)
    if len(pc):
        pc["direction"] = pc.apply(_direction, axis=1); pc["status"] = "CONFIRMATORY" if confirmatory else "exploratory"; pc["family"] = family; pc["metric"] = metric
        os.makedirs(out, exist_ok=True); pc.to_csv(os.path.join(out, f"{family}_{metric}.csv"), index=False)
        if not quiet:
            _show(pc[["scenario", "density", "A", "B", "n_units", "mean_delta_db", "ci_lo", "ci_hi", "rel_gain_pct", "p_holm", "direction", "underpowered"]],
                  f"{family} | {metric} | {'CONFIRMATORY' if confirmatory else 'exploratory'} | {len(pc)} tests, Holm over all (delta>0: A better)")
            print(f"  {int((pc.direction == 'A better (Holm)').sum())} A better, {int((pc.direction == 'A WORSE (Holm)').sum())} A worse, "
                  f"{int(pc.direction.str.startswith('n.s.').sum())} not significant; {int(pc.underpowered.sum())} underpowered")
    return pc

def outage_report(df, family, out, primary, contrasts, min_skill=0.2, ref="idw", infer=None, pooled_only=False):
    """Outage / coverage metrics are shown ONLY in (scenario, density) cells where the reference interpolator has skill >= min_skill
    (skill = 1 - MSE / variance of the test labels). Elsewhere the target is mostly unpredictable and a coverage map would be chance."""
    d = df[df.status == "ok"]
    if "skill" not in d or ref not in set(d.name): print(f"[{family}] no skill / reference rows"); return None
    sk = rx.agg_metric(d, "skill")[ref]; keep = set(sk[sk >= min_skill].index)
    _show(sk.to_frame(f"{ref}_skill").assign(reported=[i in keep for i in sk.index]), f"{family}: outage reported only where {ref} skill >= {min_skill}")
    dg = d[[(s, k) in keep for s, k in zip(d.scenario, d.density)]]
    cols = [c for c in d.columns if c.startswith(("f1_", "bacc_", "acc_", "area_"))]
    for c in cols:
        try: rx.agg_metric(dg, c).to_csv(os.path.join(out, f"{family}_{c}.csv"))
        except Exception: pass
    if dg.empty: print("  no predictable cell -> no outage table (this is the result, not a gap)"); return None
    for c in [c for c in cols if c.startswith((primary[:2], "bacc_" + primary.split("_")[1]))]:
        try: _show(rx.agg_metric(dg, c), f"{family}: {c} (gated cells only; higher is better)")
        except Exception: pass
    if primary in dg:
        return contrast_report(dg, contrasts, primary, f"{family}_outage", out, infer=infer, pooled_only=pooled_only, higher_is_better=True)

def report_family(df, tag, split, fd, out, PRE):
    """All tables for one (family, split): RMSE / MAE by method (every method, losses included), contrasts, outage."""
    fam = f"{tag}_{split}"; d = df[df.status == "ok"]; print(f"\n##### {fam}: {len(df)} job rows, {len(d)} ok, {len(df) - len(d)} failed")
    if len(df) - len(d): print(df[df.status != "ok"].groupby("name").size().to_string())
    for col in ("rmse", "mae", "skill"):
        t = rx.agg_metric(d, col); t.to_csv(os.path.join(out, f"{fam}_{col}.csv")); _show(t, f"{col}, parent-level mean (all methods)")
    kinds = d[d.name == "idw"].groupby(["density", "trend_kind"]).size().unstack(fill_value=0); _show(kinds, "trend chosen by CV (rows of the idw job)")
    pcs = {"main": contrast_report(d, PRE["contrasts_common"], "rmse", f"{fam}_main", out, infer=fd.get("infer"))}
    if fd["side"]: pcs["side"] = contrast_report(d, PRE["contrasts_side"], "rmse", f"{fam}_side", out, infer=fd.get("infer"))
    pcs["outage"] = outage_report(d, fam, out, fd["primary"], [c for c in PRE["contrasts_outage"]] + ([("gat_del_side_tuned", "rk_side")] if fd["side"] else []),
                                  PRE["outage"]["min_skill"], PRE["outage"]["ref"], fd.get("infer"))
    return pcs

def report_transfer(df, tag, split, out, contrasts, PRE):
    fam = f"{tag}_xfer_{split}"; d = rx.by_layout(df[df.status == "ok"])
    for col in ("rmse",): _show(rx.agg_metric(d, col), f"{fam}: {col} by held-out layout and pooled (ALL)")
    return contrast_report(d, contrasts, "rmse", fam, out, pooled_only=True)

def master_table(entries):
    """entries: [(family, split, label, pc)] -> one long table of every contrast that was tested."""
    fr = [e[3].assign(dataset=e[0], split=e[1], block=e[2]) for e in entries if e[3] is not None and len(e[3])]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()

def layout_summary(family_defs):
    rows = []
    for tag, fd in family_defs.items():
        for n in fd["names"]:
            u = rx.get_unit_ext(n, fd["cfg"]["cache_dir"]); rows.append(dict(family=tag, layout=u["layout"], parent=u["parent"]))
    d = pd.DataFrame(rows)
    return d.groupby("family").agg(units=("parent", "size"), parents=("parent", "nunique"), layouts=("layout", "nunique"))
