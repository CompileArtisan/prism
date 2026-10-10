"""rmbench_core: numpy/scipy/pandas part of the DeepMIMO radio-map benchmark (no torch import here).

Design rules (see notebook section 0):
  * statistical unit = one transmitter's radio map ("unit"); tuning units and evaluation units are disjoint;
  * every model sees the SAME trend, features, labels and test set (``prepare`` + ``split_for``);
  * the label-derived legacy features (path_count, delay_spread) are NOT used anywhere;
  * workers are importable top-level functions (``run_task``) so process pools never pickle notebook state.
"""
from __future__ import annotations
import os, json, time, zlib, math, traceback, itertools
from functools import lru_cache
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy import sparse, stats as sstats

# ----------------------------------------------------------------------------- config
DEFAULT_CFG = dict(
    cache_dir="rmbench_cache", max_points=3000,       # per-unit pool cap (rx points kept at load time)
    n_test=1000,                                       # fixed hold-out per unit (evaluation-size cap)
    densities=(30, 100, 300), n_rep=5,                 # nested label sets, replicates per unit
    min_unit_points=600,                               # unit dropped if fewer points survive noise filter
    ptx_dbm=20.0, noise_bw_hz=1e8, noise_nf_db=7.0,    # noise floor = -174 + 10log10(BW) + NF  (~ -87 dBm)
    clip=4.0, clip_audit_max=0.05,                     # feature dropped if >5% of values clip at +-4 sd
    tune_frac=0.25, seed=0)

def stable_hash(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF

def fspl_db(d, f_ghz):
    return 20 * np.log10(np.maximum(d, 1e-3)) + 20 * np.log10(f_ghz) + 32.44

# ----------------------------------------------------------------------------- power / preprocessing
def sum_path_power_dbm(p_paths_dbm):
    """Per-path powers (N,P) [dBm, NaN = no path] -> incoherent power sum in dBm. NaN where no path at all.
    (Legacy code averaged |h|^2 over paths, which divides by the *padded* path count -> not a power sum.)"""
    p = np.asarray(p_paths_dbm, float)
    lin = np.nansum(np.where(np.isfinite(p), 10 ** (p / 10), 0.0), axis=1)
    out = np.full(len(p), np.nan)
    ok = lin > 0
    out[ok] = 10 * np.log10(lin[ok])
    return out

def calibrate_power_offset(d, p_dbm, los, f_ghz, ptx_dbm):
    """Offset (dB) that makes near-field LOS power match free space for a 0 dBi link at ptx_dbm.
    Assumption (documented limitation): isotropic antennas; only used for the noise-floor test, never for RMSE
    (all methods work on residuals, so a constant offset cancels)."""
    m = np.isfinite(p_dbm) & (los > 0.5)
    if m.sum() < 20:
        return 0.0
    near = m & (d <= np.quantile(d[m], 0.2))
    return float(np.median(ptx_dbm - fspl_db(d[near], f_ghz) - p_dbm[near]))

def noise_floor_dbm(cfg):
    return -174.0 + 10 * math.log10(cfg["noise_bw_hz"]) + cfg["noise_nf_db"]

def preprocess_unit(raw, cfg):
    """raw: dict(pos3 (N,3), tx3 (3,), power_paths_dbm (N,P), los (N,), freq_ghz, name, scenario, tx).
    Returns unit dict (2-D positions for interpolation, 3-D distance for the trend) or None if degenerate."""
    pos3, tx3 = np.asarray(raw["pos3"], float), np.asarray(raw["tx3"], float).reshape(-1)[:3]
    d = np.linalg.norm(pos3 - tx3, axis=1)
    p = sum_path_power_dbm(raw["power_paths_dbm"])
    los = (np.asarray(raw["los"]) == 1).astype(float)          # DeepMIMO codes -1 = no path, 0 = NLOS, 1 = LOS
    off = calibrate_power_offset(d, p, los, raw["freq_ghz"], cfg["ptx_dbm"])
    y = p + off
    keep = np.isfinite(y) & (y > noise_floor_dbm(cfg))
    if keep.sum() < cfg["min_unit_points"]:
        return None
    rel = pos3[keep] - tx3
    return dict(name=raw["name"], scenario=raw["scenario"], tx=raw["tx"], freq_ghz=float(raw["freq_ghz"]),
                pos=pos3[keep, :2], tx_xy=tx3[:2].copy(), d=d[keep], az=np.arctan2(rel[:, 1], rel[:, 0]),
                los=los[keep], y=y[keep], offset_db=off, frac_kept=float(keep.mean()))

def save_unit(u, cache_dir):
    os.makedirs(os.path.join(cache_dir, "units"), exist_ok=True)
    path = os.path.join(cache_dir, "units", u["name"].replace("|", "__") + ".npz")
    np.savez_compressed(path, **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in u.items()})
    return path

def load_unit_file(path):
    z = np.load(path, allow_pickle=False)
    u = {k: z[k] for k in z.files}
    for k in ("name", "scenario"):
        u[k] = str(u[k])
    u["tx"] = int(u["tx"]); u["freq_ghz"] = float(u["freq_ghz"])
    return u

# ----------------------------------------------------------------------------- synthetic scenarios
def _shadow_field(rng, xy, ell, sigma, M=160):
    """Exponential-covariance Gaussian field (Matern-1/2 in 2-D) via random Fourier features."""
    z = rng.normal(size=(M, 2)); g = np.abs(rng.normal(size=(M, 1))) + 1e-3   # multivariate-t(1) frequencies
    w = z / (ell * g); ph = rng.uniform(0, 2 * np.pi, M)
    return sigma * math.sqrt(2.0 / M) * np.cos(xy @ w.T + ph).sum(1)

def make_synthetic_raw(scenario="A", tx=0, n=2500, seed=0):
    """Indoor-like synthetic unit: log-distance trend + wall losses + correlated shadowing + an angular term,
    emitted as per-path powers in an arbitrary (-30 dB) reference so the power-sum/offset code is exercised."""
    f_ghz = {"A": 28.0, "B": 2.5}.get(scenario, 6.0)
    rng = np.random.default_rng(stable_hash("syn", scenario, tx, seed))
    L, W = (24.0, 14.0) if scenario == "A" else (30.0, 20.0)
    xy = np.column_stack([rng.uniform(0, L, n), rng.uniform(0, W, n)])
    tx_xy = np.array([rng.uniform(0.2 * L, 0.8 * L), rng.uniform(0.2 * W, 0.8 * W)])
    tx3 = np.append(tx_xy, 3.0); pos3 = np.column_stack([xy, np.full(n, 1.5)])
    d = np.linalg.norm(pos3 - tx3, axis=1); rel = xy - tx_xy; az = np.arctan2(rel[:, 1], rel[:, 0])
    nexp = rng.uniform(1.8, 2.8)
    ncross = np.zeros(n)
    for cx, g0, g1 in [(L * rng.uniform(0.3, 0.45), W * 0.4, W * 0.55), (L * rng.uniform(0.6, 0.75), W * 0.55, W * 0.7)]:
        opp = (tx_xy[0] - cx) * (xy[:, 0] - cx) < 0
        t = (cx - tx_xy[0]) / np.where(np.abs(xy[:, 0] - tx_xy[0]) < 1e-9, 1e-9, xy[:, 0] - tx_xy[0])
        yc = tx_xy[1] + t * (xy[:, 1] - tx_xy[1])
        ncross += (opp & ~((yc >= g0) & (yc <= g1))).astype(float)
    los = (ncross == 0).astype(float)
    shadow = _shadow_field(rng, xy, ell=rng.uniform(2.0, 4.0), sigma=rng.uniform(3.0, 5.0))
    shadow += 2.0 * np.sin(3 * az + rng.uniform(0, 6.28))                        # angular (TX-polar) component
    base = args_base = 20.0 - fspl_db(d, f_ghz) * (nexp / 2.0) - 9.0 * ncross + shadow   # dBm, ptx = 20
    P = 4
    paths = np.full((n, P), np.nan)
    paths[:, 0] = base - 30.0                                                    # arbitrary -30 dB raw reference
    for j in range(1, P):
        has = rng.random(n) < 0.7
        paths[has, j] = base[has] - 30.0 - rng.uniform(4, 15, has.sum())
    paths[los < 0.5, 0] = np.where(rng.random((los < 0.5).sum()) < 0.5, np.nan, paths[los < 0.5, 0] - 8)
    losflag = np.where(np.isfinite(paths).any(1), np.where(los > 0.5, 1, 0), -1)
    return dict(pos3=pos3, tx3=tx3, power_paths_dbm=paths, los=losflag, freq_ghz=f_ghz,
                name=f"syn|{scenario}|{tx}", scenario=f"syn{scenario}", tx=tx)

# ----------------------------------------------------------------------------- DeepMIMO loader (needs network; untested offline)
def load_deepmimo_units(scenario, cfg, seed=0):
    """One raw dict per transmitter. Uses per-path power (``dataset.power``) when available -> no heavy
    compute_channels call. VERIFY on first run: attribute names / dBm units / los coding of your DeepMIMO version."""
    import re, deepmimo as dm
    dm.download(scenario)
    rx_set = next(s for s in dm.get_txrx_sets(scenario) if s.is_rx and not s.is_tx)
    n_pts = rx_set.num_points
    if n_pts > cfg["max_points"]:
        keep = np.sort(np.random.default_rng(seed).choice(n_pts, cfg["max_points"], replace=False))
        data = dm.load(scenario, rx_sets={rx_set.id: keep})
    else:
        data = dm.load(scenario, rx_sets="rx_only")
    dsets = data.datasets if isinstance(data, dm.MacroDataset) else [data]
    out = []
    for k, ds in enumerate(dsets):
        try:
            pw = np.asarray(ds.power, float)
        except Exception:                                    # fallback: sum of per-path |h|^2 (mean over antennas)
            ch = np.abs(np.asarray(ds.compute_channels())) ** 2
            pw = 10 * np.log10(ch.mean(axis=(1, 2)) + 1e-30)
        f = None
        for a in ("carrier_freq", "freq", "frequency"):
            try:
                v = getattr(ds, a)
            except Exception:
                continue
            if v is not None:
                f = float(v) / 1e9 if float(v) > 1000 else float(v); break
        if f is None:
            m = re.search(r"_(\d+)p?(\d*)", scenario); f = float(f"{m.group(1)}.{m.group(2) or 0}")
        out.append(dict(pos3=np.asarray(ds.rx_pos), tx3=np.asarray(ds.tx_pos).reshape(-1)[:3], power_paths_dbm=pw,
                        los=np.asarray(ds.los), freq_ghz=f, name=f"dm|{scenario}|{k}", scenario=scenario, tx=k))
    return out

def prefetch_units(unit_specs, cfg):
    """Build/refresh the on-disk unit cache. unit_specs: list of ('syn', scenario, tx, seed) or ('dm', scenario).
    Returns (list of unit names that are usable, list of (name, reason) dropped)."""
    ok, dropped = [], []
    for s in unit_specs:
        raws = [make_synthetic_raw(s[1], s[2], seed=s[3])] if s[0] == "syn" else load_deepmimo_units(s[1], cfg)
        for r in raws:
            u = preprocess_unit(r, cfg)
            if u is None:
                dropped.append((r["name"], "below noise floor / too few points")); continue
            save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
    return ok, dropped

@lru_cache(maxsize=64)
def get_unit(name, cache_dir):
    return load_unit_file(os.path.join(cache_dir, "units", name.replace("|", "__") + ".npz"))

def split_units(names, tune_frac, seed=0):
    """Deterministic transmitter-level split per scenario: tune units never appear in evaluation."""
    tune, evalu = [], []
    by = {}
    for n in sorted(names):
        by.setdefault(n.split("|")[1], []).append(n)
    for scen, ns in by.items():
        ns = sorted(ns, key=lambda s: stable_hash(seed, s))
        k = int(round(tune_frac * len(ns))) if len(ns) >= 4 else 0       # tiny scenarios: keep all for evaluation
        tune += ns[:k]; evalu += ns[k:]
    return tune, evalu

def make_patches(u, nx=2, ny=2, min_pts=400):
    """Optional: spatial patches of one unit as descriptive sub-units (aggregated to their parent downstream)."""
    out = []
    xe = np.quantile(u["pos"][:, 0], np.linspace(0, 1, nx + 1)); ye = np.quantile(u["pos"][:, 1], np.linspace(0, 1, ny + 1))
    for i, j in itertools.product(range(nx), range(ny)):
        m = (u["pos"][:, 0] >= xe[i]) & (u["pos"][:, 0] <= xe[i + 1]) & (u["pos"][:, 1] >= ye[j]) & (u["pos"][:, 1] <= ye[j + 1])
        if m.sum() < min_pts: continue
        v = {k: (val[m] if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(m) else val) for k, val in u.items()}
        v["name"] = f"{u['name']}#p{i}{j}"; v["parent"] = u["name"]; out.append(v)
    return out

# ----------------------------------------------------------------------------- features, trend, preparation
def feature_matrix(u, fset="geo"):
    """Label-free features. 'geo' = [log10 d, cos az, sin az, dx/L, dy/L]; 'geo_los' adds the ray-tracer LOS flag."""
    L = float(np.ptp(u["pos"], axis=0).max()) + 1e-9
    rel = (u["pos"] - u["tx_xy"]) / L
    cols = [np.log10(np.maximum(u["d"], 1e-3)), np.cos(u["az"]), np.sin(u["az"]), rel[:, 0], rel[:, 1]]
    if fset == "geo_los":
        cols.append(u["los"])
    return np.column_stack(cols)

def standardize(F, tr, clip=4.0):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Z = (F - mu) / sd
    frac = (np.abs(Z) > clip).mean(0)
    return np.clip(Z, -clip, clip), frac

def audit_clipping(u, fsets=("geo", "geo_los"), n_draws=5, n_labeled=30, clip=4.0, max_frac=0.05):
    """Fraction of values clipped (over ALL points) when standardising on a small train-only sample."""
    rows = []
    for fs in fsets:
        F = feature_matrix(u, fs)
        for r in range(n_draws):
            tr = np.random.default_rng(r).choice(len(F), n_labeled, replace=False)
            _, fr = standardize(F, tr, clip)
            for j, f in enumerate(fr):
                rows.append(dict(unit=u["name"], fset=fs, col=j, frac_clipped=float(f)))
    df = pd.DataFrame(rows).groupby(["unit", "fset", "col"], as_index=False).frac_clipped.mean()
    df["flag"] = df.frac_clipped > max_frac
    return df

def fit_trend(d, y):
    """y = a - 10 n log10(d). n is constrained to [0, 8]: received power must not rise with distance, so a positive
    least-squares slope is clipped to n=0 and the intercept refit (direction logic is unit-tested)."""
    x = 10 * np.log10(np.maximum(d, 1e-3))
    if len(y) < 3 or np.ptp(x) < 1e-6:
        return float(np.mean(y)), 0.0
    slope, a = np.polyfit(x, y, 1)
    n = float(np.clip(-slope, 0.0, 8.0))
    if n != -slope:
        a = float(np.mean(y + 10 * n * x))
    return float(a), n

def trend_eval(d, a, n):
    return a - 10 * n * np.log10(np.maximum(d, 1e-3))

def prepare(u, tr, fset="geo", clip=4.0):
    """Everything a model may use, computed from TRAIN labels only."""
    a, n = fit_trend(u["d"][tr], u["y"][tr])
    res = u["y"] - trend_eval(u["d"], a, n)                      # residual; models only ever index res[tr]
    sigma = float(res[tr].std() + 1e-6)
    Z, fr = standardize(feature_matrix(u, fset), tr, clip)
    return dict(a=a, n=n, res=res, sigma=sigma, X=Z, clipped=fr, trend=trend_eval(u["d"], a, n))

# ----------------------------------------------------------------------------- splits
def split_for(u, density, rep, n_test):
    """Fixed test set per unit (independent of method/density/rep); labeled sets nested in density, new per rep."""
    N = len(u["y"]); nt = min(n_test, N - density - 1)
    test = np.sort(np.random.default_rng(stable_hash("test", u["name"])).choice(N, nt, replace=False))
    pool = np.setdiff1d(np.arange(N), test)
    perm = np.random.default_rng([stable_hash("rep", u["name"]), rep]).permutation(pool)
    return perm[:density], test

# ----------------------------------------------------------------------------- classical interpolators (on trend residuals)
def idw_predict(Pq, Pr, rr, power=2.0, k=None):
    D = cdist(Pq, Pr)
    if k is not None and k < D.shape[1]:
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        Dk = np.take_along_axis(D, idx, 1); rk = rr[idx]
    else:
        Dk, rk = D, np.broadcast_to(rr, D.shape)
    w = 1.0 / np.maximum(Dk, 1e-6) ** power
    return (w * rk).sum(1) / w.sum(1)

def tune_idw_loo(Pr, rr, powers=(1.0, 2.0, 3.0, 4.0), ks=(8, 16, None)):
    """Leave-one-out choice of (power, k) from the task's own labels only."""
    D = cdist(Pr, Pr); np.fill_diagonal(D, np.inf)
    best, arg = np.inf, (2.0, None)
    for k in ks:
        if k is not None and k >= len(Pr) - 1: continue
        if k is None:
            Dk, rk = D, np.broadcast_to(rr, D.shape)
        else:
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]; Dk = np.take_along_axis(D, idx, 1); rk = rr[idx]
        for p in powers:
            w = 1.0 / np.maximum(Dk, 1e-6) ** p
            e = np.sqrt(np.mean(((w * rk).sum(1) / w.sum(1) - rr) ** 2))
            if e < best: best, arg = e, (p, k)
    return arg

def _gamma(h, par, model):
    nug, ps, rg = par
    r = np.minimum(h / rg, 1e6)
    if model == "exp": g = 1 - np.exp(-3 * r)
    elif model == "sph": g = np.where(r < 1, 1.5 * r - 0.5 * r ** 3, 1.0)
    elif model == "gauss": g = 1 - np.exp(-3 * r ** 2)
    else: raise ValueError(model)
    return np.where(h > 0, nug + ps * g, 0.0)

def empirical_variogram(P, r, nbins=12, max_frac=0.5):
    D = cdist(P, P); iu = np.triu_indices(len(P), 1)
    h = D[iu]; g = 0.5 * (r[:, None] - r[None, :])[iu] ** 2
    hmax = max_frac * h.max()
    edges = np.linspace(0, hmax, nbins + 1); b = np.digitize(h, edges) - 1
    ok = (b >= 0) & (b < nbins)
    cnt = np.bincount(b[ok], minlength=nbins).astype(float)
    hb = np.bincount(b[ok], weights=h[ok], minlength=nbins); gb = np.bincount(b[ok], weights=g[ok], minlength=nbins)
    m = cnt >= 3
    return hb[m] / cnt[m], gb[m] / cnt[m], cnt[m], hmax

def fit_variogram(P, r, model="exp"):
    """Weighted least squares on the binned empirical variogram; moment-based initial values.
    Returns (nugget, partial_sill, practical_range) -- falls back to the initial moments if the fit fails."""
    var = float(np.var(r)) + 1e-9
    try:
        h, g, c, hmax = empirical_variogram(P, r)
        if len(h) < 3: raise ValueError("too few bins")
        init = np.array([max(0.0, 0.3 * g[0]), max(var - 0.3 * g[0], 0.2 * var), hmax / 2])
        lo, hi = [0.0, 1e-6, hmax * 0.02], [var * 2, var * 4, hmax * 10]
        init = np.clip(init, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        s = least_squares(lambda p: np.sqrt(c) * (_gamma(h, p, model) - g), init, bounds=(lo, hi))
        if np.all(np.isfinite(s.x)): return tuple(s.x)
    except Exception:
        pass
    return (0.3 * var, 0.7 * var, float(np.ptp(P, axis=0).max()) / 3 + 1e-6)

def ok_predict(Pq, Pr, rr, par, model="exp", K=None, chunk=400):
    """Ordinary kriging with the semivariogram system [Gamma 1; 1' 0][w; mu] = [gamma_0; 1]. K=None -> global.
    Returns (pred, n_fallback): non-finite / singular rows fall back to IDW (counted, never silent)."""
    nr = len(Pr); out = np.empty(len(Pq)); nfb = 0
    eps = 1e-6 * (par[0] + par[1] + 1e-9)
    if K is None or K >= nr:
        G = _gamma(cdist(Pr, Pr), par, model) + eps * np.eye(nr)
        A = np.ones((nr + 1, nr + 1)); A[:nr, :nr] = G; A[nr, nr] = 0
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        for s in range(0, len(Pq), 2000):
            g0 = _gamma(cdist(Pr, Pq[s:s + 2000]), par, model)
            rhs = np.vstack([g0, np.ones((1, g0.shape[1]))])
            w = (Ainv @ rhs)[:nr]
            out[s:s + 2000] = w.T @ rr
    else:
        for s in range(0, len(Pq), chunk):
            q = Pq[s:s + chunk]; D = cdist(q, Pr)
            idx = np.argpartition(D, K - 1, axis=1)[:, :K]
            R = Pr[idx]; rk = rr[idx]; d0 = np.take_along_axis(D, idx, 1)
            H = np.linalg.norm(R[:, :, None, :] - R[:, None, :, :], axis=-1)
            A = np.ones((len(q), K + 1, K + 1)); A[:, :K, :K] = _gamma(H, par, model) + eps * np.eye(K); A[:, K, K] = 0
            rhs = np.concatenate([_gamma(d0, par, model), np.ones((len(q), 1))], 1)
            try:
                sol = np.linalg.solve(A, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                sol = np.stack([np.linalg.lstsq(A[i], rhs[i], rcond=None)[0] for i in range(len(q))])
            out[s:s + chunk] = (sol[:, :K] * rk).sum(1)
    bad = ~np.isfinite(out) | (np.abs(out) > 50 * (np.std(rr) + 1e-6) + np.abs(rr).max())
    if bad.any():
        nfb = int(bad.sum()); out[bad] = idw_predict(Pq[bad], Pr, rr, 2.0)
    return out, nfb

def kfold_cv_rmse(u, tr, fn, k=5, seed=0):
    """Leak-free k-fold CV *inside the labeled set*: trend is refit on every fold's training part.
    fn(P_train, res_train, P_val) -> residual predictions."""
    rng = np.random.default_rng(seed); folds = np.array_split(rng.permutation(tr), k); err = []
    for f in range(k):
        va = folds[f]; fi = np.concatenate([folds[j] for j in range(k) if j != f])
        a, n = fit_trend(u["d"][fi], u["y"][fi]); rfi = u["y"][fi] - trend_eval(u["d"][fi], a, n)
        pred = fn(u["pos"][fi], rfi, u["pos"][va]) + trend_eval(u["d"][va], a, n)
        err.append((pred - u["y"][va]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(err))))

# ----------------------------------------------------------------------------- method dispatch (shared by tuning + evaluation)
CLASSICAL = ("trend", "idw", "krig_global", "krig_local")
NEURAL = ("gcn", "kgnn", "krig_untrained")

def predict_method(method, hp, u, tr, te, prep, seed=0, graph_cache=None):
    """Returns (pred_dbm[te], info dict). Each model sees prep['res'][tr] only."""
    info = {}
    if method == "trend":
        return prep["trend"][te], info
    P, r = u["pos"], prep["res"]
    if method == "idw":
        p, k = tune_idw_loo(P[tr], r[tr]); info.update(idw_power=p, idw_k=k)
        return prep["trend"][te] + idw_predict(P[te], P[tr], r[tr], p, k), info
    if method in ("krig_global", "krig_local"):
        model = hp.get("model", "exp"); K = None if method == "krig_global" else int(hp.get("K", 24))
        par = fit_variogram(P[tr], r[tr], model)
        pred, nfb = ok_predict(P[te], P[tr], r[tr], par, model, K)
        info.update(nugget=par[0], psill=par[1], vrange=par[2], n_fallback=nfb)
        return prep["trend"][te] + pred, info
    if method in NEURAL:
        from rmbench_nn import predict_neural
        pred, info = predict_neural(method, hp, u, tr, te, prep, seed, graph_cache)
        return prep["trend"][te] + pred, info
    raise ValueError(method)

# ----------------------------------------------------------------------------- tasks + resumable parallel executor
_GRAPHS = {}

def run_task(task):
    """Top-level worker (importable -> picklable). task: dict(id, unit, density, rep, jobs=[{name,method,hp,fset?}], cfg).
    One failure never kills the task: each job returns its own row with status ok/fail."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = task["cfg"]; rows = []
    t0 = time.time()
    try:
        u = get_unit(task["unit"], cfg["cache_dir"])
        tr, te = split_for(u, task["density"], task["rep"], cfg["n_test"])
    except Exception:
        return [dict(task_id=task["id"], unit=task["unit"], density=task["density"], rep=task["rep"], name=j["name"],
                     method=j["method"], status="fail", err=traceback.format_exc(limit=3)) for j in task["jobs"]]
    preps = {}
    for j in task["jobs"]:
        row = dict(task_id=task["id"], unit=u["name"], scenario=u["scenario"], parent=u.get("parent", u["name"]),
                   density=task["density"], rep=task["rep"], name=j["name"], method=j["method"], status="ok", err="",
                   hp=json.dumps(j.get("hp", {}), sort_keys=True))
        t1 = time.time()
        try:
            fs = j.get("hp", {}).get("fset", "geo")
            if fs not in preps: preps[fs] = prepare(u, tr, fs, cfg["clip"])
            seed = stable_hash(u["name"], task["rep"], j["name"]) % 100000
            pred, info = predict_method(j["method"], j.get("hp", {}), u, tr, te, preps[fs], seed, _GRAPHS)
            e = pred - u["y"][te]
            if not np.all(np.isfinite(e)): raise FloatingPointError("non-finite prediction")
            row.update(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), n_test=len(te),
                       **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in info.items()})
        except Exception:
            row.update(status="fail", err=traceback.format_exc(limit=4), rmse=np.nan, mae=np.nan)
        row["seconds"] = time.time() - t1
        rows.append(row)
    return rows

def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass

def done_task_ids(path):
    if not os.path.exists(path): return set()
    ids = set()
    with open(path) as f:
        for line in f:
            try: ids.add(json.loads(line)[0]["task_id"])
            except Exception: pass                  # truncated last line after a crash -> task simply re-runs
    return ids

def run_tasks(tasks, out_path, n_jobs=1, log_every=10):
    """Resumable: appends one JSON line per finished task; skips task ids already present. Returns n executed."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    todo = [t for t in tasks if t["id"] not in done_task_ids(out_path)]
    print(f"[run_tasks] {len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run (n_jobs={n_jobs})")
    t0 = time.time(); n_fail = 0
    def _write(rows, fh):
        nonlocal n_fail
        n_fail += sum(r["status"] != "ok" for r in rows)
        fh.write(json.dumps(rows, default=lambda o: None if o != o else float(o) if isinstance(o, np.floating) else str(o)) + "\n"); fh.flush()
    with open(out_path, "a") as fh:
        if n_jobs <= 1:
            for i, t in enumerate(todo, 1):
                _write(run_task(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  failed jobs: {n_fail}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(n_jobs, initializer=_init_worker) as ex:
                futs = {ex.submit(run_task, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:                                     # worker crash (e.g. OOM) -> record, keep going
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"],
                                                  method=j["method"], status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    _write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  failed jobs: {n_fail}")
    return len(todo)

def read_results(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try: rows += json.loads(line)
                except Exception: pass
    return pd.DataFrame(rows)

def make_tasks(units, densities, n_rep, jobs, cfg, prefix):
    return [dict(id=f"{prefix}|{un}|{d}|{r}", unit=un, density=int(d), rep=r, jobs=jobs, cfg=cfg)
            for un in units for d in densities for r in range(n_rep)]

# ----------------------------------------------------------------------------- tuning helpers
def random_search_space(space, n_trials, seed=0):
    """space: {name: ('choice', [..]) | ('loguniform', lo, hi)}; trial 0 is always the defaults (first choice / geometric mean)."""
    rng = np.random.default_rng(seed); out = []
    for t in range(n_trials):
        hp = {}
        for k, s in space.items():
            if s[0] == "choice": hp[k] = s[1][0] if t == 0 else s[1][int(rng.integers(len(s[1])))]
            else: hp[k] = float(math.sqrt(s[1] * s[2])) if t == 0 else float(math.exp(rng.uniform(math.log(s[1]), math.log(s[2]))))
        out.append(hp)
    return out

GCN_SPACE = dict(hidden=("choice", [32, 16, 64]), layers=("choice", [2, 3]), lr=("loguniform", 3e-3, 3e-2),
                 dropout=("choice", [0.1, 0.0, 0.3]), wd=("loguniform", 1e-5, 1e-2), k=("choice", [8, 6, 12]),
                 fset=("choice", ["geo", "geo_los"]), steps=("choice", [150]))
KGNN_SPACE = dict(Kmax=("choice", [12, 8, 16]), lr=("loguniform", 3e-3, 3e-2), s2_init=("choice", [0.05, 0.3]),
                  wd=("loguniform", 1e-4, 1e-2), steps=("choice", [150]))

def tune_summary(df, group=("name",)):
    ok = df[df.status == "ok"]
    unit_level = ok.groupby(list(group) + ["unit", "density"], as_index=False).rmse.mean()
    return unit_level.groupby(list(group) + ["density"], as_index=False).rmse.mean()

# ----------------------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def paired_test(a, b, rng_seed=0):
    """Two-sided paired test on unit-level RMSE. Wilcoxon signed-rank when >=6 non-zero differences, otherwise an exact
    sign-flip permutation test (min attainable p = 2/2^n is reported, so under-powered cells are visible)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[np.isfinite(d)]; nz = d[d != 0]; n = len(d)
    if n < 2 or len(nz) == 0: return dict(n=n, p=1.0, test="none", min_p=1.0)
    min_p = min(1.0, 2.0 / 2 ** len(nz))
    if len(nz) >= 6:
        return dict(n=n, p=float(sstats.wilcoxon(nz, zero_method="wilcox", method="auto").pvalue), test="wilcoxon", min_p=min_p)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(nz))))
    obs = abs(nz.mean()); p = float(np.mean(np.abs((signs * np.abs(nz)).mean(1)) >= obs - 1e-12))
    return dict(n=n, p=p, test="signflip-exact", min_p=min_p)

def unit_level(df):
    """Average replicates, then patches -> parent: one number per (scenario, parent unit, density, method)."""
    ok = df[df.status == "ok"]
    u = ok.groupby(["scenario", "parent", "unit", "density", "name"], as_index=False).rmse.mean()
    return u.groupby(["scenario", "parent", "density", "name"], as_index=False).rmse.mean()

def paired_contrasts(df, contrasts, alpha=0.05, max_tests=24, seed=0):
    """contrasts: list of (A, B). delta = RMSE_B - RMSE_A  (>0 means A better). Holm over ALL tests (pre-capped)."""
    ul = unit_level(df); rows = []
    for (A, B) in contrasts:
        for (scen, dens), g in ul.groupby(["scenario", "density"]):
            w = g.pivot(index="parent", columns="name", values="rmse")
            if A not in w or B not in w: continue
            w = w[[A, B]].dropna()
            if len(w) < 2: continue
            t = paired_test(w[B].values, w[A].values)
            delta = (w[B] - w[A]).values; rng = np.random.default_rng(seed)
            bs = [rng.choice(delta, len(delta)).mean() for _ in range(2000)]
            rows.append(dict(scenario=scen, density=dens, A=A, B=B, n_units=len(w), mean_delta_db=delta.mean(),
                             median_delta_db=float(np.median(delta)), ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                             rel_gain_pct=100 * delta.mean() / w[B].mean(), p=t["p"], test=t["test"], min_attainable_p=t["min_p"]))
    out = pd.DataFrame(rows)
    if len(out) > max_tests: raise ValueError(f"{len(out)} tests exceed the pre-specified cap of {max_tests}")
    if len(out):
        out["p_holm"] = holm(out.p.values); out["significant"] = out.p_holm < alpha
        out["underpowered"] = out.min_attainable_p >= alpha
    return out

def rank_table(df):
    ul = unit_level(df)
    t = ul.groupby(["scenario", "density", "name"], as_index=False).agg(rmse=("rmse", "mean"), se=("rmse", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan), n_units=("rmse", "size"))
    t["rank"] = t.groupby(["scenario", "density"]).rmse.rank(method="min").astype(int)
    return t.sort_values(["scenario", "density", "rank"])

def estimate_budget(timing_s, n_units, densities, n_rep, methods, n_jobs):
    """timing_s: {method: seconds per prediction (at the largest density)}; returns total core-hours / wall-hours."""
    per = sum(timing_s[m] for m in methods)
    core_s = per * n_units * len(densities) * n_rep
    return dict(core_hours=core_s / 3600, wall_hours=core_s / 3600 / max(n_jobs, 1), per_task_s=per)
