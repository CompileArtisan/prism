"""rmbench_ext5: regime-map statistics, extra baselines, strawman diagnostics and the training-scale experiment.

Sits next to rmbench_core / rmbench_nn / rmbench_ext / rmbench_ext2 (all unchanged). Only ADDS code.
Pure numpy/pandas/sklearn parts (sections 1-3) import without torch; torch parts are guarded by HAVE_TORCH.
Every new predictor has the signature   f(u, tr, te, prep, hp, seed) -> (prediction in dBm at te, info dict)   and only reads
prep['res'][tr] (labels at tr). Test labels enter only through the RMSE (see leak_check).
"""
from __future__ import annotations
import os, re, glob, json, time, math, warnings, traceback
import numpy as np, pandas as pd
import rmbench_core as rc

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                   # analysis-only environment
    HAVE_TORCH = False

MARGIN_PCT = 2.0            # equivalence margin used in rounds 2-4
_GRAPHS = {}

# =============================================================================== 1. unit covariates (descriptive, never fed to a model)
def _poly2(P):
    x = (P[:, 0] - P[:, 0].mean()) / (P[:, 0].std() + 1e-9); y = (P[:, 1] - P[:, 1].mean()) / (P[:, 1].std() + 1e-9)
    return np.column_stack([np.ones(len(P)), x, y, x * x, x * y, y * y])

def unit_covariates(u, max_pts=400, seed=0, trend="poly2"):
    """Properties of the *target* that decide whether any interpolator can win.
    trend='poly2' (default): residual of a quadratic surface fit, identical definition for every dataset (also unknown-TX Wi-Fi).
    noise_frac = nugget / (nugget + partial sill) of an exponential variogram of that residual (1 = pure noise, 0 = smooth field)."""
    N = len(u["y"]); rng = np.random.default_rng(seed); idx = np.sort(rng.choice(N, min(N, max_pts), replace=False))
    P = np.asarray(u["pos"], float)[idx]; y = np.asarray(u["y"], float)[idx]
    if trend == "logd" and "d" in u and np.ptp(u["d"]) > 1e-6 and int(np.asarray(u.get("tx_known", 1))) == 1:
        d = np.asarray(u["d"], float)[idx]; a, n = rc.fit_trend(d, y); r = y - rc.trend_eval(d, a, n)
    else:
        A = _poly2(P); r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    nug, ps, rg = rc.fit_variogram(P, r, "exp")
    full = np.asarray(u["pos"], float); area = float(np.ptp(full[:, 0]) * np.ptp(full[:, 1]) + 1e-9)
    out = dict(noise_frac=float(nug / (nug + ps + 1e-12)), vrange=float(rg), area=area, n_pool=N, resid_std=float(r.std()),
               y_std=float(np.std(u["y"])), los_frac=float(np.mean(u["los"])) if "los" in u else np.nan, side_r2=np.nan)
    if "side" in u:
        S = np.asarray(u["side"], float)[idx]; A = np.column_stack([np.ones(len(S)), S]); b = np.linalg.lstsq(A, r, rcond=None)[0]
        out["side_r2"] = float(1 - ((r - A @ b) ** 2).sum() / (((r - r.mean()) ** 2).sum() + 1e-12))
    return out

def covariate_table(units, cache_dirs, **kw):
    rows = []
    for n in sorted(set(units)):
        u = None
        for cd in cache_dirs:
            if os.path.exists(os.path.join(cd, "units", n.replace("|", "__") + ".npz")):
                u = rc.get_unit(n, cd); break
        if u is None: continue
        try: rows.append(dict(unit=n, **unit_covariates(u, **kw)))
        except Exception as e: rows.append(dict(unit=n, err=repr(e)))
    return pd.DataFrame(rows).set_index("unit") if rows else pd.DataFrame()

def add_density_covariates(g):
    """labels per squared correlation range (the 'how many labels does a correlation cell contain' number) and friends."""
    g = g.copy()
    g["lpr"] = g.density * g.vrange ** 2 / g.area; g["log_lpr"] = np.log10(g.lpr.clip(lower=1e-6))
    g["spacing_ratio"] = np.sqrt(g.area / g.density) / g.vrange; g["log_density"] = np.log10(g.density)
    g["is_blocked"] = (g["split"] == "blocked").astype(float) if "split" in g else 0.0
    return g

# =============================================================================== 2. results loading, paired gains, verdicts
def load_results(patterns):
    frames = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            d = rc.read_results(p)
            if len(d): d["src"] = os.path.basename(p); d["family"] = os.path.basename(os.path.dirname(p)); frames.append(d)
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["split"] = np.where(df.src.str.contains("blocked"), "blocked", "random")
    return df

def unit_rmse(df):
    ok = df[df.status == "ok"]; keys = [c for c in ("family", "split", "scenario", "parent", "unit", "density") if c in ok]
    return ok.groupby(keys + ["name"], as_index=False).rmse.mean()

def paired_gain(df, A, B):
    """One row per (unit, density[, family, split]); gain > 0 means A better. Replicates are averaged first."""
    u = unit_rmse(df); keys = [c for c in u.columns if c not in ("name", "rmse")]
    w = u.pivot_table(index=keys, columns="name", values="rmse", dropna=False)
    if A not in w.columns or B not in w.columns or A == B: return pd.DataFrame()
    w = w[[A, B]].dropna().reset_index(); w["gain_db"] = w[B] - w[A]; w["gain_pct"] = 100 * w.gain_db / w[B]
    w["rmse_A"], w["rmse_B"] = w[A], w[B]; w["A"], w["B"] = A, B
    return w.drop(columns=[A, B])

def _cluster_mean_ci(values, clusters, n_boot=4000, seed=0):
    d = pd.DataFrame(dict(v=np.asarray(values, float), c=np.asarray(clusters))).dropna()
    m = d.groupby("c").v.mean().values
    if len(m) < 2: return np.nan, np.nan, np.nan, len(m)
    bs = np.random.default_rng(seed).choice(m, (n_boot, len(m))).mean(1)
    return float(m.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), len(m)

def verdict(lo, hi, margin=MARGIN_PCT):
    if not (np.isfinite(lo) and np.isfinite(hi)): return "too few clusters"
    if lo > -margin and hi < margin: return f"equivalent (CI within +/-{margin:g}%)"
    if lo > 0: return "A better"
    if hi < 0: return "A worse"
    return "inconclusive"

def contrast_table(df, contrasts, by=("family", "split", "scenario", "density"), margin=MARGIN_PCT, min_clusters=4):
    """Per group: cluster (parent) bootstrap CI of the mean relative gain, dB gain, Wilcoxon/sign-flip p (Holm over the whole table)."""
    rows = []
    for A, B in contrasts:
        w = paired_gain(df, A, B)
        if not len(w): continue
        by_ = [c for c in by if c in w]
        for key, g in w.groupby(by_):
            key = key if isinstance(key, tuple) else (key,)
            m, lo, hi, nc = _cluster_mean_ci(g.gain_pct, g.parent)
            pm = g.groupby("parent")[["rmse_A", "rmse_B"]].mean()
            t = rc.paired_test(pm.rmse_B.values, pm.rmse_A.values) if nc >= 2 else dict(p=1.0)
            rows.append(dict(zip(by_, key), A=A, B=B, n_clusters=nc, mean_gain_pct=m, ci_lo=lo, ci_hi=hi, mean_gain_db=float(g.groupby("parent").gain_db.mean().mean()),
                             p=t["p"], verdict=verdict(lo, hi, margin) if nc >= min_clusters else "too few clusters"))
    out = pd.DataFrame(rows)
    if len(out): out["p_holm"] = rc.holm(out.p.values)
    return out

# =============================================================================== 3. regime regression + bins
def cluster_ols(w, y, xs, cluster="parent", n_boot=2000, seed=0, max_nan_frac=0.3):
    """OLS of y on standardised covariates; CIs by bootstrap over clusters (parents). Degenerate / mostly-missing covariates are dropped and listed."""
    keep, dropped = [], []
    for x in xs:
        if x not in w or w[x].isna().mean() > max_nan_frac or w[x].nunique() < 2: dropped.append(x)
        else: keep.append(x)
    d = w[[y, cluster] + keep].replace([np.inf, -np.inf], np.nan).dropna()
    if len(keep) == 0 or len(d) < 12 or d[cluster].nunique() < 5: return pd.DataFrame(), dropped
    mu, sd = d[keep].mean(), d[keep].std().replace(0, 1.0)
    X = np.column_stack([np.ones(len(d)), ((d[keep] - mu) / sd).values]); Y = d[y].values
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; groups = [np.where(d[cluster].values == c)[0] for c in d[cluster].unique()]
    rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        B.append(np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0])
    B = np.array(B); r2 = 1 - ((Y - X @ beta) ** 2).sum() / (((Y - Y.mean()) ** 2).sum() + 1e-12)
    out = pd.DataFrame(dict(term=["intercept"] + keep, coef_per_sd=beta, ci_lo=np.quantile(B, .025, 0), ci_hi=np.quantile(B, .975, 0),
                            p_boot=[min(1.0, 2 * min((b > 0).mean(), (b < 0).mean())) for b in B.T]))
    out["n_rows"] = len(d); out["n_clusters"] = len(groups); out["r2_in_sample"] = r2
    return out, dropped

def lofo_r2(w, y, xs, group="family"):
    """Leave-one-family-out R^2 of the regime model (does the map predict gains in a dataset it has never seen?)."""
    d = w[[y, group] + xs].replace([np.inf, -np.inf], np.nan).dropna(); rows = []
    for gname in d[group].unique():
        tr, te = d[d[group] != gname], d[d[group] == gname]
        if len(tr) < 12 or len(te) < 5: continue
        mu, sd = tr[xs].mean(), tr[xs].std().replace(0, 1.0)
        Xa = np.column_stack([np.ones(len(tr)), ((tr[xs] - mu) / sd).values]); b = np.linalg.lstsq(Xa, tr[y].values, rcond=None)[0]
        pred = np.column_stack([np.ones(len(te)), ((te[xs] - mu) / sd).values]) @ b
        sse = ((te[y].values - pred) ** 2).sum(); sst = ((te[y].values - tr[y].mean()) ** 2).sum() + 1e-12
        rows.append(dict(held_out=gname, n=len(te), r2_vs_train_mean=1 - sse / sst))
    return pd.DataFrame(rows)

def binned_gain(w, x, q=5, y="gain_pct", n_boot=2000, seed=0):
    d = w[[x, y, "parent"]].replace([np.inf, -np.inf], np.nan).dropna()
    if d[x].nunique() < q or len(d) < 3 * q: return pd.DataFrame()
    d["bin"] = pd.qcut(d[x], q, duplicates="drop"); rows = []
    for b, g in d.groupby("bin", observed=True):
        m, lo, hi, nc = _cluster_mean_ci(g[y], g.parent, n_boot, seed)
        rows.append(dict(x=float(g[x].median()), gain=m, ci_lo=lo, ci_hi=hi, n_clusters=nc, n=len(g)))
    return pd.DataFrame(rows)

def crossovers(bt):
    """x values where the binned mean gain changes sign (linear interpolation)."""
    out = []
    for i in range(len(bt) - 1):
        a, b = bt.gain.iloc[i], bt.gain.iloc[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            out.append(float(bt.x.iloc[i] + (0 - a) * (bt.x.iloc[i + 1] - bt.x.iloc[i]) / (b - a)))
    return out

def plot_regime(gains, methods, xs=("noise_frac", "log_lpr"), margin=MARGIN_PCT, q=5, fname=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(xs), 1, figsize=(6.4, 3.2 * len(xs)), squeeze=False)
    for r, x in enumerate(xs):
        a = ax[r, 0]; a.axhline(0, color="k", lw=.8); a.axhspan(-margin, margin, color="grey", alpha=.15)
        for m in methods:
            w = gains[gains.A == m]; bt = binned_gain(w, x, q)
            if len(bt): a.errorbar(bt.x, bt.gain, yerr=[bt.gain - bt.ci_lo, bt.ci_hi - bt.gain], marker="o", capsize=2, label=m)
        a.set_xlabel({"noise_frac": "nugget share of residual variance", "log_lpr": "log10 labels per squared correlation range"}.get(x, x))
        a.set_ylabel("RMSE gain over reference (%)"); a.legend(fontsize=7)
    fig.tight_layout()
    if fname: fig.savefig(fname, dpi=200)
    return fig

# =============================================================================== 4. extra classical / neural baselines
def _norm_pos(P, tr):
    P = np.asarray(P, float); lo = P[tr].min(0); span = float(np.ptp(P[tr], axis=0).max()) + 1e-9
    return (P - lo) / span

def _idw_fallback(u, tr, te, prep):
    p, k = rc.tune_idw_loo(u["pos"][tr], prep["res"][tr]); return prep["trend"][te] + rc.idw_predict(u["pos"][te], u["pos"][tr], prep["res"][tr], p, k)

def predict_gp_ml(u, tr, te, prep, hp, seed=0):
    """Zero-mean GP on the trend residual, Matern(nu) with ARD length scales + white noise, hyper-parameters by maximum marginal likelihood."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    P = np.asarray(u["pos"], float); s = prep["sigma"]; y = prep["res"][tr] / s; mu0 = float(y.mean())
    span = float(np.ptp(P[tr], axis=0).max()) + 1e-9
    ls0 = [0.25 * span] * 2 if hp.get("ard", True) else 0.25 * span
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(ls0, (0.02 * span, 5 * span), nu=float(hp.get("nu", 1.5))) + WhiteKernel(0.1, (1e-3, 2.0))
    nfit = int(hp.get("fit_max", 400)); rng = np.random.default_rng(seed)     # O(n^3): fit hyper-parameters on <= fit_max labels, then condition on ALL labels
    sub = np.arange(len(tr)) if len(tr) <= nfit else np.sort(rng.choice(len(tr), nfit, replace=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(k, alpha=1e-8, n_restarts_optimizer=int(hp.get("restarts", 2)), random_state=int(seed)).fit(P[tr][sub], (y - mu0)[sub])
        if len(sub) < len(tr): gp = GaussianProcessRegressor(gp.kernel_, alpha=1e-8, optimizer=None).fit(P[tr], y - mu0)
        m = gp.predict(P[te])
    pred = prep["trend"][te] + (m + mu0) * s
    return pred, dict(gp_logml=float(gp.log_marginal_likelihood_value_))

def predict_rbf_spline(u, tr, te, prep, hp, seed=0):
    """Thin-plate (or other RBF) smoothing spline; smoothing parameter by 5-fold CV inside the labelled set."""
    from scipy.interpolate import RBFInterpolator
    if len(tr) < 8: return _idw_fallback(u, tr, te, prep), dict(rbf_fallback=1.0)
    Pn = _norm_pos(u["pos"], tr); s = prep["sigma"]; y = prep["res"][tr] / s; Ptr = Pn[tr]
    kern = hp.get("kernel", "thin_plate_spline"); grid = hp.get("smooth_grid", (1e-3, 1e-2, 1e-1, 1.0, 10.0))
    rng = np.random.default_rng(seed); Ptr = Ptr + rng.normal(0, 1e-9, Ptr.shape)
    folds = np.array_split(rng.permutation(len(tr)), 5); best = (np.inf, grid[0])
    for sm in grid:
        err = []
        try:
            for f in folds:
                fi = np.setdiff1d(np.arange(len(tr)), f)
                err.append((RBFInterpolator(Ptr[fi], y[fi], kernel=kern, smoothing=sm, degree=1)(Ptr[f]) - y[f]) ** 2)
            sc = float(np.mean(np.concatenate(err)))
            if sc < best[0]: best = (sc, sm)
        except Exception:
            continue
    if not np.isfinite(best[0]): return _idw_fallback(u, tr, te, prep), dict(rbf_fallback=1.0)
    m = RBFInterpolator(Ptr, y, kernel=kern, smoothing=best[1], degree=1)(Pn[te])
    return prep["trend"][te] + m * s, dict(rbf_smooth=float(best[1]), rbf_cv=best[0])

if HAVE_TORCH:
    def predict_coord_mlp(u, tr, te, prep, hp, seed=0):
        """Coordinate network (random Fourier features + 2 hidden layers) fitted to the residual of ONE unit; validation early stopping."""
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        Pn = _norm_pos(u["pos"], tr); s = prep["sigma"]; r = torch.as_tensor((prep["res"] / s).astype(np.float32))
        nf, sc, hid = int(hp.get("n_fourier", 64)), float(hp.get("fourier_scale", 3.0)), int(hp.get("hidden", 64))
        Bm = torch.tensor(rng.normal(0, sc, (2, nf)), dtype=torch.float32)
        def feat(Z): Zt = torch.as_tensor(Z, dtype=torch.float32); a = 2 * math.pi * Zt @ Bm; return torch.cat([Zt, torch.sin(a), torch.cos(a)], 1)
        Ft = feat(Pn); net = nn.Sequential(nn.Linear(Ft.shape[1], hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        perm = rng.permutation(tr); nv = max(3, int(0.2 * len(tr))) if len(tr) >= 12 else 0; va, fi = perm[:nv], perm[nv:]
        va_t, fi_t = torch.as_tensor(va, dtype=torch.long), torch.as_tensor(fi, dtype=torch.long)
        opt = torch.optim.AdamW(net.parameters(), lr=float(hp.get("lr", 3e-3)), weight_decay=float(hp.get("wd", 1e-3)))
        best, state, bad = float("inf"), None, 0
        for it in range(int(hp.get("steps", 400))):
            opt.zero_grad(); F.huber_loss(net(Ft[fi_t]).squeeze(-1), r[fi_t], delta=1.5).backward(); opt.step()
            if nv and it % 10 == 9:
                with torch.no_grad(): v = F.huber_loss(net(Ft[va_t]).squeeze(-1), r[va_t], delta=1.5).item()
                if v < best - 1e-5: best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8: break
        if state is not None: net.load_state_dict(state)
        with torch.no_grad(): m = net(Ft[torch.as_tensor(te, dtype=torch.long)]).squeeze(-1).numpy()
        return prep["trend"][te] + m * s, dict(mlp_val=float(best))

    # ------------------------------------------------------------------ neural process (attentive, meta-trained across units; zero-shot at test time)
    class NPNet(nn.Module):
        def __init__(self, hidden=64):
            super().__init__(); h = hidden; self.h = h
            self.ex = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h))
            self.exy = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h))
            self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
            self.dec = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
        def forward(self, xc, yc, xt):
            ht = self.ex(xt); hc = self.ex(xc); r = self.exy(torch.cat([xc, yc[:, None]], 1))
            att = torch.softmax(self.q(ht) @ self.k(hc).T / math.sqrt(self.h), dim=1)
            out = self.dec(torch.cat([ht, att @ self.v(r), r.mean(0, keepdim=True).expand(len(xt), -1)], 1))
            return out[:, 0], 0.02 + F.softplus(out[:, 1])

    def make_np_episodes(names, cfg, reps=2, seed_base=700):
        """Source episodes (normalised positions, normalised residual, labelled idx, held-out idx) from units that are NOT evaluation units."""
        import rmbench_ext as rx, rmbench_ext2 as r2
        eps = []
        for n in names:
            u0 = rx.get_unit_ext(n, cfg["cache_dir"])
            for d in cfg["densities"]:
                for r in range(reps):
                    try:
                        tr, te = r2.split_for2(u0, d, seed_base + r, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
                        u, spec = r2.view_for_split2(u0, tr); prep = r2.prepare2(u, tr, "geo", cfg["clip"], spec)
                        eps.append(dict(P=_norm_pos(u["pos"], tr).astype(np.float32), y=(prep["res"] / prep["sigma"]).astype(np.float32), tr=tr, te=te))
                    except Exception:
                        continue
        return eps

    def _np_loss(model, ep, rng, n_tgt=128):
        tr, te = ep["tr"], ep["te"]; m = int(np.clip(round(rng.uniform(0.4, 1.0) * len(tr)), 3, len(tr)))
        ctx = rng.choice(tr, m, replace=False); pool = np.concatenate([te, np.setdiff1d(tr, ctx)]); tg = rng.choice(pool, min(n_tgt, len(pool)), replace=False)
        P, y = torch.as_tensor(ep["P"]), torch.as_tensor(ep["y"])
        mu, sd = model(P[ctx], y[ctx], P[tg])
        return (0.5 * ((y[tg] - mu) / sd) ** 2 + torch.log(sd)).mean()

    def pretrain_np(episodes, steps, seed, out_path, hidden=64, batch=4, lr=1e-3, eval_every=250, log=print):
        torch.manual_seed(seed); rng = np.random.default_rng(seed); model = NPNet(hidden); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        order = rng.permutation(len(episodes)); nv = max(1, len(episodes) // 10); va = [episodes[i] for i in order[:nv]]; trn = [episodes[i] for i in order[nv:]]
        best, state = float("inf"), None
        for s in range(steps):
            model.train(); opt.zero_grad()
            loss = sum(_np_loss(model, trn[int(rng.integers(len(trn)))], rng) for _ in range(batch)) / batch
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            if s % eval_every == eval_every - 1 or s == steps - 1:
                model.eval(); vr = np.random.default_rng(123)
                with torch.no_grad(): v = float(np.mean([_np_loss(model, e, vr).item() for e in va for _ in range(2)]))
                if v < best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
                log(f"    NP step {s + 1}/{steps} train {loss.item():.3f} val {v:.3f}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True); torch.save(dict(state=state, hidden=hidden, val=best), out_path); return out_path

    _NP_CACHE = {}
    def predict_np_prior(u, tr, te, prep, hp, seed=0):
        path = hp["np_path"]
        if path not in _NP_CACHE:
            ck = torch.load(path, map_location="cpu"); mdl = NPNet(ck["hidden"]); mdl.load_state_dict(ck["state"]); mdl.eval(); _NP_CACHE[path] = mdl
        mdl = _NP_CACHE[path]; Pn = torch.as_tensor(_norm_pos(u["pos"], tr), dtype=torch.float32); s = prep["sigma"]
        y = torch.as_tensor((prep["res"] / s).astype(np.float32))
        with torch.no_grad(): mu, sd = mdl(Pn[torch.as_tensor(tr)], y[torch.as_tensor(tr)], Pn[torch.as_tensor(te)])
        return prep["trend"][te] + mu.numpy() * s, dict(np_sd=float(sd.mean()))

    # ------------------------------------------------------------------ GNN residual on top of ridge+IDW ("rk_side"-style base): the fair side-information GNN
    def predict_rk_gcn(u, tr, te, prep, hp, seed=0):
        """Label-aware GCN whose zero-initialised head ADDS a correction to a ridge(prep['X']) + LOO-IDW base, validation early stopping includes step 0.
        steps=0 returns the base itself (an in-sample-ridge variant of rk_side). If this loses to rk_side the loss is selection noise, not capacity."""
        import rmbench_nn as rn
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        N = len(u["y"]); g = rn.get_graph(u, int(hp.get("k", 8)), _GRAPHS); X = torch.tensor(prep["X"], dtype=torch.float32)
        sig = prep["sigma"]; res = np.asarray(prep["res"]); Pt = torch.tensor(np.asarray(u["pos"]), dtype=torch.float32)
        rn_ = torch.zeros(N); rn_[torch.as_tensor(tr)] = torch.tensor(res[tr] / sig, dtype=torch.float32)
        A = np.column_stack([np.ones(N), prep["X"]]); lam = float(hp.get("lam", 1.0)); pen = np.eye(A.shape[1]); pen[0, 0] = 0
        def base_for(vis):
            vis = np.asarray(vis); At = A[vis]; beta = np.linalg.solve(At.T @ At + lam * pen, At.T @ res[vis]); fit = A @ beta
            r2_ = torch.tensor((res[vis] - fit[vis]) / sig, dtype=torch.float32)
            return torch.tensor(fit / sig, dtype=torch.float32) + rn.idw_torch(Pt, torch.as_tensor(vis, dtype=torch.long), r2_, 2.0, 8)
        model = rn.GCNNet(X.shape[1] + 9, int(hp.get("hidden", 32)), int(hp.get("layers", 2)), float(hp.get("dropout", 0.1)))
        steps = int(hp.get("steps", 150)); fit, val = rn._split_val(tr, rng); val_t = torch.as_tensor(val, dtype=torch.long)
        def run(vis, train):
            m = torch.zeros(N); m[torch.as_tensor(np.asarray(vis), dtype=torch.long)] = 1.0; model.train(train)
            return model(X, rn_ * m, m, g, base_for(vis))
        moved = 0.0
        if steps > 0:
            opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-4))); best, state, bad = float("inf"), None, 0
            if len(val):
                with torch.no_grad(): best = F.huber_loss(run(fit, False)[val_t], rn_[val_t], delta=1.5).item()
                state = {k: t.clone() for k, t in model.state_dict().items()}
            for it in range(steps):
                q = int(np.clip(round(rng.uniform(0.15, 0.5) * len(fit)), 1, max(1, len(fit) - 2))); pp = rng.permutation(fit); Q, V = pp[:q], pp[q:]
                opt.zero_grad(); out = run(V, True); Qt = torch.as_tensor(Q, dtype=torch.long)
                F.huber_loss(out[Qt], rn_[Qt], delta=1.5).backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                if len(val) and it % 5 == 4:
                    with torch.no_grad(): v = F.huber_loss(run(fit, False)[val_t], rn_[val_t], delta=1.5).item()
                    if v < best - 1e-5: best, state, bad, moved = v, {k: t.clone() for k, t in model.state_dict().items()}, 0, 1.0
                    else:
                        bad += 1
                        if bad >= 8: break
            if state is not None: model.load_state_dict(state)
        with torch.no_grad(): pred = run(tr, False)[torch.as_tensor(te, dtype=torch.long)].numpy() * sig
        return prep["trend"][te] + pred, dict(left_step0=moved)

NEW_METHODS = {"gp_ml": predict_gp_ml, "rbf_spline": predict_rbf_spline}
if HAVE_TORCH: NEW_METHODS.update(coord_mlp=predict_coord_mlp, np_prior=predict_np_prior, rk_gcn=predict_rk_gcn)

# =============================================================================== 5. job lists
def extra_jobs(side=False):
    J = [dict(name="gp_ml", method="gp_ml", hp=dict(nu=1.5, ard=True, restarts=2)), dict(name="rbf_spline", method="rbf_spline", hp={}),
         dict(name="coord_mlp", method="coord_mlp", hp=dict(steps=400))]
    if side: J += [dict(name="rk_gcn", method="rk_gcn", hp=dict(fset="geo_side", steps=150)), dict(name="rk_gcn_s0", method="rk_gcn", hp=dict(fset="geo_side", steps=0))]
    return J

def diag_jobs():
    """Selection-noise diagnostic: gcn_krig starts at IDW (zero-initialised head). steps=0 IS that starting point."""
    return [dict(name="gcn_krig", method="gcn_krig", hp=dict(steps=150)), dict(name="gcn_krig_s0", method="gcn_krig", hp=dict(steps=0))]

# =============================================================================== 6. executor (mirrors run_task_ext2, plus the new methods)
def dispatch(j, u, tr, te, prep, seed):
    import rmbench_ext as rx, rmbench_ext2 as r2
    m, hp = j["method"], j.get("hp", {})
    if m == "gnn": p, info = rx.fit_predict_graph(u, tr, te, prep, hp, seed, rx._GRAPHS); return prep["trend"][te] + p, info
    if m == "rk_side": return r2.predict_rk_side(u, tr, te, prep, hp)
    if m in NEW_METHODS: return NEW_METHODS[m](u, tr, te, prep, hp, seed)
    return rc.predict_method(m, hp, u, tr, te, prep, seed, rx._GRAPHS)

def _view_prep(u0, tr, cfg, hp, views, preps):
    import rmbench_ext as rx, rmbench_ext2 as r2
    pol = hp.get("trend_policy", cfg.get("trend_policy", "robust")); fs = hp.get("fset", "geo")
    if pol not in views: views[pol] = (rx.view_for_split(u0, tr), None) if pol == "legacy" else r2.view_for_split2(u0, tr, force="const" if pol == "const" else None)
    u, spec = views[pol]
    if (pol, fs) not in preps: preps[(pol, fs)] = rc.prepare(u, tr, fs, cfg["clip"], cfg.get("trend_mode", "fit")) if pol == "legacy" else r2.prepare2(u, tr, fs, cfg["clip"], spec)
    return u, spec, preps[(pol, fs)], pol

def run_task5(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if HAVE_TORCH: torch.set_num_threads(1)
    import rmbench_ext as rx, rmbench_ext2 as r2
    cfg, rows = task["cfg"], []
    try:
        u0 = rx.get_unit_ext(task["unit"], cfg["cache_dir"])
        tr, te = r2.split_for2(u0, task["density"], task["rep"], cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
    except Exception:
        return [dict(task_id=task["id"], unit=task["unit"], density=task["density"], rep=task["rep"], name=j["name"], method=j["method"], status="fail", err=traceback.format_exc(limit=3)) for j in task["jobs"]]
    views, preps = {}, {}; yte = u0["y"][te]; vte = float(yte.var() + 1e-9)
    for j in task["jobs"]:
        hp = j.get("hp", {})
        row = dict(task_id=task["id"], unit=u0["name"], scenario=u0["scenario"], layout=u0["layout"], parent=u0["parent"], density=task["density"], rep=task["rep"],
                   name=j["name"], method=j["method"], status="ok", err="", hp=json.dumps(hp, sort_keys=True, default=str))
        t1 = time.time()
        try:
            u, spec, prep, pol = _view_prep(u0, tr, cfg, hp, views, preps); row["trend_policy"] = pol
            seed = rc.stable_hash(u["name"], task["rep"], j["name"]) % 100000
            pred, info = dispatch(j, u, tr, te, prep, seed); e = pred - yte
            if not np.all(np.isfinite(e)): raise FloatingPointError("non-finite prediction")
            mse = float(np.mean(e ** 2))
            row.update(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(e))), n_test=len(te), skill=1.0 - mse / vte, tx_known=u0["tx_known"],
                       **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in info.items()})
            try: row.update(r2.cov_metrics2(pred, yte, u["y"][tr], cfg))
            except Exception: pass
        except Exception:
            row.update(status="fail", err=traceback.format_exc(limit=4), rmse=np.nan, mae=np.nan)
        row["seconds"] = time.time() - t1; rows.append(row)
    return rows

def make_tasks5(units, densities, reps, jobs, cfg, prefix):
    return [dict(id=f"{prefix}|{un}|{d}|{r}", unit=un, density=int(d), rep=int(r), jobs=jobs, cfg=cfg) for un in units for d in densities for r in reps]

def run_tasks5(tasks, out_path, n_jobs=1, log_every=25):
    """Resumable JSONL executor, same file protocol as rc.run_tasks (one JSON list of rows per line)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks5] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
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
                write(run_task5(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task5, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"], status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

def leak_check(jobs, unit, cfg, density=10, rep=0, noise_db=50.0):
    """Predictions must be bit-identical when the TEST labels are corrupted. Returns {job name: True/False}."""
    import rmbench_ext as rx, rmbench_ext2 as r2
    u0 = rx.get_unit_ext(unit, cfg["cache_dir"]); tr, te = r2.split_for2(u0, density, rep, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
    out = {}
    for j in jobs:
        preds = []
        for corrupt in (False, True):
            try:
                import rmbench_ext as _rx; _rx._GRAPHS.clear()          # v2: a cached graph must not survive between the clean and the corrupted pass
            except Exception: pass
            u1 = dict(u0); y = np.array(u0["y"], float)
            if corrupt: y[te] += np.random.default_rng(1).normal(0, noise_db, len(te))
            u1["y"] = y; u, spec, prep, _ = _view_prep(u1, tr, cfg, j.get("hp", {}), {}, {})
            preds.append(dispatch(j, u, tr, te, prep, 7)[0])
        out[j["name"]] = bool(np.array_equal(preds[0], preds[1]))
    return out

# =============================================================================== 7. diagnostics: selection noise
def selection_harm(df, trained="gcn_krig", step0="gcn_krig_s0", margin=MARGIN_PCT):
    """Per fit: RMSE(trained) - RMSE(step 0 = the IDW-based starting point). Positive = training made it worse. Validation picked the checkpoint; this is its cost."""
    ok = df[df.status == "ok"]; keys = [c for c in ("family", "split", "scenario", "parent", "unit", "density", "rep") if c in ok]
    w = ok.pivot_table(index=keys, columns="name", values="rmse")
    if trained not in w or step0 not in w: return pd.DataFrame()
    w = w[[trained, step0]].dropna().reset_index(); w["harm_pct"] = 100 * (w[trained] - w[step0]) / w[step0]
    rows = []
    for d, g in w.groupby("density"):
        m, lo, hi, nc = _cluster_mean_ci(g.harm_pct, g.parent)
        rows.append(dict(density=d, n_fits=len(g), mean_harm_pct=m, ci_lo=lo, ci_hi=hi, share_worse=float((g.harm_pct > 0).mean()), share_worse_gt_margin=float((g.harm_pct > margin).mean()),
                         share_better_gt_margin=float((g.harm_pct < -margin).mean())))
    return pd.DataFrame(rows)

# =============================================================================== 8. training-scale experiment
def build_scale_pool(cfg, n_eval=16, n_train=64, n_tx=4, n_pts=3000, seed=11):
    """One wall-simulator pool: layouts L00..L(n_eval-1) are EVALUATION layouts, the rest are SOURCE layouts (never evaluated)."""
    import rmbench_ext2 as r2
    names = r2.build_wall_units(cfg, n_eval + n_train, n_tx, seed=seed, n_pts=n_pts)
    lay = lambda n: int(re.search(r"\|L(\d+)\|", n).group(1))
    ev = [n for n in names if lay(n) < n_eval]; tr_by = {}
    for n in names:
        if lay(n) >= n_eval: tr_by.setdefault(lay(n), []).append(n)
    return ev, [tr_by[k] for k in sorted(tr_by)]

def pretrain_scale(train_layouts, Ks, cfg_src, pre_cfgs, out_dir, gnn_steps, np_steps, np_hidden=64):
    """For every K: pretrain every GNN in pre_cfgs and one neural process on the first K source layouts. Returns (gnn_paths[(K, tag)], np_paths[K])."""
    import rmbench_ext2 as r2
    gp, npp = {}, {}
    for K in Ks:
        names = [n for L in train_layouts[:K] for n in L]; print(f"=== K={K} source layouts, {len(names)} source units")
        plan = [dict(layout=f"K{K}", cross_id=f"K{K}", eval=[], cross=names, within=[])]
        paths = r2.pretrain_for_plan2(plan, cfg_src, pre_cfgs, os.path.join(out_dir, "gnn"), steps=gnn_steps)
        for tag in pre_cfgs:
            if (f"K{K}", "cross", tag) in paths: gp[(K, tag)] = paths[(f"K{K}", "cross", tag)]
        if HAVE_TORCH:
            p = os.path.join(out_dir, "np", f"np_K{K}.pt")
            if not os.path.exists(p): pretrain_np(make_np_episodes(names, cfg_src), np_steps, seed=K, out_path=p, hidden=np_hidden)
            npp[K] = p
    return gp, npp

def scale_jobs(Ks, gnn_paths, np_paths, pre_cfgs, finetune_steps=30, side=True):
    import rmbench_ext as rx, rmbench_ext2 as r2
    J = rx.base_jobs() + extra_jobs(side) + rx.graph_jobs(("gcn_knn", "gat_del")) + (r2.side_jobs() if side else [])
    for K in Ks:
        for tag, hp in pre_cfgs.items():
            p = gnn_paths.get((K, tag))
            if p:
                J.append(dict(name=f"xfer{K}_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": 0}))
                J.append(dict(name=f"xferft{K}_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": finetune_steps}))
        if K in np_paths: J.append(dict(name=f"np_prior{K}", method="np_prior", hp=dict(np_path=np_paths[K])))
    return J

_SCALE_RE = re.compile(r"^(xferft|xfer|np_prior)(\d+)(?:_(.+))?$")
def ref_for(name):
    m = _SCALE_RE.match(name); tag = (m.group(3) or "") if m else name
    return "rk_side" if ("side" in tag or name in ("rk_gcn", "rk_gcn_s0", "kgnn_side")) else "idw"

def scale_table(df, margin=MARGIN_PCT):
    """Gain of every method over its reference (idw, or rk_side for side-information models) per density, with the source-layout count K parsed from the name."""
    rows = []
    for name in sorted(df.name.unique()):
        if name in ("idw", "trend", "rk_side"): continue
        B = ref_for(name); w = paired_gain(df, name, B)
        if not len(w): continue
        m = _SCALE_RE.match(name); kind, K = (m.group(1), int(m.group(2))) if m else (name, 0); tag = (m.group(3) or "") if m else ""
        for d, g in w.groupby("density"):
            mean, lo, hi, nc = _cluster_mean_ci(g.gain_pct, g.parent)
            rows.append(dict(method=name, kind=kind, tag=tag, K=K, ref=B, density=d, n_clusters=nc, mean_gain_pct=mean, ci_lo=lo, ci_hi=hi, mean_gain_db=float(g.gain_db.mean()), verdict=verdict(lo, hi, margin)))
    return pd.DataFrame(rows)

def plot_scale(tab, fname=None, classical=("krig_global", "gp_ml", "rbf_spline", "kgnn", "coord_mlp"), margin=MARGIN_PCT):
    import matplotlib.pyplot as plt
    dens = sorted(tab.density.unique()); fig, ax = plt.subplots(1, len(dens), figsize=(4.2 * len(dens), 3.6), squeeze=False, sharey=True)
    for a, d in zip(ax[0], dens):
        t = tab[tab.density == d]; a.axhline(0, color="k", lw=.8); a.axhspan(-margin, margin, color="grey", alpha=.15)
        for (kind, tag), g in t[t.K > 0].groupby(["kind", "tag"]):
            g = g.sort_values("K"); a.errorbar(g.K, g.mean_gain_pct, yerr=[g.mean_gain_pct - g.ci_lo, g.ci_hi - g.mean_gain_pct], marker="o", capsize=2, label=f"{kind}_{tag}".strip("_"))
        for c in classical:
            r = t[(t.method == c)]
            if len(r): a.axhline(float(r.mean_gain_pct.iloc[0]), ls="--", lw=.8, label=c)
        a.set_xscale("log"); a.set_title(f"{d} labels / unit"); a.set_xlabel("source layouts used for pre-training (K)")
    ax[0, 0].set_ylabel("RMSE gain over idw / rk_side (%)"); ax[0, -1].legend(fontsize=6, loc="best"); fig.tight_layout()
    if fname: fig.savefig(fname, dpi=200)
    return fig

# =============================================================================== 9. run-integrity helpers (v2)
def check_run(path, max_fail_frac=0.01, strict=True):
    """Count failed jobs in a results JSONL. v1 silently analysed whatever survived; this raises instead."""
    if not os.path.exists(path): raise FileNotFoundError(path)
    n = nf = 0; by = {}; first = ""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            for r in json.loads(line):
                n += 1
                if r.get("status") != "ok":
                    nf += 1; by[r.get("name")] = by.get(r.get("name"), 0) + 1; first = first or str(r.get("err", ""))[-400:]
    frac = nf / max(n, 1)
    print(f"[check_run] {path}: {n} rows, {nf} failed ({frac:.2%})" + (f" by job {by}" if nf else ""))
    if strict and (n == 0 or frac > max_fail_frac):
        raise RuntimeError(f"{path}: {nf}/{n} failed jobs ({frac:.1%}) or no rows. Last error:\n{first}")
    return dict(path=path, rows=n, failed=nf, frac=frac)

def sanity_scale(df, rtol=1e-7):
    """Pre-trained priors must (a) change the prediction, (b) differ across K and across architectures. Returns a list of problems (empty = ok).
    v1 smoke run: xfer2_* == xfer8_* == xfer*_gcn_knn to the last digit, which means the prior was not used."""
    ok = df[df.status == "ok"]; keys = [c for c in ("unit", "density", "rep") if c in ok]
    w = ok.pivot_table(index=keys, columns="name", values="rmse"); pri = sorted(c for c in w.columns if _SCALE_RE.match(c)); probs = []
    same = lambda a, b: len(w[[a, b]].dropna()) > 0 and np.allclose(w[[a, b]].dropna()[a], w[[a, b]].dropna()[b], rtol=rtol, atol=1e-9)
    if "trend" in w:
        probs += [f"{c} is identical to 'trend' (the prior adds nothing / is not loaded)" for c in pri if same(c, "trend")]
    for i, a in enumerate(pri):
        for b in pri[i + 1:]:
            ma, mb = _SCALE_RE.match(a), _SCALE_RE.match(b)
            if ma.group(1) == mb.group(1) and (ma.group(2), ma.group(3)) != (mb.group(2), mb.group(3)) and same(a, b):
                probs.append(f"{a} and {b} are identical (K or architecture has no effect)")
    return probs
