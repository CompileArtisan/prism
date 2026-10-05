import sys, time, warnings, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
from pykrige.ok import OrdinaryKriging
from kgnn_radiomap import kgnn_predict, fit_trend, trend_eval

# ---------------------------------------------------------------- synthetic indoor field
REGIMES = {
    # name: (walls?, wall_loss_mean, shadow_sigma, shadow_len, nugget_sigma)
    "A_open_smooth":   (False, 0.0, 3.0, 4.0, 0.5),    # stationary: kriging's home turf
    "B_walls":         (True, 8.0, 2.0, 2.5, 0.5),     # wall steps + mild shadowing
    "C_walls_noisy":   (True, 8.0, 2.0, 2.5, 2.5),     # same, but white-noise floor dominates
}

def _segments(rng, W, H, walls):
    if not walls: return np.zeros((0, 4)), np.zeros(0)
    segs, loss = [], []
    def add_line(x0, y0, x1, y1, L):
        # one wall with a door gap
        t = rng.uniform(0.2, 0.8); gap = 1.4 / np.hypot(x1 - x0, y1 - y0)
        a, b = max(t - gap / 2, 0), min(t + gap / 2, 1)
        for (s0, s1) in ((0, a), (b, 1)):
            segs.append([x0 + (x1 - x0) * s0, y0 + (y1 - y0) * s0, x0 + (x1 - x0) * s1, y0 + (y1 - y0) * s1]); loss.append(L)
    for xv in (W * rng.uniform(0.28, 0.38), W * rng.uniform(0.62, 0.72)):
        add_line(xv, 0, xv, H, rng.normal(8.0, 1.5))
    yh = H * rng.uniform(0.4, 0.6); add_line(0, yh, W, yh, rng.normal(8.0, 1.5))
    for _ in range(3):                                       # a few short partitions
        x0, y0 = rng.uniform(1, W - 1), rng.uniform(1, H - 1)
        ang = rng.choice([0, np.pi / 2]); ln = rng.uniform(3, 7)
        segs.append([x0, y0, x0 + ln * np.cos(ang), y0 + ln * np.sin(ang)]); loss.append(rng.normal(6.0, 1.5))
    return np.array(segs), np.array(loss)

def _crossings(tx, pts, segs, loss):
    if len(segs) == 0: return np.zeros(len(pts))
    p, r = tx[None, :], pts - tx[None, :]                   # ray p + t r
    q, s = segs[None, :, :2], (segs[:, 2:] - segs[:, :2])[None]
    rxs = r[:, None, 0] * s[..., 1] - r[:, None, 1] * s[..., 0]
    qp = q - p[:, None, :]
    t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / np.where(np.abs(rxs) < 1e-12, np.inf, rxs)
    u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / np.where(np.abs(rxs) < 1e-12, np.inf, rxs)
    hit = (t > 0) & (t < 1) & (u > 0) & (u < 1)
    return (hit * loss[None, :]).sum(1)

def make_field(regime, seed, N=3000, W=30.0, H=20.0):
    walls, _, sh_sig, sh_len, nug = REGIMES[regime]
    rng = np.random.default_rng(10_000 + seed)
    pos = np.column_stack([rng.uniform(0, W, N), rng.uniform(0, H, N)])
    tx = np.array([rng.uniform(3, W - 3), rng.uniform(3, H - 3)])
    segs, loss = _segments(rng, W, H, walls)
    d3 = np.sqrt(((pos - tx) ** 2).sum(1) + 1.5 ** 2)
    n_exp = rng.uniform(2.0, 3.0)
    base = -40.0 - 10 * n_exp * np.log10(d3) - _crossings(tx, pos, segs, loss)
    M = 200; om = rng.normal(0, 1 / sh_len, (M, 2)); ph = rng.uniform(0, 2 * np.pi, M)
    shadow = sh_sig * np.sqrt(2 / M) * np.cos(pos @ om.T + ph).sum(1)
    y = base + shadow + rng.normal(0, nug, N)
    return pos, d3, tx, y

# ---------------------------------------------------------------- classical baselines (tuned only on the labeled set)
def idw(ptr, vtr, pte, power, nn):
    k = len(ptr) if nn is None else min(nn, len(ptr))
    d, i = cKDTree(ptr).query(pte, k=k)
    if k == 1: d, i = d[:, None], i[:, None]
    w = 1.0 / (d + 1e-6) ** power
    return (w * vtr[i]).sum(1) / w.sum(1)

def idw_loo(ptr, vtr, power, nn):
    n = len(ptr); k = n - 1 if nn is None else min(nn, n - 1)
    d, i = cKDTree(ptr).query(ptr, k=k + 1)
    d, i = d[:, 1:], i[:, 1:]
    w = 1.0 / (d + 1e-6) ** power
    return (w * vtr[i]).sum(1) / w.sum(1)

IDW_GRID = [(p, nn) for p in (1.0, 1.5, 2.0, 3.0) for nn in (8, 16, None)]
def tune_idw(ptr, vtr):
    best = min(IDW_GRID, key=lambda g: np.sqrt(np.mean((idw_loo(ptr, vtr, *g) - vtr) ** 2)))
    return best

def krig(ptr, vtr, pte, model, nlags):
    ok = OrdinaryKriging(ptr[:, 0], ptr[:, 1], vtr, variogram_model=model, nlags=nlags, enable_plotting=False)
    out = np.empty(len(pte))
    for s in range(0, len(pte), 2000):
        z, _ = ok.execute("points", pte[s:s + 2000, 0], pte[s:s + 2000, 1]); out[s:s + 2000] = np.asarray(z)
    return out

KRIG_GRID = [(m, nl) for m in ("spherical", "exponential", "gaussian") for nl in (6, 12)]
def tune_krig(ptr, vtr, rng, folds=5):
    n = len(ptr); folds = min(folds, n); idx = np.array_split(rng.permutation(n), folds); best, bs = None, np.inf
    for g in KRIG_GRID:
        sse = 0.0; ok = True
        for f in idx:
            tr = np.setdiff1d(np.arange(n), f)
            try: pr = krig(ptr[tr], vtr[tr], ptr[f], *g)
            except Exception: ok = False; break
            sse += np.sum((pr - vtr[f]) ** 2)
        if ok and sse < bs: bs, best = sse, g
    return best or ("spherical", 6)

def run_classical(pos, d3, y, tr, te, seed):
    out = {}
    ptr, pte = pos[tr], pos[te]
    g = tune_idw(ptr, y[tr]); out["IDW(raw)"] = idw(ptr, y[tr], pte, *g)
    a, n = fit_trend(d3[tr], y[tr]); res = y[tr] - trend_eval(d3[tr], a, n); tt = trend_eval(d3[te], a, n)
    g = tune_idw(ptr, res); out["IDW(detr)"] = idw(ptr, res, pte, *g) + tt
    rng = np.random.default_rng(seed)
    try:
        g = tune_krig(ptr, y[tr], rng); out["Krig(raw)"] = krig(ptr, y[tr], pte, *g)
        g = tune_krig(ptr, res, rng); out["Krig(detr)"] = krig(ptr, res, pte, *g) + tt
    except Exception as e:
        print("krig fail", e)
    return out

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

if __name__ == "__main__":
    regimes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(REGIMES)
    dens = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [20, 60, 150]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rows = []
    for rg in regimes:
        for n in dens:
            for s in range(seeds):
                pos, d3, tx, y = make_field(rg, s)
                rng = np.random.default_rng(s); perm = rng.permutation(len(y)); tr, te = perm[:n], perm[n:]
                cl = run_classical(pos, d3, y, tr, te, s)
                t0 = time.time()
                lk = kgnn_predict(pos, d3, y, tx, tr, te, seed=s, corr=False, Kmax=24)
                cl["KGNN"] = lk
                for m, p in cl.items(): rows.append(dict(regime=rg, n=n, seed=s, method=m, rmse=rmse(p, y[te])))
            print(rg, n, "done", f"{time.time()-t0:.1f}s/lk-fit", flush=True)
    df = pd.DataFrame(rows); df.to_csv("/home/claude/proxy_v1.csv", index=False)
    print(df.pivot_table(index=["regime", "n"], columns="method", values="rmse").round(3))
