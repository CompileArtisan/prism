"""rmbench_rms.py -- RadioMapSeer study: do cross-map-trained graph models beat
side-information regression-kriging (`rk_side`)?   (plan: PLAN_radiomapseer_DRAFT.md)

Standalone: imports NOTHING from the other rmbench_* modules, so it cannot break them.
Written by the cell `%%writefile rmbench_rms.py` of rmbench_rms.ipynb.  Adds only.

Layout of this file
  1 config / utils            5 labels, trend, metrics        9 training + tuning
  2 data (real + synthetic)   6 classical arms                10 executors (resumable JSONL)
  3 geometry + features       7 graph builder, GATv2, nets    11 statistics, families, verdicts
  4 context cache             8 learned-arm inference         12 pre-registration, checks
"""
import os, sys, json, math, time, hashlib, threading, queue, itertools, warnings, importlib.util
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as TF
except Exception:                                   # classical-only use is still possible
    torch = None
    nn = None

# =====================================================================================
# 1  CONFIG / UTILS
# =====================================================================================
ARMS_CLASSICAL = ["idw", "krig_global", "rk_side", "rk_gbm"]
ARMS_LEARNED = ["gnn_kim", "gnn_kim_oldedge", "rgnn_xmap", "radiounet", "cnn_feat"]
FEATURE_NAMES = ["log_dist_tx", "los", "blocked_len", "n_crossings", "cover_30m", "dist_nearest_bld"]

DEFAULT_CFG = dict(
    # ---- data ([VERIFY] items are checked by the notebook, section 3) ----
    synthetic=False, data_root="RadioMapSeer", sim="DPM", H=256,
    floor_mode="clip",            # [CHOOSE] "clip" = RadioUNet convention (pixels below noise floor set to the floor); "none"
    noise_floor_pix=51.0,         # 0.2 * 255, RadioUNet loaders.py `thresh=0.2`
    db_lo=-147.0, db_hi=-47.84,   # [VERIFY] pixel 0..255 -> dB (NOT in the RadioUNet code; only rescales dB figures, not gains/tests)
    split_source="loader",        # "loader" = RadioUNet ordering (seed 42, 501/100/99); "synthetic" for smoke
    n_tx_total=80,
    # ---- task ----
    salt="rms1", alphas=[0.005, 0.02, 0.05], train_alpha_range=[0.003, 0.06],
    n_tx_train=8, n_tx_test=4, n_draws=5, n_seeds=3, min_labels=8,
    block=64,                     # blocked split: contiguous held-out square (px)
    # ---- classical hyper-parameters (defaults = trial 0; tuned on val maps by the notebook) ----
    classical_hp=dict(idw=dict(p=2.0, K=16), krig_global=dict(K=16, global_max_n=1500),
                      rk_side=dict(ridge=1.0, p=2.0, K=16), rk_gbm=dict(max_iter=100, lr=0.1, p=2.0, K=16)),
    # ---- graph / networks ----
    tile=64, stride=64, gnn_batch=4, gnn_infer_tiles=4, res_scale=10.0,
    gnn_hp=dict(lr=1e-3, wd=1e-5, width=64, depth=3, r2=18.0),
    radiounet_impl="public",      # "public" = RonLevie/RadioUNet RadioWNet(phase='firstU'); "compact" = small U-Net (smoke)
    radiounet_repo="RadioUNet", radiounet_batch=4,
    radiounet_hp=dict(lr=1e-4, wd=0.0, base=32, depth=4),
    # ---- budgets [CHOOSE] ----
    tune_n_cfg=30, tune_steps=1500, final_steps=8000, ckpt_every=500, ckpt_val_maps=12,
    tune_val_maps=100, n_workers_data=2, device="auto",
    # ---- paths / parallelism ----
    out_dir="results_rms", n_jobs=1,
)


def make_cfg(**kw):
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def smoke_cfg(out_dir="results_rms_smoke", **kw):
    """Tiny profile on synthetic maps. Tests plumbing only; says nothing about science."""
    base = dict(synthetic=True, H=96, split_source="synthetic", n_tx_total=6, n_tx_train=2, n_tx_test=2,
                n_draws=2, n_seeds=1, tile=32, stride=32, gnn_batch=4, gnn_infer_tiles=8,
                tune_n_cfg=2, tune_steps=25, final_steps=40, ckpt_every=20, ckpt_val_maps=3, tune_val_maps=4,
                n_workers_data=1, radiounet_impl="compact", out_dir=out_dir, min_labels=8,
                n_synth_maps=26, split_counts=[12, 6, 8], block=32,
                gnn_hp=dict(width=16, depth=2, r2=8.0), radiounet_hp=dict(base=8, depth=3))
    base.update(kw)
    return make_cfg(**base)


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def get_device(cfg):
    if torch is None:
        return None
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =====================================================================================
# 2  DATA
# =====================================================================================
def loader_split():
    """RadioUNet loaders.py: maps_inds = arange(0,700,int16); seed 42; shuffle; train=[0:501], val=[501:601],
    test=[601:700] (the loader's ind1/ind2 are inclusive); dataset id = maps_inds + 1."""
    inds = np.arange(0, 700, 1, dtype=np.int16)
    np.random.RandomState(42).shuffle(inds)              # == np.random.seed(42); np.random.shuffle(inds)
    ids = (inds.astype(int) + 1)
    return dict(train=ids[0:501].tolist(), val=ids[501:601].tolist(), test=ids[601:700].tolist())


class RMSData:
    """Real RadioMapSeer. Files (checked against RadioUNet loaders.py):
       png/buildings_complete/{id}.png, png/antennas/{id}_{tx}.png, gain/{DPM|IRT2|IRT4}/{id}_{tx}.png"""
    def __init__(self, cfg):
        from PIL import Image
        self.Image = Image
        self.cfg = cfg
        self.root = Path(cfg["data_root"])
        self._bld = {}

    def _png(self, p):
        return np.asarray(self.Image.open(p).convert("L"))

    def bld(self, m):
        if m not in self._bld:
            if len(self._bld) > 64:
                self._bld.clear()
            self._bld[m] = self._png(self.root / "png" / "buildings_complete" / f"{m}.png") > 127
        return self._bld[m]

    def tx_xy(self, m, t):
        a = self._png(self.root / "png" / "antennas" / f"{m}_{t}.png")
        iy, ix = np.unravel_index(int(np.argmax(a)), a.shape)
        return int(iy), int(ix)

    def exists(self, m, t, sim):
        return (self.root / "gain" / sim / f"{m}_{t}.png").exists()

    def y_db(self, m, t, sim=None):
        c = self.cfg
        sim = sim or c["sim"]
        pix = self._png(self.root / "gain" / sim / f"{m}_{t}.png").astype(np.float32)
        if c["floor_mode"] == "clip":
            pix = np.maximum(pix, c["noise_floor_pix"])
        return (c["db_lo"] + pix / 255.0 * (c["db_hi"] - c["db_lo"])).astype(np.float32)


class SynthData:
    """Random rectangles + log-distance law + per-crossing wall loss + smooth shadowing. Plumbing tests only."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.H = cfg["H"]
        self.n = cfg.get("n_synth_maps", 24)

    def bld(self, m):
        H = self.H
        rng = np.random.default_rng(stable_seed("synth_bld", m))
        b = np.zeros((H, H), bool)
        for _ in range(int(rng.integers(8, 15))):
            h, w = rng.integers(max(4, H // 16), max(6, H // 5), size=2)
            y0, x0 = rng.integers(0, H - h), rng.integers(0, H - w)
            b[y0:y0 + h, x0:x0 + w] = True
        return b

    def tx_xy(self, m, t):
        rng = np.random.default_rng(stable_seed("synth_tx", m, t))
        b = self.bld(m)
        free = np.flatnonzero(~b.ravel())
        k = int(rng.choice(free))
        return int(k // self.H), int(k % self.H)

    def exists(self, m, t, sim):
        return True

    def y_db(self, m, t, sim=None):
        c = self.cfg
        b = self.bld(m)
        txy = self.tx_xy(m, t)
        los, blocked, cross = tx_geometry(b, txy)
        yy, xx = np.mgrid[0:self.H, 0:self.H]
        d = np.sqrt((yy - txy[0]) ** 2 + (xx - txy[1]) ** 2)
        rng = np.random.default_rng(stable_seed("synth_sh", m, t))
        sh = ndimage.gaussian_filter(rng.normal(size=b.shape), 4.0)
        sh = sh / (sh.std() + 1e-9) * 2.0
        y = -60.0 - 28.0 * np.log10(1.0 + d) - 9.0 * cross - 0.15 * blocked + sh
        return np.clip(y, c["db_lo"], c["db_hi"]).astype(np.float32)


def make_data(cfg):
    return SynthData(cfg) if cfg["synthetic"] else RMSData(cfg)


def get_split(cfg):
    if cfg["split_source"] == "loader" and not cfg["synthetic"]:
        return loader_split()
    n = cfg.get("n_synth_maps", 24)
    a, b, c = cfg.get("split_counts", [12, 6, 6])
    ids = np.arange(1, n + 1)
    np.random.RandomState(42).shuffle(ids)
    return dict(train=ids[:a].tolist(), val=ids[a:a + b].tolist(), test=ids[a + b:a + b + c].tolist())


def pick_tx(cfg, m, n, role):
    rng = np.random.default_rng(stable_seed("pick_tx", cfg["salt"], role, m))
    return sorted(int(v) for v in rng.choice(cfg["n_tx_total"], size=min(n, cfg["n_tx_total"]), replace=False))


# =====================================================================================
# 3  GEOMETRY AND LABEL-FREE FEATURES   (inputs: building raster + Tx position ONLY)
# =====================================================================================
def tx_geometry(bld, txy, n_samp=None):
    """Segment Tx -> every pixel, sampled at ~<=1 px spacing. Returns (los bool, blocked_len px, n_crossings)."""
    H, W = bld.shape
    n_samp = n_samp or int(1.5 * max(H, W))
    ty, tx = txy
    yy, xx = np.mgrid[0:H, 0:W]
    dy = (yy - ty).astype(np.float32)
    dx = (xx - tx).astype(np.float32)
    dist = np.sqrt(dy * dy + dx * dx)
    cnt = np.zeros((H, W), np.int32)
    cross = np.zeros((H, W), np.uint8)
    prev = np.zeros((H, W), bool)
    for s in range(1, n_samp + 1):
        t = s / (n_samp + 1.0)
        py = np.clip(np.floor(ty + t * dy + 0.5).astype(np.int32), 0, H - 1)
        px = np.clip(np.floor(tx + t * dx + 0.5).astype(np.int32), 0, W - 1)
        cur = bld[py, px]
        cnt += cur
        cross += (cur & ~prev)
        prev = cur
    blocked = cnt.astype(np.float32) / n_samp * dist
    los = (cnt == 0) & ~bld
    return los, blocked.astype(np.float32), cross


def map_geometry(bld):
    """Per-map (Tx-independent) features: building cover within 30 px (box), distance to nearest building."""
    cover = ndimage.uniform_filter(bld.astype(np.float32), size=61, mode="constant", cval=0.0)
    if bld.any():
        dist = ndimage.distance_transform_edt(~bld).astype(np.float32)
    else:
        dist = np.full(bld.shape, 1e3, np.float32)
    return cover.astype(np.float32), dist


def feature_stack(txy, los, blocked, cross, cover, dist):
    """6 normalised label-free features (fixed constants; no statistics from any data)."""
    H, W = los.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - txy[0]) ** 2 + (xx - txy[1]) ** 2).astype(np.float32)
    F = np.stack([np.log10(1.0 + d) / 2.5,
                  los.astype(np.float32),
                  np.log1p(blocked) / np.log(100.0),
                  np.log1p(cross.astype(np.float32)) / np.log(6.0),
                  cover,
                  np.log1p(dist) / np.log(100.0)]).astype(np.float32)
    return F, d


# =====================================================================================
# 4  CONTEXT (one map x one transmitter), disk + LRU cache
# =====================================================================================
_STATE = dict(cfg=None, data=None, lru=OrderedDict(), device=None)


def init(cfg):
    """Call once per process (the notebook calls it; workers call it through the pool initializer)."""
    _STATE["cfg"] = cfg
    _STATE["data"] = make_data(cfg)
    _STATE["lru"] = OrderedDict()
    Path(cache_dir(cfg)).mkdir(parents=True, exist_ok=True)
    if torch is not None:
        _STATE["device"] = get_device(cfg)


def cache_dir(cfg):
    return str(Path(cfg["out_dir"]) / ("cache_synth" if cfg["synthetic"] else "cache_real"))


class Ctx:
    pass


def get_ctx(m, t, sim=None, cfg=None, with_y=True):
    cfg = cfg or _STATE["cfg"]
    sim = sim or cfg["sim"]
    key = (m, t, sim if with_y else None)
    lru = _STATE["lru"]
    if key in lru:
        lru.move_to_end(key)
        return lru[key]
    data = _STATE["data"]
    cd = Path(cache_dir(cfg))
    bld = data.bld(m)
    pm = cd / f"map_{m}.npz"
    if pm.exists():
        z = np.load(pm); cover, dist = z["cover"].astype(np.float32), z["dist"].astype(np.float32)
    else:
        cover, dist = map_geometry(bld)
        np.savez(pm, cover=cover.astype(np.float16), dist=np.minimum(dist, 6e4).astype(np.float16))
    pt = cd / f"tx_{m}_{t}.npz"
    if pt.exists():
        z = np.load(pt); los, blocked, cross, txy = z["los"], z["blocked"].astype(np.float32), z["cross"], tuple(int(v) for v in z["txy"])
    else:
        txy = data.tx_xy(m, t)
        los, blocked, cross = tx_geometry(bld, txy)
        np.savez(pt, los=los, blocked=blocked.astype(np.float16), cross=cross, txy=np.array(txy))
    c = Ctx()
    c.m, c.t, c.sim, c.bld, c.txy = m, t, sim, bld, txy
    c.H, c.W = bld.shape
    c.out = ~bld
    c.los, c.dist_bld = los, dist
    c.F, c.d = feature_stack(txy, los, blocked, cross, cover, dist)
    c.logd = np.log10(1.0 + c.d).astype(np.float32)
    c.y = data.y_db(m, t, sim) if with_y else None
    c.tg_cache = {}
    lru[key] = c
    if len(lru) > 48:
        lru.popitem(last=False)
    return c


# =====================================================================================
# 5  LABELS, SHARED TREND, METRICS
# =====================================================================================
class Lab:
    pass


def fit_trend(logd_lab, y_lab, fallback=None):
    n = len(y_lab)
    if n >= 3 and np.ptp(logd_lab) > 1e-6:
        A = np.stack([np.ones(n), logd_lab], 1)
        coef, *_ = np.linalg.lstsq(A, y_lab.astype(np.float64), rcond=None)
        return float(coef[0]), float(coef[1])
    if fallback is not None:
        return fallback
    return float(np.mean(y_lab)) if n else 0.0, 0.0


def make_lab(ctx, alpha, rng, split="random", block=64, label_mode="normal", min_labels=8, global_trend=None):
    """Draw labels among outdoor pixels. `split='blocked'` holds out one contiguous square as the test region.
       label_mode: 'normal' | 'shuffle' (label values permuted across label positions) | 'zero' (no labels)."""
    H, W = ctx.H, ctx.W
    out = ctx.out.copy()
    test_region = np.ones((H, W), bool)
    if split == "blocked":
        y0 = int(rng.integers(0, H - block + 1)); x0 = int(rng.integers(0, W - block + 1))
        test_region = np.zeros((H, W), bool); test_region[y0:y0 + block, x0:x0 + block] = True
        cand = np.flatnonzero((out & ~test_region).ravel())
        test_pool = np.flatnonzero((out & test_region).ravel())
    else:
        cand = np.flatnonzero(out.ravel())
        test_pool = cand
    lab = Lab()
    n_out = int(out.sum())
    if label_mode == "zero" or alpha <= 0:
        n = 0
    else:
        n = int(min(len(cand), max(min_labels, round(alpha * n_out))))
    idx = np.sort(rng.choice(cand, size=n, replace=False)) if n else np.zeros(0, np.int64)
    ylab = ctx.y.ravel()[idx].astype(np.float64) if n else np.zeros(0)
    if label_mode == "shuffle" and n:
        ylab = ylab[rng.permutation(n)]
    lab.idx, lab.ylab, lab.n = idx, ylab, n
    lab.logd = ctx.logd.ravel()[idx].astype(np.float64)
    if n:
        lab.coef = fit_trend(lab.logd, ylab, global_trend)
    else:
        lab.coef = global_trend if global_trend is not None else (float(np.mean(ctx.y[ctx.out])), 0.0)
    lab.rlab = ylab - (lab.coef[0] + lab.coef[1] * lab.logd)
    lab.test_idx = np.setdiff1d(test_pool, idx, assume_unique=False)
    lab.thr = float(np.percentile(ylab, 25)) if n else float(np.percentile(ctx.y[ctx.out], 25))
    lab.rc = None
    return lab


def coords_of(idx, W):
    return np.stack([idx // W, idx % W], 1).astype(np.float64)


def trend_map(ctx, lab):
    return (lab.coef[0] + lab.coef[1] * ctx.logd).astype(np.float32)


def label_draw_rng(cfg, m, t, ai, draw, role="eval"):
    return np.random.default_rng(stable_seed("labels", cfg["salt"], role, m, t, ai, draw))


def compute_metrics(ctx, lab, yhat):
    ti = lab.test_idx
    y = ctx.y.ravel()[ti]
    yh = yhat.ravel()[ti].astype(np.float64)
    e = yh - y
    res = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), n_test=int(len(ti)))
    los = ctx.los.ravel()[ti]
    res["rmse_los"] = float(np.sqrt(np.mean(e[los] ** 2))) if los.sum() >= 8 else np.nan
    res["rmse_nlos"] = float(np.sqrt(np.mean(e[~los] ** 2))) if (~los).sum() >= 8 else np.nan
    near = ctx.dist_bld.ravel()[ti] <= 10
    res["rmse_edge10"] = float(np.sqrt(np.mean(e[near] ** 2))) if near.sum() >= 8 else np.nan
    tc, pc = y >= lab.thr, yh >= lab.thr
    if tc.sum() >= 8 and (~tc).sum() >= 8:
        res["bacc_q25"] = float(0.5 * (np.mean(pc[tc]) + np.mean(~pc[~tc])))
    else:
        res["bacc_q25"] = np.nan
    return res


# =====================================================================================
# 6  CLASSICAL ARMS  (fit per map on that map's labels only)
# =====================================================================================
def idw_predict(cl, rl, cq, p, K):
    n = len(rl)
    if n == 0:
        return np.zeros(len(cq))
    tree = cKDTree(cl)
    k = min(K, n)
    d, i = tree.query(cq, k=k)
    if k == 1:
        d, i = d[:, None], i[:, None]
    w = 1.0 / np.maximum(d, 1e-3) ** p
    return (w * rl[i]).sum(1) / w.sum(1)


def fit_variogram(cl, rl, rng, max_pts=1200):
    n = len(rl)
    if n > max_pts:
        s = rng.choice(n, max_pts, replace=False); cl, rl = cl[s], rl[s]
        n = max_pts
    var = float(np.var(rl)) + 1e-6
    default = dict(nug=0.2 * var, sill=0.8 * var, rng=30.0)
    if n < 12:
        return default
    dm = np.sqrt(((cl[:, None, :] - cl[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    h = dm[iu]; g = 0.5 * (rl[:, None] - rl[None, :])[iu] ** 2
    hmax = np.percentile(h, 50)
    bins = np.linspace(0, hmax, 16)
    bi = np.digitize(h, bins) - 1
    hs, gs, ws = [], [], []
    for b in range(15):
        m = bi == b
        if m.sum() >= 20:
            hs.append(h[m].mean()); gs.append(g[m].mean()); ws.append(m.sum())
    if len(hs) < 4:
        return default
    f = lambda hh, nug, sill, r: nug + sill * (1 - np.exp(-hh / r))
    try:
        po, _ = curve_fit(f, np.array(hs), np.array(gs), p0=[0.2 * var, 0.8 * var, 20.0],
                          bounds=([0, 1e-6, 1.0], [var * 2, var * 5, 400.0]), sigma=1.0 / np.sqrt(ws), maxfev=2000)
        return dict(nug=float(po[0]), sill=float(po[1]), rng=float(po[2]))
    except Exception:
        return default


def krig_predict(cl, rl, cq, vg, K, global_max_n, chunk=4096):
    """Ordinary kriging, exponential covariance + nugget. Global (all labels) if n<=global_max_n, else K nearest."""
    n = len(rl)
    if n == 0:
        return np.zeros(len(cq))
    nug, sill, rg = vg["nug"], vg["sill"], vg["rng"]
    cov = lambda h: sill * np.exp(-h / rg)
    out = np.zeros(len(cq))
    if n <= global_max_n:
        D = np.sqrt(((cl[:, None, :] - cl[None, :, :]) ** 2).sum(-1))
        A = np.zeros((n + 1, n + 1)); A[:n, :n] = cov(D) + np.eye(n) * (nug + 1e-9); A[:n, n] = 1; A[n, :n] = 1
        Ainv = np.linalg.pinv(A)
        for s in range(0, len(cq), chunk):
            q = cq[s:s + chunk]
            c0 = np.ones((len(q), n + 1)); c0[:, :n] = cov(np.sqrt(((q[:, None, :] - cl[None, :, :]) ** 2).sum(-1)))
            out[s:s + chunk] = (c0 @ Ainv)[:, :n] @ rl
        return out
    tree = cKDTree(cl)
    k = min(K, n)
    for s in range(0, len(cq), chunk):
        q = cq[s:s + chunk]
        d, i = tree.query(q, k=k)
        P = cl[i]
        D = np.sqrt(((P[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1))
        A = np.zeros((len(q), k + 1, k + 1)); A[:, :k, :k] = cov(D) + np.eye(k)[None] * (nug + 1e-9)
        A[:, :k, k] = 1; A[:, k, :k] = 1
        b = np.ones((len(q), k + 1)); b[:, :k] = cov(d)
        w = np.linalg.solve(A, b[:, :, None])[:, :k, 0]
        out[s:s + chunk] = (w * rl[i]).sum(1)
    return out


def ridge_fit(X, r, alpha):
    mu, rm = X.mean(0), r.mean()
    Xc = X - mu
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (r - rm))
    return w, rm - mu @ w


def rk_side_residual(ctx, lab, qidx, hp, mean_model="ridge"):
    """Residual-from-trend prediction at flat pixel indices qidx:  ridge(features) + IDW(leftover residual)."""
    W = ctx.W
    Fl = ctx.F.reshape(6, -1)
    Xl = Fl[:, lab.idx].T.astype(np.float64)
    Xq = Fl[:, qidx].T.astype(np.float64)
    if lab.n < 3:
        return np.zeros(len(qidx))
    if mean_model == "ridge":
        w, b = ridge_fit(Xl, lab.rlab, hp["ridge"])
        ml, mq = Xl @ w + b, Xq @ w + b
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        msl = int(min(20, max(2, lab.n // 10)))
        g = HistGradientBoostingRegressor(max_iter=int(hp["max_iter"]), learning_rate=hp["lr"], max_depth=3,
                                          min_samples_leaf=msl, early_stopping=False, random_state=0)
        g.fit(Xl, lab.rlab)
        ml, mq = g.predict(Xl), g.predict(Xq)
    left = lab.rlab - ml
    return mq + idw_predict(coords_of(lab.idx, W), left, coords_of(qidx, W), hp["p"], hp["K"])


def predict_classical(arm, ctx, lab, cfg, rng_vg=None):
    """Full-map prediction (dB) = shared trend + arm's residual prediction (evaluated on outdoor pixels)."""
    hp = cfg["classical_hp"][arm]
    qidx = np.flatnonzero(ctx.out.ravel())
    W = ctx.W
    if arm == "idw":
        r = idw_predict(coords_of(lab.idx, W), lab.rlab, coords_of(qidx, W), hp["p"], hp["K"])
    elif arm == "krig_global":
        rng = rng_vg or np.random.default_rng(stable_seed("vg", ctx.m, ctx.t, lab.n))
        vg = fit_variogram(coords_of(lab.idx, W), lab.rlab, rng)
        r = krig_predict(coords_of(lab.idx, W), lab.rlab, coords_of(qidx, W), vg, hp["K"], hp["global_max_n"])
    elif arm == "rk_side":
        r = rk_side_residual(ctx, lab, qidx, hp, "ridge")
    elif arm == "rk_gbm":
        r = rk_side_residual(ctx, lab, qidx, hp, "gbm")
    else:
        raise ValueError(arm)
    yhat = np.zeros(ctx.H * ctx.W, np.float32)
    yhat[qidx] = (trend_map(ctx, lab).ravel()[qidx] + r)
    return yhat


def rk_base_map(ctx, lab, cfg):
    """rk_side residual prediction for ALL outdoor pixels (0 at buildings) -- the base of rgnn_xmap."""
    qidx = np.flatnonzero(ctx.out.ravel())
    r = rk_side_residual(ctx, lab, qidx, cfg["classical_hp"]["rk_side"], "ridge")
    base = np.zeros(ctx.H * ctx.W, np.float32)
    base[qidx] = r
    return base


# =====================================================================================
# 7  GRAPH BUILDER (Kim edge rule), GATv2, NETWORKS
# =====================================================================================
def edge_offsets(r2):
    R = int(math.floor(math.sqrt(r2)))
    return [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if 0 < dy * dy + dx * dx <= r2]


def line_cells(dy, dx):
    """Intermediate cells of the discrete segment (0,0)->(dy,dx); the edge is blocked if any is a building."""
    n = max(abs(dy), abs(dx))
    return [(int(math.floor(s * dy / n + 0.5)), int(math.floor(s * dx / n + 0.5))) for s in range(1, n)]


def build_graph(bld, los, r2, same_los, sigma=None):
    """Disjoint union of B tiles. Nodes = outdoor pixels only. Edge (neighbour -> node) iff
       dist <= sqrt(r2)  AND  no building on the segment  AND  (same_los=True) both nodes share the LOS/NLOS state.
       Self loops added. Edge attribute = Gaussian distance weight."""
    dev = bld.device
    B, h, w = bld.shape
    R = int(math.ceil(math.sqrt(r2)))
    bp = TF.pad(bld.to(torch.uint8), (R, R, R, R), value=1).bool()          # outside the tile counts as building
    lp = TF.pad(los.to(torch.uint8), (R, R, R, R), value=0).bool()
    out = ~bld
    node_flat = out.reshape(-1).nonzero().squeeze(1)
    nid = torch.full((B * h * w,), -1, dtype=torch.long, device=dev)
    nid[node_flat] = torch.arange(len(node_flat), device=dev)
    sigma = sigma or math.sqrt(r2) / 2.0
    base_idx = torch.arange(B * h * w, device=dev).view(B, h, w)
    sh = lambda a, oy, ox: a[:, R + oy:R + oy + h, R + ox:R + ox + w]
    srcs, dsts, attrs = [], [], []
    for dy, dx in edge_offsets(r2):
        valid = out & ~sh(bp, dy, dx)
        for cy, cx in line_cells(dy, dx):
            valid = valid & ~sh(bp, cy, cx)
        if same_los:
            valid = valid & (sh(lp, dy, dx) == los)
        dst_flat = base_idx[valid]
        if len(dst_flat) == 0:
            continue
        src_flat = dst_flat + dy * w + dx
        srcs.append(nid[src_flat]); dsts.append(nid[dst_flat])
        attrs.append(torch.full((len(dst_flat),), math.exp(-(dy * dy + dx * dx) / (2 * sigma ** 2)), device=dev))
    nn_ = len(node_flat)
    ar = torch.arange(nn_, device=dev)
    srcs.append(ar); dsts.append(ar); attrs.append(torch.ones(nn_, device=dev))
    ei = torch.stack([torch.cat(srcs), torch.cat(dsts)])
    return dict(node_flat=node_flat, ei=ei, ea=torch.cat(attrs).unsqueeze(1), n_nodes=nn_)


if nn is not None:
    class GATv2(nn.Module):
        def __init__(s, cin, c, heads, edge_dim=1):
            super().__init__()
            s.h, s.c = heads, c
            s.Wl = nn.Linear(cin, heads * c); s.Wr = nn.Linear(cin, heads * c)
            s.We = nn.Linear(edge_dim, heads * c, bias=False)
            s.att = nn.Parameter(torch.empty(heads, c)); nn.init.xavier_uniform_(s.att)
            s.bias = nn.Parameter(torch.zeros(heads * c))

        def forward(s, x, ei, ea):
            N = x.shape[0]
            src, dst = ei
            xl = s.Wl(x).view(N, s.h, s.c); xr = s.Wr(x).view(N, s.h, s.c)
            m = xl[src] + xr[dst] + s.We(ea).view(-1, s.h, s.c)
            e = (TF.leaky_relu(m, 0.2) * s.att).sum(-1)
            emax = torch.full((N, s.h), -1e30, device=x.device).scatter_reduce(
                0, dst[:, None].expand(-1, s.h), e.detach(), reduce="amax", include_self=True)
            ex = torch.exp(e - emax[dst])
            den = torch.zeros(N, s.h, device=x.device).index_add_(0, dst, ex)
            a = ex / (den[dst] + 1e-16)
            o = torch.zeros(N, s.h, s.c, device=x.device).index_add_(0, dst, xl[src] * a.unsqueeze(-1))
            return o.reshape(N, -1) + s.bias

    class GraphNet(nn.Module):
        """Kim et al.: 3 x GATv2 (4,4,4 heads; hidden 64,64,40) + linear head. width/depth are tunable."""
        def __init__(s, cin, width=64, depth=3, heads=4, zero_head=False):
            super().__init__()
            last = max(heads, int(round(width * 0.625 / heads)) * heads)
            widths = [max(heads, width // heads * heads)] * (depth - 1) + [last]
            s.layers = nn.ModuleList(); cp = cin
            for wd in widths:
                s.layers.append(GATv2(cp, wd // heads, heads)); cp = wd
            s.head = nn.Linear(cp, 1)
            if zero_head:
                nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)

        def forward(s, x, ei, ea):
            for l in s.layers:
                x = TF.elu(l(x, ei, ea))
            return s.head(x).squeeze(-1)

    def _cb(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True))

    class CompactUNet(nn.Module):
        """Small RadioUNet-style U-Net (smoke / fallback). Input H,W divisible by 2**(depth-1)."""
        def __init__(s, cin, base=32, depth=4):
            super().__init__()
            ws = [base * 2 ** i for i in range(depth)]
            s.enc = nn.ModuleList([_cb(cin if i == 0 else ws[i - 1], ws[i]) for i in range(depth)])
            s.up = nn.ModuleList([nn.ConvTranspose2d(ws[i], ws[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
            s.dec = nn.ModuleList([_cb(2 * ws[i - 1], ws[i - 1]) for i in range(depth - 1, 0, -1)])
            s.out = nn.Conv2d(ws[0], 1, 1)

        def forward(s, x):
            skips = []
            for i, e in enumerate(s.enc):
                x = e(x)
                if i < len(s.enc) - 1:
                    skips.append(x); x = TF.max_pool2d(x, 2)
            for u, d in zip(s.up, s.dec):
                x = u(x); x = d(torch.cat([x, skips.pop()], 1))
            return s.out(x)

    class PublicRadioUNet(nn.Module):
        """Wraps RonLevie/RadioUNet lib/modules.py RadioWNet(phase='firstU') = the RadioUNet. Output = first U.
           (The public forward still evaluates the detached second U; that is wasted compute, not a behaviour change.)"""
        def __init__(s, repo, cin):
            super().__init__()
            p = Path(repo) / "lib" / "modules.py"
            if not p.exists():
                raise FileNotFoundError(f"{p} not found: git clone https://github.com/RonLevie/RadioUNet.git {repo}")
            spec = importlib.util.spec_from_file_location("radiounet_public_modules", str(p))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            s.net = mod.RadioWNet(inputs=cin, phase="firstU")

        def forward(s, x):
            return s.net(x)[0]


def is_gnn(arm):
    return arm in ("gnn_kim", "gnn_kim_oldedge", "rgnn_xmap")


def default_hp(arm, cfg):
    return dict(cfg["gnn_hp"]) if is_gnn(arm) else dict(cfg["radiounet_hp"])


def build_model(arm, hp, cfg):
    if is_gnn(arm):
        cin = 2 + 6 + (1 if arm == "rgnn_xmap" else 0)
        return GraphNet(cin, width=int(hp["width"]), depth=int(hp["depth"]), heads=int(hp.get("heads", 4)),
                        zero_head=(arm == "rgnn_xmap"))
    cin = 4 + (len(FEATURE_NAMES) if arm == "cnn_feat" else 0)     # cnn_feat = same U-net + the label-free features as extra channels
    if cfg["radiounet_impl"] == "public":
        return PublicRadioUNet(cfg["radiounet_repo"], cin)
    return CompactUNet(cin, base=int(hp["base"]), depth=int(hp["depth"]))


def n_params(model):
    """Parameters actually used by the arm (for the public RadioUNet: the first U only, not the unused second U)."""
    if nn is not None and isinstance(model, PublicRadioUNet):
        return int(sum(p.numel() for n, p in model.named_parameters() if not n.split(".")[1].startswith("W")))
    return int(sum(p.numel() for p in model.parameters()))


# =====================================================================================
# 8  INFERENCE FOR LEARNED ARMS (whole map)
# =====================================================================================
def tile_starts(H, T, stride):
    s = list(range(0, H - T + 1, stride))
    if s[-1] != H - T:
        s.append(H - T)
    return s


def tile_window(T):
    a = np.minimum(np.arange(T) + 1, T - np.arange(T)).astype(np.float32)
    w = np.outer(a, a)
    return (w / w.max()).ravel()


def gnn_chunks(ctx, hp, cfg, same_los, dev, cache=True):
    key = (float(hp["r2"]), bool(same_los))
    if cache and key in ctx.tg_cache:
        return ctx.tg_cache[key]
    T, H, W = cfg["tile"], ctx.H, ctx.W
    st = tile_starts(H, T, cfg["stride"])
    tiles = [(y0, x0) for y0 in st for x0 in st]
    win = tile_window(T)
    chunks = []
    for c0 in range(0, len(tiles), cfg["gnn_infer_tiles"]):
        tl = tiles[c0:c0 + cfg["gnn_infer_tiles"]]
        bld = torch.from_numpy(np.stack([ctx.bld[y:y + T, x:x + T] for y, x in tl])).to(dev)
        los = torch.from_numpy(np.stack([ctx.los[y:y + T, x:x + T] for y, x in tl])).to(dev)
        g = build_graph(bld, los, hp["r2"], same_los)
        nf = g["node_flat"].cpu().numpy()
        b, rem = nf // (T * T), nf % (T * T)
        y0s = np.array([y for y, _ in tl]); x0s = np.array([x for _, x in tl])
        gy, gx = y0s[b] + rem // T, x0s[b] + rem % T
        chunks.append(dict(g=g, gidx=torch.from_numpy(gy * W + gx).to(dev),
                           Fn=torch.from_numpy(ctx.F[:, gy, gx].T.copy()).to(dev),
                           w=torch.from_numpy(win[rem]).to(dev)))
    if cache:
        ctx.tg_cache[key] = chunks
    return chunks


@torch.no_grad() if torch is not None else (lambda f: f)
def predict_learned(arm, model, hp, ctx, lab, cfg, cache_graphs=True):
    """Full-map prediction (dB) = shared trend + model residual (rgnn_xmap: trend + rk_side + correction)."""
    dev = _STATE["device"]
    model.eval()
    H, W, scale = ctx.H, ctx.W, cfg["res_scale"]
    sparse = torch.zeros(H * W, device=dev); mask = torch.zeros(H * W, device=dev)
    if lab.n:
        ii = torch.from_numpy(lab.idx).to(dev)
        sparse[ii] = torch.from_numpy(lab.rlab / scale).float().to(dev); mask[ii] = 1.0
    if is_gnn(arm):
        base = torch.from_numpy(rk_base_map(ctx, lab, cfg) / scale).to(dev) if arm == "rgnn_xmap" else None
        acc = torch.zeros(H * W, device=dev); ws = torch.zeros(H * W, device=dev)
        for ch in gnn_chunks(ctx, hp, cfg, arm != "gnn_kim_oldedge", dev, cache_graphs):
            g = ch["gidx"]
            parts = [sparse[g][:, None], mask[g][:, None], ch["Fn"]]
            if base is not None:
                parts.append(base[g][:, None])
            o = model(torch.cat(parts, 1), ch["g"]["ei"], ch["g"]["ea"])
            if base is not None:
                o = o + base[g]
            acc.index_add_(0, g, o * ch["w"]); ws.index_add_(0, g, ch["w"])
        res = (acc / ws.clamp_min(1e-9) * scale).cpu().numpy()
    else:
        x = torch.zeros(1, 4 + (len(FEATURE_NAMES) if arm == "cnn_feat" else 0), H, W, device=dev)
        x[0, 0] = torch.from_numpy(ctx.bld.astype(np.float32)).to(dev)
        x[0, 1, ctx.txy[0], ctx.txy[1]] = 1.0
        x[0, 2] = sparse.view(H, W); x[0, 3] = mask.view(H, W)
        if arm == "cnn_feat":
            x[0, 4:] = torch.from_numpy(np.ascontiguousarray(ctx.F)).to(dev)
        res = (model(x)[0, 0] * scale).cpu().numpy().ravel()
    yhat = np.zeros(H * W, np.float32)
    o = ctx.out.ravel()
    yhat[o] = trend_map(ctx, lab).ravel()[o] + res[o]
    return yhat


# =====================================================================================
# 9  TRAINING (across maps) AND TUNING
# =====================================================================================
_CTX_LOCK = threading.RLock()


def get_ctx_ts(m, t, sim=None):
    with _CTX_LOCK:
        return get_ctx(m, t, sim)


def precompute_geometry(cfg, maps, txs_fn, n_jobs=1, sim=None):
    """Fill the disk cache (per-map and per-(map,tx) features) so that training never waits for geometry."""
    todo = [(m, t) for m in maps for t in txs_fn(m)]
    t0 = time.time()
    for i, (m, t) in enumerate(todo):
        get_ctx(m, t, sim, with_y=False)
        if (i + 1) % 200 == 0:
            log(f"geometry {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)")
    return len(todo)


def alpha_train(rng, cfg):
    lo, hi = cfg["train_alpha_range"]
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def train_item(arm, hp, cfg, ctx, rng, label_mode="normal", gt=None):
    """One training example for a (map, tx): a fresh label draw, trend, and a random tile (GNN) or the full map (CNN)."""
    a = 0.0 if label_mode == "zero" else alpha_train(rng, cfg)
    lab = make_lab(ctx, a, rng, "random", min_labels=cfg["min_labels"], label_mode=label_mode, global_trend=gt)
    H, W, sc = ctx.H, ctx.W, cfg["res_scale"]
    R = ((ctx.y - trend_map(ctx, lab)) / sc).astype(np.float32)
    sparse = np.zeros(H * W, np.float32); mask = np.zeros(H * W, np.float32)
    sparse[lab.idx] = (lab.rlab / sc).astype(np.float32); mask[lab.idx] = 1.0
    sparse, mask = sparse.reshape(H, W), mask.reshape(H, W)
    lossm = ctx.out & (mask == 0)
    if not is_gnn(arm):
        return dict(bld=ctx.bld, tx=ctx.txy, sparse=sparse, mask=mask, target=R, lossm=lossm,
                    F=(ctx.F if arm == "cnn_feat" else None))
    T = cfg["tile"]
    for _ in range(40):
        y0 = int(rng.integers(0, H - T + 1)); x0 = int(rng.integers(0, W - T + 1))
        if ctx.out[y0:y0 + T, x0:x0 + T].mean() >= 0.3:
            break
    sl = (slice(y0, y0 + T), slice(x0, x0 + T))
    item = dict(bld=ctx.bld[sl], los=ctx.los[sl], F=ctx.F[:, sl[0], sl[1]], sparse=sparse[sl], mask=mask[sl],
                target=R[sl], lossm=lossm[sl])
    if arm == "rgnn_xmap":
        base = np.zeros((T, T), np.float32)
        yy, xx = np.mgrid[y0:y0 + T, x0:x0 + T]
        q = (yy * W + xx).ravel()
        q = q[ctx.out.ravel()[q]]
        r = rk_side_residual(ctx, lab, q, cfg["classical_hp"]["rk_side"], "ridge") / sc
        base.ravel()[(q // W - y0) * T + (q % W - x0)] = r
        item["base"] = base
    return item


def collate(arm, hp, cfg, items, dev):
    T = lambda a: torch.from_numpy(np.stack(a)).to(dev)
    if not is_gnn(arm):
        B, H, W = len(items), items[0]["bld"].shape[0], items[0]["bld"].shape[1]
        x = torch.zeros(B, 4 + (len(FEATURE_NAMES) if arm == "cnn_feat" else 0), H, W, device=dev)
        x[:, 0] = T([it["bld"].astype(np.float32) for it in items])
        for i, it in enumerate(items):
            x[i, 1, it["tx"][0], it["tx"][1]] = 1.0
        x[:, 2] = T([it["sparse"] for it in items]); x[:, 3] = T([it["mask"] for it in items])
        if arm == "cnn_feat":
            x[:, 4:] = T([it["F"] for it in items])
        return dict(x=x, target=T([it["target"] for it in items]).unsqueeze(1), lossm=T([it["lossm"] for it in items]).unsqueeze(1))
    bld, los = T([it["bld"] for it in items]), T([it["los"] for it in items])
    g = build_graph(bld, los, hp["r2"], arm != "gnn_kim_oldedge")
    nf = g["node_flat"]
    Fl = T([it["F"] for it in items]).permute(1, 0, 2, 3).reshape(6, -1)
    flat = lambda k: T([it[k] for it in items]).reshape(-1)[nf]
    parts = [flat("sparse")[:, None], flat("mask")[:, None], Fl[:, nf].T]
    base = flat("base") if "base" in items[0] else None
    if base is not None:
        parts.append(base[:, None])
    return dict(x=torch.cat(parts, 1), g=g, target=flat("target"), lossm=flat("lossm").bool(), base=base)


class Prefetch:
    def __init__(self, fn, n_threads, seed, qsize=6):
        self.q = queue.Queue(qsize); self.stop = False; self.err = None
        def work(i):
            rng = np.random.default_rng(stable_seed("prefetch", seed, i))
            try:
                while not self.stop:
                    self.q.put(fn(rng))
            except Exception as e:                      # surfaced in get()
                self.err = e
                self.q.put(None)
        self.th = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(max(1, n_threads))]
        for t in self.th:
            t.start()

    def get(self):
        v = self.q.get()
        if v is None:
            raise RuntimeError(f"data worker failed: {self.err!r}")
        return v

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def val_score(arm, model, hp, cfg, ctxs, role="ckpt", label_mode="normal", gt=None, per_alpha=False):
    """Mean test RMSE (dB) of a learned arm on a fixed set of val contexts, one label draw per alpha."""
    res = {a: [] for a in cfg["alphas"]}
    for ctx in ctxs:
        for ai, a in enumerate(cfg["alphas"]):
            lab = make_lab(ctx, a, label_draw_rng(cfg, ctx.m, ctx.t, ai, 0, role), "random", min_labels=cfg["min_labels"],
                           label_mode=label_mode, global_trend=gt)
            yhat = predict_learned(arm, model, hp, ctx, lab, cfg, cache_graphs=False)
            res[a].append(compute_metrics(ctx, lab, yhat)["rmse"])
        ctx.tg_cache.clear()
    pa = {a: float(np.mean(v)) for a, v in res.items()}
    return pa if per_alpha else float(np.mean(list(pa.values())))


def train_model(arm, hp, seed, steps, cfg, pool, ckpt_ctxs, label_mode="normal", gt=None, verbose=True):
    """pool = list of (map, tx) training pairs. Early-selection on `ckpt_ctxs` (val maps only)."""
    dev = _STATE["device"]
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    model = build_model(arm, hp, cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["wd"])
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    B = cfg["gnn_batch"] if is_gnn(arm) else cfg["radiounet_batch"]

    def make(rng):
        its = []
        for _ in range(B):
            m, t = pool[int(rng.integers(len(pool)))]
            its.append(train_item(arm, hp, cfg, get_ctx_ts(m, t), rng, label_mode, gt))
        return its

    pf = Prefetch(make, cfg["n_workers_data"], seed)
    best, best_state, hist, t0 = np.inf, None, [], time.time()
    try:
        for step in range(1, steps + 1):
            model.train()
            b = collate(arm, hp, cfg, pf.get(), dev)
            if is_gnn(arm):
                out = model(b["x"], b["g"]["ei"], b["g"]["ea"])
                if b["base"] is not None:
                    out = out + b["base"]
                err = (out - b["target"]) ** 2
                lm = b["lossm"]
            else:
                err = (model(b["x"]) - b["target"]) ** 2
                lm = b["lossm"]
            loss = err[lm].mean() if lm.any() else err.mean() * 0.0
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            if step % cfg["ckpt_every"] == 0 or step == steps:
                v = val_score(arm, model, hp, cfg, ckpt_ctxs, "ckpt", label_mode, gt)
                hist.append((step, float(loss.item()), v))
                if verbose:
                    log(f"  {arm} seed{seed} step {step}/{steps} loss {loss.item():.4f} val_rmse {v:.3f} dB  ({time.time() - t0:.0f}s)")
                if v < best:
                    best = v; best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    finally:
        pf.close()
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(arm=arm, seed=seed, hp=hp, steps=steps, params=n_params(model), train_sec=time.time() - t0,
                       best_ckpt_val_rmse=float(best), hist=hist, label_mode=label_mode)


def sample_hp(arm, cfg, rng):
    lg = lambda lo, hi: float(10 ** rng.uniform(lo, hi))
    if is_gnn(arm):
        return dict(lr=lg(-3.5, -2.5), wd=lg(-6, -3), width=int(rng.choice([32, 48, 64, 96])),
                    depth=int(rng.choice([2, 3, 4])), r2=float(rng.choice([8.0, 18.0, 32.0])))
    if cfg["radiounet_impl"] == "public":
        return dict(lr=lg(-4.5, -3.3), wd=lg(-6, -3), base=32, depth=4)
    return dict(lr=lg(-4, -2.5), wd=lg(-6, -3), base=int(rng.choice([16, 32, 48])), depth=int(rng.choice([3, 4])))


def tune_arm(arm, cfg, pool, ckpt_ctxs, val_ctxs, n_cfg=None, steps=None):
    """Random search, trial 0 = defaults. Every trial: train `tune_steps`, score on val contexts (val maps only)."""
    n_cfg = n_cfg or cfg["tune_n_cfg"]; steps = steps or cfg["tune_steps"]
    rng = np.random.default_rng(stable_seed("tune", cfg["salt"], arm))
    trials = [default_hp(arm, cfg)] + [sample_hp(arm, cfg, rng) for _ in range(n_cfg - 1)]
    rows = []
    for i, hp in enumerate(trials):
        t0 = time.time()
        try:
            model, info = train_model(arm, hp, 1000 + i, steps, cfg, pool, ckpt_ctxs, verbose=False)
            pa = val_score(arm, model, hp, cfg, val_ctxs, "tune", per_alpha=True)
            score = float(np.mean(list(pa.values())))
            rows.append(dict(arm=arm, trial=i, val_rmse=score, **{f"val_rmse_a{a}": v for a, v in pa.items()},
                             params=info["params"], sec=time.time() - t0, hp=json.dumps(hp), ok=True))
            log(f"tune {arm} trial {i}/{n_cfg - 1}: val_rmse {score:.3f} dB  hp={hp}")
        except Exception as e:                           # a bad configuration must not kill the search
            rows.append(dict(arm=arm, trial=i, val_rmse=np.inf, hp=json.dumps(hp), ok=False, err=repr(e)))
            log(f"tune {arm} trial {i} FAILED: {e!r}")
    df = pd.DataFrame(rows)
    best = df.loc[df["val_rmse"].idxmin()]
    return json.loads(best["hp"]), df


def classical_val_score(arm, cfg, ctxs, role="tune"):
    v = []
    for ctx in ctxs:
        for ai, a in enumerate(cfg["alphas"]):
            lab = make_lab(ctx, a, label_draw_rng(cfg, ctx.m, ctx.t, ai, 0, role), "random", min_labels=cfg["min_labels"])
            v.append(compute_metrics(ctx, lab, predict_classical(arm, ctx, lab, cfg))["rmse"])
    return float(np.mean(v))


def tune_classical(cfg, val_ctxs):
    """Matching grid on the val maps, coordinate-wise (ridge first, then IDW power/K). Returns (hp dict, trials df)."""
    hp = json.loads(json.dumps(cfg["classical_hp"]))
    rows = []

    def run(arm, grid, key):
        best = None
        for g in grid:
            c = json.loads(json.dumps(cfg)); c["classical_hp"][arm].update(hp[arm]); c["classical_hp"][arm].update(g)
            s = classical_val_score(arm, c, val_ctxs)
            rows.append(dict(arm=arm, stage=key, hp=json.dumps(c["classical_hp"][arm]), val_rmse=s))
            if best is None or s < best[0]:
                best = (s, g)
        hp[arm].update(best[1])

    pk = [dict(p=p, K=K) for p in (1.0, 1.5, 2.0, 3.0) for K in (8, 16, 32)]
    run("idw", pk, "p,K")
    run("krig_global", [dict(K=K) for K in (8, 16, 32)], "K")
    run("rk_side", [dict(ridge=r) for r in (0.1, 1.0, 10.0, 100.0)], "ridge")
    run("rk_side", pk, "p,K")
    run("rk_gbm", [dict(max_iter=m, lr=l) for m in (50, 100, 200) for l in (0.05, 0.1)], "gbm")
    run("rk_gbm", pk, "p,K")
    return hp, pd.DataFrame(rows)


def global_trend_from(ctxs, n_px=2000, seed=0):
    rng = np.random.default_rng(seed)
    L, Y = [], []
    for c in ctxs:
        o = np.flatnonzero(c.out.ravel()); s = rng.choice(o, min(n_px, len(o)), replace=False)
        L.append(c.logd.ravel()[s]); Y.append(c.y.ravel()[s])
    return fit_trend(np.concatenate(L).astype(np.float64), np.concatenate(Y).astype(np.float64))


def feat_only_model(ctxs, n_px=1500, seed=0):
    """Features-only reference for the zero-label control: HGB on the 6 features (+ nothing else), trained across maps."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for c in ctxs:
        o = np.flatnonzero(c.out.ravel()); s = rng.choice(o, min(n_px, len(o)), replace=False)
        X.append(c.F.reshape(6, -1)[:, s].T); Y.append(c.y.ravel()[s])
    g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=4, random_state=0)
    g.fit(np.concatenate(X), np.concatenate(Y))
    return g


# =====================================================================================
# 10  EXECUTORS (resumable JSONL)
# =====================================================================================
def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(jsonable(r), allow_nan=True) + "\n")


def eval_ctx_rows(arm_fn, arms, ctx, cfg, split, seed, sim):
    rows = []
    for ai, alpha in enumerate(cfg["alphas"]):
        for d in range(cfg["n_draws"]):
            lab = make_lab(ctx, alpha, label_draw_rng(cfg, ctx.m, ctx.t, ai, d), split, cfg["block"], min_labels=cfg["min_labels"])
            for arm in arms:
                t0 = time.time()
                yhat = arm_fn(arm, ctx, lab)
                dt = time.time() - t0
                rows.append(dict(arm=arm, seed=seed, sim=sim, split=split, map=int(ctx.m), tx=int(ctx.t), alpha=alpha, draw=d,
                                 n_lab=lab.n, t_sec=dt, **compute_metrics(ctx, lab, yhat)))
    return rows


def _task_classical(args):
    m, t, split, sim, arms = args
    cfg = _STATE["cfg"]
    ctx = get_ctx(m, t, sim)
    rows = eval_ctx_rows(lambda a, c, l: predict_classical(a, c, l, cfg), arms, ctx, cfg, split, -1, sim)
    _STATE["lru"].clear()
    return rows


def _init_worker(cfg):
    try:
        if torch is not None:
            torch.set_num_threads(1)
    except Exception:
        pass
    init(cfg)


def tx_list(cfg, m, role="test"):
    return pick_tx(cfg, m, cfg["n_tx_test"], role)


def run_classical(cfg, maps, split, sim, out_path, arms=None, role="test"):
    arms = arms or ARMS_CLASSICAL
    done = read_jsonl(out_path)
    have = set()
    if len(done):
        d = done[(done.sim == sim) & (done.split == split)]
        for (m, t), g in d.groupby(["map", "tx"]):
            if set(arms) <= set(g.arm):
                have.add((int(m), int(t)))
    todo = []
    for m in maps:
        for t in tx_list(cfg, m, role):
            if (m, t) in have:
                continue
            if not _STATE["data"].exists(m, t, sim):
                continue
            todo.append((m, t, split, sim, arms))
    log(f"classical[{sim}/{split}]: {len(todo)} (map,tx) tasks to run, {len(have)} already done")
    t0 = time.time()
    if cfg["n_jobs"] > 1 and todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(cfg["n_jobs"], mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(cfg,)) as ex:
            for i, rows in enumerate(ex.map(_task_classical, todo, chunksize=1)):
                append_jsonl(out_path, rows)
                if (i + 1) % 25 == 0:
                    log(f"  {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)")
    else:
        for i, a in enumerate(todo):
            append_jsonl(out_path, _task_classical(a))
            if (i + 1) % 25 == 0:
                log(f"  {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)")
    return read_jsonl(out_path)


def run_learned(cfg, arm, seed, model, hp, maps, split, sim, out_path, role="test", label_mode="normal", gt=None):
    done = read_jsonl(out_path)
    have = set()
    if len(done):
        d = done[(done.arm == arm) & (done.seed == seed) & (done.sim == sim) & (done.split == split)]
        have = set(zip(d["map"].astype(int), d["tx"].astype(int)))
    n = 0; t0 = time.time()
    for m in maps:
        for t in tx_list(cfg, m, role):
            if (m, t) in have or not _STATE["data"].exists(m, t, sim):
                continue
            ctx = get_ctx(m, t, sim)
            rows = []
            for ai, alpha in enumerate(cfg["alphas"]):
                for d in range(cfg["n_draws"]):
                    lab = make_lab(ctx, alpha, label_draw_rng(cfg, m, t, ai, d), split, cfg["block"], min_labels=cfg["min_labels"],
                                   label_mode=label_mode, global_trend=gt)
                    t1 = time.time()
                    yhat = predict_learned(arm, model, hp, ctx, lab, cfg)
                    rows.append(dict(arm=arm, seed=seed, sim=sim, split=split, map=int(m), tx=int(t), alpha=alpha, draw=d,
                                     n_lab=lab.n, t_sec=time.time() - t1, **compute_metrics(ctx, lab, yhat)))
            ctx.tg_cache.clear()
            append_jsonl(out_path, rows); n += 1
            if n % 25 == 0:
                log(f"  {arm} seed{seed}: {n} (map,tx) done ({time.time() - t0:.0f}s)")
    return read_jsonl(out_path)


def save_model(path, model, hp, info):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), hp=hp, info=jsonable(info)), path)


def load_model(path, arm, cfg):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(arm, ck["hp"], cfg)
    model.load_state_dict(ck["state"])
    return model.to(_STATE["device"]).eval(), ck["hp"], ck["info"]


# =====================================================================================
# 11  STATISTICS, FAMILIES, VERDICTS   (unit of inference = test building map)
# =====================================================================================
STAT = dict(n_perm=200000, n_boot=10000, margin=2.0, ni_margin=5.0, alpha=0.05, seed=20261022)

# Plan section 7. Each contrast is run at every density in cfg['alphas'].
FAMILIES = {
    "primary":      dict(pairs=[("gnn_kim", "rk_side"), ("rgnn_xmap", "rk_side")]),
    "vs_classical": dict(pairs=[("gnn_kim", "idw"), ("rgnn_xmap", "idw")]),
    "feat":         dict(pairs=[("rk_side", "idw")]),
    "topo":         dict(pairs=[("gnn_kim", "gnn_kim_oldedge")]),
    "deep":         dict(pairs=[("gnn_kim", "radiounet"), ("rgnn_xmap", "radiounet")], ni_margin=5.0),
    "deep_feat":    dict(pairs=[("gnn_kim", "cnn_feat"), ("rgnn_xmap", "cnn_feat")], ni_margin=5.0),
}


def signflip_p(g, n_perm, rng):
    """Two-sided sign-flip test of mean(g)=0. Exact enumeration if n<=16, else Monte-Carlo (exact is 2^n)."""
    n = len(g); obs = abs(float(np.mean(g)))
    if n <= 16:
        s = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        return float(np.mean(np.abs((s * g).mean(1)) >= obs - 1e-12))
    cnt, done = 0, 0
    while done < n_perm:
        k = min(20000, n_perm - done)
        s = rng.choice([-1.0, 1.0], size=(k, n))
        cnt += int(np.sum(np.abs((s * g).mean(1)) >= obs - 1e-12)); done += k
    return (cnt + 1) / (n_perm + 1)


def min_attainable_p(n, n_perm):
    return max(2.0 / 2 ** n, 1.0 / (n_perm + 1)) if n > 16 else 2.0 / 2 ** n


def boot_ci(g, n_boot, rng):
    n = len(g); out = []
    for s in range(0, n_boot, 2000):
        k = min(2000, n_boot - s)
        out.append(g[rng.integers(0, n, size=(k, n))].mean(1))
    out = np.concatenate(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def verdict_label(lo, hi, margin):
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if lo > 0:
        return "A better"
    if hi < 0:
        return "A worse"
    return "inconclusive"


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def unit_means(df, metric="rmse"):
    """(arm, alpha, map) -> mean over transmitters, label draws and training seeds."""
    return df.groupby(["arm", "alpha", "map"], as_index=False)[metric].mean()


def contrast(U, A, B, alpha, metric, st=STAT, kind="rel"):
    ua = U[(U.arm == A) & (np.isclose(U.alpha, alpha))].set_index("map")[metric]
    ub = U[(U.arm == B) & (np.isclose(U.alpha, alpha))].set_index("map")[metric]
    common = ua.index.intersection(ub.index)
    a, b = ua[common].values.astype(float), ub[common].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    g = 100.0 * (b - a) / b if kind == "rel" else 100.0 * (a - b)       # rel: RMSE gain %; diff: points (higher=better)
    rng = np.random.default_rng(stable_seed(st["seed"], A, B, alpha, metric))
    if len(g) < 3:
        return dict(n_maps=len(g))
    lo, hi = boot_ci(g, st["n_boot"], rng)
    return dict(n_maps=len(g), mean_A=float(a.mean()), mean_B=float(b.mean()), delta=float((a - b).mean()),
                gain_pct=float(g.mean()), ci_lo=lo, ci_hi=hi, p_raw=signflip_p(g, st["n_perm"], rng),
                frac_maps_A_better=float(np.mean(g > 0)))


def run_families(df, alphas, metric="rmse", families=None, st=STAT, kind="rel", label="confirmatory"):
    families = families or FAMILIES
    U = unit_means(df, metric)
    rows = []
    for fam, spec in families.items():
        fr = []
        for A, B in spec["pairs"]:
            for a in alphas:
                r = contrast(U, A, B, a, metric, st, kind)
                if "gain_pct" not in r:
                    continue
                r.update(family=fam, A=A, B=B, alpha=a, metric=metric, status=label)
                r["verdict"] = verdict_label(r["ci_lo"], r["ci_hi"], st["margin"])
                if "ni_margin" in spec:
                    r["noninferior"] = bool(r["ci_lo"] > -spec["ni_margin"])
                fr.append(r)
        if fr:
            ph = holm([r["p_raw"] for r in fr])
            for r, p in zip(fr, ph):
                r["p_holm"] = float(p); r["sig"] = bool(p < st["alpha"])
            rows += fr
    cols = ["family", "A", "B", "alpha", "metric", "n_maps", "mean_A", "mean_B", "delta", "gain_pct", "ci_lo", "ci_hi",
            "p_raw", "p_holm", "sig", "verdict", "noninferior", "frac_maps_A_better", "status"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]] if len(out) else out


def family_feasibility(alphas, n_maps, st=STAT, families=None):
    """Freeze-time check: can each family reach p_holm < 0.05 with n_maps maps?"""
    families = families or FAMILIES
    rows = []
    for fam, spec in families.items():
        m = len(spec["pairs"]) * len(alphas)
        mp = min_attainable_p(n_maps, st["n_perm"])
        rows.append(dict(family=fam, n_tests=m, n_maps=n_maps, min_p_raw=mp, min_p_holm=min(1.0, mp * m), feasible=bool(mp * m < st["alpha"])))
    return pd.DataFrame(rows)


def positive_result(prim, st=STAT, min_gain=2.0, need=2):
    """Plan section 2. prim = run_families(...) rows of family 'primary' (A vs rk_side)."""
    out = {}
    for arm in ("gnn_kim", "rgnn_xmap"):
        r = prim[(prim.A == arm) & (prim.B == "rk_side")]
        win = r[(r.p_holm < st["alpha"]) & (r.gain_pct >= min_gain) & (r.ci_lo > 0)]
        worse = r[(r.p_holm < st["alpha"]) & (r.ci_hi < 0)]
        losers_ok = len(worse) == 0                  # not significantly worse than rk_side at any density
        out[arm] = dict(n_win=int(len(win)), n_dens=int(len(r)), sig_worse_at=sorted(worse.alpha.tolist()),
                        positive=bool(len(win) >= need and losers_ok))
    out["POSITIVE_GNN_RESULT"] = bool(any(v["positive"] for v in out.values()))
    return out


def cluster_bootstrap(df, city_map, A, B, alpha, metric="rmse", n_boot=10000, seed=1):
    """Sensitivity: resample CITIES (6 clusters), keep all maps of a sampled city. Descriptive only."""
    U = unit_means(df, metric)
    ua = U[(U.arm == A) & np.isclose(U.alpha, alpha)].set_index("map")[metric]
    ub = U[(U.arm == B) & np.isclose(U.alpha, alpha)].set_index("map")[metric]
    common = ua.index.intersection(ub.index)
    g = pd.Series(100.0 * (ub[common] - ua[common]) / ub[common], index=common)
    cm = pd.Series({int(k): v for k, v in city_map.items()}).reindex(g.index)
    cities = cm.dropna().unique()
    groups = {c: g[cm == c].values for c in cities}
    rng = np.random.default_rng(seed); means = []
    for _ in range(n_boot):
        pick = rng.choice(len(cities), len(cities), replace=True)
        means.append(np.mean(np.concatenate([groups[cities[i]] for i in pick])))
    return dict(n_clusters=len(cities), gain_pct=float(g.mean()), ci_lo=float(np.percentile(means, 2.5)), ci_hi=float(np.percentile(means, 97.5)))


# =====================================================================================
# 12  PRE-REGISTRATION AND CHECKS
# =====================================================================================
def module_hash():
    return sha256_file(__file__)


def freeze_prereg(path, obj):
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} already exists: a frozen plan is never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(obj), sort_keys=True, indent=1)
    p.write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    Path(str(p) + ".sha256").write_text(h)
    return h


def check_frozen(path):
    """The confirmatory runner calls this first: file must exist, hash must match, code must be the frozen code."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"{p} is missing: run the freeze step first")
    text = p.read_text()
    h = hashlib.sha256(text.encode()).hexdigest()
    rec = Path(str(p) + ".sha256").read_text().strip()
    if h != rec:
        raise RuntimeError("prereg file does not match its recorded sha256")
    obj = json.loads(text)
    if obj.get("module_sha256") != module_hash():
        raise RuntimeError("rmbench_rms.py changed after the freeze (module hash differs)")
    return obj, h


def assert_disjoint(splits):
    s = {k: set(v) for k, v in splits.items()}
    for a, b in itertools.combinations(s, 2):
        assert not (s[a] & s[b]), f"map ids shared between {a} and {b}: {sorted(s[a] & s[b])[:5]}"
    return True


def edge_unit_test(r2=18.0, size=24, verbose=True):
    """Check 6: on a toy map no edge crosses a building and (same_los=True) no edge joins a LOS and an NLOS node.
       The expected edge set is rebuilt by brute force and must equal the graph's edge set."""
    bld = np.zeros((size, size), bool)
    bld[9:12, 3:20] = True; bld[2:5, 15:19] = True
    txy = (1, 1)
    los, _, _ = tx_geometry(bld, txy)
    res = {}
    for same in (True, False):
        g = build_graph(torch.from_numpy(bld)[None], torch.from_numpy(los)[None], r2, same)
        nf = g["node_flat"].numpy(); ei = g["ei"].numpy()
        got = set((int(nf[s]), int(nf[d])) for s, d in ei.T if s != d)
        exp = set()
        for y in range(size):
            for x in range(size):
                if bld[y, x]:
                    continue
                for dy, dx in edge_offsets(r2):
                    y2, x2 = y + dy, x + dx
                    if not (0 <= y2 < size and 0 <= x2 < size) or bld[y2, x2]:
                        continue
                    n = max(abs(dy), abs(dx))
                    cells = [(y + int(math.floor(s * dy / n + 0.5)), x + int(math.floor(s * dx / n + 0.5))) for s in range(1, n)]
                    if any(bld[cy, cx] for cy, cx in cells):
                        continue
                    if same and los[y, x] != los[y2, x2]:
                        continue
                    exp.add((y2 * size + x2, y * size + x))
        assert got == exp, f"edge set differs from brute force (same_los={same}): {len(got)} vs {len(exp)}"
        mism = sum(1 for s, d in got if los[s // size, s % size] != los[d // size, d % size])
        crossing = 0                                    # independent geometric check: dense sampling of the true segment
        for s, d in got:
            (y1, x1), (y2, x2) = divmod(s, size), divmod(d, size)
            k = 4 * max(abs(y2 - y1), abs(x2 - x1)) + 1
            tt = np.linspace(0, 1, k)
            cy = np.floor(y1 + tt * (y2 - y1) + 0.5).astype(int); cx = np.floor(x1 + tt * (x2 - x1) + 0.5).astype(int)
            crossing += int(bld[cy, cx].mean() > 0.25)   # > 25% of the segment inside buildings (corner grazing allowed)
        assert crossing == 0, "an edge runs through a building"
        if same:
            assert mism == 0, "an edge joins a LOS and an NLOS node"
        res[f"same_los={same}"] = dict(n_edges=len(got), los_nlos_edges=mism, edges_through_building=crossing)
    assert res["same_los=False"]["los_nlos_edges"] > 0, "toy map does not exercise the LOS/NLOS rule"
    if verbose:
        log("edge-rule unit test passed:", res)
    return res


def perturb_test(arm, ctx, lab, cfg, model=None, hp=None, atol=1e-4):
    """Check 3 (generalised): the prediction may depend on the labels and label-free inputs only. Rebuild the
       context with y = None (any access to a pixel value would raise) and require an identical prediction."""
    import copy
    c2 = copy.copy(ctx); c2.y = None; c2.tg_cache = {}
    if arm in ARMS_CLASSICAL:
        a, b = predict_classical(arm, ctx, lab, cfg), predict_classical(arm, c2, lab, cfg)
    else:
        a, b = predict_learned(arm, model, hp, ctx, lab, cfg), predict_learned(arm, model, hp, c2, lab, cfg)
    return bool(np.allclose(a[ctx.out.ravel()], b[ctx.out.ravel()], atol=atol))


# =====================================================================================
# 13  SMALL HELPERS USED BY THE NOTEBOOK
# =====================================================================================
def make_pool(cfg, maps, n_tx=None):
    n_tx = n_tx or cfg["n_tx_train"]
    return [(m, t) for m in maps for t in pick_tx(cfg, m, n_tx, "train")]


def make_val_ctxs(cfg, maps, n, role):
    return [get_ctx(m, pick_tx(cfg, m, 1, role)[0]) for m in maps[:n]]


def classical_val_by_alpha(arm, cfg, ctxs, role="tune"):
    out = {}
    for ai, a in enumerate(cfg["alphas"]):
        v = []
        for ctx in ctxs:
            lab = make_lab(ctx, a, label_draw_rng(cfg, ctx.m, ctx.t, ai, 0, role), "random", min_labels=cfg["min_labels"])
            v.append(compute_metrics(ctx, lab, predict_classical(arm, ctx, lab, cfg))["rmse"])
        out[a] = float(np.mean(v))
    return out
