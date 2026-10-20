"""rmbench_r7 -- exported from rmbench_r7.ipynb (sections 1-6)."""
from __future__ import annotations
import os, json, time, math, traceback
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import rmbench_core as rc
import rmbench_ext5 as x5

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

print("torch available:", HAVE_TORCH)

LAMS = (0.0, 0.25, 0.5, 1.0)   # candidate blend weights for the correction (0.0 = pure rk_side)
_GCACHE: dict = {}             # graph cache, keyed by (unit name, kind, k, beta)
_RCACHE: dict = {}             # raster cache, keyed by (raster_dir, slug)

def _sym_edges(idx, w):
    """idx (N,k) neighbour ids, w (N,k) weights -> symmetric normalised COO (row, col, val)
    with self loops, as int64 / float32 numpy arrays."""
    N, k = idx.shape
    r = np.repeat(np.arange(N), k)
    c = idx.ravel()
    v = w.ravel()
    A = sp.coo_matrix((v, (r, c)), shape=(N, N)).tocsr()
    A = A.maximum(A.T).tolil()
    A.setdiag(1.0)
    A = A.tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    val = A.data / np.sqrt(deg[A.row] * deg[A.col])
    return A.row.astype(np.int64), A.col.astype(np.int64), val.astype(np.float32)


def knn_graph(pos, k=8):
    k = min(k, len(pos) - 1)
    d, i = cKDTree(pos).query(pos, k=k + 1)
    d, i = d[:, 1:], i[:, 1:]
    sc = np.median(d[:, 0]) + 1e-9
    return _sym_edges(i, np.exp(-(d / (3 * sc)) ** 2))


def feat_graph(pos, S, k=8, beta=1.0):
    """kNN in [pos/ell, beta*z(S)].  ell = median 8th-neighbour distance, so one unit ~ one
    neighbourhood radius.  z(S) is taken over ALL nodes of the unit (covariates only)."""
    k = min(k, len(pos) - 1)
    d8 = cKDTree(pos).query(pos, k=min(9, len(pos)))[0][:, -1]
    ell = float(np.median(d8)) + 1e-9
    Z = (np.asarray(S, float) - np.mean(S, 0)) / (np.std(S, 0) + 1e-9)
    E = np.column_stack([pos / ell, beta * Z])
    d, i = cKDTree(E).query(E, k=k + 1)
    d, i = d[:, 1:], i[:, 1:]
    sc = np.median(d[:, 0]) + 1e-9
    return _sym_edges(i, np.exp(-(d / (3 * sc)) ** 2))

def _cells(R, P):
    ix = np.clip(np.floor((P[..., 0] - R["x0"]) / R["res"]).astype(int), 0, R["occ"].shape[1] - 1)
    iy = np.clip(np.floor((P[..., 1] - R["y0"]) / R["res"]).astype(int), 0, R["occ"].shape[0] - 1)
    return iy, ix


def blocked_length(R, A, B):
    """Length (m) of the segments A[i]->B[i] that run through occupied raster cells
    (A, B: (M,2) metres).  Samples every ~res/2."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    L = np.linalg.norm(B - A, axis=1)
    K = int(np.ceil(L.max() / (0.5 * R["res"]))) + 2
    t = np.linspace(0.0, 1.0, K)
    P = A[:, None, :] + (B - A)[:, None, :] * t[None, :, None]
    iy, ix = _cells(R, P)
    return R["occ"][iy, ix].mean(1) * L


def wall_graph(pos, R, k=8, beta=3.0, n_cand=24, chunk=4000):
    """Neighbours = the k smallest of  d + beta * blocked_length  among the n_cand
    Euclidean-nearest nodes.  beta fixed a priori (or tuned on source+val only)."""
    n_cand = min(n_cand, len(pos) - 1)
    k = min(k, n_cand)
    d, i = cKDTree(pos).query(pos, k=n_cand + 1)
    d, i = d[:, 1:], i[:, 1:]
    N = len(pos)
    bl = np.zeros_like(d)
    K = int(np.ceil(float(d.max()) / (0.5 * R["res"]))) + 2
    chunk = int(max(50, min(chunk, 3_000_000 // max(1, n_cand * K))))      # bound memory
    for a in range(0, N, chunk):
        sl = slice(a, min(N, a + chunk))
        n = sl.stop - sl.start
        A = np.repeat(pos[sl], n_cand, axis=0)
        B = pos[i[sl].ravel()]
        bl[sl] = blocked_length(R, A, B).reshape(n, n_cand)
    de = d + beta * bl
    o = np.argsort(de, axis=1)[:, :k]
    ii = np.take_along_axis(i, o, 1)
    dd = np.take_along_axis(de, o, 1)
    sc = np.median(dd[:, 0]) + 1e-9
    return _sym_edges(ii, np.exp(-(dd / (3 * sc)) ** 2))


def load_raster(raster_dir, slug):
    key = (raster_dir, slug)
    if key not in _RCACHE:
        z = np.load(os.path.join(raster_dir, f"{slug}.npz"))
        _RCACHE[key] = dict(occ=z["occ"].astype(bool), hgt=z["hgt"],
                            x0=float(z["x0"]), y0=float(z["y0"]), res=float(z["res"]))
    return _RCACHE[key]


def build_rasters(slugs, out_dir="rmbench_cache_city/rasters", cfg=None):
    """Run ONCE where DeepMIMO scenes are downloadable (needs network + the original build
    code).  Saves occ/hgt/x0/y0/res per scene.  UNTESTED in the dev sandbox."""
    import rmbench_city as cc
    cfg = cfg or cc.CITY_CFG
    os.makedirs(out_dir, exist_ok=True)
    for slug in slugs:
        ds = next(iter(rc._dm_datasets(slug, cfg)))
        pos3 = np.asarray(ds.rx_pos, float)
        tx3 = np.asarray(ds.tx_pos, float).reshape(-1)[:3]
        R = cc.raster_for(getattr(ds, "scene", None), pos3, tx3)
        np.savez_compressed(os.path.join(out_dir, f"{slug}.npz"),
                            occ=R["occ"], hgt=R["hgt"], x0=R["x0"], y0=R["y0"], res=R["res"])
        print("raster", slug, R["occ"].shape)

def get_graph(u, hp):
    kind = hp.get("graph", "knn")
    k = int(hp.get("k", 8))
    beta = float(hp.get("beta", 1.0 if kind == "feat" else 3.0))
    key = (u["name"], kind, k, beta)
    if key in _GCACHE:
        return _GCACHE[key]
    pos = np.asarray(u["pos"], float)
    if kind == "knn":
        g = knn_graph(pos, k)
    elif kind == "feat":
        # stored side features if present and non-empty, else fall back to the LOS flag
        S = np.asarray(u["side"], float) if "side" in u else np.empty(0)
        S = S.reshape(len(S), -1) if S.size else np.asarray(u["los"], float)[:, None]
        g = feat_graph(pos, S, k, beta)
    elif kind == "wall":
        rd = hp.get("raster_dir")
        if not rd:
            raise ValueError("graph='wall' needs hp['raster_dir'] (see build_rasters)")
        g = wall_graph(pos, load_raster(rd, str(u["scenario"])), k, beta)
    else:
        raise ValueError(kind)
    _GCACHE[key] = g
    return g

def crossfit_residuals(u, tr, prep, K=5, seed=0, hp=None):
    import rmbench_ext2 as r2
    tr = np.asarray(tr)
    n = len(tr)
    K = int(min(K, max(2, n)))
    perm = np.random.default_rng(seed).permutation(n)
    out = np.zeros(n)
    y = np.asarray(u["y"], float)
    for f in np.array_split(np.arange(n), K):
        hold = perm[f]
        keep = np.ones(n, bool)
        keep[hold] = False
        p, _ = r2.predict_rk_side(u, tr[keep], tr[hold], prep, hp or dict(fset="geo_side"))
        out[hold] = y[tr[hold]] - p
    return out

if HAVE_TORCH:
    class _Net(nn.Module):
        def __init__(s, din, h=32, L=2, p=0.1):
            super().__init__()
            s.inp = nn.Linear(din, h)
            s.layers = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(L)])
            s.p = p
            s.out = nn.Linear(h, 1)
            nn.init.zeros_(s.out.weight)       # step 0 == rk_side
            nn.init.zeros_(s.out.bias)

        def forward(s, x, A):
            z = F.relu(s.inp(x))
            for lin in s.layers:
                z = F.dropout(F.relu(lin(torch.cat([z, torch.sparse.mm(A, z)], 1))), s.p, s.training)
            return s.out(z).squeeze(-1)

    def _adj(g, N):
        r, c, v = g
        return torch.sparse_coo_tensor(torch.from_numpy(np.stack([r, c])),
                                       torch.from_numpy(v), (N, N)).coalesce()

    def _inputs(X, A, rn, vis_mask):
        """[covariates, visible residual, visible flag, neighbourhood mean of visible residuals]."""
        m = vis_mask
        nb = (torch.sparse.mm(A, (rn * m)[:, None]).squeeze(-1)
              / (torch.sparse.mm(A, m[:, None]).squeeze(-1) + 1e-3))
        return torch.cat([X, (rn * m)[:, None], m[:, None], nb[:, None]], 1)

    def fit_correction(u, tr, te, prep, r_cf, hp, seed):
        """Train on cross-fitted residuals r_cf (dBm, aligned with tr);
        return the correction c(x) in dBm at te."""
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        N = len(u["y"])
        tr = np.asarray(tr)
        te = np.asarray(te)
        sig = float(prep["sigma"])

        A = _adj(get_graph(u, hp), N)
        X = torch.tensor(prep["X"], dtype=torch.float32)
        rn = torch.zeros(N)
        rn[torch.from_numpy(tr)] = torch.tensor(r_cf / sig, dtype=torch.float32)

        # fit / validation split of the labelled nodes
        n = len(tr)
        perm = rng.permutation(n)
        nv = int(round(0.2 * n)) if n >= 20 else 0
        val_i, fit_i = perm[:nv], perm[nv:]

        net = _Net(X.shape[1] + 3, int(hp.get("hidden", 32)), int(hp.get("layers", 2)),
                   float(hp.get("dropout", 0.1)))
        opt = torch.optim.AdamW(net.parameters(), lr=float(hp.get("lr", 1e-2)),
                                weight_decay=float(hp.get("wd", 1e-4)))
        steps = int(hp.get("steps", 150))

        def vis(idx):
            m = torch.zeros(N)
            m[torch.from_numpy(tr[idx])] = 1.0
            return m

        def val_loss():
            vi = torch.from_numpy(tr[val_i])
            with torch.no_grad():
                net.eval()
                out = net(_inputs(X, A, rn, vis(fit_i)), A)[vi]
                return F.huber_loss(out, rn[vi], delta=1.5).item()

        best = val_loss() if nv else float("inf")
        state = {k: t.clone() for k, t in net.state_dict().items()}
        bad = 0

        for s in range(steps):
            q = int(np.clip(round(rng.uniform(0.15, 0.5) * len(fit_i)), 1, max(1, len(fit_i) - 2)))
            pp = rng.permutation(fit_i)
            Q, V = pp[:q], pp[q:]                      # Q: hidden targets, V: visible context
            net.train()
            opt.zero_grad()
            qi = torch.from_numpy(tr[Q])
            F.huber_loss(net(_inputs(X, A, rn, vis(V)), A)[qi], rn[qi], delta=1.5).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()

            if nv and s % 5 == 4:                      # early stopping on held-out labelled nodes
                v = val_loss()
                if v < best - 1e-5:
                    best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8:
                        break

        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            c = net(_inputs(X, A, rn, vis(np.arange(n))), A)[torch.from_numpy(te)].numpy() * sig
        return c

def _fit_pair(u, tr, te, prep, hp, seed):
    """Returns (rk_side prediction, GCN correction, rk_side info) at te."""
    import rmbench_ext2 as r2
    rk, info = r2.predict_rk_side(u, tr, te, prep, hp)
    r_cf = crossfit_residuals(u, tr, prep, int(hp.get("cf_folds", 5)), seed, hp)
    return rk, fit_correction(u, tr, te, prep, r_cf, hp, seed), info


def predict_rgnn(u, tr, te, prep, hp, seed=0):
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for rgnn")
    tr = np.asarray(tr)
    te = np.asarray(te)
    n = len(tr)
    y = np.asarray(u["y"], float)
    rk_hp = dict(hp, fset=hp.get("fset", "geo_side"))
    info = {}

    lam, gate = 1.0, hp.get("gate", True)
    if gate:
        lam = 0.0
        if n >= int(hp.get("min_labels", 30)):
            K = int(hp.get("gate_folds", 5))
            perm = np.random.default_rng(seed + 1).permutation(n)
            E = np.zeros((len(LAMS), n))                 # squared CV error per lambda, per labelled point
            for f in np.array_split(np.arange(n), K):
                hold = perm[f]
                keep = np.ones(n, bool)
                keep[hold] = False
                trf, tef = tr[keep], tr[hold]
                rk, c, _ = _fit_pair(u, trf, tef, prep, rk_hp, seed)
                for i, l in enumerate(LAMS):
                    E[i, hold] = (rk + l * c - y[tef]) ** 2

            mse = E.mean(1)
            j = int(np.argmin(mse))
            d = E[0] - E[j]
            z = float(d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12))
            gain = float(d.mean() / (E[0].mean() + 1e-12))
            ok = j > 0 and z > float(hp.get("z", 2.5)) and gain > float(hp.get("rel", 0.01))
            lam = float(LAMS[j]) if ok else 0.0
            info.update(cv_z=z, cv_gain_rel=gain, lam_cv_best=float(LAMS[j]))

    rk, c, rk_info = _fit_pair(u, tr, te, prep, rk_hp, seed)
    info.update(lam=lam, gate_open=float(lam > 0), corr_rms=float(np.sqrt(np.mean(c ** 2))),
                **{k: v for k, v in rk_info.items() if k.startswith(("rk_", "idw_"))})
    return rk + lam * c, info

def r7_jobs(raster_dir=None, tuned=None, diagnostics=True):
    base = dict(fset="geo_side", k=8, hidden=32, layers=2, dropout=0.1, lr=1e-2, steps=150,
                gate=True, z=2.5, rel=0.01, min_labels=30)
    if tuned:
        base.update(tuned)
    J = [dict(name="rgnn_knn",  method="rgnn", hp=dict(base, graph="knn")),
         dict(name="rgnn_feat", method="rgnn", hp=dict(base, graph="feat", beta=1.0))]
    if raster_dir:
        J.append(dict(name="rgnn_wall", method="rgnn",
                      hp=dict(base, graph="wall", beta=3.0, raster_dir=raster_dir)))
    if diagnostics:
        J += [dict(name=j["name"] + "_ungated", method="rgnn", hp=dict(j["hp"], gate=False))
              for j in list(J)]
    return J


def install():
    x5.NEW_METHODS["rgnn"] = predict_rgnn


install()


def run_task7(task):
    """Register rgnn, then run the stock task executor. Calls x5.run_task5 directly (what x6.run_task6 does):
    calling x6.run_task6 here would recurse forever, because run_tasks7 below patches x6.run_task6 to point at this function."""
    install()
    import rmbench_ext6 as x6
    x6.install()
    return x5.run_task5(task)


def run_tasks7(tasks, out_path, n_jobs=1, log_every=25):
    """x6.run_tasks6, but workers import this module (so rgnn is registered in every
    spawned process).  NOTE: with n_jobs > 1, workers can only import this from a .py
    module -- run section 9 first and call rmbench_r7.run_tasks7 from the exported file."""
    import rmbench_ext6 as x6
    old = x6.run_task6
    x6.run_task6 = run_task7
    try:
        return x6.run_tasks6(tasks, out_path, n_jobs=n_jobs, log_every=log_every)
    finally:
        x6.run_task6 = old


def leak_check7(jobs, unit, cfg, density=100, rep=0):
    """Predictions must be bit-identical when the TEST labels are corrupted."""
    install()
    import rmbench_ext6 as x6
    x6.install()
    _GCACHE.clear()
    return x5.leak_check(jobs, unit, cfg, density=density, rep=rep)
