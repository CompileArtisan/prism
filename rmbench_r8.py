"""rmbench_r8.py -- Round 8: a LABEL-CENTRIC, obstruction-aware attention interpolator ('rattn'). Hand-written module (not exported from a notebook).
Needs rmbench_r7.py (blocked_length, load_raster, install) next to it. ADDS ONLY; frozen modules untouched.

Why a different model: in Round 7 the GNNs lived on a graph over EVERY grid cell (~2 m apart). A 2-layer GCN then sees ~4 m, so with 30-300 labels most nodes see
no labelled neighbour and the GNN is blind exactly where labels are scarce. Here messages travel directly from the K nearest LABELLED points to each query:

  pred(q) = trend(q) + ridge(q) + sum_j  w_qj * r_j            (r_j = ridge residual at labelled point j; exactly what rk_side feeds to IDW)
  w_q.    = softmax( -p0 * log(d_qj / ell)  +  MLP(edge features_qj)  |  null logit(q) )      (null slot = shrink to 0)

p0 is the IDW power tuned by LOO on the labelled set, the MLP's last layer starts at zero, so training starts at ~IDW (= rk_side's interpolator) and can only
move away if leave-one-out error on the labelled points says so. The output is LINEAR in the residuals (a learned, non-stationary kriging-like kernel), not a free-form net.
Edge features by arm:
  rattn_geo   log distance, neighbour rank                                        (a learned isotropic kernel: tests 'learned kernel vs fixed IDW')
  rattn_side  + query map features, query-minus-source map features (5 each)      (tests the 5 label-free features inside the kernel)
  rattn_wall  + blocked length / blocked fraction of the segment q->j (raster)    (tests obstruction-aware kernels; needs rasters)
Training = leave-one-out on the labelled set (each labelled point is a query, sources = all OTHER labelled points), Huber loss, 20% of queries held out for early stopping.
Gate: pred = rk_side + lam * (rattn - rk_side), lam in {0, 0.5, 1} by 5-fold CV inside the labelled set (whole pipeline refitted per fold); lam > 0 only if paired z > 2.5 and
relative gain > 1% (same rule as the earlier gates). Labels are read at `tr` only (leak_check8).

  import rmbench_r8 as r8;  jobs = r8.r8_jobs(raster_dir);  r8.run_tasks8(tasks, "results_city/r8.jsonl", n_jobs=1)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree
import rmbench_core as rc, rmbench_ext5 as x5
import rmbench_r7 as r7

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

LAMS8 = (0.0, 0.5, 1.0)

# ================================================================================================ numpy side: ridge, neighbours, edge features
def _ridge_parts(prep, tr):
    """Identical to the ridge step of r2.predict_rk_side. Returns (fit at ALL nodes, in-sample ridge residual at tr, chosen lambda)."""
    X = np.asarray(prep["X"], float); res = np.asarray(prep["res"], float); A = np.column_stack([np.ones(len(X)), X]); At, yt = A[tr], res[tr]
    pen = np.eye(A.shape[1]); pen[0, 0] = 0.0; best = (np.inf, None, None)
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        Minv = np.linalg.solve(At.T @ At + lam * pen, At.T); H = At @ Minv; e = yt - H @ yt
        score = float(np.mean((e / np.clip(1 - np.diag(H), 1e-6, None)) ** 2))
        if score < best[0]: best = (score, lam, Minv @ yt)
    _, lam, beta = best; fit = A @ beta
    return fit, yt - fit[tr], float(lam)

def _side_z(u):
    S = np.asarray(u["side"], float) if "side" in u else np.zeros((len(u["y"]), 0))
    S = S.reshape(len(S), -1); return (S - S.mean(0)) / (S.std(0) + 1e-9) if S.size else S

def _nbr(pos, q_idx, src_idx, K, drop_self):
    """K nearest sources of each query (Euclidean candidates). drop_self: q_idx IS src_idx (leave-one-out), the first hit is the point itself."""
    tree = cKDTree(pos[src_idx]); kk = min(K + int(drop_self), len(src_idx)); d, j = tree.query(pos[q_idx], k=kk)
    d = np.asarray(d).reshape(len(q_idx), -1); j = np.asarray(j).reshape(len(q_idx), -1)
    return (d[:, 1:], j[:, 1:]) if drop_self else (d, j)

def _feats(u, pos, S, q_idx, src_idx, d, j, ell, kind, raster_dir):
    """-> E (Nq,K,F) edge features, Q (Nq,Fq) query features. Positions, map features and the raster only; no label."""
    Nq, K = d.shape; ld = np.log(d / ell + 1e-3)
    F_ = [ld[..., None] / 2.0, np.broadcast_to((np.arange(K) / max(K, 1))[None, :, None], (Nq, K, 1))]; Q_ = [ld.mean(1, keepdims=True)]
    if kind in ("side", "wall") and S.shape[1]:
        Sq = S[q_idx]; Sj = S[src_idx][j]; F_ += [np.broadcast_to(Sq[:, None, :], (Nq, K, Sq.shape[1])), Sq[:, None, :] - Sj]; Q_.append(Sq)
    if kind == "wall":
        R = r7.load_raster(raster_dir, str(u["scenario"])); A = np.repeat(pos[q_idx], K, axis=0); B = pos[src_idx][j].reshape(-1, 2)
        bl = np.concatenate([r7.blocked_length(R, A[a:a + 4000], B[a:a + 4000]) for a in range(0, len(A), 4000)]).reshape(Nq, K)
        F_ += [np.log1p(bl / 10.0)[..., None], np.clip(bl / (d + 1e-6), 0.0, 1.0)[..., None]]
    return np.concatenate(F_, -1).astype(np.float32), np.concatenate(Q_, 1).astype(np.float32)

# ================================================================================================ torch side: the kernel
if HAVE_TORCH:
    class _Kern(nn.Module):
        def __init__(s, F_, Fq, h=16):
            super().__init__()
            s.w = nn.Sequential(nn.Linear(F_, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)); s.n = nn.Sequential(nn.Linear(Fq, h), nn.ReLU(), nn.Linear(h, 1))
            for m in (s.w[-1], s.n[-1]): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)            # start = IDW-like kernel, null slot ~ off
            s.null0 = nn.Parameter(torch.tensor(-3.0))
        def forward(s, E, base, Q, Rv):
            lg = base + s.w(E).squeeze(-1); ln = (s.null0 + s.n(Q).squeeze(-1))[:, None]
            W = torch.softmax(torch.cat([lg, ln], 1), 1); return (W[:, :-1] * Rv).sum(1)

    def fit_kernel(u, tr, prep, r_tr, hp, seed):
        torch.manual_seed(int(seed)); rng = np.random.default_rng(int(seed)); pos = np.asarray(u["pos"], float); tr = np.asarray(tr); n = len(tr)
        K = int(min(int(hp.get("K", 16)), n - 1)); kind = hp.get("edge", "side"); sig = float(prep["sigma"]); S = _side_z(u)
        ell = float(np.median(cKDTree(pos[tr]).query(pos[tr], k=2)[0][:, 1])) + 1e-9
        d, j = _nbr(pos, tr, tr, K, True); E, Q = _feats(u, pos, S, tr, tr, d, j, ell, kind, hp.get("raster_dir"))
        p0, _ = rc.tune_idw_loo(pos[tr], r_tr); base = (-float(p0) * np.log(d / ell + 1e-3)).astype(np.float32)
        y_n = (np.asarray(r_tr, float) / sig).astype(np.float32); Rv = y_n[j]
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)); Et, Qt, Bt, Rt, Yt = T(E), T(Q), T(base), T(Rv), T(y_n)
        net = _Kern(E.shape[2], Q.shape[1], int(hp.get("hidden", 16))); opt = torch.optim.AdamW(net.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-3)))
        perm = rng.permutation(n); nv = int(round(0.2 * n)) if n >= 30 else 0; vi, fi = torch.from_numpy(perm[:nv]), torch.from_numpy(perm[nv:])
        L = lambda idx: F.huber_loss(net(Et[idx], Bt[idx], Qt[idx], Rt[idx]), Yt[idx], delta=1.5)
        state = {k: t.clone() for k, t in net.state_dict().items()}; best, bad = float("inf"), 0
        if nv:
            net.eval()
            with torch.no_grad(): best = L(vi).item()
        for s in range(int(hp.get("steps", 200))):
            net.train(); opt.zero_grad(); L(fi).backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            if nv and s % 5 == 4:
                net.eval()
                with torch.no_grad(): v = L(vi).item()
                if v < best - 1e-6: best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10: break
        if nv: net.load_state_dict(state)
        net.eval(); return dict(net=net, ell=ell, p0=float(p0), K=K, kind=kind, S=S, sig=sig, raster_dir=hp.get("raster_dir"), n=n)

    def predict_kernel(M, u, tr, te, r_tr):
        pos = np.asarray(u["pos"], float); tr = np.asarray(tr); te = np.asarray(te); d, j = _nbr(pos, te, tr, M["K"], False)
        E, Q = _feats(u, pos, M["S"], te, tr, d, j, M["ell"], M["kind"], M["raster_dir"]); base = (-M["p0"] * np.log(d / M["ell"] + 1e-3)).astype(np.float32)
        Rv = (np.asarray(r_tr, float) / M["sig"])[j].astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        with torch.no_grad(): c = M["net"](T(E), T(base), T(Q), T(Rv)).numpy()
        return c.astype(float) * M["sig"]

# ================================================================================================ gated predictor
def _fit_predict(u, tr, te, prep, hp, seed):
    import rmbench_ext2 as r2
    rk, rk_info = r2.predict_rk_side(u, tr, te, prep, dict(fset="geo_side")); fit, r_tr, lam_r = _ridge_parts(prep, tr)
    M = fit_kernel(u, tr, prep, r_tr, hp, seed); c = predict_kernel(M, u, tr, te, r_tr)
    return rk, np.asarray(prep["trend"])[te] + fit[te] + c, dict(p0=M["p0"], ell=M["ell"])

def predict_rattn(u, tr, te, prep, hp, seed=0):
    if not HAVE_TORCH: raise RuntimeError("torch is required for rattn")
    tr = np.asarray(tr); te = np.asarray(te); n = len(tr); y = np.asarray(u["y"], float); info = {}; gate = hp.get("gate", True); lam = 1.0
    if gate:
        lam = 0.0
        if n >= int(hp.get("min_labels", 30)):
            K = int(hp.get("gate_folds", 5)); perm = np.random.default_rng(seed + 1).permutation(n); E = np.zeros((len(LAMS8), n))
            for f in np.array_split(np.arange(n), K):
                hold = perm[f]; keep = np.ones(n, bool); keep[hold] = False; trf, tef = tr[keep], tr[hold]
                rk, ra, _ = _fit_predict(u, trf, tef, prep, hp, seed)
                for i, l in enumerate(LAMS8): E[i, hold] = (rk + l * (ra - rk) - y[tef]) ** 2
            mse = E.mean(1); jj = int(np.argmin(mse)); dd = E[0] - E[jj]; z = float(dd.mean() / (dd.std(ddof=1) / math.sqrt(n) + 1e-12)); g = float(dd.mean() / (E[0].mean() + 1e-12))
            ok = jj > 0 and z > float(hp.get("z", 2.5)) and g > float(hp.get("rel", 0.01)); lam = float(LAMS8[jj]) if ok else 0.0
            info.update(cv_z=z, cv_gain_rel=g, lam_cv_best=float(LAMS8[jj]))
    rk, ra, fi = _fit_predict(u, tr, te, prep, hp, seed)
    info.update(lam=lam, gate_open=float(lam > 0), corr_rms=float(np.sqrt(np.mean((ra - rk) ** 2))), idw_p0=fi["p0"])
    return rk + lam * (ra - rk), info

# ================================================================================================ jobs / install / executors
def r8_jobs(raster_dir=None, diagnostics=True, **over):
    base = dict(fset="geo_side", K=16, hidden=16, steps=200, lr=1e-2, wd=1e-3, gate=True, z=2.5, rel=0.01, min_labels=30); base.update(over)
    J = [dict(name="rattn_geo", method="rattn", hp=dict(base, edge="geo")), dict(name="rattn_side", method="rattn", hp=dict(base, edge="side"))]
    if raster_dir: J.append(dict(name="rattn_wall", method="rattn", hp=dict(base, edge="wall", raster_dir=raster_dir)))
    if diagnostics: J += [dict(name=j["name"] + "_ng", method="rattn", hp=dict(j["hp"], gate=False)) for j in list(J)]
    return J

def install():
    r7.install(); x5.NEW_METHODS["rattn"] = predict_rattn

install()

def run_task8(task):
    install(); import rmbench_ext6 as x6; x6.install(); return x5.run_task5(task)          # NOT x6.run_task6: run_tasks8 patches that name (would recurse)

def run_tasks8(tasks, out_path, n_jobs=1, log_every=25):
    import rmbench_ext6 as x6; old = x6.run_task6; x6.run_task6 = run_task8
    try: return x6.run_tasks6(tasks, out_path, n_jobs=n_jobs, log_every=log_every)
    finally: x6.run_task6 = old

def leak_check8(jobs, unit, cfg, density=100, rep=0):
    """Predictions must be bit-identical when the TEST labels are corrupted."""
    install(); import rmbench_ext6 as x6; x6.install(); return x5.leak_check(jobs, unit, cfg, density=density, rep=rep)
