"""rmbench_ext4 -- fourth layer (round 3: a pre-specified test of the one model family that was ever ahead of IDW).

rmbench_core.py, rmbench_nn.py, rmbench_ext.py, rmbench_ext2.py, rmbench_ext3.py and run_ext.py stay UNTOUCHED; this file only adds.

Why this file exists
  Rounds 1-2: label-aware GCN / GAT lose to IDW; KGNN (kriging-initialised message passing = a one-layer kernel attention over the K nearest
  labelled nodes, whose "attention weights" are the solution of the kriging system) wins by 2-8 %, but (a) its kernel has 6 parameters that
  are learned from 5-100 labels of ONE unit, (b) kgnn_side never beat rk_side (ridge + IDW on the same covariates), (c) nobody tested a
  gradient-boosting baseline. This layer tests three concrete, falsifiable ideas:

    1. META-LEARNED KERNEL PRIOR ("kgnn_prior*"). The same KGNN kernel, but its parameters are pre-trained by episodic masked-label training on
       units of OTHER layouts (leave-one-layout-out / leave-a-group-of-layouts-out, exactly the protocol of the GNN transfer experiments) and
       then used zero-shot (steps=0) or fine-tuned for 30 steps with step 0 as an early-stopping candidate. Hypothesis: at 5-20 labels the
       kernel is not identifiable from the unit's own labels, so a prior from other buildings helps.
    2. PROPAGATION-AWARE KERNEL ("kgnn_prior_sk"). The kernel is multiplied by exp(-sum_k w_k |s_k(i) - s_k(j)|) where s are the standardised
       side-information covariates (walls crossed, LOS flag). Two points separated by a wall are less correlated. w is shared across layouts
       through the prior, so it is identifiable even when a single unit has 20 labels. (Product of valid kernels -> still positive definite.)
    3. REVIEWER BASELINES: gradient boosting on the same covariates + IDW on the out-of-fold residual ("gb_geo" / "gb_side"), and fixed
       50/50 ensembles ("ens_geo" = kgnn + idw, "ens_side" = kgnn_side + rk_side, "ens_sk" = kgnn_prior_sk + rk_side; nothing is fitted).

  Nothing here is guaranteed to win. The notebook pre-registers the contrasts and a result-to-wording table BEFORE any confirmatory run.

Everything returns full dBm predictions (trend included) as rc.predict_method does, and only ever indexes labels at the labelled set `tr`.
"""
from __future__ import annotations
import os, json, time, traceback
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import rmbench_core as rc, rmbench_ext as rx, rmbench_ext2 as r2, rmbench_ext3 as r3
from rmbench_nn import _sp, _isp, _norm, _split_val

# =============================================================================== A. kernel model (KGNN + optional side-information kernel)
class KGNNP(nn.Module):
    """Same parameterisation and forward as rmbench_nn.KGNN. With n_side = 0 it is numerically identical to KGNN (tested in the notebook);
    with n_side > 0 the whole covariance is multiplied by exp(-sum_k softplus(ws_k) |dS_k|)."""
    def __init__(self, polar=True, s2_init=0.3, n_side=0, w_init=0.1):
        super().__init__()
        self.polar, self.n_side = polar, int(n_side)
        self.l1 = nn.Parameter(torch.tensor(_isp(3.0))); self.s1 = nn.Parameter(torch.tensor(_isp(0.8)))
        self.lr = nn.Parameter(torch.tensor(_isp(3.0))); self.lt = nn.Parameter(torch.tensor(_isp(3.0)))
        self.s2 = nn.Parameter(torch.tensor(_isp(s2_init))); self.tau = nn.Parameter(torch.tensor(_isp(0.2)))
        if self.n_side: self.ws = nn.Parameter(torch.full((self.n_side,), _isp(w_init)))

    def cov(self, dxy, dr, dtan, dS=None):
        k = _sp(self.s1) ** 2 * torch.exp(-_norm(dxy) / _sp(self.l1))
        if self.polar:
            k = k + _sp(self.s2) ** 2 * torch.exp(-torch.sqrt((dr / _sp(self.lr)) ** 2 + (dtan / _sp(self.lt)) ** 2 + 1e-9))
        if dS is not None and self.n_side:
            k = k * torch.exp(-(dS.abs() * _sp(self.ws)).sum(-1))
        return k

    def forward(self, Pq, Pr, yr, scale, tx, K, Sq=None, Sr=None):
        import math
        B = Pq.shape[0]
        _, idx = torch.topk(torch.cdist(Pq, Pr), K, dim=1, largest=False)
        pj, pq, yj, txs = Pr[idx] / scale, Pq[:, None, :] / scale, yr[idx], tx / scale
        use_s = self.n_side > 0 and Sq is not None and Sr is not None
        def polar(a):
            v = a - txs; return _norm(v), torch.atan2(v[..., 1], v[..., 0])
        rj, phj = polar(pj); rq, phq = polar(pq)
        wrap = lambda x: torch.remainder(x + math.pi, 2 * math.pi) - math.pi
        sj = Sr[idx] if use_s else None
        c_t = self.cov(pj - pq, rj - rq, 0.5 * (rj + rq) * wrap(phj - phq), (sj - Sq[:, None, :]) if use_s else None)
        dph = wrap(phj[:, :, None] - phj[:, None, :])
        C = self.cov(pj[:, :, None, :] - pj[:, None, :, :], rj[:, :, None] - rj[:, None, :], 0.5 * (rj[:, :, None] + rj[:, None, :]) * dph,
                     (sj[:, :, None, :] - sj[:, None, :, :]) if use_s else None)
        A = torch.zeros(B, K + 1, K + 1)
        A[:, :K, :K] = C + (_sp(self.tau) ** 2 + 1e-4) * torch.eye(K); A[:, :K, K] = 1; A[:, K, :K] = 1
        sol = torch.linalg.solve(A, torch.cat([c_t, torch.ones(B, 1)], 1)[..., None]).squeeze(-1)
        return (sol[:, :K] * yj).sum(1)

def side_cols(u, prep):
    """Standardised side-information columns of prep['X'] (geo_side = geo columns first, then side columns). Empty array if there are none."""
    ng = rc.feature_matrix(u, "geo").shape[1]; X = prep["X"]
    return X[:, ng:] if X.shape[1] > ng else np.zeros((len(X), 0))

def _scale_of(Pt, tr):
    Ptr = Pt[tr]; k3 = min(3, len(tr) - 1)
    return float(torch.topk(torch.cdist(Ptr, Ptr), k3 + 1, largest=False).values[:, -1].median()) + 1e-6

def fit_predict_kgnnp(u, tr, te, prep, hp, seed, S=None):
    """hp: Kmax, steps, lr, wd, s2_init, side (bool: use side kernel), init_path (pre-trained state), polar. Mirrors rmbench_nn.fit_predict_kgnn,
    including step 0 (the starting kernel: untrained or the pre-trained prior) as an early-stopping candidate."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Pt = torch.tensor(u["pos"], dtype=torch.float32); tx = torch.tensor(u["tx_xy"], dtype=torch.float32)
    sig = prep["sigma"]; rn = torch.tensor(prep["res"] / sig, dtype=torch.float32)
    use_s = bool(hp.get("side")) and S is not None and S.shape[1] > 0
    St = torch.tensor(S, dtype=torch.float32) if use_s else None
    scale = _scale_of(Pt, tr); Kmax, steps, delta = int(hp.get("Kmax", 12)), int(hp.get("steps", 150)), 1.5
    m = KGNNP(polar=bool(hp.get("polar", True)), s2_init=float(hp.get("s2_init", 0.3)), n_side=St.shape[1] if use_s else 0)
    if hp.get("init_path"): m.load_state_dict(torch.load(hp["init_path"], map_location="cpu"))
    fit, val = _split_val(tr, rng)
    sl = (lambda ix: St[ix]) if use_s else (lambda ix: None)
    if steps > 0:
        opt = torch.optim.AdamW(m.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-3)))
        def val_loss():
            m.eval()
            with torch.no_grad():
                return F.huber_loss(m(Pt[val], Pt[fit], rn[fit], scale, tx, min(Kmax, len(fit)), sl(val), sl(fit)), rn[val], delta=delta).item()
        best, state = (val_loss(), {k: t.clone() for k, t in m.state_dict().items()}) if len(val) else (float("inf"), None)
        for ep in range(steps):
            m.train(); opt.zero_grad(); n = len(fit)
            q = int(np.clip(round(rng.uniform(0.1, 0.5) * n), 2, max(2, min(48, n - 3)))); pp = rng.permutation(fit); qi, ri = pp[:q], pp[q:]
            F.huber_loss(m(Pt[qi], Pt[ri], rn[ri], scale, tx, min(Kmax, len(ri)), sl(qi), sl(ri)), rn[qi], delta=delta).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
            if len(val) and ep % 10 == 9:
                v = val_loss()
                if v < best - 1e-5: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
        if state is not None: m.load_state_dict(state)
    m.eval(); Kf = min(Kmax, len(tr))
    with torch.no_grad():
        pred = np.concatenate([m(Pt[te[i:i + 1500]], Pt[tr], rn[tr], scale, tx, Kf, sl(te[i:i + 1500]), sl(tr)).numpy() for i in range(0, len(te), 1500)]) * sig
    info = dict(steps_run=steps, l1=float(_sp(m.l1).detach()), s2=float(_sp(m.s2).detach()), tau=float(_sp(m.tau).detach()))
    if use_s: info["w_side_mean"] = float(_sp(m.ws).detach().mean()); info["w_side_max"] = float(_sp(m.ws).detach().max())
    return pred, info

def predict_kgnnp(u, tr, te, prep, hp, seed):
    """hp.ridge: ridge(side covariates) first, kernel on the leftover residual (as kgnn_side). hp.side: side-information kernel."""
    if hp.get("ridge"):
        fit, r_tr, lam = r3._ridge_side(prep, tr); res2 = prep["res"] - fit
        prep2 = dict(prep, res=res2, sigma=float(res2[tr].std() + 1e-6))
        p, info = fit_predict_kgnnp(u, tr, te, prep2, hp, seed, side_cols(u, prep)); info = dict(info); info["rk_lambda"] = lam
        return prep["trend"][te] + fit[te] + p, info
    p, info = fit_predict_kgnnp(u, tr, te, prep, hp, seed, side_cols(u, prep)); return prep["trend"][te] + p, info

# =============================================================================== B. baselines
def predict_gb_idw(u, tr, te, prep, hp, seed):
    """Gradient boosting on the same standardised covariates as the GNNs / rk_side, then IDW (LOO power, k) of the OUT-OF-FOLD residual
    (in-sample residuals would be ~0 and leave nothing for IDW). Labels at tr only."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold
    X, res, P = prep["X"], prep["res"], u["pos"]; n = len(tr); Xt, yt = X[tr], res[tr]
    mk = lambda: GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.08, subsample=0.8, min_samples_leaf=max(2, n // 15), random_state=seed)
    oof = np.zeros(n)
    for a, b in KFold(min(5, n), shuffle=True, random_state=seed).split(Xt): oof[b] = mk().fit(Xt[a], yt[a]).predict(Xt[b])
    fte = mk().fit(Xt, yt).predict(X[te]); r_left = yt - oof; p, k = rc.tune_idw_loo(P[tr], r_left)
    return prep["trend"][te] + fte + rc.idw_predict(P[te], P[tr], r_left, p, k), dict(idw_power=p, idw_k=k)

def _avg(a, b, ia, ib): return 0.5 * (a + b), {**{f"a_{k}": v for k, v in ia.items() if isinstance(v, (int, float))}, "ens": 1.0}

# =============================================================================== C. dispatch (installed inside every worker)
def install():
    """Teach rc.predict_method: 'kgnnp', 'gb_idw', 'ens_geo' (kgnn + idw), 'ens_side' (kgnn_side + rk_side), 'ens_sk' (kgnnp + rk_side). Idempotent."""
    r3.install()
    if getattr(rc.predict_method, "_ext4", False): return
    orig = rc.predict_method
    def pm4(method, hp, u, tr, te, prep, seed=0, graph_cache=None):
        if method == "kgnnp": return predict_kgnnp(u, tr, te, prep, hp, seed)
        if method == "gb_idw": return predict_gb_idw(u, tr, te, prep, hp, seed)
        if method == "ens_geo":
            a, ia = orig("kgnn", dict(Kmax=12, s2_init=0.3), u, tr, te, prep, seed, graph_cache); b, _ = orig("idw", {}, u, tr, te, prep, seed, graph_cache)
            return _avg(a, b, ia, {})
        if method == "ens_side":
            a, ia = r3.predict_kgnn_side(u, tr, te, prep, dict(Kmax=12, s2_init=0.3), seed); b, _ = r2.predict_rk_side(u, tr, te, prep, hp)
            return _avg(a, b, ia, {})
        if method == "ens_sk":
            a, ia = predict_kgnnp(u, tr, te, prep, hp, seed); b, _ = r2.predict_rk_side(u, tr, te, prep, hp)
            return _avg(a, b, ia, {})
        return orig(method, hp, u, tr, te, prep, seed, graph_cache)
    pm4._ext4 = True; pm4._ext3 = True; rc.predict_method = pm4

def run_task_ext4(task):
    install(); return r2.run_task_ext2(task)

def run_tasks4(tasks, out_path, n_jobs=1, log_every=25):
    """Same resumable JSONL protocol as r3.run_tasks3, dispatching run_task_ext4."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks4] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs}) -> {out_path}")
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
                write(run_task_ext4(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task_ext4, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"],
                                                  status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

# =============================================================================== D. meta-learning the kernel across layouts
def _episode(u0, cfg, d, r, variant, side_fs="geo_side"):
    """One (unit, labelled set) -> tensors for pre-training. variant: 'geo' (kernel on the trend residual), 'rid' (ridge first, geo kernel),
    'sk' (ridge first, side kernel). Uses the robust trend exactly as evaluation does (r2.view_for_split2 / prepare2)."""
    tr, _ = r2.split_for2(u0, d, 500 + r, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
    u, spec = r2.view_for_split2(u0, tr); fs = "geo" if variant == "geo" else side_fs
    prep = r2.prepare2(u, tr, fs, cfg["clip"], spec); res, sig = prep["res"], prep["sigma"]
    if variant != "geo":
        fit, _, _ = r3._ridge_side(prep, tr); res = res - fit; sig = float(res[tr].std() + 1e-6)
    Pt = torch.tensor(u["pos"], dtype=torch.float32); S = side_cols(u, prep)
    return dict(Pt=Pt, tx=torch.tensor(u["tx_xy"], dtype=torch.float32), rn=torch.tensor(res / sig, dtype=torch.float32), tr=np.asarray(tr),
                scale=_scale_of(Pt, tr), S=torch.tensor(S, dtype=torch.float32) if (variant == "sk" and S.shape[1] > 0) else None)

def build_prior_episodes(items, variant, reps=2):
    """items: list of (unit_name, cfg). Episodes that fail (e.g. too few points) are skipped."""
    eps = []
    for name, cfg in items:
        u0 = rx.get_unit_ext(name, cfg["cache_dir"])
        for d in cfg["densities"]:
            for r in range(reps):
                try: eps.append(_episode(u0, cfg, d, r, variant))
                except Exception: continue
    return eps

def _sl(ep, ix): return None if ep["S"] is None else ep["S"][ix]

def _loss(m, ep, Q, V, Kmax=12):
    out = m(ep["Pt"][Q], ep["Pt"][V], ep["rn"][V], ep["scale"], ep["tx"], min(Kmax, len(V)), _sl(ep, Q), _sl(ep, V))
    return F.huber_loss(out, ep["rn"][Q], delta=1.5)

def pretrain_kernel(eps_train, eps_val, steps, seed, out_path, lr=5e-3, batch=4, wd=1e-4, eval_every=50, log=print):
    """Episodic masked-label pre-training of the kernel across many units. Step 0 (untrained kernel) is a candidate for the saved weights."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n_side = 0 if eps_train[0]["S"] is None else eps_train[0]["S"].shape[1]
    m = KGNNP(polar=True, s2_init=0.3, n_side=n_side); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    vsets = []
    for i, ep in enumerate(eps_val):
        r = np.random.default_rng(1000 + i); pp = r.permutation(ep["tr"]); q = max(2, len(pp) // 3); vsets.append((ep, pp[:q], pp[q:]))
    def vloss():
        m.eval()
        with torch.no_grad(): return float(np.mean([_loss(m, ep, Q, V).item() for ep, Q, V in vsets])) if vsets else float("nan")
    best, state, t0 = float("inf"), None, time.time()
    if vsets: best, state = vloss(), {k: t.clone() for k, t in m.state_dict().items()}; v0 = best
    for s in range(steps):
        m.train(); opt.zero_grad(); loss = 0.0
        for _ in range(batch):
            ep = eps_train[int(rng.integers(len(eps_train)))]; tr = ep["tr"]; n = len(tr)
            q = int(np.clip(round(rng.uniform(0.1, 0.5) * n), 2, max(2, min(48, n - 3)))); pp = rng.permutation(tr)
            loss = loss + _loss(m, ep, pp[:q], pp[q:]) / batch
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        if vsets and (s + 1) % eval_every == 0:
            v = vloss()
            if v < best - 1e-5: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
            if (s + 1) % (eval_every * 6) == 0: log(f"    prior {os.path.basename(out_path)} step {s + 1}/{steps} val={v:.4f} best={best:.4f} (step0 {v0:.4f}) {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(state if state is not None else m.state_dict(), out_path); return out_path

def plan_groups(members, group_size=1, n_src=40, seed=0):
    """members: {tag: fd-with-cfg}. Held-out groups of layouts (a layout = building / simulated layout). For every group: eval units (per member tag)
    and source units of ALL OTHER layouts (round-robin across layouts, <= n_src). Eval units are never in the source set."""
    items = []
    for tag, fd in members.items():
        for n in fd["names"]: items.append(dict(tag=tag, name=n, cfg=fd["cfg"], layout=rx.get_unit_ext(n, fd["cfg"]["cache_dir"])["layout"]))
    layouts = sorted({it["layout"] for it in items}, key=lambda L: rc.stable_hash(seed, "lay4", L)); plan = []
    for gi in range(0, len(layouts), group_size):
        G = layouts[gi:gi + group_size]; gid = f"g{gi // group_size}_" + "-".join(G) if group_size > 1 else G[0]
        by = {L: [it for it in items if it["layout"] == L] for L in layouts if L not in G}; src = []
        for i in range(max((len(v) for v in by.values()), default=0)):
            for v in by.values():
                if i < len(v) and len(src) < n_src: src.append(v[i])
        plan.append(dict(gid=gid, layouts=G, eval=[it for it in items if it["layout"] in G], source=src))
    return plan

def conf4_jobs(side, paths=None, finetune_steps=30):
    """Round-3 methods. paths: {variant: state path} of the prior trained WITHOUT the evaluated layouts (None -> no prior methods)."""
    J = rx.base_jobs()                                                                    # trend, idw, krig_global, kgnn (defaults)
    fs = "geo_side"
    if side:
        J += [dict(name="rk_side", method="rk_side", hp=dict(fset=fs)), dict(name="krig_side", method="krig_side", hp=dict(model="exp", fset=fs)),
              dict(name="kgnn_side", method="kgnn_side", hp=dict(Kmax=12, s2_init=0.3, fset=fs)), dict(name="gb_side", method="gb_idw", hp=dict(fset=fs)),
              dict(name="ens_side", method="ens_side", hp=dict(fset=fs))]
    else:
        J += [dict(name="gb_geo", method="gb_idw", hp=dict(fset="geo")), dict(name="ens_geo", method="ens_geo", hp=dict(fset="geo"))]
    if paths:
        g = paths.get("geo")
        if g: J += [dict(name="kgnn_prior", method="kgnnp", hp=dict(init_path=g, steps=0, fset="geo")),
                    dict(name="kgnn_priorft", method="kgnnp", hp=dict(init_path=g, steps=finetune_steps, lr=5e-3, fset="geo"))]
        if side and paths.get("rid"): J += [dict(name="kgnn_prior_rid", method="kgnnp", hp=dict(init_path=paths["rid"], steps=0, ridge=True, fset=fs))]
        if side and paths.get("sk"):
            J += [dict(name="kgnn_prior_sk", method="kgnnp", hp=dict(init_path=paths["sk"], steps=0, ridge=True, side=True, fset=fs)),
                  dict(name="kgnn_priorft_sk", method="kgnnp", hp=dict(init_path=paths["sk"], steps=finetune_steps, lr=5e-3, ridge=True, side=True, fset=fs)),
                  dict(name="ens_sk", method="ens_sk", hp=dict(init_path=paths["sk"], steps=0, ridge=True, side=True, fset=fs))]
    return J

def run_prior_family(members, split, root, PRE, n_jobs, name, group_size=1, steps=1500, n_src=40, with_prior=True, log=print):
    """members: {tag: fd}. Pre-trains the kernel prior per held-out group of layouts (skips existing files), then runs every member's eval units.
    Returns {tag: results DataFrame}. Resumable (priors are files, results are JSONL)."""
    cfgs = {t: r2.family_cfg(fd, split, PRE) for t, fd in members.items()}; mem = {t: dict(fd, cfg=cfgs[t]) for t, fd in members.items()}
    side_any = any(fd["side"] for fd in members.values()); plan = plan_groups(mem, group_size, n_src); log(f"[{name}|{split}] groups:", [(p["gid"], len(p["eval"]), len(p["source"])) for p in plan])
    tasks = {t: [] for t in members}; pdir = os.path.join(root, name, f"priors_{split}")
    for p in plan:
        paths = {}
        if with_prior and p["source"]:
            srcs = sorted(p["source"], key=lambda it: rc.stable_hash("val4", it["name"])); nv = max(1, len(srcs) // 7)
            for var in (["geo"] + (["rid", "sk"] if side_any else [])):
                path = os.path.join(pdir, f"{p['gid']}__{var}.pt")
                if not os.path.exists(path):
                    log(f"  pretraining prior {p['gid']} / {var} on {len(p['source'])} source units")
                    tr_e = build_prior_episodes([(it["name"], it["cfg"]) for it in srcs[nv:]], var); va_e = build_prior_episodes([(it["name"], it["cfg"]) for it in srcs[:nv]], var, reps=1)
                    if not tr_e: log("    no usable source episodes, skipped"); continue
                    pretrain_kernel(tr_e, va_e, steps, 0, path, log=log)
                paths[var] = path
        for t, fd in members.items():
            ev = [it["name"] for it in p["eval"] if it["tag"] == t]
            if ev: tasks[t] += r2.make_tasks2(ev, cfgs[t]["densities"], PRE["reps"], conf4_jobs(fd["side"], paths if with_prior else None), cfgs[t], prefix=f"{t}|{split}|{p['gid']}")
    out = {}
    for t in members:
        os.makedirs(os.path.join(root, t), exist_ok=True); path = os.path.join(root, t, f"{t}_{split}.jsonl"); run_tasks4(tasks[t], path, n_jobs); out[t] = rc.read_results(path)
    return out

# =============================================================================== E. pre-specified contrasts
SIDE_C = [("kgnn_prior_sk", "rk_side"),            # PRIMARY (side families): the whole proposal vs the strongest classical comparator
          ("kgnn_prior_sk", "kgnn_prior_rid"),     # side-information kernel, given the prior
          ("kgnn_prior_rid", "kgnn_side"),         # meta-learned prior, given the ridge step
          ("kgnn_prior_sk", "kgnn_side"),          # proposal vs the round-2 KGNN-side
          ("kgnn_priorft_sk", "kgnn_prior_sk"),    # does 30-step fine-tuning help
          ("ens_sk", "rk_side"), ("ens_side", "rk_side"),
          ("kgnn_prior_sk", "gb_side"), ("rk_side", "gb_side")]
GEO_C = [("kgnn_prior", "kgnn"),                   # PRIMARY (geo-only families): meta-learned prior vs the same kernel learned from scratch
         ("kgnn_priorft", "kgnn"), ("kgnn_prior", "idw"), ("kgnn_prior", "krig_global"),
         ("ens_geo", "kgnn"), ("kgnn_prior", "gb_geo"), ("kgnn", "gb_geo")]
PRIMARY = [("kgnn_prior_sk", "rk_side"), ("kgnn_prior", "kgnn")]

def evaluate_claims4(PCS, CLS, PRE):
    """r3.evaluate_claims restricted to contrasts whose two methods exist in the family (others would only say 'not run')."""
    c = r3.evaluate_claims(PCS, CLS, PRE)
    return c[c.status != "not run"].reset_index(drop=True) if len(c) else c
