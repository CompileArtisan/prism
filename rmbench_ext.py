"""rmbench_ext: extensions on top of rmbench_core / rmbench_nn (both left untouched).

Adds
  A. a REAL multi-building measurement dataset (UJIIndoorLoc: 3 buildings, 4-5 floors, 520 Wi-Fi APs) turned into radio-map "units";
  B. a harder simulated target for DeepMIMO ("narrowband" single-subcarrier power incl. small-scale fading) + NLOS scenarios;
  C. GAT (own implementation, edge-aware attention) and Delaunay vs kNN graphs, plus an IDW-anchored hybrid for both;
  D. coverage / outage metrics (accuracy, outage-F1, balanced accuracy, covered-area error);
  E. cross-layout generalisation: GNNs pre-trained on OTHER layouts (leave-one-layout-out) and applied zero-shot / lightly fine-tuned,
     next to a within-layout-pretrained reference, scratch-trained GNNs and the classical interpolators;
  F. an executor (own run_task_ext, resumable JSONL) because the original run_task hard-codes the method list and metrics.

Leak rules kept from rmbench_core: models only see labels at `tr`; trend, features, standardisation, IDW parameters and (for real data,
where transmitters are unknown) the *estimated transmitter position* are all fitted from the labeled set only.
"""
from __future__ import annotations
import os, io, json, time, math, glob, zipfile, traceback, urllib.request
import numpy as np, pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import least_squares
import torch, torch.nn as nn, torch.nn.functional as F
import rmbench_core as rc
from rmbench_nn import GCNNet, label_diffusion, idw_torch, knn_edges, _split_val

H0 = 3.0                      # vertical offset (m) used to turn a 2-D distance into a pseudo-3-D distance when the transmitter height is unknown

# =============================================================================== configs
UJI_CFG = dict(rc.DEFAULT_CFG, cache_dir="rmbench_cache_uji", densities=(15, 30, 60), n_rep=5, n_test=40, min_unit_points=110,
               min_det_frac=0.5, uji_max_waps_per_floor=15, outage_q=(0.1, 0.25), outage_abs_dbm=(-85.0,))
DM_HARD_CFG = dict(rc.DEFAULT_CFG, cache_dir="rmbench_cache_dm_nb", power_def="narrowband", outage_q=(0.1, 0.25),
                   outage_abs_dbm=(-77.0,))          # -77 dBm = noise floor (-87) + 10 dB margin

# =============================================================================== A. real data: UJIIndoorLoc
UJI_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"

def fetch_uji(dest="data/ujiindoorloc"):
    """Return the folder holding trainingData.csv / validationData.csv. Downloads the UCI zip if they are not there yet
    (CC BY 4.0). If the download fails, get the zip from https://archive.ics.uci.edu/dataset/310 by hand and unzip into `dest`."""
    def find():
        hits = {os.path.basename(p).lower(): p for p in glob.glob(os.path.join(dest, "**", "*.csv"), recursive=True)}
        return hits.get("trainingdata.csv"), hits.get("validationdata.csv")
    tr, va = find()
    if tr is None:
        os.makedirs(dest, exist_ok=True)
        print("downloading", UJI_URL)
        with urllib.request.urlopen(UJI_URL, timeout=120) as r: zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
        tr, va = find()
    if tr is None: raise FileNotFoundError(f"trainingData.csv not found under {dest}")
    return os.path.dirname(tr)

def build_uji_units(cfg, csv_dir):
    """UJIIndoorLoc -> units. One unit = one Wi-Fi AP on one floor of one building (a transmitter's radio map).
    Layout = building; parent = (building, floor) (APs on one floor share the same reference points, so the PARENT is the
    statistical unit, never the AP). Repeated captures at the same reference point (many users / 25 phones) are collapsed to the
    median RSSI of the detections; a point counts as detected only if >= min_det_frac of its captures saw the AP.
    LIMITS: AP positions are unknown (estimated per task from the labeled set only, see localize_tx); RSSI 100 = not detected ->
    only detected points are in a unit (censored data: coverage claims are conditional on detection); no LOS flag (use fset='geo')."""
    frames = [pd.read_csv(os.path.join(csv_dir, f)) for f in ("trainingData.csv", "validationData.csv") if os.path.exists(os.path.join(csv_dir, f))]
    df = pd.concat(frames, ignore_index=True)
    waps = [c for c in df.columns if c.startswith("WAP")]
    ok, dropped = [], []
    for (b, f), g in df.groupby(["BUILDINGID", "FLOOR"]):
        key = pd.MultiIndex.from_arrays([g.LONGITUDE.round(1), g.LATITUDE.round(1)]).factorize()[0]
        R = g[waps].where(g[waps] != 100)                                   # 100 = not detected
        med = R.groupby(key).median(); frac = R.notna().groupby(key).mean()
        med = med.where(frac >= cfg["min_det_frac"])
        xy = g[["LONGITUDE", "LATITUDE"]].groupby(key).mean().to_numpy(float); xy = xy - xy.mean(0)
        elig = [w for w in waps if med[w].notna().sum() >= cfg["min_unit_points"]]
        elig = sorted(elig, key=lambda w: rc.stable_hash(cfg["seed"], b, f, w))[:cfg["uji_max_waps_per_floor"]]
        for w in elig:
            m = med[w].notna().to_numpy(); pos, y = xy[m], med[w].to_numpy(float)[m]
            t = pos[int(np.argmax(y))]; rel = pos - t
            u = dict(name=f"uji|b{b}f{f}|{w}", scenario="uji", layout=f"uji_b{b}", parent=f"uji|b{b}f{f}", tx=int(w[3:]), freq_ghz=2.4,
                     pos=pos, tx_xy=t, d=np.hypot(np.linalg.norm(rel, axis=1), H0), az=np.arctan2(rel[:, 1], rel[:, 0]),
                     los=np.zeros(len(y)), y=y, offset_db=0.0, frac_kept=float(m.mean()), tx_known=0)
            rc.save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
        dropped += [(f"uji|b{b}f{f}|{w}", "too few detected reference points") for w in waps if w not in elig and med[w].notna().sum() >= cfg["min_unit_points"]]
    return ok, dropped

def make_fake_uji(dest, seed=0, floors=(2, 2, 2), n_waps=24):
    """Synthetic file pair with the UJIIndoorLoc schema (for dry runs when the real data is not available). NOT real data."""
    rng = np.random.default_rng(seed); os.makedirs(dest, exist_ok=True)
    rows = []
    for b, nf in enumerate(floors):
        for f in range(nf):
            wap_xy = rng.uniform([0, 0], [60, 40], (n_waps, 2)); wall = rng.uniform(0, 12, n_waps)
            gx, gy = np.meshgrid(np.arange(0, 60, 4.0), np.arange(0, 40, 4.0)); P = np.column_stack([gx.ravel(), gy.ravel()])
            shadow = rng.normal(0, 3, (len(P), n_waps))
            for rep in range(3):
                for i, p in enumerate(P):
                    d = np.hypot(np.linalg.norm(wap_xy - p, axis=1), 3.0)
                    r = -30 - 25 * np.log10(d) - wall + shadow[i] + rng.normal(0, 2, n_waps) + rng.normal(0, 3)
                    r = np.where(r > -95, np.round(r), 100)
                    rows.append([*r, -7500 + 100 * b + p[0], 4864900 + 100 * b + p[1], f, b, 0, 0, rep, rep, 0])
    cols = [f"WAP{k + 1:03d}" for k in range(n_waps)] + ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
    df = pd.DataFrame(rows, columns=cols); df.to_csv(os.path.join(dest, "trainingData.csv"), index=False)
    df.sample(frac=0.05, random_state=seed).to_csv(os.path.join(dest, "validationData.csv"), index=False)
    return dest

# =============================================================================== B. harder DeepMIMO target
def _narrowband_power_dbm(ds):
    """Single antenna pair, first subcarrier: |H|^2 INCLUDING small-scale fading (the channel variant averages over antennas and
    subcarriers, which smooths it). Shifted by the median gap to the path-power sum so the noise-floor test stays on the dBm scale.
    ASSUMES compute_channels() -> (N, rx_ant, tx_ant, subcarriers), as rc._channel_power_dbm does. Untested on real DeepMIMO here."""
    ch = np.asarray(ds.compute_channels())
    if ch.ndim != 4: raise ValueError(f"compute_channels() returned shape {ch.shape}; expected 4-D")
    lin = np.abs(ch[:, 0, 0, 0]) ** 2
    db = np.full(len(lin), np.nan); k = lin > 0; db[k] = 10 * np.log10(lin[k])
    try:
        ref = rc.sum_path_power_dbm(np.asarray(ds.power, float)); m = np.isfinite(ref) & np.isfinite(db) & (ref > -250)
        off = float(np.median(ref[m] - db[m])) if m.sum() >= 20 else 0.0
    except Exception:
        off = 0.0
    return db + off

def load_dm_raw(scenario, cfg, seed=0):
    mode = cfg.get("power_def", "path_sum")
    if mode != "narrowband": return rc.load_deepmimo_units(scenario, cfg, seed)
    out = []
    for k, ds in enumerate(rc._dm_datasets(scenario, cfg, seed)):
        out.append(dict(pos3=np.asarray(ds.rx_pos), tx3=np.asarray(ds.tx_pos).reshape(-1)[:3], power_paths_dbm=_narrowband_power_dbm(ds)[:, None],
                        los=np.asarray(ds.los), freq_ghz=rc._carrier_ghz(ds, scenario), name=f"dm|{scenario}|{k}", scenario=scenario, tx=k))
    return out

def build_dm_units(scenarios, cfg):
    """Each scenario is tried separately; a slug the installed DeepMIMO does not know is reported, not fatal."""
    ok, dropped = [], []
    for s in scenarios:
        try: raws = load_dm_raw(s, cfg)
        except Exception as e:
            dropped.append((s, repr(e)[:200])); continue
        for r in raws:
            u = rc.preprocess_unit(r, cfg)
            if u is None: dropped.append((r["name"], "below noise floor / too few points")); continue
            u.update(layout=s, parent=u["name"], tx_known=1)
            rc.save_unit(u, cfg["cache_dir"]); ok.append(u["name"])
    return ok, dropped

def get_unit_ext(name, cache_dir):
    u = dict(rc.get_unit(name, cache_dir))
    u["layout"] = str(u["layout"]) if "layout" in u else str(u["scenario"])
    u["parent"] = str(u["parent"]) if "parent" in u else str(u["name"])
    u["tx_known"] = int(u["tx_known"]) if "tx_known" in u else 1
    return u

# =============================================================================== transmitter localisation (unknown-TX data), labels in `tr` only
def localize_tx(pos, y, h0=H0, n_max=6.0):
    """Fit y = a - 10 n log10(sqrt(|p - t|^2 + h0^2)) for (t_x, t_y, a, n) by robust least squares from the labeled points only.
    Falls back to the strongest labeled point if the fit does not beat a constant by >=10% in squared error."""
    lo, hi = pos.min(0), pos.max(0); pad = 0.3 * (hi - lo + 1.0)
    sse0 = float(((y - y.mean()) ** 2).sum()); best = (sse0 * 0.9, None)
    order = np.argsort(y)
    for s in (pos[order[-1]], pos[order[-3:]].mean(0)):
        try:
            fun = lambda th: th[2] - 10 * th[3] * np.log10(np.sqrt(((pos - th[:2]) ** 2).sum(1) + h0 ** 2)) - y
            r = least_squares(fun, [s[0], s[1], y.max() + 25 * np.log10(h0), 2.5],
                              bounds=([lo[0] - pad[0], lo[1] - pad[1], -300, 0.0], [hi[0] + pad[0], hi[1] + pad[1], 100, n_max]),
                              loss="soft_l1", f_scale=3.0, max_nfev=200)
            sse = float((fun(r.x) ** 2).sum())
            if sse < best[0]: best = (sse, r.x[:2].copy())
        except Exception:
            pass
    return best[1] if best[1] is not None else pos[order[-1]].copy()

def view_for_split(u, tr):
    """Unit as seen by the models for labeled set `tr`. Known transmitter: unchanged. Unknown: tx/d/az re-derived from tr labels only."""
    if u["tx_known"]: return u
    t = localize_tx(u["pos"][tr], u["y"][tr]); rel = u["pos"] - t
    return {**u, "tx_xy": t, "d": np.hypot(np.linalg.norm(rel, axis=1), H0), "az": np.arctan2(rel[:, 1], rel[:, 0])}

# =============================================================================== C. graphs, GAT, hybrid GNNs
def _norm_adj(N, a, b):
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N)).tocsr()
    A = A.maximum(A.T).tolil(); A.setdiag(1.0); A = A.tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    return A.row.astype(np.int64), A.col.astype(np.int64), (A.data / np.sqrt(deg[A.row] * deg[A.col])).astype(np.float32)

def delaunay_graph(pos, prune=4.0):
    """Delaunay triangulation edges; edges longer than prune x median edge length are dropped (they bridge concavities / walls).
    Falls back to kNN(6) if the points are degenerate (collinear, <4 points)."""
    try:
        S = Delaunay(pos).simplices
    except Exception:
        return knn_edges(pos, 6)
    pr = np.vstack([S[:, [0, 1]], S[:, [1, 2]], S[:, [0, 2]]]); L = np.linalg.norm(pos[pr[:, 0]] - pos[pr[:, 1]], axis=1)
    pr = pr[L <= prune * np.median(L)]
    return _norm_adj(len(pos), pr[:, 0], pr[:, 1])

def get_graph2(u, kind, k, cache):
    """(row, col, w, E) torch tensors. row = destination node, col = source node. E = [dist, dx, dy, is_self] / median edge length."""
    key = (u["name"], kind, k)
    if cache is not None and key in cache: return cache[key]
    pos = u["pos"]
    r, c, w = delaunay_graph(pos) if kind == "delaunay" else knn_edges(pos, k)
    dxy = pos[c] - pos[r]; dist = np.linalg.norm(dxy, axis=1); self_ = (r == c)
    sc = float(np.median(dist[~self_])) if (~self_).any() else 1.0
    E = np.column_stack([dist / sc, dxy / sc, self_.astype(float)]).astype(np.float32)
    g = (torch.from_numpy(r), torch.from_numpy(c), torch.from_numpy(w), torch.from_numpy(E))
    if cache is not None: cache[key] = g
    return g

class GCNWrap(GCNNet):
    def forward(self, X, ym, m, g, base=None): return super().forward(X, ym, m, g[:3], base)

class GATLayer(nn.Module):
    """Multi-head graph attention with edge features. logit_ij = LeakyReLU(a_dst.z_i + a_src.z_j + w_e.[geometry_ij, labeled_j, residual_j]),
    softmax over each node's incoming edges. The edge term is what makes the weighting propagation-aware (distance, direction, and
    whether / how much the neighbour is a measured point) instead of purely feature-similarity."""
    def __init__(self, din, dout, heads, edim):
        super().__init__(); self.H, self.dh = heads, dout // heads
        self.W = nn.Linear(din, heads * self.dh, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1); self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.we = nn.Linear(edim, heads, bias=False); self.b = nn.Parameter(torch.zeros(dout))
    def forward(self, h, row, col, ef):
        N = h.shape[0]; z = self.W(h).view(N, self.H, self.dh)
        e = F.leaky_relu((z[row] * self.a_dst).sum(-1) + (z[col] * self.a_src).sum(-1) + self.we(ef), 0.2)       # (E, H)
        mx = torch.full((N, self.H), -1e30).scatter_reduce(0, row[:, None].expand(-1, self.H), e, "amax", include_self=True)
        ex = torch.exp(e - mx[row]); den = torch.zeros(N, self.H).index_add_(0, row, ex)
        a = ex / (den[row] + 1e-9)
        return torch.zeros(N, self.H, self.dh).index_add_(0, row, a[..., None] * z[col]).reshape(N, -1) + self.b

class GATNet(nn.Module):
    def __init__(self, in_dim, hidden=32, layers=2, dropout=0.1, heads=4, edim=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden); self.gats = nn.ModuleList([GATLayer(hidden, hidden, heads, edim) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias); self.p = dropout
    def forward(self, X, ym, m, g, base=None):
        row, col, w, E = g
        parts = [X, ym[:, None], m[:, None], label_diffusion(ym, m, (row, col, w))]
        if base is not None: parts.append(base[:, None])
        h = F.relu(self.inp(torch.cat(parts, 1)))
        ef = torch.cat([E, m[col, None], ym[col, None]], 1)
        for gl in self.gats: h = h + F.dropout(F.elu(gl(h, row, col, ef)), self.p, self.training)
        out = self.out(h).squeeze(-1)
        return out if base is None else base + out

def build_model(arch, in_dim, hp):
    h, L, p = int(hp.get("hidden", 32)), int(hp.get("layers", 2)), float(hp.get("dropout", 0.1))
    return GATNet(in_dim, h, L, p, int(hp.get("heads", 4))) if arch == "gat" else GCNWrap(in_dim, h, L, p)

def make_episode(u, tr, prep, hp, cache):
    N = len(u["y"]); hyb = bool(hp.get("hybrid", False))
    X = torch.tensor(prep["X"], dtype=torch.float32)
    rn = torch.zeros(N); rn[tr] = torch.tensor(prep["res"][tr] / prep["sigma"], dtype=torch.float32)       # test residuals never enter
    ep = dict(N=N, X=X, rn=rn, tr=np.asarray(tr), g=get_graph2(u, hp.get("graph", "knn"), int(hp.get("k", 8)), cache))
    if hyb: ep["Pt"] = torch.tensor(u["pos"], dtype=torch.float32); ep["idw"] = rc.tune_idw_loo(u["pos"][tr], prep["res"][tr])
    return ep

def _forward(model, ep, visible, hyb, train):
    m = torch.zeros(ep["N"]); m[visible] = 1.0; model.train(train); base = None
    if hyb:
        vt = torch.as_tensor(np.asarray(visible), dtype=torch.long); p, k = ep["idw"]
        base = idw_torch(ep["Pt"], vt, ep["rn"][vt], p, k)
    return model(ep["X"], ep["rn"] * m, m, ep["g"], base)

def _train_step(model, opt, ep, rng, hyb):
    tr = ep["tr"]; q = int(np.clip(round(rng.uniform(0.15, 0.5) * len(tr)), 1, max(1, len(tr) - 2)))
    pp = rng.permutation(tr); Q, V = pp[:q], pp[q:]
    opt.zero_grad(); F.huber_loss(_forward(model, ep, V, hyb, True)[Q], ep["rn"][Q], delta=1.5).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()

def _in_dim(ep, hyb): return ep["X"].shape[1] + 2 + 6 + int(hyb)

def fit_predict_graph(u, tr, te, prep, hp, seed, cache):
    """hp: arch gcn|gat, graph knn|delaunay, k, hidden, layers, heads, dropout, lr, wd, steps, hybrid (IDW-anchored), init_path
    (pre-trained weights; steps=0 -> zero-shot, steps>0 -> fine-tune with step 0 as an early-stopping candidate)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    hyb = bool(hp.get("hybrid", False)); init = hp.get("init_path")
    ep = make_episode(u, tr, prep, hp, cache); model = build_model(hp.get("arch", "gcn"), _in_dim(ep, hyb), hp)
    if init: model.load_state_dict(torch.load(init, map_location="cpu"))
    steps = int(hp.get("steps", 150)); fit, val = _split_val(tr, rng); epf = {**ep, "tr": fit}
    if steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-4)))
        best, state, bad = float("inf"), None, 0
        if (hyb or init) and len(val):
            with torch.no_grad(): best = F.huber_loss(_forward(model, ep, fit, hyb, False)[val], ep["rn"][val], delta=1.5).item()
            state = {k: t.clone() for k, t in model.state_dict().items()}
        for s in range(steps):
            _train_step(model, opt, epf, rng, hyb)
            if len(val) and s % 5 == 4:
                with torch.no_grad(): v = F.huber_loss(_forward(model, ep, fit, hyb, False)[val], ep["rn"][val], delta=1.5).item()
                if v < best - 1e-5: best, state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8: break
        if state is not None: model.load_state_dict(state)
    with torch.no_grad(): pred = _forward(model, ep, tr, hyb, False)[te].numpy() * prep["sigma"]
    return pred, dict(steps_run=steps, n_graph_edges=int(ep["g"][0].numel()))

def pretrain_graph(eps_train, eps_val, hp, steps, seed, out_path, eval_every=50, log=print):
    """Episodic masked-label pre-training across many units. Each step: pick an episode (one unit x one labeled set), hide a random
    part of its LABELED nodes, predict them. Validation uses held-out source units, again only their labeled nodes (never test points).
    Returns out_path (best-validation weights)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); hyb = bool(hp.get("hybrid", False))
    model = build_model(hp.get("arch", "gcn"), _in_dim(eps_train[0], hyb), hp)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 5e-3)), weight_decay=float(hp.get("wd", 1e-4)))
    vsets = []
    for i, ep in enumerate(eps_val):
        r = np.random.default_rng(1000 + i); pp = r.permutation(ep["tr"]); q = max(1, len(pp) // 3); vsets.append((ep, pp[:q], pp[q:]))
    def vloss():
        with torch.no_grad():
            return float(np.mean([F.huber_loss(_forward(model, ep, V, hyb, False)[Q], ep["rn"][Q], delta=1.5).item() for ep, Q, V in vsets])) if vsets else float("nan")
    best, state, t0 = float("inf"), None, time.time()
    if vsets: best, state = vloss(), {k: t.clone() for k, t in model.state_dict().items()}
    for s in range(steps):
        _train_step(model, opt, eps_train[int(rng.integers(len(eps_train)))], rng, hyb)
        if vsets and (s + 1) % eval_every == 0:
            v = vloss()
            if v < best - 1e-5: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
            if (s + 1) % (eval_every * 10) == 0: log(f"    pretrain {os.path.basename(out_path)} step {s + 1}/{steps} val={v:.4f} best={best:.4f} {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(state if state is not None else model.state_dict(), out_path)
    return out_path

# =============================================================================== D. coverage / outage metrics
def outage_metrics(pred, true, thr):
    """Outage = power below thr. Returns accuracy, F1 of the outage class (NaN if the test set has no outage AND none is predicted),
    balanced accuracy, and |predicted outage fraction - true outage fraction| (covered-area error)."""
    po, to = pred < thr, true < thr
    tp, fp, fn, tn = int((po & to).sum()), int((po & ~to).sum()), int((~po & to).sum()), int((~po & ~to).sum())
    den = 2 * tp + fp + fn
    tpr = tp / (tp + fn) if tp + fn else np.nan; tnr = tn / (tn + fp) if tn + fp else np.nan
    return dict(acc=(tp + tn) / len(true), f1=(2 * tp / den) if den else np.nan, bacc=np.nanmean([tpr, tnr]), area=abs(po.mean() - to.mean()))

def cov_metrics(pred, true, y_train, cfg):
    """Thresholds: quantiles of the TRAIN labels (leak-free, adapts to each unit's dynamic range) and optional absolute dBm values."""
    out = {}
    th = {f"q{int(round(100 * q))}": float(np.quantile(y_train, q)) for q in cfg.get("outage_q", ())}
    th.update({f"abs{int(round(-a))}": float(a) for a in cfg.get("outage_abs_dbm", ())})
    for tag, t in th.items():
        for k, v in outage_metrics(pred, true, t).items(): out[f"{k}_{tag}"] = float(v)
    return out

# =============================================================================== F. executor
_GRAPHS = {}

def run_task_ext(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1"); torch.set_num_threads(1)
    cfg, rows = task["cfg"], []
    try:
        u0 = get_unit_ext(task["unit"], cfg["cache_dir"])
        tr, te = rc.split_for(u0, task["density"], task["rep"], cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0))
        u = view_for_split(u0, tr)
    except Exception:
        return [dict(task_id=task["id"], unit=task["unit"], density=task["density"], rep=task["rep"], name=j["name"], method=j["method"],
                     status="fail", err=traceback.format_exc(limit=3)) for j in task["jobs"]]
    preps = {}
    for j in task["jobs"]:
        hp = j.get("hp", {})
        row = dict(task_id=task["id"], unit=u["name"], scenario=u["scenario"], layout=u["layout"], parent=u["parent"], density=task["density"],
                   rep=task["rep"], name=j["name"], method=j["method"], status="ok", err="", hp=json.dumps(hp, sort_keys=True, default=str))
        t1 = time.time()
        try:
            fs = hp.get("fset", "geo")
            if fs not in preps: preps[fs] = rc.prepare(u, tr, fs, cfg["clip"], cfg.get("trend_mode", "fit"))
            prep = preps[fs]; seed = rc.stable_hash(u["name"], task["rep"], j["name"]) % 100000
            if j["method"] == "gnn":
                p, info = fit_predict_graph(u, tr, te, prep, hp, seed, _GRAPHS); pred = prep["trend"][te] + p
            else:
                pred, info = rc.predict_method(j["method"], hp, u, tr, te, prep, seed, _GRAPHS)
            e = pred - u["y"][te]
            if not np.all(np.isfinite(e)): raise FloatingPointError("non-finite prediction")
            row.update(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), n_test=len(te), tx_known=u0["tx_known"],
                       **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in info.items()},
                       **cov_metrics(pred, u["y"][te], u["y"][tr], cfg))
        except Exception:
            row.update(status="fail", err=traceback.format_exc(limit=4), rmse=np.nan, mae=np.nan)
        row["seconds"] = time.time() - t1; rows.append(row)
    return rows

def run_tasks_ext(tasks, out_path, n_jobs=1, log_every=25):
    """Same resumable JSONL protocol as rc.run_tasks, but dispatching run_task_ext."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = rc.done_task_ids(out_path); todo = [t for t in tasks if t["id"] not in done]
    print(f"[run_tasks_ext] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run (n_jobs={n_jobs})")
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
                write(run_task_ext(t), fh)
                if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=rc._init_worker) as ex:
                futs = {ex.submit(run_task_ext, t): t for t in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try: rows = f.result()
                    except Exception:
                        t = futs[f]; rows = [dict(task_id=t["id"], unit=t["unit"], density=t["density"], rep=t["rep"], name=j["name"], method=j["method"],
                                                  status="fail", err=traceback.format_exc(limit=2)) for j in t["jobs"]]
                    write(rows, fh)
                    if i % log_every == 0 or i == len(todo): print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s failed jobs: {nf}")

# =============================================================================== job lists
GRAPH_JOBS = {          # topology + architecture grid (all trained from scratch per task, label-aware, same inputs)
    "gcn_knn": dict(arch="gcn", graph="knn", k=8), "gcn_del": dict(arch="gcn", graph="delaunay"),
    "gat_knn": dict(arch="gat", graph="knn", k=8), "gat_del": dict(arch="gat", graph="delaunay"),
    "gat_del_idw": dict(arch="gat", graph="delaunay", hybrid=True)}

def base_jobs(with_kgnn=True):
    J = [dict(name="trend", method="trend"), dict(name="idw", method="idw"), dict(name="krig_global", method="krig_global", hp=dict(model="exp"))]
    if with_kgnn: J += [dict(name="kgnn", method="kgnn", hp=dict(Kmax=12, s2_init=0.3))]
    return J

def graph_jobs(names=None, steps=150):
    return [dict(name=n, method="gnn", hp={**hp, "steps": steps}) for n, hp in GRAPH_JOBS.items() if names is None or n in names]

# =============================================================================== E. cross-layout (leave-one-layout-out)
def layouts_of(names, cache_dir):
    by = {}
    for n in sorted(names, key=lambda s: rc.stable_hash(0, s)): by.setdefault(get_unit_ext(n, cache_dir)["layout"], []).append(n)
    return by

def lolo_plan(names, cache_dir, n_eval=20, n_src=40, min_within=6):
    """For every layout L: eval units (n_eval of L), 'cross' source = units of all OTHER layouts (round-robin, <= n_src),
    'within' source = remaining units of L (<= n_src; skipped if fewer than min_within). Eval units are never in any source."""
    by = layouts_of(names, cache_dir); plan = []
    for L, ns in by.items():
        ev = ns[:n_eval]; within = ns[n_eval:n_eval + n_src]
        others = [v for k, v in by.items() if k != L]; cross = []
        for i in range(max((len(v) for v in others), default=0)):
            for v in others:
                if i < len(v) and len(cross) < n_src: cross.append(v[i])
        plan.append(dict(layout=L, eval=ev, cross=cross, within=within if len(within) >= min_within else []))
    return plan

def build_episodes(names, cfg, hp, reps=2):
    eps, cache = [], {}
    for n in names:
        u0 = get_unit_ext(n, cfg["cache_dir"])
        for d in cfg["densities"]:
            for r in range(reps):
                try:
                    tr, _ = rc.split_for(u0, d, r, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0))
                    u = view_for_split(u0, tr); prep = rc.prepare(u, tr, hp.get("fset", "geo"), cfg["clip"], cfg.get("trend_mode", "fit"))
                    eps.append(make_episode(u, tr, prep, hp, cache))
                except Exception:
                    continue
    return eps

def pretrain_for_plan(plan, cfg, pre_cfgs, out_dir, steps=1500, seed=0):
    """pre_cfgs: {tag: hp dict with arch/graph/...}. Writes out_dir/<layout>__<scope>__<tag>.pt (skips existing). Returns {(layout, scope, tag): path}."""
    paths = {}
    for item in plan:
        for scope in ("cross", "within"):
            src = item[scope]
            if not src: continue
            srcs = sorted(src, key=lambda s: rc.stable_hash(seed, "val", s)); nv = max(1, len(srcs) // 7)
            for tag, hp in pre_cfgs.items():
                path = os.path.join(out_dir, f"{item['layout']}__{scope}__{tag}.pt"); paths[(item["layout"], scope, tag)] = path
                if os.path.exists(path): continue
                print(f"  pretraining {item['layout']} / {scope} / {tag} on {len(src)} source units")
                tr_eps = build_episodes(srcs[nv:], cfg, hp); va_eps = build_episodes(srcs[:nv], cfg, hp, reps=1)
                if not tr_eps: print("    no usable source episodes, skipped"); paths.pop((item["layout"], scope, tag)); continue
                pretrain_graph(tr_eps, va_eps, hp, steps, seed, path)
    return paths

def lolo_tasks(plan, paths, cfg, pre_cfgs, finetune_steps=30, scratch=("gcn_knn", "gat_del"), prefix="lolo"):
    tasks = []
    for item in plan:
        jobs = base_jobs() + graph_jobs(scratch)
        for tag, hp in pre_cfgs.items():
            for scope, nm in (("cross", "xfer"), ("within", "within")):
                p = paths.get((item["layout"], scope, tag))
                if p: jobs.append(dict(name=f"{nm}_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": 0}))
            p = paths.get((item["layout"], "cross", tag))
            if p: jobs.append(dict(name=f"xferft_{tag}", method="gnn", hp={**hp, "init_path": p, "steps": finetune_steps}))
        tasks += rc.make_tasks(item["eval"], cfg["densities"], cfg["n_rep"], jobs, cfg, prefix=f"{prefix}|{item['layout']}")
    return tasks

# =============================================================================== analysis helpers
def pool_all(df):
    d2 = df.copy(); d2["scenario"] = "ALL"; return pd.concat([df, d2], ignore_index=True)

def by_layout(df):
    """Per-layout analysis (scenario := layout) plus a pooled 'ALL' block."""
    d = df.copy(); d["scenario"] = d["layout"]; return pool_all(d)

def agg_metric(df, col, by=("scenario", "density")):
    """Replicates -> unit -> parent (floor / unit) -> mean over parents. One number per (by, method)."""
    ok = df[(df.status == "ok")].dropna(subset=[col])
    u = ok.groupby(["scenario", "parent", "unit", "density", "name"], as_index=False)[col].mean()
    p = u.groupby(["scenario", "parent", "density", "name"], as_index=False)[col].mean()
    return p.groupby(list(by) + ["name"], as_index=False)[col].mean().pivot_table(index=list(by), columns="name", values=col)

def paired_metric(df, col, contrasts, higher_is_better=False, max_tests=60):
    """Paired tests across PARENTS on any metric. For accuracy/F1 the loss 1-metric is tested, so delta>0 still means A better and
    rel_gain_pct is the relative reduction in error rate. Holm over the whole call; cap is pre-specified by the caller."""
    d = df.copy(); d["rmse"] = (1 - d[col]) if higher_is_better else d[col]
    return rc.paired_contrasts(d.dropna(subset=["rmse"]), contrasts, max_tests=max_tests)
