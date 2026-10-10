"""rmbench_nn: torch models. No torch_geometric: graph = three index tensors (row, col, w).

GCN  : label-aware. Inputs per node = train-standardised geometric features, the *observed* residual (masked), the mask, and
       mask-normalised multi-hop label diffusion (so a node 5 hops from the nearest label still gets a signal at 30 labels).
       Episodic masked-label training: each step hides a random subset of the labeled nodes and predicts them.
       (Legacy GCN never saw a label at inference -> it was a feature regressor; that is the low-density failure mode.)
KGNN : the supplied kriging-initialised message-passing model, with the dead variance-aware correction head removed and
       the audit fixes listed in the notebook (safe norms, no unused args, shared trend/residual). steps=0 gives the
       *untrained local kriging* reference (fixed default covariance, no learning).
Both models predict the RESIDUAL; rmbench_core adds the shared trend back.
"""
from __future__ import annotations
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import sparse
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------- graph
def knn_edges(pos, k):
    """Symmetrised kNN graph + self loops, D^-1/2 (A+I) D^-1/2. Returns numpy (row, col, w)."""
    N = len(pos); k = min(k, N - 1)
    _, idx = cKDTree(pos).query(pos, k + 1)
    A = sparse.coo_matrix((np.ones(N * k), (np.repeat(np.arange(N), k), idx[:, 1:].ravel())), shape=(N, N)).tocsr()
    A = A.maximum(A.T).tolil(); A.setdiag(1.0); A = A.tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    return A.row.astype(np.int64), A.col.astype(np.int64), (A.data / np.sqrt(deg[A.row] * deg[A.col])).astype(np.float32)

def get_graph(u, k, cache):
    key = (u["name"], k)
    if cache is not None and key in cache: return cache[key]
    r, c, w = knn_edges(u["pos"], k)
    g = (torch.from_numpy(r), torch.from_numpy(c), torch.from_numpy(w))
    if cache is not None: cache[key] = g
    return g

def prop(h, g):
    row, col, w = g
    v = h.dim() == 1
    if v: h = h[:, None]
    out = torch.zeros_like(h).index_add_(0, row, h[col] * w[:, None])
    return out[:, 0] if v else out

def label_diffusion(ym, m, g, hops=(2, 4, 8)):
    num, den, feats = ym, m, []
    for h in range(1, max(hops) + 1):
        num, den = prop(num, g), prop(den, g)
        if h in hops:
            feats += [num / (den + 1e-3), den]
    return torch.stack(feats, 1)

# ----------------------------------------------------------------------------- GCN
class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden=32, layers=2, dropout=0.1):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(layers)])
        self.out = nn.Linear(hidden, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # untrained -> trend only
        self.p = dropout

    def forward(self, X, ym, m, g):
        h = torch.cat([X, ym[:, None], m[:, None], label_diffusion(ym, m, g)], 1)
        for lin in self.lins:
            h = F.dropout(F.relu(prop(lin(h), g)), self.p, self.training)
        return self.out(h).squeeze(-1)

def _split_val(tr, rng, frac=0.2):
    perm = rng.permutation(tr); nv = max(3, int(round(len(perm) * frac))) if len(perm) >= 12 else 0
    return perm[nv:], perm[:nv]

def fit_predict_gcn(u, tr, te, prep, hp, seed, cache):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    N = len(u["y"]); g = get_graph(u, int(hp.get("k", 8)), cache)
    X = torch.tensor(prep["X"], dtype=torch.float32)
    rn = torch.zeros(N); rn[tr] = torch.tensor(prep["res"][tr] / prep["sigma"], dtype=torch.float32)   # test residuals never enter
    model = GCNNet(X.shape[1] + 2 + 6, int(hp.get("hidden", 32)), int(hp.get("layers", 2)), float(hp.get("dropout", 0.1)))
    steps = int(hp.get("steps", 150)); fit, val = _split_val(tr, rng)

    def run(visible, train):
        m = torch.zeros(N); m[visible] = 1.0
        model.train(train)
        return model(X, rn * m, m, g)
    if steps > 0:                                                 # steps == 0 is legal: returns the zero-init model (trend only)
        opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-4)))
        best, state, bad = float("inf"), None, 0
        for s in range(steps):
            q = int(np.clip(round(rng.uniform(0.15, 0.5) * len(fit)), 1, max(1, len(fit) - 2)))
            pp = rng.permutation(fit); Q, V = pp[:q], pp[q:]
            opt.zero_grad(); out = run(V, True)
            F.huber_loss(out[Q], rn[Q], delta=1.5).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            if len(val) and s % 5 == 4:
                with torch.no_grad(): v = F.huber_loss(run(fit, False)[val], rn[val], delta=1.5).item()
                if v < best - 1e-5: best, state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8: break                            # patience = 8 evaluations (40 steps)
        if state is not None: model.load_state_dict(state)
    with torch.no_grad():
        pred = run(tr, False)[te].numpy() * prep["sigma"]
    return pred, dict(steps_run=steps)

# ----------------------------------------------------------------------------- KGNN
def _sp(x): return F.softplus(x)
def _isp(v): return math.log(math.exp(v) - 1.0)
def _norm(v): return torch.sqrt((v ** 2).sum(-1) + 1e-12)     # safe at 0: autograd of linalg.norm at the diagonal is a NaN risk

class KGNN(nn.Module):
    def __init__(self, polar=True, s2_init=0.05):
        super().__init__()
        self.polar = polar
        self.l1 = nn.Parameter(torch.tensor(_isp(3.0))); self.s1 = nn.Parameter(torch.tensor(_isp(0.8)))
        self.lr = nn.Parameter(torch.tensor(_isp(3.0))); self.lt = nn.Parameter(torch.tensor(_isp(3.0)))
        self.s2 = nn.Parameter(torch.tensor(_isp(s2_init))); self.tau = nn.Parameter(torch.tensor(_isp(0.2)))

    def cov(self, dxy, dr, dtan):
        k = _sp(self.s1) ** 2 * torch.exp(-_norm(dxy) / _sp(self.l1))
        if self.polar:
            k = k + _sp(self.s2) ** 2 * torch.exp(-torch.sqrt((dr / _sp(self.lr)) ** 2 + (dtan / _sp(self.lt)) ** 2 + 1e-9))
        return k

    def forward(self, Pq, Pr, yr, scale, tx, K):
        """Pq (B,2), Pr (M,2), yr (M,) normalised residuals -> kriging-mean of the K nearest references per target."""
        B = Pq.shape[0]
        _, idx = torch.topk(torch.cdist(Pq, Pr), K, dim=1, largest=False)
        pj, pq, yj, txs = Pr[idx] / scale, Pq[:, None, :] / scale, yr[idx], tx / scale
        def polar(a):
            v = a - txs; return _norm(v), torch.atan2(v[..., 1], v[..., 0])
        rj, phj = polar(pj); rq, phq = polar(pq)
        wrap = lambda x: torch.remainder(x + math.pi, 2 * math.pi) - math.pi
        c_t = self.cov(pj - pq, rj - rq, 0.5 * (rj + rq) * wrap(phj - phq))
        dph = wrap(phj[:, :, None] - phj[:, None, :])
        C = self.cov(pj[:, :, None, :] - pj[:, None, :, :], rj[:, :, None] - rj[:, None, :],
                     0.5 * (rj[:, :, None] + rj[:, None, :]) * dph)
        A = torch.zeros(B, K + 1, K + 1)
        A[:, :K, :K] = C + (_sp(self.tau) ** 2 + 1e-4) * torch.eye(K); A[:, :K, K] = 1; A[:, K, :K] = 1
        sol = torch.linalg.solve(A, torch.cat([c_t, torch.ones(B, 1)], 1)[..., None]).squeeze(-1)
        return (sol[:, :K] * yj).sum(1)

def fit_predict_kgnn(u, tr, te, prep, hp, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Pt = torch.tensor(u["pos"], dtype=torch.float32); tx = torch.tensor(u["tx_xy"], dtype=torch.float32)
    sig = prep["sigma"]; rn = torch.tensor(prep["res"] / sig, dtype=torch.float32)         # uses res[tr] only (indexed below)
    Ptr = Pt[tr]; k3 = min(3, len(tr) - 1)
    scale = float(torch.topk(torch.cdist(Ptr, Ptr), k3 + 1, largest=False).values[:, -1].median()) + 1e-6
    Kmax, steps, delta = int(hp.get("Kmax", 12)), int(hp.get("steps", 150)), 1.5
    m = KGNN(polar=bool(hp.get("polar", True)), s2_init=float(hp.get("s2_init", 0.05)))
    fit, val = _split_val(tr, rng)
    if steps > 0:
        opt = torch.optim.AdamW(m.parameters(), lr=float(hp.get("lr", 1e-2)), weight_decay=float(hp.get("wd", 1e-3)))
        def val_loss():
            m.eval()
            with torch.no_grad():
                return F.huber_loss(m(Pt[val], Pt[fit], rn[fit], scale, tx, min(Kmax, len(fit))), rn[val], delta=delta).item()
        best, state = (val_loss(), {k: t.clone() for k, t in m.state_dict().items()}) if len(val) else (float("inf"), None)  # step 0 = kriging is a candidate
        for ep in range(steps):
            m.train(); opt.zero_grad(); n = len(fit)
            q = int(np.clip(round(rng.uniform(0.1, 0.5) * n), 2, max(2, min(48, n - 3)))); pp = rng.permutation(fit); qi, ri = pp[:q], pp[q:]
            F.huber_loss(m(Pt[qi], Pt[ri], rn[ri], scale, tx, min(Kmax, len(ri))), rn[qi], delta=delta).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
            if len(val) and ep % 10 == 9:
                v = val_loss()
                if v < best - 1e-5: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
        if state is not None: m.load_state_dict(state)
    m.eval(); Kf = min(Kmax, len(tr))
    with torch.no_grad():
        pred = np.concatenate([m(Pt[te[i:i + 1500]], Pt[tr], rn[tr], scale, tx, Kf).numpy() for i in range(0, len(te), 1500)]) * sig
    return pred, dict(steps_run=steps, l1=float(_sp(m.l1).detach()), s2=float(_sp(m.s2).detach()), tau=float(_sp(m.tau).detach()))

# ----------------------------------------------------------------------------- dispatch used by rmbench_core.predict_method
def predict_neural(method, hp, u, tr, te, prep, seed, cache):
    torch.set_num_threads(1)
    if method == "gcn": return fit_predict_gcn(u, tr, te, prep, hp, seed, cache)
    if method == "kgnn": return fit_predict_kgnn(u, tr, te, prep, hp, seed)
    if method == "krig_untrained": return fit_predict_kgnn(u, tr, te, prep, {**hp, "steps": 0, "polar": hp.get("polar", False)}, seed)
    raise ValueError(method)
