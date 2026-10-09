"""KGNN: kriging-initialised message passing with a LEARNED, TX-polar-anisotropic covariance.
Per target, the K nearest labeled references exchange information through the kriging system
[C_rr + tau^2 I, 1; 1^T, 0] w = [c_tr; 1]  (messages = kriging weights, i.e. declustered), where
C = s1^2 exp(-d_euclid/l1) + s2^2 exp(-sqrt((dr/lr)^2 + (rbar*dphi/lt)^2)).
Hyper-parameters (+ an optional zero-initialised variance-aware MLP correction) are learned by episodic
masked-label training on the task's own labels."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def fit_trend(d, y):
    """y = a - 10 n log10(d) by least squares on the supplied (train) points only."""
    x = 10 * np.log10(np.maximum(d, 1e-3))
    if np.ptp(x) < 1e-9:
        return float(np.mean(y)), 0.0
    slope, a = np.polyfit(x, y, 1)
    return float(a), float(-slope)


def trend_eval(d, a, n):
    return a - 10 * n * np.log10(np.maximum(d, 1e-3))


def _sp(x): return F.softplus(x)
def _isp(v): return math.log(math.exp(v) - 1.0)

class KGNN(nn.Module):
    def __init__(self, hidden=16, corr=True, polar=True, s2_init=0.05):
        super().__init__()
        self.polar, self.corr = polar, corr
        self.l1 = nn.Parameter(torch.tensor(_isp(3.0))); self.s1 = nn.Parameter(torch.tensor(_isp(0.8)))
        self.lr = nn.Parameter(torch.tensor(_isp(3.0))); self.lt = nn.Parameter(torch.tensor(_isp(3.0)))
        self.s2 = nn.Parameter(torch.tensor(_isp(s2_init)))       # polar component amplitude (0.05 = ~off; larger lets lr/lt receive signal)
        self.tau = nn.Parameter(torch.tensor(_isp(0.2)))
        self.head = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def cov(self, dxy, rbar, dr, dtan):
        k = _sp(self.s1) ** 2 * torch.exp(-torch.linalg.norm(dxy, dim=-1) / _sp(self.l1))
        if self.polar:
            dm = torch.sqrt((dr / _sp(self.lr)) ** 2 + (dtan / _sp(self.lt)) ** 2 + 1e-9)
            k = k + _sp(self.s2) ** 2 * torch.exp(-dm)
        return k

    def forward(self, Pq, Pr, yr, scale, tx, R, sigma, K):
        D = torch.cdist(Pq, Pr); d, idx = torch.topk(D, K, dim=1, largest=False)
        pj = Pr[idx] / scale; pq = Pq[:, None, :] / scale; yj = yr[idx] / sigma; txs = tx / scale
        def polar(a):  # (...,2) -> r, phi
            v = a - txs; return torch.linalg.norm(v, dim=-1), torch.atan2(v[..., 1], v[..., 0])
        rj, phj = polar(pj); rq, phq = polar(pq)
        # target-ref covariance
        dxy_t = pj - pq; dph = torch.remainder(phj - phq + math.pi, 2 * math.pi) - math.pi
        c_t = self.cov(dxy_t, None, rj - rq, 0.5 * (rj + rq) * dph)
        # ref-ref covariance
        dxy_r = pj[:, :, None, :] - pj[:, None, :, :]
        dphr = torch.remainder(phj[:, :, None] - phj[:, None, :] + math.pi, 2 * math.pi) - math.pi
        C = self.cov(dxy_r, None, rj[:, :, None] - rj[:, None, :], 0.5 * (rj[:, :, None] + rj[:, None, :]) * dphr)
        tau2 = _sp(self.tau) ** 2; tot = _sp(self.s1) ** 2 + (_sp(self.s2) ** 2 if self.polar else 0.0)
        B = Pq.shape[0]
        A = torch.zeros(B, K + 1, K + 1)
        A[:, :K, :K] = C + (tau2 + 1e-4) * torch.eye(K); A[:, :K, K] = 1; A[:, K, :K] = 1
        rhs = torch.cat([c_t, torch.ones(B, 1)], dim=1)
        sol = torch.linalg.solve(A, rhs[..., None]).squeeze(-1)
        lam, mu = sol[:, :K], sol[:, K]
        mean = (lam * yj).sum(1)
        var = (tot - (lam * c_t).sum(1) - mu).clamp(min=1e-6)
        out = mean
        if self.corr:
            wstd = yj.std(1) if K > 1 else torch.zeros(B)
            f = torch.stack([mean, var.sqrt(), wstd, (rq / (R / scale)).squeeze(-1) if rq.dim() > 1 else rq / (R / scale),
                             torch.cos(phq.squeeze(-1)), torch.sin(phq.squeeze(-1))], dim=1)
            out = mean + 0.5 * torch.tanh(self.head(f).squeeze(-1))
        return out * sigma

def kgnn_predict(pos, d_tx, y, tx_xy, train_idx, test_idx, seed=0, steps=200, lr=1e-2, wd=1e-3, Kmax=12,
                 delta=1.5, val_frac=0.2, corr=True, polar=True, n_ens=1, s2_init=0.05):
    train_idx = np.asarray(train_idx); test_idx = np.asarray(test_idx)
    a, nexp = fit_trend(d_tx[train_idx], y[train_idx]); res = y - trend_eval(d_tx, a, nexp)
    sigma = float(np.std(res[train_idx]) + 1e-6)
    Pt = torch.tensor(pos, dtype=torch.float32); yres = torch.tensor(res, dtype=torch.float32); tx = torch.tensor(tx_xy, dtype=torch.float32)
    Ptr = Pt[train_idx]; Dm = torch.cdist(Ptr, Ptr); k3 = min(3, len(train_idx) - 1)
    scale = float(torch.topk(Dm, k3 + 1, largest=False).values[:, -1].median()) + 1e-6
    R = float(torch.linalg.norm(Ptr - tx, dim=1).mean()) + 1e-6
    preds = []
    for e in range(n_ens):
        s = seed * 1000 + e; torch.manual_seed(s); rng = np.random.default_rng(s)
        perm = rng.permutation(train_idx); nv = max(3, int(round(len(perm) * val_frac))) if len(perm) >= 12 else 0
        val_idx, fit_idx = (perm[:nv], perm[nv:]) if nv else (perm[:0], perm)
        m = KGNN(corr=corr, polar=polar, s2_init=s2_init); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        best, state = float("inf"), None
        def eval_val():
            m.eval()
            with torch.no_grad():
                pv = m(Pt[val_idx], Pt[fit_idx], yres[fit_idx], scale, tx, R, sigma, min(Kmax, len(fit_idx)))
                return F.huber_loss(pv / sigma, yres[val_idx] / sigma, delta=delta).item()
        if nv: best, state = eval_val(), {k: t.clone() for k, t in m.state_dict().items()}   # step-0 == kriging is a candidate
        for ep in range(steps):
            m.train(); opt.zero_grad()
            n = len(fit_idx); q = int(np.clip(round(rng.uniform(0.1, 0.5) * n), 2, min(48, n - 3)))
            pp = rng.permutation(fit_idx); qi, ri = pp[:q], pp[q:]
            pr = m(Pt[qi], Pt[ri], yres[ri], scale, tx, R, sigma, min(Kmax, len(ri)))
            F.huber_loss(pr / sigma, yres[qi] / sigma, delta=delta).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
            if nv and ep % 10 == 9:
                v = eval_val()
                if v < best - 1e-5: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
        if state is not None: m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            Kf = min(Kmax, len(train_idx))
            preds.append(np.concatenate([m(Pt[test_idx[i:i + 1500]], Pt[train_idx], yres[train_idx], scale, tx, R, sigma, Kf).numpy()
                                         for i in range(0, len(test_idx), 1500)]))
    return np.mean(preds, 0) + trend_eval(d_tx[test_idx], a, nexp)
