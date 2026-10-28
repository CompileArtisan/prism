"""Mechanism test (synthetic, CPU): can the public RadioWNet first-U fit a SIGNED pointwise target with and without its final ReLU?
Inputs mimic the pipeline (building mask, tx impulse, sparse labels, mask, 6 smooth 'feature' channels); target = signed function of two feature channels."""
import importlib.util, sys, time, numpy as np, torch, torch.nn as nn
from scipy import ndimage
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("rnm", "RadioUNet/lib/modules.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def make_net(cin, keep_relu, seed):
    torch.manual_seed(seed)
    net = mod.RadioWNet(inputs=cin, phase="firstU")
    if not keep_relu:
        c, r, p = net.conv_up000; assert isinstance(r, nn.ReLU); net.conv_up000 = nn.Sequential(c, p)
    return net

def batch(rng, B=4, H=64):
    x = torch.zeros(B, 10, H, H); y = torch.zeros(B, 1, H, H)
    for b in range(B):
        f = [ndimage.gaussian_filter(rng.standard_normal((H, H)), 4) for _ in range(6)]
        f = [(a - a.min()) / (a.max() - a.min()) for a in f]
        x[b, 0] = torch.from_numpy((ndimage.gaussian_filter(rng.standard_normal((H, H)), 4) > 0.2).astype(np.float32))
        x[b, 1, rng.integers(H), rng.integers(H)] = 1.0
        m = rng.random((H, H)) < 0.02; x[b, 3] = torch.from_numpy(m.astype(np.float32))
        for k in range(6): x[b, 4 + k] = torch.from_numpy(f[k].astype(np.float32))
        t = 2.0 * (f[0] - 0.5) + 1.0 * (f[1] - 0.5)          # signed, mean ~ 0
        y[b, 0] = torch.from_numpy(t.astype(np.float32)); x[b, 2] = x[b, 3] * y[b, 0]
    return x, y

def run(keep_relu, steps=300, seed=0):
    net = make_net(10, keep_relu, seed); opt = torch.optim.AdamW(net.parameters(), lr=3e-4); rng = np.random.default_rng(1)
    hist = []; t0 = time.time()
    for s in range(1, steps + 1):
        x, y = batch(rng); out = net(x)[0]
        loss = ((out - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0: hist.append((s, float(loss.detach()))); print(f"  keep_relu={keep_relu} step {s} mse {float(loss.detach()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    xv, yv = batch(np.random.default_rng(99), B=4)
    with torch.no_grad(): ov = net(xv)[0]
    return dict(keep_relu=keep_relu, val_mse=float(((ov - yv) ** 2).mean()), var_target=float(yv.var()), min_out=float(ov.min()), frac_target_neg=float((yv < 0).float().mean()))

if __name__ == "__main__":
    res = [run(True), run(False)]
    for r in res: print(r)
