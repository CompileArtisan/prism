"""Standalone timing proxy for one GCN fit (no notebook, no DeepMIMO). Run: python bench_fit.py
Edit N_NODES / HIDDEN / LAYERS / EPOCHS to match your tuned values (see derived.json once available)."""
import time, numpy as np, torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv

N_NODES, N_LAB, K, HIDDEN, LAYERS, EPOCHS, FEATS = 1500, 107, 8, 64, 2, 300, 8
rng = np.random.default_rng(0)
pos = rng.uniform(0, 50, (N_NODES, 2)); x = torch.tensor(rng.normal(size=(N_NODES, FEATS)), dtype=torch.float32)
y = torch.tensor(rng.normal(size=N_NODES), dtype=torch.float32)
A = kneighbors_graph(pos, K, mode="connectivity").tocoo()
ei = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
lab = torch.tensor(rng.choice(N_NODES, N_LAB, replace=False))

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        dims = [FEATS] + [HIDDEN] * LAYERS
        s.convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        s.out = torch.nn.Linear(HIDDEN, 1)
    def forward(s, x, ei):
        for c in s.convs: x = torch.relu(c(x, ei))
        return s.out(x).squeeze(-1)

def fit():
    m = Net(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(EPOCHS):
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(m(x, ei)[lab], y[lab]); loss.backward(); opt.step()

for th in (1, 4, 8, 16):
    torch.set_num_threads(th); fit()                      # warm-up
    t = time.time(); [fit() for _ in range(3)]; dt = (time.time() - t) / 3
    print(f"threads={th:2d}: {dt:.2f} s/fit -> main loops ~{9400*dt/3600:.1f} h (serial)")
