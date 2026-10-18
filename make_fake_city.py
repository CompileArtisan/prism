"""Synthetic city units for plumbing + sensitivity tests of d0_city.py. amp>0 plants a GLOBAL nonlinear function of the 5 map features (shared by all layouts)."""
import os, sys, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import rmbench_core as rc, rmbench_city as cc
def make(root, amp, n_layouts=26, seed=0, hard=False):
    shutil.rmtree(root, ignore_errors=True); os.makedirs(root); cache = os.path.join(root, cc.CITY_CFG["cache_dir"]); cfg = dict(cc.CITY_CFG, cache_dir=cache)
    names = [f"city_{i}_fake{i}_3p5" for i in range(n_layouts)]; rows = []
    roles = ["source"] * 4 + ["val"] + ["dev"] + ["source"] * 2 + ["confirm"]          # p % 8 pattern, repeated
    for i, slug in enumerate(names):
        role = ["source","source","source","source","val","dev","confirm","confirm"][i % 8]; rows.append(dict(slug=slug, role=role, eligible=True, n_units=2))
        for k in range(2):
            raw = rc.make_synthetic_raw("B", tx=k, n=2500, seed=100 * i + k); raw["name"] = f"dm|{slug}|{k}"; raw["scenario"] = slug
            u = rc.preprocess_unit(raw, cfg); rng = np.random.default_rng(1000 * i + k); N = len(u["y"]); P = u["pos"]
            def smooth():
                W = rng.normal(size=(12, 2)) / rng.uniform(4, 9); ph = rng.uniform(0, 6.28, 12); return np.sqrt(2 / 12) * np.cos(P @ W.T + ph).sum(1)
            S = np.column_stack([smooth(), smooth(), rng.normal(size=N), smooth(), rng.normal(size=N)]) if hard else np.column_stack([(1 - u["los"]) * rng.uniform(0.5, 3, N), u["d"] / 20.0, smooth(), rng.normal(size=N), smooth()])
            f = 3.0 * np.tanh(S[:, 0] * S[:, 2]) + 2.5 * np.tanh(S[:, 1] - 1) * np.tanh(S[:, 4]) + 2.0 * np.tanh(S[:, 0] - 1.5) * np.tanh(S[:, 1] - 0.8)
            if hard:   # many high-gain interaction terms of the 5 features: too rich for ~100 labels of one map, learnable from many layouts
                g = np.random.default_rng(7); f = np.zeros(N)
                for _ in range(24):
                    ia, ib = g.choice(5, 3, replace=False), g.choice(5, 3, replace=False); f += g.normal() * np.tanh(3 * (S[:, ia] - 1) @ g.normal(size=3)) * np.tanh(3 * (S[:, ib] - 1) @ g.normal(size=3))
                f = f / (f.std() + 1e-9) * 3.0                 # 3 dB std, comparable to the shadowing; side columns are unrelated to y otherwise
            u["y"] = u["y"] + amp * f
            u.update(layout=slug, parent=slug, tx_known=1, side=S.astype(np.float32), has_scene=1, n_buildings=100); rc.save_unit(u, cache)
    pd.DataFrame(rows).to_csv(os.path.join(root, cc.MANIFEST), index=False); return root
if __name__ == "__main__": make(sys.argv[1], float(sys.argv[2]), hard=len(sys.argv) > 3)
