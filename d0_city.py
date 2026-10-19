"""d0_city.py -- D0 on the city pool: can a prior pre-trained on OTHER cities predict a held-out city's residual better than any within-map fit?

  uv run d0_city.py pretrain          # trains two priors on the 'source' cities (validation on 'val'): full (geometry + 5 map features) and geo-only
  uv run d0_city.py d0 [--n-units 24] # scores them on the 'dev' cities ONLY (never 'confirm') and prints the verdict

Reads city_manifest.csv (roles from `rmbench_city.py roles`), units from rmbench_cache_city/. Reuses ext5/ext6: the prior, the global scaling constants and the
comparators are the same code as the planted-signal rounds. A plant with amp = 0 and d_extra = 0 is used ONLY as the container of fixed GLOBAL scaling
constants (fit on source cities); it changes no label.

Target = the REAL labels of a held-out dev city. The prior uses no label of that city; every other arm uses n labels of it. Per dev unit and n:
  final-metric RMSE (dB) of: idw, rk_side, rk_onehot, rk_spline, rk_gbm (within-map arms), prior_full + IDW of the remainder, prior_geo + IDW of the remainder
  (the 'geo' prior is trained on geometry only, side columns zeroed: it shows what the 5 map features add; both priors are UNGATED here, Stage 2 adds the gate)
  descriptive only: covariate R2 of the residual. NOTE the within-map fits receive position features too, so they also capture spatial shadowing; that is why
  the R2 columns are NOT used for the verdict.
PRE-DECLARED verdict (fixed here, before any result): units are averaged within a city; at n = 100 labels, with best = the lowest RMSE of the five within-map
arms in that city (an oracle choice, deliberately hard on the prior), gain = 100 * (best - prior_full) / best. PASS iff
  median gain >= 2 % (the 2 % margin of the earlier rounds)  AND  gain > 0 in >= 60 % of dev cities  AND  median (gain_full - gain_geo) >= 1 point
  (the map features, not just geometry, must carry it).
If it does not pass, cross-city transfer has nothing to add and Stage 2 must not be run: report the negative result.
"""
import os, sys, glob, json, math, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import rmbench_core as rc, rmbench_dmfix                      # noqa: F401
import rmbench_city as cc, rmbench_ext as rx, rmbench_ext2 as r2, rmbench_ext5 as x5, rmbench_ext6 as x6

CFG = dict(cc.CITY_CFG)
OUT = "results_city"
VERDICT = dict(n_ref=100, min_gain_pct=2.0, min_frac_pos=0.6, min_full_minus_geo=1.0)

def pool(manifest=cc.MANIFEST, cache=None):
    """{role: [layout = [(unit name, cache dir), ...], ...]} for source / val / dev / confirm / confirm2, layouts in numeric city order."""
    cache = cache or CFG["cache_dir"]; m = pd.read_csv(manifest); role = dict(zip(m.slug, m.role)); by = {}
    for f in sorted(glob.glob(os.path.join(cache, "units", "*.npz"))):
        name = os.path.basename(f)[:-4].replace("__", "|"); slug = name.split("|")[1]
        if role.get(slug) in ("source", "val", "dev", "confirm", "confirm2"): by.setdefault(slug, []).append(name)
    key = lambda s: (cc.parse_slug(s)[0] if s.startswith("city_") else 10 ** 6, s)
    out = {r: [] for r in ("source", "val", "dev", "confirm", "confirm2")}
    for slug in sorted(by, key=key): out[role[slug]].append([(n, cache) for n in sorted(by[slug])])
    return out

def _gbm_cov_fit(u, tr, te, prep, seed, stages=(25, 50, 100, 200, 300)):
    """Covariate-only gradient boosting of the residual (no spatial interpolation); stage count chosen by 5-fold CV inside the labelled set."""
    from sklearn.ensemble import GradientBoostingRegressor
    X = np.asarray(prep["X"], float); y = np.asarray(prep["res"], float); n = len(tr); perm = np.random.default_rng(seed).permutation(n); oof = np.zeros((len(stages), n))
    mk = lambda: GradientBoostingRegressor(n_estimators=max(stages), learning_rate=0.05, max_depth=3, subsample=0.7, min_samples_leaf=3, random_state=int(seed))
    for h in np.array_split(perm, 5):
        keep = np.ones(n, bool); keep[h] = False; st = list(mk().fit(X[tr[keep]], y[tr[keep]]).staged_predict(X[tr[h]]))
        for i, s in enumerate(stages): oof[i, h] = st[s - 1]
    s = stages[int(np.argmin(((oof - y[tr][None]) ** 2).mean(1)))]; return list(mk().fit(X[tr], y[tr]).staged_predict(X[te]))[s - 1]

def _cov_fit(kind, u, tr, te, prep, seed=0):
    if kind == "gbm": return _gbm_cov_fit(u, tr, te, prep, seed)
    if kind == "linear":
        G, S = x6._split_X(u, prep); A = np.column_stack([np.ones(len(G)), G, S])
    else: A = x6._design(kind, u, tr, prep)
    _, beta = x6._ridge_fit(A, tr, np.asarray(prep["res"], float)); return (A @ beta)[te]

def _r2(pred, target): return float(1 - np.mean((np.asarray(pred, float) - target) ** 2) / (np.var(target) + 1e-12))

def d0_real_table(prior_full, prior_geo, eval_pairs, cfg, n_labels=(100, 300), n_units=24, log=print):
    """One row per (dev unit, n_labels). eval_pairs = [(unit name, cache dir)]. Uses ONLY the labels of the evaluation map (at tr) and the pre-trained priors."""
    ck = {"full": x5._load_prior(prior_full), "geo": x5._load_prior(prior_geo) if prior_geo else None}; pl = ck["full"]["plant"]; rows = []
    for n, c in eval_pairs[:n_units]:
        u0 = x6.plant6(rx.get_unit_ext(n, c), dict(plant6=pl))
        for nl in n_labels:
            if nl > len(u0["y"]) - cfg["n_test"] - 5: continue
            tr, te = r2.split_for2(u0, nl, 900, cfg["n_test"], cfg.get("split_mode", "random"), cfg.get("block_buffer", 0.0), cfg.get("test_salt", ""))
            u, spec = r2.view_for_split2(u0, tr); prep = r2.prepare2(u, tr, "geo_side", cfg["clip"], spec); sig = float(prep["sigma"]); res = np.asarray(prep["res"], float); y = np.asarray(u["y"], float)
            row = dict(unit=n, layout=str(u0["layout"]), n_labels=nl); target = res[te] / sig
            preds = {}
            for tag, ckk in ck.items():
                if ckk is None: continue
                X = x6._X6(u, prep, ckk["plant"]); p = np.asarray(ckk["net"].predict(X), float) * sig; preds[tag] = p
                row[f"r2_prior_{tag}"] = _r2(p[te] / sig, target)
            for k in ("linear", "onehot", "spline", "gbm"): row[f"r2_{k}"] = _r2(_cov_fit(k, u, tr, te, prep) / sig, target)
            rmse = lambda pred: float(np.sqrt(np.mean((np.asarray(pred, float) - y[te]) ** 2)))
            row["rmse_idw"] = rmse(x6._idw_on(u, tr, te, prep, np.zeros(len(te)), res[tr])[0])
            row["rmse_rk_side"] = rmse(r2.predict_rk_side(u, tr, te, prep, dict(fset="geo_side"))[0])
            for tag, p in preds.items(): row[f"rmse_prior_{tag}_idw"] = rmse(x6._idw_on(u, tr, te, prep, p[te], res[tr] - p[tr])[0])
            for k in ("onehot", "spline"): row[f"rmse_rk_{k}"] = rmse(x6._rk_ridge(k, u, tr, te, prep)[0])
            row["rmse_rk_gbm"] = rmse(x6._rk_gbm(u, tr, te, prep, 0)[0]); rows.append(row)
        log(f"  D0 {n}: done")
    return pd.DataFrame(rows)

COMPS = ("rmse_idw", "rmse_rk_side", "rmse_rk_onehot", "rmse_rk_spline", "rmse_rk_gbm")

def d0_real_verdict(D, v=VERDICT):
    """City-level medians (units averaged within a city first). PASS rule = the pre-declared one in the module docstring."""
    s = D[D.n_labels == v["n_ref"]]
    if not len(s): raise ValueError(f"no rows at n_labels={v['n_ref']}")
    by = s.groupby("layout").mean(numeric_only=True); best = by[list(COMPS)].min(axis=1); out = dict(n_cities=int(len(by)), n_ref=v["n_ref"])
    g = 100.0 * (best - by["rmse_prior_full_idw"]) / best; out["median_gain_full_pct"] = float(g.median()); out["frac_cities_gain_pos"] = float((g > 0).mean())
    if "rmse_prior_geo_idw" in by:
        gg = 100.0 * (best - by["rmse_prior_geo_idw"]) / best; out["median_gain_geo_pct"] = float(gg.median()); out["median_full_minus_geo_pts"] = float((g - gg).median())
    out["median_rmse"] = {k[5:]: float(by[k].median()) for k in by.columns if k.startswith("rmse_")}
    out["pass"] = bool(out["median_gain_full_pct"] >= v["min_gain_pct"] and out["frac_cities_gain_pos"] >= v["min_frac_pos"] and out.get("median_full_minus_geo_pts", 0.0) >= v["min_full_minus_geo"]); return out

def cmd_pretrain(out=OUT):
    P = pool(); S, V = P["source"], P["val"]; print(f"source cities {len(S)} ({sum(map(len, S))} units), val cities {len(V)}, dev {len(P['dev'])}, confirm {len(P['confirm'])} (untouched)")
    pairs = [p for L in S for p in L]; pl = x6.make_plant(pairs, amp=0.0, d_extra=0, seed=0); K = len(S); os.makedirs(out, exist_ok=True)
    pl_geo = dict(pl, sd=[1e12] * len(pl["sd"]))                       # side columns collapse to ~0: a geometry-only prior with the same code path
    for tag, plant in (("full", pl), ("geo", pl_geo)): x6.pretrain_prior_scale6(S, V, [K], CFG, out, f"city_{tag}", plant)
    print("priors ->", os.path.join(out, f"city_full_K{K}.pkl"), "and city_geo_K%d.pkl" % K)

def cmd_d0(n_units=24, out=OUT):
    P = pool(); K = len(P["source"]); full = os.path.join(out, f"city_full_K{K}.pkl"); geo = os.path.join(out, f"city_geo_K{K}.pkl")
    dev = [p for L in P["dev"] for p in L]; print(f"D0 on {len(P['dev'])} dev cities ({len(dev)} units); 'confirm' cities are not touched")
    D = d0_real_table(full, geo if os.path.exists(geo) else None, dev, CFG, n_units=n_units); D.to_csv(os.path.join(out, "d0_city.csv"), index=False)
    pd.set_option("display.width", 220); cols = [c for c in D.columns if c.startswith(("r2_", "rmse_"))]
    print("\nmedian over dev cities (units averaged within a city):"); print(D.groupby(["n_labels", "layout"]).mean(numeric_only=True).groupby("n_labels")[cols].median().round(3).T.to_string())
    v = d0_real_verdict(D); json.dump(v, open(os.path.join(out, "d0_city_verdict.json"), "w"), indent=1); print("\nVERDICT:", json.dumps(v, indent=1))
    print("\n" + ("PASS: Stage 2 may be run on the CONFIRM cities (freeze the plan first)." if v["pass"] else "NO PASS: cross-city transfer has nothing to add here. Do NOT run Stage 2; report the negative result."))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pretrain", "d0"]); ap.add_argument("--n-units", type=int, default=24); a = ap.parse_args()
    cmd_pretrain() if a.cmd == "pretrain" else cmd_d0(a.n_units)
