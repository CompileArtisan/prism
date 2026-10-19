"""rmbench_city.py -- DeepMIMO city pool for the cross-layout test. Adds only; every other rmbench_* file stays untouched.

  uv run rmbench_city.py build --first 24        # download + build units for the first 24 groups (in numeric city order); writes the manifest
  uv run rmbench_city.py build                   # all candidates (~80 downloads, 50-220 MB each: several GB, an hour or more)
  uv run rmbench_city.py roles [--freeze]        # assign source / val / dev / confirm roles from the manifest; --freeze writes the hashed plan

WHAT IS FIXED HERE, BEFORE ANY MODEL RESULT EXISTS (data availability and map geometry only; never a target value or a model):
  * candidates = plain `city_<n>_<name>_3p5` scenarios from your dm.search output; the _s / _lwm / _dyn / _gp / _10cm variants and repeats are excluded
  * map = the 'buildings' objects' TRUE horizontal faces (ear-clipped triangles) with roof z >= 2 m, rasterised at 2 m. The 2 m roof cut is
    justified by diagnose_raster.py (2026-10-18, New York and Los Angeles: ~2/3 of horizontal faces are <1 m flat patches; the ray tracer had
    max_transmissions = 0, so a receiver with a signal cannot be inside a building). It was chosen from geometry and path existence, before any model result. Axis-aligned bounding boxes are NOT used:
    on your scenes one object can be a concave city-block outline whose box covers 40-100 % of the receiver grid (diagnose_scene.py, 2026-10-18)
  * eligibility = >= 1 transmitter with >= 1000 points above the noise floor, >= 10 'buildings' objects, and an alignment test: <= 10 % of the
    receivers that have a signal lie on a building cell, and the share of path-less receivers on a building cell exceeds that by >= 0.25.
    (Path existence is used ONLY to validate that scene and receiver grid line up; it is not a feature and enters no model.)
  * layouts that share a city (aliases below) stay together in ONE role; roles by group position p % 8: 0-3 source, 4 validation, 5 development, 6-7 confirmation
  * unit of inference = the city (layout); `parent` = `layout` = scenario slug
  * side information = 5 label-free map features per receiver (env_features): 3-D-aware line-of-sight obstruction counts / length / height from the
    raster (the ray runs from the transmitter height to the receiver height, so a roof-top BS is not blocked by a lower building), building cover
    within 30 m, distance to the nearest building. No path counts, delays or other channel outputs.
UNTESTED against real DeepMIMO scenes: the glue to ds.scene (bounding boxes of objects labelled 'buildings'). The rasteriser, the features and the role
assignment were tested on synthetic DeepMIMO Face/PhysicalElement objects.
"""
import os, sys, re, json, math, time, hashlib, argparse
import numpy as np
sys.path.insert(0, os.getcwd())
import rmbench_core as rc
import rmbench_dmfix                                                  # carrier-frequency fix; must precede any load

CITY_CFG = dict(rc.DEFAULT_CFG, cache_dir="rmbench_cache_city", max_points=12000, min_unit_points=1000, n_test=500,
                densities=(30, 100, 300), n_rep=5)
MANIFEST, PLAN = "city_manifest.csv", "results_city/prereg_city.json"
ENV_COLS = ["log1p_n_edges", "log1p_blocked_len", "log1p_max_obstruction", "cover_30m", "log1p_d_nearest"]
RES, PAD, R_COVER = 2.0, 40.0, 30.0
MIN_ROOF = 2.0                                   # m: horizontal faces below the 2 m receiver height are flat ground-level patches, not obstructions
FEAT_VERSION = "raster-v2"                      # manifest rows from another feature version are rebuilt, never reused

_US = ["newyork", "losangeles", "chicago", "houston", "phoenix", "philadelphia", "miami", "sandiego", "dallas", "sanfrancisco", "austin",
       "santaclara", "fortworth", "columbus", "charlotte", "indianapolis", "sanfrancisco", "seattle", "denver", "oklahoma"]
_REST = {21: "taito_city", 22: "rome", 23: "beijing", 24: "cairo", 25: "instanbul", 26: "dubai", 27: "rio_de_janeiro", 28: "mumbai", 29: "centro",
         30: "singapore", 31: "barcelona", 32: "sydney", 33: "hong_kong", 34: "amsterdam", 35: "san_francisco", 36: "bangkok", 37: "seoul",
         38: "toronto", 39: "jerusalem", 40: "prague", 41: "kyoto", 42: "san_nicolas", 43: "cape_town", 44: "lisboa", 45: "stockholm",
         46: "la_habana", 47: "chicago", 48: "gurbchen", 49: "new_delhi", 50: "edinburgh", 51: "firenze", 52: "marrakesh", 53: "sankt-peterburg",
         54: "north_jakarta", 55: "madrid", 56: "montreal", 57: "taipei", 58: "sumida_city", 59: "roma", 60: "toronto", 61: "hatsukaichi",
         62: "santiago", 63: "athens", 64: "new_york", 65: "granada", 66: "bruxelles", 67: "fujiyoshida", 68: "reykjavik", 69: "warszawa",
         70: "stockholm", 71: "helsinki", 72: "capetown", 73: "casablanca", 74: "chiyoda", 75: "dongcheng", 76: "houston", 77: "melbourne",
         78: "newtaipei", 79: "parow", 80: "philadelphia", 81: "seoul", 82: "shibuya", 83: "tempe", 84: "baoding", 85: "fuyang", 86: "ankara",
         87: "jinan", 88: "tongshan", 89: "nairobi", 90: "nanjing", 91: "xiangyang", 93: "rangoon", 94: "monterrey", 95: "delhi", 96: "osaka",
         97: "chennai", 98: "guangzhou", 99: "shanghai", 101: "jakarta", 102: "huanggang"}
CANDIDATES = [f"city_{i}_{n}_3p5" for i, n in enumerate(_US)] + [f"city_{i}_{n}_3p5" for i, n in sorted(_REST.items())]
# not auto-included (non-ASCII or duplicated index in the listing): city_92_s\u00e3opaulo_3p5, city_94_zhanjiang_3p5, city_100_bogot\u00e1_3p5

_ALIAS = {"roma": "rome", "new_york": "newyork", "san_francisco": "sanfrancisco", "taito_city": "tokyo", "chiyoda": "tokyo", "sumida_city": "tokyo",
          "shibuya": "tokyo", "north_jakarta": "jakarta", "new_delhi": "delhi", "cape_town": "capetown", "newtaipei": "taipei"}

def parse_slug(s):
    m = re.match(r"city_(\d+)_(.+)_3p5$", s); return int(m.group(1)), _ALIAS.get(m.group(2), m.group(2))

def groups(slugs):
    """[(group_name, [slugs])] ordered by the smallest city index in the group; same-city layouts stay together."""
    g = {}
    for s in slugs: i, n = parse_slug(s); g.setdefault(n, []).append((i, s))
    return sorted(((n, [s for _, s in sorted(v)]) for n, v in g.items()), key=lambda t: min(parse_slug(x)[0] for x in t[1]))

def assign_roles(eligible):
    """Role per slug from the position of its group among the ELIGIBLE groups. Fixed rule, no randomness."""
    role = {}
    for p, (_, ss) in enumerate(groups(eligible)):
        r = "source" if p % 8 <= 3 else "val" if p % 8 == 4 else "dev" if p % 8 == 5 else "confirm"
        for s in ss: role[s] = r
    return role

# ------------------------------------------------------------------------------------------------------------ label-free map features
def _normal(v):
    """Newell normal (robust to collinear first vertices and concave outlines)."""
    v = np.asarray(v, float); n = np.cross(v, np.roll(v, -1, axis=0)).sum(0); nn = np.linalg.norm(n); return n / nn if nn > 1e-12 else np.zeros(3)

def rasterize(scene, x0, y0, nx, ny, res=RES, label="buildings", min_roof=MIN_ROOF):
    """-> (occ bool (ny,nx), hgt float (ny,nx)): cell centres covered by a horizontal face of any `label` object; hgt = highest covering face z.
    Horizontal = |normal_z| > 0.7 and mean z >= min_roof (DeepMIMO 'buildings' include flat patches ~0.3 m high that cover streets). Concave faces use Face.triangular_faces (ear clipping). Cell-centre sampling: footprints thinner than ~res can vanish."""
    occ = np.zeros((ny, nx), bool); hgt = np.zeros((ny, nx))
    for o in scene.get_objects(label):
        for f in o.faces:
            v = np.asarray(f.vertices, float)
            if len(v) < 3 or abs(_normal(v)[2]) < 0.7: continue
            z = float(v[:, 2].mean())
            if z < min_roof: continue                                   # flat ground patch / building bottom face, not a roof
            for tri in f.triangular_faces:
                T = np.asarray(tri, float)[:, :2]
                ix0 = max(int(np.floor((T[:, 0].min() - x0) / res)), 0); ix1 = min(int(np.ceil((T[:, 0].max() - x0) / res)), nx - 1)
                iy0 = max(int(np.floor((T[:, 1].min() - y0) / res)), 0); iy1 = min(int(np.ceil((T[:, 1].max() - y0) / res)), ny - 1)
                if ix1 < ix0 or iy1 < iy0: continue
                gx, gy = np.meshgrid(x0 + (np.arange(ix0, ix1 + 1) + 0.5) * res, y0 + (np.arange(iy0, iy1 + 1) + 0.5) * res)
                (ax, ay), (bx, by), (cx, cy) = T; den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(den) < 1e-12: continue
                l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den; l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den; l3 = 1 - l1 - l2
                m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0); sub = (slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
                occ[sub] |= m; hgt[sub] = np.where(m, np.maximum(hgt[sub], z), hgt[sub])
    return occ, hgt

def raster_for(scene, pos3, tx3, res=RES, pad=PAD):
    """Raster covering the receivers and the transmitter plus `pad` metres. Returns dict(occ, hgt, x0, y0, res)."""
    P = np.vstack([np.asarray(pos3, float)[:, :2], np.asarray(tx3, float)[None, :2]]); x0, y0 = P.min(0) - pad; x1, y1 = P.max(0) + pad
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res)); occ, hgt = rasterize(scene, x0, y0, nx, ny, res)
    return dict(occ=occ, hgt=hgt, x0=x0, y0=y0, res=res)

def _cells(R, P):
    ix = np.clip(np.floor((P[..., 0] - R["x0"]) / R["res"]).astype(int), 0, R["occ"].shape[1] - 1)
    iy = np.clip(np.floor((P[..., 1] - R["y0"]) / R["res"]).astype(int), 0, R["occ"].shape[0] - 1); return iy, ix

def on_building(R, P):
    iy, ix = _cells(R, np.asarray(P, float)); return R["occ"][iy, ix]

def env_features(R, rx3, tx3, chunk=1500, r_cover=R_COVER):
    """rx3 (N,3), tx3 (3,) -> (N,5) = ENV_COLS. The ray runs from tx height to rx height; a cell blocks when its roof is above the ray.
    The run of occupied cells that starts at the transmitter (its own building) is ignored. Depends on positions and the map only."""
    from scipy import ndimage
    rx3 = np.asarray(rx3, float); tx3 = np.asarray(tx3, float).reshape(3); occ, hgt, res = R["occ"], R["hgt"], R["res"]
    dist = ndimage.distance_transform_edt(~occ) * res; cov = ndimage.uniform_filter(occ.astype(float), size=int(2 * r_cover / res) + 1, mode="constant")
    L = np.hypot(rx3[:, 0] - tx3[0], rx3[:, 1] - tx3[1]); K = int(np.ceil(L.max() / res)) + 2; t = np.linspace(0.0, 1.0, K); out = np.zeros((len(rx3), 5))
    for a in range(0, len(rx3), chunk):
        P = rx3[a:a + chunk]; pts = tx3[None, None, :2] + (P[:, None, :2] - tx3[None, None, :2]) * t[None, :, None]; iy, ix = _cells(R, pts)
        z = tx3[2] + (P[:, 2:3] - tx3[2]) * t[None, :]; o = occ[iy, ix]; h = hgt[iy, ix]; own = np.cumprod(o, axis=1).astype(bool)
        over = np.where(own | ~o, 0.0, np.maximum(h - z, 0.0)); blk = over > 0
        edges = (blk[:, 1:] & ~blk[:, :-1]).sum(1) + blk[:, 0]
        riy, rix = _cells(R, P[:, :2])
        out[a:a + chunk] = np.column_stack([np.log1p(edges), np.log1p(blk.mean(1) * L[a:a + chunk]), np.log1p(over.max(1)), cov[riy, rix], np.log1p(dist[riy, rix])])
    return out

# ------------------------------------------------------------------------------------------------------------ build
def build_city(slug, cfg=CITY_CFG):
    """-> (list of unit names saved, info dict). One unit per transmitter. Raster built once per scene."""
    names, drops, nb, R, k_in, np_in = [], [], 0, None, [], []
    for k, ds in enumerate(rc._dm_datasets(slug, cfg)):
        pos3 = np.asarray(ds.rx_pos, float); tx3 = np.asarray(ds.tx_pos, float).reshape(-1)[:3]
        raw = dict(pos3=pos3, tx3=tx3, power_paths_dbm=rc._path_power_dbm(ds), los=np.asarray(ds.los), freq_ghz=rc._carrier_ghz(ds, slug),
                   name=f"dm|{slug}|{k}", scenario=slug, tx=k)
        u = rc.preprocess_unit(raw, cfg)
        if u is None: drops.append(k); continue
        d = np.linalg.norm(pos3 - tx3, axis=1); p = rc.sum_path_power_dbm(raw["power_paths_dbm"]); los = (np.asarray(raw["los"]) == 1).astype(float)
        y = p + rc.calibrate_power_offset(d, p, los, raw["freq_ghz"], cfg["ptx_dbm"]); keep = np.isfinite(y) & (y > rc.noise_floor_dbm(cfg))
        assert keep.sum() == len(u["y"]), "keep mask does not reproduce preprocess_unit"
        if R is None:
            sc = getattr(ds, "scene", None); nb = len(sc.get_objects("buildings")) if sc is not None else 0
            R = raster_for(sc, pos3, tx3) if nb else None
        side = env_features(R, pos3[keep], tx3) if R is not None else np.zeros((int(keep.sum()), len(ENV_COLS)))
        if R is not None:                                              # alignment validation (path existence is NOT a feature)
            nopath = np.asarray(raw["los"]) == -1
            k_in.append(float(on_building(R, pos3[keep, :2][::5]).mean())); np_in.append(float(on_building(R, pos3[nopath, :2][::5]).mean()) if nopath.sum() else float("nan"))
        u.update(layout=slug, parent=slug, tx_known=1, side=side.astype(np.float32), has_scene=int(R is not None), n_buildings=nb)
        rc.save_unit(u, cfg["cache_dir"]); names.append(u["name"])
    ik = float(np.mean(k_in)) if k_in else float("nan"); inp = float(np.nanmean(np_in)) if np_in and not all(v != v for v in np_in) else float("nan")
    return names, dict(slug=slug, n_units=len(names), n_dropped_units=len(drops), n_buildings=nb, inside_kept=ik, inside_nopath=inp)

def eligible_row(r):
    ik, inp = r.get("inside_kept", float("nan")), r.get("inside_nopath", float("nan"))
    return bool(r["n_units"] >= 1 and r["n_buildings"] >= 10 and ik == ik and inp == inp and ik <= 0.10 and (inp - ik) >= 0.25)

def build_with_retry(slug, cfg=CITY_CFG, waits=(45, 120, 300), pause=4.0):
    """build_city, retrying when the download was refused (HTTP 429 'Too Many Requests' shows up as a missing scenario folder). Pauses `pause` s before every
    attempt that has to download, so a long build does not trip the server's rate limit."""
    import deepmimo as dm
    for i, w in enumerate((0,) + tuple(waits)):
        if w: print(f"  [{slug}] download refused, waiting {w}s (attempt {i + 1}/{len(waits) + 1})"); time.sleep(w)
        try:
            if not os.path.isdir(os.path.join(dm.get_scenarios_dir(), slug)): time.sleep(pause)
            return build_city(slug, cfg)
        except Exception as e:
            if i == len(waits) or ("not found" not in repr(e).lower() and "429" not in repr(e)): raise

def cmd_build(first=None, cfg=CITY_CFG):
    import pandas as pd
    todo = CANDIDATES if first is None else [s for _, ss in groups(CANDIDATES)[:first] for s in ss]
    allrows = pd.read_csv(MANIFEST).to_dict("records") if os.path.exists(MANIFEST) else []
    rows = [r for r in allrows if r.get("feat_version") == FEAT_VERSION and not (isinstance(r.get("error"), str) and r["error"])]; done = {r["slug"] for r in rows}   # error rows are retried
    if len(allrows) != len(rows): print(f"[city] {len(allrows) - len(rows)} manifest rows come from an older feature version, or failed earlier, and will be (re)built")
    for s in todo:
        if s in done: continue
        t0 = time.time()
        try: _, info = build_with_retry(s, cfg)
        except Exception as e: info = dict(slug=s, n_units=0, n_dropped_units=-1, n_buildings=0, inside_kept=float("nan"), inside_nopath=float("nan"), error=repr(e)[:160])
        info["eligible"] = eligible_row(info); info["feat_version"] = FEAT_VERSION; rows.append(info); pd.DataFrame(rows).to_csv(MANIFEST, index=False)
        print(f"{s}: units={info['n_units']} dropped={info['n_dropped_units']} bldgs={info['n_buildings']} on_bldg(signal)={info['inside_kept']:.2f} on_bldg(no path)={info['inside_nopath']:.2f} "
              f"eligible={info['eligible']} ({time.time() - t0:.0f}s) {info.get('error', '')}")
    df = pd.read_csv(MANIFEST); print(f"\n{int(df.eligible.sum())}/{len(df)} eligible -> {MANIFEST}")

def cmd_roles(freeze=False):
    """Before the plan is frozen: roles from assign_roles(eligible). AFTER it is frozen: the frozen roles are read back and NEVER recomputed; a slug that
    became eligible later (e.g. a download that was rate-limited the first time) gets role 'confirm2' = an extra fresh pool declared here, untouched by any
    earlier analysis. Re-running `roles --freeze` after freezing never changes the plan."""
    import pandas as pd
    df = pd.read_csv(MANIFEST); el = df[df.eligible].slug.tolist()
    if os.path.exists(PLAN):
        frozen = json.load(open(PLAN))["plan"]["roles"]; role = {s_: frozen.get(s_, "confirm2") for s_ in el}
        print(f"plan already frozen ({json.load(open(PLAN))['sha256'][:16]}): {len(frozen)} roles kept, {sum(1 for s_ in el if s_ not in frozen)} newly eligible -> confirm2")
        if freeze: print("(--freeze ignored: the plan is already frozen)")
    else: role = assign_roles(el)
    df["role"] = df.slug.map(role).fillna("ineligible"); df.to_csv(MANIFEST, index=False)
    print(df.groupby("role").slug.agg(["size", lambda x: ", ".join(x.str.extract(r"city_(\d+)_")[0].head(12))]).to_string())
    if os.path.exists(PLAN): return
    plan = dict(cfg={k: v for k, v in CITY_CFG.items() if k != "cache_dir"}, env_cols=ENV_COLS, roles=role, rule="p%8: 0-3 source, 4 val, 5 dev, 6-7 confirm; groups by alias")
    h = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
    if freeze:
        os.makedirs(os.path.dirname(PLAN), exist_ok=True); json.dump(dict(sha256=h, plan=plan, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(PLAN, "w"), indent=1)
        print("frozen", h[:16], "->", PLAN)
    else: print("(not frozen; add --freeze once you have read the table)", h[:16])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "roles"]); ap.add_argument("--first", type=int); ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args(); cmd_build(a.first) if a.cmd == "build" else cmd_roles(a.freeze)
