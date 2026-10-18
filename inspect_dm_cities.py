"""inspect_dm_cities.py -- run next to rmbench_core.py. Downloads a FEW candidate scenarios and prints, per scenario, what is needed to
decide whether they can form an independent-layout pool. It reads labels' summary statistics only; it fits and compares NOTHING.

  python inspect_dm_cities.py                      # default 6 scenarios (see CANDIDATES)
  python inspect_dm_cities.py city_2_chicago_3p5   # or name your own

Output: one line per scenario + one line per transmitter, then a CSV (dm_inspect.csv). Send me the printed text (or the CSV).
Downloads can be large: the script prints the on-disk size after each one so you can stop early (Ctrl-C).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd())
import rmbench_core as rc
import deepmimo as dm

CANDIDATES = ["city_0_newyork_3p5", "city_1_losangeles_3p5", "city_21_taito_city_3p5", "city_22_rome_3p5",
              "city_0_newyork_3p5_s", "city_0_newyork_3p5_lwm"]          # the last two: to learn what the _s / _lwm variants are

def folder_mb(path):
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return tot / 1e6

def main(slugs):
    cfg = dict(rc.DEFAULT_CFG)
    rows = []
    for s in slugs:
        print(f"\n===== {s}")
        try:
            t0 = time.time(); dm.download(s)
            try: print(f"  folder: {folder_mb(dm.get_scenario_folder(s)):.0f} MB, download {time.time() - t0:.0f}s")
            except Exception as e: print("  (size unavailable)", repr(e)[:80])
            try:
                for st in dm.get_txrx_sets(s): print("  txrx set:", {k: getattr(st, k, None) for k in ("id", "name", "is_tx", "is_rx", "num_points")})
            except Exception as e: print("  txrx sets unavailable:", repr(e)[:100])
            raws = rc.load_deepmimo_units(s, cfg)
            for r in raws:
                u = rc.preprocess_unit(r, cfg)
                if u is None: print(f"  tx {r['tx']}: dropped (below noise floor / too few points)"); continue
                row = dict(scenario=s, tx=r["tx"], freq_ghz=r["freq_ghz"], n_pts=len(u["y"]), los_frac=float(np.mean(u["los"])), y_std=float(np.std(u["y"])),
                           x_span=float(np.ptp(u["pos"][:, 0])), y_span=float(np.ptp(u["pos"][:, 1])), d_max=float(np.max(u["d"])))
                rows.append(row); print("  ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
        except Exception as e:
            print("  FAILED:", repr(e)[:300])
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows); df.to_csv("dm_inspect.csv", index=False)
        print("\n", df.groupby("scenario").agg(n_tx=("tx", "size"), freq=("freq_ghz", "first"), pts=("n_pts", "median"), los=("los_frac", "mean"), y_std=("y_std", "mean")).round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1:] or CANDIDATES)
