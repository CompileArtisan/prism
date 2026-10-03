"""Pass-1 progress from derived.json (read-only, safe while the kernel runs).  Usage:  watch -n 30 python3 watch_progress.py"""
import json, os, time, sys
P = sys.argv[1] if len(sys.argv) > 1 else "derived.json"
try:
    d = json.load(open(P))
except Exception as e:
    print("cannot read", P, "->", e, "(it may be mid-write; try again)"); sys.exit()

def done(e):  # derived, not a legacy default / placeholder
    return e["cell"] != "legacy" and not str(e["source"]).startswith("assumed")

def cnt(name, model=None, cell=None):
    n = 0
    for k, e in d.items():
        nm, scn, dens, mo = (k.split("|") + [""] * 4)[:4]
        if nm == name and scn and dens and (model is None or mo == model) and done(e) and (cell is None or e["cell"] == cell):
            n += 1
    return n

def flag(name, model=""):
    e = d.get(f"{name}|||{model}"); return bool(e and done(e))

rows = [
    ("N7  knn_k per scn x density",        cnt("knn_k"), 12),
    ("N8  GCN tuned (scn x dens)",         cnt("hidden", "GCN"), 12),
    ("N8  GAT tuned (scn x dens)",         cnt("hidden", "GAT"), 12),
    ("N8  Hybrid tuned (scn x dens)",      cnt("hidden", "Hybrid"), 12),
    ("N8  val_scheme (scn x dens)",        cnt("val_scheme"), 12),
    ("N8  patience GCN / GAT",             int(flag("patience", "GCN")) + int(flag("patience", "GAT")), 2),
    ("N9  huber_sigma (scn x dens)",       cnt("huber_sigma_db"), 12),
    ("N9  Star tuned (scn x dens)",        cnt("star_K"), 12),
    ("N10 spatial_folds / kmeans_n_init",  int(flag("spatial_folds")) + int(flag("kmeans_n_init")), 2),
    ("N10 num_seeds",                      int(flag("num_seeds")), 1),
]
age = time.time() - os.path.getmtime(P)
print(f"{P}: {len(d)} entries, last write {age/60:.1f} min ago")
for name, a, b in rows:
    print(f"  {'#'*int(10*a/b):<10} {a:>2}/{b:<2}  {name}")
print("COMPLETE" if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(P)), "derived.complete")) else "not complete yet (derived.complete missing)")
