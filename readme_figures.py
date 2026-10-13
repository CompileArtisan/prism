"""make_readme_figures.py -- regenerates every figure in README.md.

Provenance: every number below is transcribed from the saved notebook outputs
(radiomap_6g_sweep, radiomap_ext_real_crosslayout, radiomap_ext_conf2, radiomap_ext_conf3_kriging_attention) or from
the previous README (which itself came from those notebooks). Nothing is re-estimated here. Where a value is computed
from the parent-level mean RMSE tables (not from a Holm-tested contrast) the comment says so.
Run:  python make_readme_figures.py   (writes ./figures/*.png)
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
COL = {"trend": "#9aa0a6", "idw": "#1f77b4", "krig_global": "#ff7f0e", "kgnn": "#d62728", "kgnn_prior": "#9467bd", "gb_geo": "#8c564b", "ens_geo": "#2ca02c"}

def heat(ax, M, rows, cols, stars=None, vmax=None, cmap="RdBu", fmt="{:+.1f}"):
    M = np.array(M, float); vmax = vmax or np.nanmax(np.abs(M))
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]): continue
            s = (stars[i][j] if stars is not None else "")
            ax.text(j, i, fmt.format(M[i, j]) + s, ha="center", va="center", fontsize=8.5, color="black" if abs(M[i, j]) < 0.65 * vmax else "white")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

# ---------------------------------------------------------------------------------------------------------------- Fig 1
rows = ["DeepMIMO i1_2p5, main sweep (30/100/300)", "DeepMIMO, round 1 tuned (30/100/300)", "DeepMIMO, round 2 (30/100/300)",
        "Wall sim, round 1 tuned (20/50/100)", "Wall sim, round 2 (20/50/100)",
        "UJI Wi-Fi, round 1 tuned (5/10/20)", "UJI Wi-Fi, round 2 (5/10/20)", "UJI Wi-Fi, round 3 (5/10/20) \u2020",
        "SOD Wi-Fi, round 2 (5/10/20)", "SOD Wi-Fi, round 3 (5/10/20) \u2020"]
M = [[6.5, 7.3, 5.7], [7.1, 6.9, 5.8], [5.6, 6.5, 5.6], [6.0, 3.4, 2.7], [4.4, 3.4, 2.8],
     [4.2, 2.7, 1.3], [4.0, 2.1, 1.4],
     [(9.622 - 9.310) / 9.622 * 100, (7.610 - 7.454) / 7.610 * 100, (6.718 - 6.710) / 6.718 * 100],     # computed from mean-RMSE table (round 3), not Holm-tested
     [2.7, 4.1, 4.8],
     [(9.570 - 9.300) / 9.570 * 100, (7.326 - 6.989) / 7.326 * 100, (6.061 - 5.810) / 6.061 * 100]]     # computed from mean-RMSE table (round 3), not Holm-tested
S = [["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"], ["*", "", ""], ["*", "*", ""], ["", "", ""], ["", "", ""], ["", "", ""]]
fig, ax = plt.subplots(figsize=(6.6, 4.3))
heat(ax, M, rows, ["fewest labels", "middle", "most labels"], S, vmax=9, cmap="Reds")
ax.set_title("KGNN vs IDW, random split: relative RMSE gain (%), positive = KGNN better", fontsize=9.5)
fig.text(0.01, -0.02, "* Holm-significant at parent level.  \u2020 computed from the parent-level mean-RMSE table, no paired test run.\nSOD round 2/3 has 5 building-floors only, so no test can reach significance.", fontsize=7.5, ha="left", va="top")
fig.savefig("figures/fig1_kgnn_vs_idw_replication.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 2  (round 3 mean RMSE, parent-level)
R3 = {
 "UJI Wi-Fi, random split": ([5, 10, 20], {"trend": [10.478, 8.265, 7.137], "idw": [9.622, 7.610, 6.718], "krig_global": [9.730, 7.789, 6.761],
                              "kgnn": [9.310, 7.454, 6.710], "kgnn_prior": [9.150, 7.511, 6.733], "gb_geo": [12.753, 9.009, 8.016]}),
 "UJI Wi-Fi, blocked split": ([5, 10, 20], {"trend": [10.835, 9.955, 9.238], "idw": [10.428, 9.449, 8.694], "krig_global": [10.479, 9.585, 8.805],
                              "kgnn": [10.124, 9.330, 8.388], "kgnn_prior": [9.957, 9.331, 8.701], "gb_geo": [12.362, 9.894, 8.891]}),
 "SOD Wi-Fi, random split": ([5, 10, 20, 60], {"trend": [11.880, 8.686, 7.367, 6.643], "idw": [9.570, 7.326, 6.061, 4.986], "krig_global": [10.402, 7.436, 6.051, 4.959],
                              "kgnn": [9.300, 6.989, 5.810, 4.825], "kgnn_prior": [9.489, 7.543, 6.483, 5.423], "gb_geo": [12.468, 7.705, 6.177, 4.762]}),
}
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, (t, (x, d)) in zip(axs, R3.items()):
    for k, v in d.items():
        ax.plot(x, v, "-o", ms=3.5, lw=2 if k in ("kgnn", "idw") else 1.2, color=COL[k], label=k, ls="--" if k == "trend" else "-")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.set_title(t, fontsize=9.5); ax.set_xlabel("labelled points per unit")
axs[0].set_ylabel("RMSE (dB), parent-level mean"); axs[2].legend(fontsize=7.5, frameon=False)
fig.suptitle("Round 3, real Wi-Fi: absolute RMSE (all differences between the top methods are 0.1-0.8 dB)", fontsize=10, y=1.03)
fig.savefig("figures/fig2_real_wifi_rmse_vs_labels.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 3  (round 2, per building, random split)
bld = ["SOD b1 (CETC331)", "SOD b2 (HCXY)", "SOD b3 (SYL)", "UJI b0", "UJI b1", "UJI b2"]
G = {5: [11.57, 8.99, 7.35, 9.17, 6.42, 8.65], 10: [6.87, 3.40, 1.60, 2.56, 0.54, 3.52], 20: [5.21, 4.64, 6.68, -0.11, 2.12, 1.31]}
fig, ax = plt.subplots(figsize=(6.4, 3.4)); w = 0.26
for i, (k, c) in enumerate(zip(G, ["#08519c", "#3182bd", "#9ecae1"])):
    ax.bar(np.arange(6) + (i - 1) * w, G[k], w, label=f"{k} labels", color=c)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(6)); ax.set_xticklabels(bld, rotation=20, ha="right")
ax.set_ylabel("RMSE gain of KGNN over global kriging (%)"); ax.legend(frameon=False, title="round 2, random split", fontsize=8)
ax.set_title("Per-building view: KGNN vs ordinary kriging", fontsize=9.5)
fig.savefig("figures/fig3_per_building_kgnn_vs_kriging.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 4  (GNNs vs IDW, random split)
rows = ["gcn  (DeepMIMO, main sweep)", "gcn_krig  (DeepMIMO, main sweep)",
        "gcn_knn tuned  (UJI)", "gat_del tuned  (UJI)", "gcn_knn tuned  (wall)", "gat_del tuned  (wall)", "gcn_knn tuned  (DeepMIMO)", "gat_del tuned  (DeepMIMO)"]
M = [[-13.0, -9.5, -9.3], [-2.5, -0.5, -0.2], [-3.2, -4.1, -3.3], [-2.3, -7.3, -6.1], [-10.3, -8.7, -9.1], [-10.2, -8.2, -9.0], [-13.8, -10.3, -9.8], [-10.2, -7.4, -6.6]]
S = [["*", "*", "*"], ["*", "", ""], ["", "", ""], ["", "*", "*"], ["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
heat(ax, M, rows, ["fewest labels", "middle", "most labels"], S, vmax=14, cmap="RdBu")
ax.set_title("Graph networks vs IDW, random split: RMSE gain (%), negative = GNN worse", fontsize=9.5)
fig.text(0.01, -0.02, "* Holm-significant. No GNN variant is significantly better than IDW in any cell.", fontsize=7.5, ha="left", va="top")
fig.savefig("figures/fig4_gnn_vs_idw.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 5  (round 3, meta-learned prior vs from-scratch KGNN, per building)
rows = ["SOD b1 (CETC331)", "SOD b2 (HCXY)", "SOD b3 (SYL)", "UJI b0", "UJI b1", "UJI b2"]
M = [[-5.55, -12.28, -16.17], [4.87, -1.50, -1.10], [-1.14, -5.07, -11.35], [2.81, -0.02, 0.30], [2.14, -1.98, -0.54], [0.67, -0.88, -0.82]]
fig, ax = plt.subplots(figsize=(4.8, 3.1))
heat(ax, M, rows, ["5 labels", "10", "20"], None, vmax=16, cmap="RdBu")
ax.set_title("Round 3: meta-learned prior vs KGNN from scratch\nrandom split, RMSE gain (%), negative = prior worse", fontsize=9)
fig.savefig("figures/fig5_prior_vs_scratch_per_building.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 6  (wall simulator, round 3 + round 2, random split; A vs rk_side)
dens = ["20 labels", "50", "100"]
V = {"kgnn_side (round 2)": [-0.5, 0.5, 1.3], "ens_side = kgnn_side + rk_side": [0.0, 0.9, 1.5], "ens_sk = kgnn_prior_sk + rk_side": [-0.0, -0.1, -0.1], "kgnn_prior_sk (side-aware prior kernel)": [-0.6, -1.5, -2.5]}
fig, ax = plt.subplots(figsize=(6.4, 3.0)); w = 0.2
for i, (k, v) in enumerate(V.items()): ax.bar(np.arange(3) + (i - 1.5) * w, v, w, label=k)
ax.axhline(0, color="k", lw=0.8); ax.axhspan(-2, 2, color="grey", alpha=0.12, lw=0); ax.set_xticks(range(3)); ax.set_xticklabels(dens)
ax.set_ylabel("RMSE gain over rk_side (%)"); ax.legend(frameon=False, fontsize=7.2, loc="lower left"); ax.set_ylim(-3.2, 2.2)
ax.set_title("Wall simulator: nothing reliably beats ridge + IDW (grey band = \u00b12% equivalence margin)", fontsize=9)
fig.savefig("figures/fig6_wall_side_information.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- Fig 7  (cost)
t = {"idw": .004, "kgnn_prior": .006, "kgnn_prior_sk": .012, "rk_side": .012, "krig_global": .172, "gb_geo": .253, "kgnn_priorft": .121, "kgnn": .697, "kgnn_side": .825, "ens_geo": .421}
t = dict(sorted(t.items(), key=lambda kv: kv[1]))
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.barh(list(t), list(t.values()), color="#4c72b0"); ax.set_xscale("log"); ax.set_xlabel("seconds per prediction job (training included, 1 core)")
ax.set_title("Round 3 cost per prediction", fontsize=9.5)
for i, v in enumerate(t.values()): ax.text(v * 1.08, i, f"{v:.3f}", va="center", fontsize=7.5)
fig.savefig("figures/fig7_cost.png"); plt.close(fig)
print("wrote", sorted(os.listdir("figures")))

