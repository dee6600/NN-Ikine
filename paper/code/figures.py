"""Generate all paper figures from results/*.json|npz (plus small sweeps)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ikcore import REACH, KerasMLP, fk_rpp, fk_rrr, ik_rpp_closed, rrr_branch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "..", "..", "devlopment_project")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical order (slots 1-3) + neutral inks
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 8.5,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def ecdf(ax, e, color, label, ls="-"):
    e = np.sort(e)
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=color, label=label, ls=ls)


audit = json.load(open(os.path.join(RES, "audit.json")))
d = pd.read_csv(os.path.join(DEV, "train_3link.csv")).values
X3, Q3 = d[:, :3], d[:, 3:6]

# ---------------------------------------------------------------------------
# Fig. 2  Dataset pathologies
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15), gridspec_kw=dict(wspace=0.42))
rng = np.random.default_rng(0)
cube = rng.uniform(-15, 15, (400000, 3))
n = np.linalg.norm(cube, axis=1)
ax[0].hist(n[n <= REACH], bins=60, color=C1, label="reachable")
ax[0].hist(n[n > REACH], bins=12, color=C2, label="unreachable")
ax[0].axvline(REACH, color=INK2, lw=0.8, ls="--")
ax[0].set_xlabel(r"target distance $\|\mathbf{p}\|$")
ax[0].set_ylabel("samples")
ax[0].set_yticks([])
ax[0].set_title("(a) sampling cube vs. workspace", loc="left")
ax[0].legend(loc="upper left")
ymax = ax[0].get_ylim()[1]
ax[0].set_ylim(0, ymax * 1.45)
ax[0].annotate(f"{100*np.mean(n>REACH):.1f}% outside\nreach $a_2+a_3$", xy=(21.5, ymax * 0.35), xytext=(23.2, ymax * 1.05), color=INK2, fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))

# (b) elbow branch in a thin z-slab: label discontinuity
sh, el = rrr_branch(Q3, X3)
slab = np.abs(X3[:, 2]) < 1.5
for s, c, lab in ((1, C1, r"elbow $\theta_3\geq0$"), (-1, C2, r"elbow $\theta_3<0$")):
    m = slab & (el == s)
    ax[1].scatter(X3[m, 0], X3[m, 1], s=4, color=c, lw=0, label=lab)
ax[1].set_aspect("equal")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title(r"(b) solver branch, $|z|<1.5$", loc="left")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, handletextpad=0.1, columnspacing=0.6, markerscale=2.5)

# (c) raw joint labels: unwrapped outliers
for i, (c, lab) in enumerate(((C1, r"$q_1$"), (C2, r"$q_2$"), (C3, r"$q_3$"))):
    v = np.sort(np.abs(Q3[:, i]))[::-1]
    ax[2].plot(np.arange(1, len(v) + 1), v, color=c, label=lab)
ax[2].axhline(np.pi, color=INK2, lw=0.8, ls="--")
ax[2].text(3000, np.pi * 1.35, r"$\pi$", color=INK2)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("rank")
ax[2].set_ylabel(r"$|q_i|$ label (rad)")
ax[2].set_title("(c) unwrapped labels", loc="left")
ax[2].legend(loc="lower left")
save(fig, "fig_dataset")

# ---------------------------------------------------------------------------
# Fig. 4  Task-space error of the shipped RRR model
# ---------------------------------------------------------------------------
t = np.load(os.path.join(RES, "rrr20000_test.npz"))
from sklearn.model_selection import train_test_split
Xtr, Xte, Qtr, Qte = train_test_split(X3, Q3, test_size=0.2, random_state=1)
m20 = KerasMLP(os.path.join(DEV, "model", "model20000_rrr.h5"))
e_tr = np.linalg.norm(fk_rrr(m20(Xtr)) - Xtr, axis=1)
e_te = t["cart"]
e_c = np.linalg.norm(fk_rrr(np.tile(Qtr.mean(0), (len(Xte), 1))) - Xte, axis=1)
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw=dict(wspace=0.3))
ecdf(ax[0], e_tr, C1, "train split")
ecdf(ax[0], e_te, C2, "test split")
ecdf(ax[0], e_c, C3, "constant-mean predictor")
ax[0].set_xscale("log")
ax[0].set_xlabel("end-effector position error (length units)")
ax[0].set_ylabel("fraction of targets")
ax[0].set_title("(a) error CDF, model20000_rrr", loc="left")
ax[0].axvline(0.1, color=INK2, lw=0.7, ls=":")
ax[0].legend(loc="upper left")
rho, nrm = t["rho"], t["nrm"]
# (b) median error vs. target distance (singular set at the origin: folded elbow)
for v, c, lab in ((nrm, C1, r"target distance $\|\mathbf{p}\|$"), (rho, C2, r"distance to $z$-axis $\rho$")):
    bins = np.quantile(v, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(v, bins) - 1, 0, 7)
    mid = [np.median(v[idx == k]) for k in range(8)]
    med = [np.median(e_te[idx == k]) for k in range(8)]
    ax[1].plot(mid, med, color=c, marker="o", ms=3.5, label=lab)
ax[1].set_xlabel("distance (length units; octile bins)")
ax[1].set_ylabel("median test error")
ax[1].set_ylim(0, None)
ax[1].set_title("(b) median error vs. target geometry", loc="left")
ax[1].legend(loc="lower left")
save(fig, "fig_rrr_error")

# ---------------------------------------------------------------------------
# Fig. 5  RPP: input-range bug and extrapolation
# ---------------------------------------------------------------------------
mr = KerasMLP(os.path.join(DEV, "model", "model8000_rpp_rad.h5"))
xs = np.linspace(-50, 200, 501)
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.0), gridspec_kw=dict(wspace=0.3))
for yz, c in ((50.0, C1), (100.0, C2), (150.0, C3)):
    P = np.stack([xs, np.full_like(xs, yz), np.full_like(xs, yz)], 1)
    e = np.linalg.norm(fk_rpp(mr(P)) - P, axis=1)
    ax[0].plot(xs, e, color=c, label=f"$y=z={yz:.0f}$")
ax[0].axvspan(0, 2 * np.pi, color=GRID, lw=0)
ax[0].text(9, ax[0].get_ylim()[1] * 0.85, r"training range $x\in[0,2\pi]$", color=INK2, fontsize=7)
ax[0].set_xlabel("target $x$")
ax[0].set_ylabel("position error")
ax[0].set_title("(a) model8000_rpp_rad along an $x$-sweep", loc="left")
ax[0].legend(loc="upper left")
Xr = pd.read_csv(os.path.join(DEV, "train_rpp_rad.csv"), header=None).values[:, :3]
th = np.degrees(ik_rpp_closed(Xr)[:, 0])
P = np.random.default_rng(3).uniform([-200, -200, 5], [200, 200, 200], (20000, 3))
P = P[np.hypot(P[:, 0], P[:, 1]) <= 200]
ax[1].hist(np.degrees(ik_rpp_closed(P)[:, 0]), bins=72, range=(-180, 180), color=C1, density=True,
           label=r"reachable $\theta_1$ (full workspace)")
ax[1].hist(th, bins=72, range=(-180, 180), color=C2, density=True, alpha=0.9, label=r"$\theta_1$ in train_rpp_rad")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"base angle $\theta_1$ (deg)")
ax[1].set_ylabel("density")
ax[1].set_title(r"(b) training covers $\theta_1\in[-49^\circ,0^\circ]$", loc="left")
ax[1].legend(loc="upper left")
save(fig, "fig_rpp")

# ---------------------------------------------------------------------------
# Part II figures (only if the controlled experiments have finished)
# ---------------------------------------------------------------------------
p2 = os.path.join(RES, "improved.json")
if os.path.exists(p2):
    I = json.load(open(p2))
    regs = [("solver_raw", "solver labels (raw)"), ("solver_wrapped", "solver labels, wrapped"),
            ("mixed_branch", "random branch per sample"), ("consistent", "consistent branch, angles"),
            ("consistent_sincos", r"consistent branch, $(\cos,\sin)$")]
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.2), gridspec_kw=dict(wspace=0.75, width_ratios=[1.25, 1]))
    for k, (r, lab) in enumerate(regs):
        med = np.array([row["fresh"]["median"] for row in I[f"{r}|mse"]])
        p95 = np.array([row["fresh"]["p95"] for row in I[f"{r}|mse"]])
        yk = len(regs) - 1 - k
        ax[0].plot([med.min(), med.max()], [yk + 0.12] * 2, color=C1, lw=1.2)
        ax[0].scatter(med, [yk + 0.12] * 3, color=C1, s=12, zorder=4, label="median" if k == 0 else None)
        ax[0].plot([p95.min(), p95.max()], [yk - 0.12] * 2, color=C2, lw=1.2)
        ax[0].scatter(p95, [yk - 0.12] * 3, color=C2, s=12, marker="D", zorder=4, label="95th pct." if k == 0 else None)
    ax[0].set_yticks(range(len(regs)))
    ax[0].set_yticklabels([l for _, l in regs][::-1])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("position error, 20k fresh targets")
    ax[0].set_title("(a) same inputs, different labels (L2)", loc="left")
    ax[0].legend(loc="lower right")
    ax[0].grid(axis="y", visible=False)
    Ns = sorted(int(k) for k in I["scaling"])
    for key, c, lab, mk in (("median", C1, "median", "o"), ("p95", C2, "95th pct.", "D"), ("p99", C3, "99th pct.", "s")):
        m = [np.mean([row["fresh"][key] for row in I["scaling"][str(N)]]) for N in Ns]
        ax[1].plot(Ns, m, color=c, marker=mk, ms=3.5, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("training samples $N$ (exact labels)")
    ax[1].set_ylabel("position error")
    ax[1].set_title("(b) scaling, consistent $(\\cos,\\sin)$", loc="left")
    ax[1].legend(loc="upper right")
    save(fig, "fig_ablation")

print("figures written to", FIG)
