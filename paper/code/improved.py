"""Part II: controlled experiments on what makes learned IK work for the RRR arm.

All regimes share the SAME input positions (the original 80% training split of
train_3link.csv) and the same network/optimiser; only the target labels change.
A final regime scales up with exact closed-form labels. Writes results/improved.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from ikcore import (REACH, KerasMLP, fk_rrr, ik_rrr_all, ik_rrr_closed,
                    ikine_lm, rrr_branch, timeit, wrap)
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "..", "..", "devlopment_project")
OUT = os.path.join(HERE, "..", "results")
torch.set_num_threads(4)

d = pd.read_csv(os.path.join(DEV, "train_3link.csv")).values.astype(np.float64)
X, Q = d[:, :3], d[:, 3:6]
Xtr, Xte, Qtr, Qte = train_test_split(X, Q, test_size=0.2, random_state=1)

# fresh, uniformly distributed reachable test targets (same cube, |p| <= 20)
rng = np.random.default_rng(123)
Pf = rng.uniform(-15, 15, size=(40000, 3))
Pf = Pf[np.linalg.norm(Pf, axis=1) <= REACH][:20000]


def labels(regime, P, Qsolver=None, rng=None):
    if regime == "solver_raw":
        return Qsolver
    if regime == "solver_wrapped":
        return wrap(Qsolver)
    if regime == "mixed_branch":
        allq = ik_rrr_all(P)                       # (4,N,3)
        k = rng.integers(0, 4, len(P))
        return allq[k, np.arange(len(P))]
    if regime in ("consistent", "consistent_sincos"):
        return ik_rrr_closed(P, +1, +1)
    raise ValueError(regime)


def encode(Y, sincos):
    return np.concatenate([np.cos(Y), np.sin(Y)], 1) if sincos else Y


def decode(Z, sincos):
    return np.arctan2(Z[:, 3:], Z[:, :3]) if sincos else Z


class MLP(torch.nn.Module):
    def __init__(self, out, width=256, depth=3):
        super().__init__()
        layers, w = [], 3
        for _ in range(depth):
            layers += [torch.nn.Linear(w, width), torch.nn.ReLU()]
            w = width
        layers.append(torch.nn.Linear(w, out))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x / 15.0)                       # fixed input scaling


def train(P, Y, sincos, seed, steps=40000, bs=512, lr=2e-3, loss_fn="mse"):
    torch.manual_seed(seed)
    Z = torch.tensor(encode(Y, sincos), dtype=torch.float32)
    Pt = torch.tensor(P, dtype=torch.float32)
    net = MLP(Z.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        idx = torch.randint(0, len(Pt), (bs,), generator=g)
        r = net(Pt[idx]) - Z[idx]
        loss = torch.mean(r ** 2) if loss_fn == "mse" else torch.mean(torch.abs(r))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return net


def predict(net, P, sincos):
    with torch.no_grad():
        Z = net(torch.tensor(P, dtype=torch.float32)).double().numpy()
    return decode(Z, sincos)


def cart(Y, P):
    return np.linalg.norm(fk_rrr(Y) - P, axis=1)


def summ(e):
    return dict(mean=float(e.mean()), median=float(np.median(e)),
                p95=float(np.percentile(e, 95)), p99=float(np.percentile(e, 99)),
                lt_0p1=float(np.mean(e < 0.1)), lt_0p5=float(np.mean(e < 0.5)))


SEEDS = (0, 1, 2)
REGIMES = [("solver_raw", False), ("solver_wrapped", False), ("mixed_branch", False),
           ("consistent", False), ("consistent_sincos", True)]
RES = {"n_train": int(len(Xtr)), "n_test_split": int(len(Xte)), "n_fresh": int(len(Pf))}
nets = {}
for loss_fn in ("l1", "mse"):
    for reg, sc in REGIMES:
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            Y = labels(reg, Xtr, Qtr, np.random.default_rng(seed))
            net = train(Xtr, Y, sc, seed, loss_fn=loss_fn)
            e_te = cart(predict(net, Xte, sc), Xte)
            e_fr = cart(predict(net, Pf, sc), Pf)
            rows.append(dict(seed=seed, test=summ(e_te), fresh=summ(e_fr), secs=time.time() - t0))
            print(loss_fn, reg, seed, rows[-1]["test"]["median"], rows[-1]["fresh"]["median"],
                  f"{time.time()-t0:.0f}s", flush=True)
        RES[f"{reg}|{loss_fn}"] = rows

# Scaling with exact labels: N samples drawn uniformly in the reachable cube
SCALE = {}
for N in (7344, 30000, 120000):
    rows = []
    for seed in SEEDS:
        r = np.random.default_rng(1000 + seed)
        P = r.uniform(-15, 15, size=(int(N * 1.2), 3))
        P = P[np.linalg.norm(P, axis=1) <= REACH][:N]
        net = train(P, ik_rrr_closed(P, +1, +1), True, seed, steps=60000)
        e_fr = cart(predict(net, Pf, True), Pf)
        e_te = cart(predict(net, Xte, True), Xte)
        rows.append(dict(seed=seed, test=summ(e_te), fresh=summ(e_fr)))
        nets[("scale", N, seed)] = net
        print("scale", N, seed, rows[-1]["fresh"]["median"], flush=True)
    SCALE[N] = rows
RES["scaling"] = SCALE
best = nets[("scale", 120000, 0)]

# Original shipped model on the fresh set, for reference
m20 = KerasMLP(os.path.join(DEV, "model", "model20000_rrr.h5"))
RES["original_model20000_fresh"] = summ(cart(m20(Pf), Pf))
RES["original_model20000_test"] = summ(cart(m20(Xte), Xte))

# Error vs geometry for the best learned model and the original model
Yb = predict(best, Pf, True)
np.savez(os.path.join(OUT, "geometry_errors.npz"), P=Pf, e_best=cart(Yb, Pf), e_orig=cart(m20(Pf), Pf))

# ---------------------------------------------------------------------------
# Hybrid: learned warm start + LM refinement, vs. LM from q0 = 0
# ---------------------------------------------------------------------------
sub = Pf[:3000]
H = {}
inits = {
    "zero": np.zeros_like(sub),
    "nn_original": m20(sub),
    "nn_consistent": predict(best, sub, True),
}
for name, Q0 in inits.items():
    its, fails, branch_ok = [], 0, []
    for p, q0 in zip(sub, Q0):
        q, it = ikine_lm("rrr", p, q0=q0)
        if q is None:
            fails += 1
            continue
        its.append(it)
        s, e = rrr_branch(q[None], p[None])
        branch_ok.append(s[0] == 1 and e[0] == 1)
    its = np.array(its)
    H[name] = dict(fail_rate=fails / len(sub), iters_mean=float(its.mean()),
                   iters_median=float(np.median(its)), iters_p95=float(np.percentile(its, 95)),
                   frac_on_front_elbowup_branch=float(np.mean(branch_ok)))
    print("hybrid", name, H[name], flush=True)
RES["hybrid"] = H

# ---------------------------------------------------------------------------
# Latency (single query, Python/NumPy on CPU) and batched throughput
# ---------------------------------------------------------------------------
p1 = Pf[17]
W = [(l.weight.detach().double().numpy().T, l.bias.detach().double().numpy())
     for l in best.f if isinstance(l, torch.nn.Linear)]


def np_best(x):
    h = x / 15.0
    for i, (Wi, bi) in enumerate(W):
        h = h @ Wi + bi
        if i < len(W) - 1:
            h = np.maximum(h, 0)
    return np.arctan2(h[..., 3:], h[..., :3])


q_nn = np_best(p1[None])[0]
T = dict(
    closed_form=timeit(lambda: ik_rrr_closed(p1[None])),
    nn_original_233k=timeit(lambda: m20(p1[None])),
    nn_consistent_133k=timeit(lambda: np_best(p1[None])),
    lm_from_zero=timeit(lambda: ikine_lm("rrr", p1), n=300),
    lm_from_nn=timeit(lambda: ikine_lm("rrr", p1, q0=q_nn), n=300),
)
B = Pf[:10000]
T["batched_per_query"] = dict(
    closed_form=timeit(lambda: ik_rrr_closed(B), n=20, warmup=3) / len(B),
    nn_original_233k=timeit(lambda: m20(B), n=20, warmup=3) / len(B),
    nn_consistent_133k=timeit(lambda: np_best(B), n=20, warmup=3) / len(B),
)
RES["latency_s"] = T
RES["n_params_consistent"] = int(sum(p.numel() for p in best.parameters()))
print(json.dumps(T, indent=1))

with open(os.path.join(OUT, "improved.json"), "w") as f:
    json.dump(RES, f, indent=1)
torch.save(best.state_dict(), os.path.join(OUT, "rrr_consistent_sincos_120k.pt"))
