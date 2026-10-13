"""Part I: audit and re-evaluation of the original 2021 artifacts.

Reads the datasets and trained Keras models exactly as shipped in
../../devlopment_project and writes results/audit.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from ikcore import (REACH, D1, KerasMLP, fk_rpp, fk_rrr, ik_rpp_closed,
                    ik_rrr_all, ik_rrr_closed, ikine_lm, keras_categorical_accuracy,
                    rrr_branch, wrap)

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "..", "..", "devlopment_project")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
R = {}


def load(name, header):
    d = pd.read_csv(os.path.join(DEV, name), header=0 if header else None)
    v = d.values.astype(np.float64)
    return v[:, :3], v[:, 3:6]


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(np.max(x)))


# ---------------------------------------------------------------------------
# 1. Dataset audit
# ---------------------------------------------------------------------------
X3, Q3 = load("train_3link.csv", True)
Xr, Qr = load("train_rpp_rad.csv", False)
Xd, Qd = load("train_rpp.csv", False)
Xa, Qa = load("train8000test.csv", True)

res3 = np.linalg.norm(fk_rrr(Q3) - X3, axis=1)
sh, el = rrr_branch(Q3, X3)
R["data_rrr"] = dict(
    n_requested=10000, n_rows=len(X3), n_failed=10000 - len(X3),
    fail_rate=(10000 - len(X3)) / 10000,
    label_fk_residual=stats(res3),
    max_norm_p=float(np.linalg.norm(X3, axis=1).max()),
    frac_unwrapped_any=float(np.mean(np.any(np.abs(Q3) > np.pi, axis=1))),
    frac_unwrapped_q1=float(np.mean(np.abs(Q3[:, 0]) > np.pi)),
    frac_unwrapped_q2=float(np.mean(np.abs(Q3[:, 1]) > np.pi)),
    frac_unwrapped_q3=float(np.mean(np.abs(Q3[:, 2]) > np.pi)),
    q_min=Q3.min(0).tolist(), q_max=Q3.max(0).tolist(), q_std=Q3.std(0).tolist(),
    branch_counts={f"{s:+d}{e:+d}": int(np.sum((sh == s) & (el == e)))
                   for s in (1, -1) for e in (1, -1)},
)

# Monte-Carlo: fraction of the sampling cube [-15,15]^3 outside the workspace
rng = np.random.default_rng(0)
cube = rng.uniform(-15, 15, size=(2_000_000, 3))
R["data_rrr"]["mc_frac_cube_unreachable"] = float(np.mean(np.linalg.norm(cube, axis=1) > REACH))

for tag, X, Q in (("data_rpp_rad", Xr, Qr), ("data_rpp_deg", Xd, Qd), ("data_rpp_analytic", Xa, Qa)):
    res = np.linalg.norm(fk_rpp(Q) - X, axis=1) if tag != "data_rpp_analytic" else None
    R[tag] = dict(n_rows=len(X), x_range=[float(X[:, 0].min()), float(X[:, 0].max())],
                  y_range=[float(X[:, 1].min()), float(X[:, 1].max())],
                  z_range=[float(X[:, 2].min()), float(X[:, 2].max())],
                  q_min=Q.min(0).tolist(), q_max=Q.max(0).tolist())
    if res is not None:
        R[tag]["label_fk_residual"] = stats(res)
        R[tag]["frac_q3_negative"] = float(np.mean(Q[:, 2] < 0))
        R[tag]["frac_q1_unwrapped"] = float(np.mean(np.abs(Q[:, 0]) > np.pi))
# analytic generator (gen.py) uses atan2(y,-x) which is not the IK of the DH model
qa_true = ik_rpp_closed(Xa)
R["data_rpp_analytic"]["q1_mismatch_vs_dh_ik_deg"] = stats(np.degrees(np.abs(wrap(Qa[:, 0] - qa_true[:, 0]))))
R["data_rpp_analytic"]["fk_residual_of_labels"] = stats(np.linalg.norm(fk_rpp(Qa) - Xa, axis=1))

# ---------------------------------------------------------------------------
# 2. Replicating the data-generation solver run (gen.m) with the LM re-implementation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2021)
P = rng.uniform(-15, 15, size=(10000, 3))
fails, iters, unreach, sols = [], [], [], []
for p in P:
    q, it = ikine_lm("rrr", p)
    fails.append(q is None)
    iters.append(it)
    unreach.append(np.linalg.norm(p) > REACH)
    sols.append(q if q is not None else np.full(3, np.nan))
fails, iters, unreach, sols = map(np.array, (fails, iters, unreach, sols))
ok = ~fails
sh2, el2 = rrr_branch(sols[ok], P[ok])
R["replicate_gen_m"] = dict(
    n=10000, fail_rate=float(fails.mean()),
    frac_unreachable=float(unreach.mean()),
    fails_unreachable=int(np.sum(fails & unreach)),
    fails_reachable=int(np.sum(fails & ~unreach)),
    reachable_failed_frac=float(np.sum(fails & ~unreach) / np.sum(~unreach)),
    iters_success=stats(iters[ok]),
    frac_unwrapped_any=float(np.mean(np.any(np.abs(sols[ok]) > np.pi, axis=1))),
    branch_counts={f"{s:+d}{e:+d}": int(np.sum((sh2 == s) & (el2 == e)))
                   for s in (1, -1) for e in (1, -1)},
)
np.savez(os.path.join(OUT, "replicate_gen_m.npz"), P=P, fails=fails, iters=iters, sols=sols)

# RPP replications (gen_rpp.m: x~U[0,2pi]; earlier version x~U[0,360])
for tag, xmax in (("replicate_gen_rpp_rad", 2 * np.pi), ("replicate_gen_rpp_deg", 360.0)):
    rng = np.random.default_rng(7)
    Pp = np.stack([rng.uniform(0, xmax, 3000), rng.uniform(5, 200, 3000), rng.uniform(5, 200, 3000)], 1)
    f = []
    it_ok = []
    for p in Pp:
        q, it = ikine_lm("rpp", p)
        f.append(q is None)
        if q is not None:
            it_ok.append(it)
    R[tag] = dict(n=3000, fail_rate=float(np.mean(f)), iters_success=stats(it_ok))

# ---------------------------------------------------------------------------
# 3. Re-evaluating the shipped models
# ---------------------------------------------------------------------------
def eval_model(path, X, Q, robot, split_seed):
    m = KerasMLP(path)
    Xtr, Xte, Qtr, Qte = train_test_split(X, Q, test_size=0.2, random_state=split_seed)
    out = dict(arch=m.arch, n_params=int(m.n_params),
               loss=m.train_cfg["loss"], metrics=m.train_cfg["metrics"])
    fk = fk_rrr if robot == "rrr" else fk_rpp
    for split, (Xs, Qs) in dict(train=(Xtr, Qtr), test=(Xte, Qte)).items():
        Yp = m(Xs)
        err_raw = np.abs(Yp - Qs)
        d = dict(n=len(Xs),
                 keras_accuracy=keras_categorical_accuracy(Qs, Yp),
                 mae_raw=err_raw.mean(0).tolist(),
                 cart_err=stats(np.linalg.norm(fk(Yp) - Xs, axis=1)))
        if robot == "rrr":
            d["mae_wrapped_deg"] = np.degrees(np.abs(wrap(Yp - Qs)).mean(0)).tolist()
            d["frac_cart_err_lt_0p1"] = float(np.mean(np.linalg.norm(fk(Yp) - Xs, axis=1) < 0.1))
            d["frac_cart_err_lt_1"] = float(np.mean(np.linalg.norm(fk(Yp) - Xs, axis=1) < 1.0))
            # error relative to workspace
            d["cart_err_rel_reach"] = float(np.median(np.linalg.norm(fk(Yp) - Xs, axis=1)) / REACH)
            # baseline: predict the training-set mean joint vector
            d["cart_err_const_baseline"] = stats(np.linalg.norm(fk(np.tile(Qtr.mean(0), (len(Xs), 1))) - Xs, axis=1))
        else:
            d["mae_deg_q1"] = float(np.degrees(np.abs(wrap(Yp[:, 0] - Qs[:, 0])).mean()))
        out[split] = d
    # accuracy of the argmax heuristic "always predict the majority argmax class"
    out["keras_accuracy_majority_class"] = float(np.bincount(np.argmax(Qte, 1)).max() / len(Qte))
    return out, m


MD = os.path.join(DEV, "model")
R["models"] = {}
for name in ("model2000_rrr", "model10000_rrr", "model15000_rrr", "model20000_rrr"):
    R["models"][name], _ = eval_model(os.path.join(MD, name + ".h5"), X3, Q3, "rrr", 1)
R["models"]["model8000_rpp_rad"], m_rpp = eval_model(os.path.join(MD, "model8000_rpp_rad.h5"), Xr, Qr, "rpp", 1)
R["models"]["model8000_rpp"], _ = eval_model(os.path.join(MD, "model8000_rpp.h5"), Xd, Qd, "rpp", 1)
R["models"]["model2000"], _ = eval_model(os.path.join(MD, "model2000.h5"), Xa, Qa, "rpp", 0)

# Error breakdown of the flagship RRR model on its test split
m20 = KerasMLP(os.path.join(MD, "model20000_rrr.h5"))
Xtr, Xte, Qtr, Qte = train_test_split(X3, Q3, test_size=0.2, random_state=1)
Yp = m20(Xte)
cart = np.linalg.norm(fk_rrr(Yp) - Xte, axis=1)
unwr = np.any(np.abs(Qte) > np.pi, axis=1)
R["rrr20000_breakdown"] = dict(
    cart_err_wrapped_labels=stats(cart[~unwr]),
    cart_err_unwrapped_labels=stats(cart[unwr]),
    n_unwrapped=int(unwr.sum()),
)
# Spatial coherence of the solver's branch choice and its effect on the model
from sklearn.neighbors import NearestNeighbors
sh_all, el_all = rrr_branch(Q3, X3)
br_all = sh_all * 10 + el_all
itr_i, ite_i = train_test_split(np.arange(len(X3)), test_size=0.2, random_state=1)
dist, nbr = NearestNeighbors(n_neighbors=1).fit(X3[itr_i]).kneighbors(X3[ite_i])
s_p, e_p = rrr_branch(Yp, Xte)
same = (s_p * 10 + e_p) == br_all[ite_i]
R["rrr20000_breakdown"].update(
    nn1_median_dist=float(np.median(dist)),
    nn1_same_branch=float(np.mean(br_all[itr_i][nbr[:, 0]] == br_all[ite_i])),
    nn1_same_elbow=float(np.mean(el_all[itr_i][nbr[:, 0]] == el_all[ite_i])),
    nn1_same_shoulder=float(np.mean(sh_all[itr_i][nbr[:, 0]] == sh_all[ite_i])),
    pred_on_label_branch=float(np.mean(same)),
    cart_err_on_label_branch=stats(cart[same]),
    cart_err_off_label_branch=stats(cart[~same]),
    frac_labels_with_negative_joint=float(np.mean(np.any(Q3 < 0, axis=1))),
)
# Error vs distance to singular sets
rho = np.hypot(Xte[:, 0], Xte[:, 1])
nrm = np.linalg.norm(Xte, axis=1)
np.savez(os.path.join(OUT, "rrr20000_test.npz"), X=Xte, Q=Qte, Y=Yp, cart=cart, rho=rho, nrm=nrm)

# ---------------------------------------------------------------------------
# 4. The headline test pose [1, 4, 9]
# ---------------------------------------------------------------------------
p = np.array([[1.0, 4.0, 9.0]])
case = dict(norm_p=float(np.linalg.norm(p)))
allq = ik_rrr_all(p)[:, 0, :]
case["closed_form_deg"] = {f"shoulder{s:+d}_elbow{e:+d}": np.degrees(allq[i]).round(2).tolist()
                           for i, (s, e) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])}
q_lm, it = ikine_lm("rrr", p[0])
case["ikine_lm_from_zero_deg"] = np.degrees(q_lm).round(2).tolist()
case["ikine_lm_iters"] = int(it)
ynn = m20(p)[0]
case["nn20000_deg"] = np.degrees(ynn).round(2).tolist()
case["nn20000_cart_err"] = float(np.linalg.norm(fk_rrr(ynn) - p))
case["nn20000_fk"] = fk_rrr(ynn)[0].round(3).tolist()
slide = np.radians([78.7, 4.3, 120.1])
case["slide_claim_fk"] = fk_rrr(slide)[0].round(3).tolist()
case["slide_claim_cart_err"] = float(np.linalg.norm(fk_rrr(slide) - p))
# what rrr_3_link_NN.m actually animates: hard-coded q=[1,0.05,1] rad
hard = np.array([1.0, 0.05, 1.0])
case["nn_script_hardcoded_deg"] = np.degrees(hard).round(2).tolist()
case["nn_script_hardcoded_fk"] = fk_rrr(hard)[0].round(3).tolist()
case["nn_script_hardcoded_cart_err"] = float(np.linalg.norm(fk_rrr(hard) - p))


def animated_final(qdeg):
    """Final pose reached by `for qi = deg2rad(0:5:q)` loops (MATLAB colon semantics).
    Returns None if any loop is empty (q<0 -> undefined loop var -> catch -> 'Singularity' msg)."""
    out = []
    for v in qdeg:
        if v < 0:
            return None
        out.append(np.floor(v / 5.0) * 5.0)
    return out


qa = animated_final(np.degrees(q_lm))
case["toolbox_script_animated_final_deg"] = qa
if qa is not None:
    case["toolbox_script_animated_cart_err"] = float(np.linalg.norm(fk_rrr(np.radians(qa)) - p))
qa = animated_final(np.degrees(hard))
case["nn_script_animated_final_deg"] = qa
case["nn_script_animated_cart_err"] = float(np.linalg.norm(fk_rrr(np.radians(qa)) - p))
R["case_149"] = case

# Fraction of LM solutions (from q0=0) that the animation loop would reject as "singular"
sols_ok = sols[ok]
R["replicate_gen_m"]["frac_solutions_with_negative_joint"] = float(np.mean(np.any(sols_ok < 0, axis=1)))

# RPP demo inputs: [1,4,9] (hard-coded in py_rpp.py) and [200,200,100] (script comment)
rpp_cases = {}
for p in ([1.0, 4.0, 9.0], [200.0, 200.0, 100.0], [3.0, 100.0, 100.0]):
    pp = np.array([p])
    y = m_rpp(pp)[0]
    rpp_cases[str(p)] = dict(nn=y.round(4).tolist(), exact=ik_rpp_closed(pp)[0].round(4).tolist(),
                             cart_err=float(np.linalg.norm(fk_rpp(y) - pp)))
R["case_rpp"] = rpp_cases

# ---------------------------------------------------------------------------
# 5. Provenance of the slide-9 comparison plot
# ---------------------------------------------------------------------------
idx_all = np.arange(len(X3))
itr, ite = train_test_split(idx_all, test_size=0.2, random_state=1)
first10 = np.arange(10)
Y10 = m20(X3[:10])
R["slide9"] = dict(
    rows=first10.tolist(),
    in_train_split=[bool(i in set(itr)) for i in first10],
    q_label=Q3[:10].round(4).tolist(),
    q_pred=Y10.round(4).tolist(),
    cart_err=np.linalg.norm(fk_rrr(Y10) - X3[:10], axis=1).round(4).tolist(),
)
np.savez(os.path.join(OUT, "slide9.npz"), Q=Q3[:10], Y=Y10)

with open(os.path.join(OUT, "audit.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps(R, indent=1))
