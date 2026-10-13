"""Emit LaTeX tables (paper/tables/*.tex) directly from results/*.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(TAB, exist_ok=True)
A = json.load(open(os.path.join(RES, "audit.json")))


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ---------------------------------------------------------------------------
# Table: shipped models, Keras "accuracy" vs task-space error
# ---------------------------------------------------------------------------
rows = [
    ("model2000\\_rrr", "RRR", "test"), ("model10000\\_rrr", "RRR", "test"),
    ("model15000\\_rrr", "RRR", "test"), ("model20000\\_rrr", "RRR", "test"),
    ("model20000\\_rrr", "RRR", "train"),
    ("model8000\\_rpp\\_rad", "RPP", "test"), ("model8000\\_rpp", "RPP", "test"),
]
L = []
for name, robot, split in rows:
    m = A["models"][name.replace("\\_", "_")][split]
    M = A["models"][name.replace("\\_", "_")]
    c = m["cart_err"]
    joint = (", ".join(f"{v:.0f}" for v in m["mae_wrapped_deg"]) if robot == "RRR"
             else f"{m['mae_deg_q1']:.1f} ($q_1$)")
    lt1 = pct(m["frac_cart_err_lt_1"]) if robot == "RRR" else "--"
    L.append(f"\\texttt{{{name}}} & {robot} & {split} & {pct(m['keras_accuracy'])} & "
             f"{pct(M['keras_accuracy_majority_class'])} & {joint} & {f(c['median'])} & "
             f"{f(c['mean'])} & {f(c['p95'])} & {lt1} \\\\")
    if name == "model20000\\_rrr" and split == "train":
        L.append("\\midrule")
L.insert(4, "")
open(os.path.join(TAB, "tab_models.tex"), "w").write("\n".join(l for l in L if l) + "\n")

# ---------------------------------------------------------------------------
# Tables from the controlled study
# ---------------------------------------------------------------------------
p2 = os.path.join(RES, "improved.json")
if os.path.exists(p2):
    I = json.load(open(p2))
    regs = [("solver_raw", "Solver labels, as generated"),
            ("solver_wrapped", "Solver labels, wrapped to $(-\\pi,\\pi]$"),
            ("mixed_branch", "Closed-form, random branch per sample"),
            ("consistent", "Closed-form, one branch, angle output"),
            ("consistent_sincos", "Closed-form, one branch, $(\\cos,\\sin)$ output")]

    def ms(rows, split, key):
        v = np.array([r[split][key] for r in rows])
        return v.mean(), v.std()

    L = []
    for reg, lab in regs:
        cells = []
        for loss in ("l1", "mse"):
            rr = I[f"{reg}|{loss}"]
            m, s = ms(rr, "fresh", "median")
            p, ps = ms(rr, "fresh", "p95")
            t, ts = ms(rr, "fresh", "lt_0p1")
            cells += [f"{m:.3f}\\,$\\pm$\\,{s:.3f}", f"{p:.2f}\\,$\\pm$\\,{ps:.2f}", f"{100*t:.1f}"]
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(TAB, "tab_ablation.tex"), "w").write("\n".join(L) + "\n")

    L = []
    o = I["original_model20000_fresh"]
    L.append(f"Original \\texttt{{model20000\\_rrr}} (233k params, 7.3k solver labels) & 1 & "
             f"{o['median']:.3f} & {o['p95']:.2f} & {o['p99']:.2f} & {100*o['lt_0p1']:.1f} \\\\")
    L.append("\\midrule")
    for N in sorted(I["scaling"], key=int):
        rr = I["scaling"][N]
        m, s = ms(rr, "fresh", "median")
        p, _ = ms(rr, "fresh", "p95")
        q, _ = ms(rr, "fresh", "p99")
        t, _ = ms(rr, "fresh", "lt_0p1")
        Ns = f"{int(N):,}".replace(",", "{,}")
        L.append(f"Consistent $(\\cos,\\sin)$, $N={Ns}$ exact labels & 3 & "
                 f"{m:.3f}\\,$\\pm$\\,{s:.3f} & {p:.2f} & {q:.2f} & {100*t:.1f} \\\\")
    open(os.path.join(TAB, "tab_scaling.tex"), "w").write("\n".join(L) + "\n")

    H = I["hybrid"]
    names = [("zero", "$\\mathbf{q}_0=\\mathbf{0}$ (as in the original scripts)"),
             ("nn_original", "$\\mathbf{q}_0=$ original \\texttt{model20000\\_rrr}"),
             ("nn_consistent", "$\\mathbf{q}_0=$ consistent $(\\cos,\\sin)$ network")]
    L = []
    for k, lab in names:
        h = H[k]
        L.append(f"{lab} & {pct(h['fail_rate'], 2)} & {h['iters_mean']:.1f} & {h['iters_median']:.0f} & "
                 f"{h['iters_p95']:.0f} & {pct(h['frac_on_front_elbowup_branch'])} \\\\")
    open(os.path.join(TAB, "tab_hybrid.tex"), "w").write("\n".join(L) + "\n")

    T = I["latency_s"]
    B = T["batched_per_query"]
    L = [
        f"Closed-form IK (Eq.~\\ref{{eq:ik-rrr}}) & {1e6*T['closed_form']:.1f} & {1e6*B['closed_form']:.3f} \\\\",
        f"MLP, original architecture (233k params) & {1e6*T['nn_original_233k']:.1f} & {1e6*B['nn_original_233k']:.3f} \\\\",
        f"MLP, consistent $(\\cos,\\sin)$ (134k params) & {1e6*T['nn_consistent_133k']:.1f} & {1e6*B['nn_consistent_133k']:.3f} \\\\",
        f"LM \\texttt{{ikine}} replica, $\\mathbf{{q}}_0=\\mathbf{{0}}$ & {1e6*T['lm_from_zero']:.0f} & -- \\\\",
        f"LM \\texttt{{ikine}} replica, learned warm start & {1e6*T['lm_from_nn']:.0f} & -- \\\\",
    ]
    open(os.path.join(TAB, "tab_latency.tex"), "w").write("\n".join(L) + "\n")
print("tables written to", TAB)

# Compact warm-start table for the short paper
if os.path.exists(p2):
    H = json.load(open(p2))["hybrid"]
    rows = [("zero", "All zeros (baseline)"), ("nn_original", "Baseline network"),
            ("nn_consistent", "Improved network")]
    L = [f"{lab} & {H[k]['iters_mean']:.1f} & {H[k]['iters_p95']:.0f} & {pct(H[k]['frac_on_front_elbowup_branch'])} \\\\"
         for k, lab in rows]
    open(os.path.join(TAB, "tab_hybrid_short.tex"), "w").write("\n".join(L) + "\n")
