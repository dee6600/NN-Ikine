"""Core kinematics, solvers and model loading shared by all experiments.

Everything here mirrors the robot definitions in the original MATLAB scripts
(gen.m, gen_rpp.m, rrr_3_link_*.m, rpp_*.m) so that results are comparable.
"""
import json
import time

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Robot parameters (taken verbatim from the MATLAB scripts)
# ---------------------------------------------------------------------------
A2, A3 = 10.0, 10.0      # RRR link lengths: Link([th,0,10,0,0]) x2
D1 = 5.0                 # RPP column height: Link([th1,5,0,0,0])
REACH = A2 + A3          # RRR outer workspace radius


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# Forward kinematics and Jacobians
# ---------------------------------------------------------------------------
def fk_rrr(q):
    q = np.atleast_2d(q)
    c1, s1 = np.cos(q[:, 0]), np.sin(q[:, 0])
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    c23, s23 = np.cos(q[:, 1] + q[:, 2]), np.sin(q[:, 1] + q[:, 2])
    r = A2 * c2 + A3 * c23
    return np.stack([c1 * r, s1 * r, A2 * s2 + A3 * s23], axis=1)


def jac_rrr(q):
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    c23, s23 = np.cos(q[1] + q[2]), np.sin(q[1] + q[2])
    r = A2 * c2 + A3 * c23
    dr2 = -A2 * s2 - A3 * s23
    dr3 = -A3 * s23
    return np.array([
        [-s1 * r, c1 * dr2, c1 * dr3],
        [c1 * r, s1 * dr2, s1 * dr3],
        [0.0, A2 * c2 + A3 * c23, A3 * c23],
    ])


def fk_rpp(q):
    """DH: L1=[th1,d1,0,0] R ; L2=[0,d2,0,-pi/2] P ; L3=[0,d3,0,0] P."""
    q = np.atleast_2d(q)
    s1, c1 = np.sin(q[:, 0]), np.cos(q[:, 0])
    return np.stack([-s1 * q[:, 2], c1 * q[:, 2], D1 + q[:, 1]], axis=1)


def jac_rpp(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    return np.array([
        [-c1 * q[2], 0.0, -s1],
        [-s1 * q[2], 0.0, c1],
        [0.0, 1.0, 0.0],
    ])


ROBOTS = {
    "rrr": (fk_rrr, jac_rrr),
    "rpp": (fk_rpp, jac_rpp),
}


# ---------------------------------------------------------------------------
# Numerical IK: re-implementation of Robotics Toolbox (v10) SerialLink.ikine
# Levenberg-Marquardt scheme with the options used in the project:
#   mask=[1 1 1 0 0 0], ilimit=1000, rlimit=100, tol=1e-10, lambda0=0.1
# Step:  dq = (J'WJ + lambda I)^-1 J'W e ; accept -> lambda/2, reject -> 2*lambda
# Rejected steps count toward the iteration limit, as in the toolbox loop.
# Returns (q or None, iterations).
# ---------------------------------------------------------------------------
def ikine_lm(robot, p_target, q0=(0.0, 0.0, 0.0), ilimit=1000, rlimit=100,
             tol=1e-10, lam0=0.1):
    fk, jac = ROBOTS[robot]
    q = np.array(q0, dtype=float)
    e = p_target - fk(q)[0]
    lam = lam0
    rej = 0
    it = 0
    while True:
        it += 1
        if it > ilimit:
            return None, it - 1
        en = np.linalg.norm(e)
        if en < tol:
            return q, it
        J = jac(q)
        JtJ = J.T @ J
        dq = np.linalg.solve(JtJ + lam * np.eye(3), J.T @ e)
        qn = q + dq
        enew = p_target - fk(qn)[0]
        if np.linalg.norm(enew) < en:
            q, e = qn, enew
            lam /= 2.0
            rej = 0
        else:
            lam *= 2.0
            rej += 1
            if rej > rlimit:
                return None, it


# ---------------------------------------------------------------------------
# Closed-form IK
# ---------------------------------------------------------------------------
def ik_rrr_closed(p, shoulder=+1, elbow=+1):
    """Closed-form IK for the anthropomorphic arm.

    shoulder=+1: theta1 = atan2(py,px) ('front'); -1: theta1 + pi ('back').
    elbow=+1:    theta3 in [0, pi];  -1: theta3 in [-pi, 0].
    Unreachable targets are clipped to the workspace boundary.
    """
    p = np.atleast_2d(p)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    rho = np.hypot(px, py)
    th1 = np.arctan2(py, px)
    r = rho
    if shoulder < 0:
        th1 = wrap(th1 + np.pi)
        r = -rho
    c3 = (px ** 2 + py ** 2 + pz ** 2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)
    c3 = np.clip(c3, -1.0, 1.0)
    s3 = elbow * np.sqrt(1 - c3 ** 2)
    th3 = np.arctan2(s3, c3)
    th2 = np.arctan2(pz, r) - np.arctan2(A3 * s3, A2 + A3 * c3)
    return np.stack([th1, wrap(th2), th3], axis=1)


def ik_rrr_all(p):
    """All four closed-form branches, shape (4, N, 3)."""
    return np.stack([ik_rrr_closed(p, s, e) for s in (+1, -1) for e in (+1, -1)])


def ik_rpp_closed(p):
    p = np.atleast_2d(p)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([np.arctan2(-px, py), pz - D1, np.hypot(px, py)], axis=1)


def rrr_branch(q, p):
    """Classify a joint solution into (shoulder, elbow) branch labels."""
    q = np.atleast_2d(q)
    front = np.abs(wrap(q[:, 0] - np.arctan2(p[:, 1], p[:, 0]))) < np.pi / 2
    elbow_up = np.sin(q[:, 2]) >= 0
    return np.where(front, 1, -1), np.where(elbow_up, 1, -1)


# ---------------------------------------------------------------------------
# Keras .h5 -> NumPy MLP (no TensorFlow needed)
# ---------------------------------------------------------------------------
_ACT = {
    "relu": lambda x: np.maximum(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "linear": lambda x: x,
}


class KerasMLP:
    def __init__(self, path):
        with h5py.File(path, "r") as h:
            cfg = json.loads(h.attrs["model_config"])
            self.train_cfg = json.loads(h.attrs["training_config"])
            self.layers = []
            for l in cfg["config"]["layers"]:
                if l["class_name"] != "Dense":
                    continue
                name = l["config"]["name"]
                g = h["model_weights"][name]
                ds = {}
                g.visititems(lambda k, v: ds.__setitem__(k.split("/")[-1], v[()])
                             if isinstance(v, h5py.Dataset) else None)
                self.layers.append((ds["kernel:0"].astype(np.float64),
                                    ds["bias:0"].astype(np.float64),
                                    l["config"]["activation"]))
        self.n_params = sum(W.size + b.size for W, b, _ in self.layers)

    @property
    def arch(self):
        return [(W.shape[1], a) for W, _, a in self.layers]

    def __call__(self, x):
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        for W, b, a in self.layers:
            h = _ACT[a](h @ W + b)
        return h


def keras_categorical_accuracy(y_true, y_pred):
    """What `metrics=['accuracy']` resolves to in tf.keras 2.x for a
    3-unit output with a non-crossentropy loss: argmax(y_true)==argmax(y_pred)."""
    return float(np.mean(np.argmax(y_true, 1) == np.argmax(y_pred, 1)))


def timeit(fn, n=2000, warmup=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n
