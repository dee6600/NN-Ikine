# Inverse Kinematics with Neural Networks — Technical Report

Oct 13, 2026 · @Durgesh

**Authors:** Rumitkumar Patel, Durgesh Reddiyar, Priyank Patel — Arizona State University, MAE 547 (Modeling and Control of Robots), Project 2, Fall 2021

## Abstract

A neural network trained on 10,000 solver-generated samples reproduced the numerical inverse kinematics (IK) of a 3-DOF RRR arm to within about 1° on a test pose. We compared two IK approaches on an RRR (anthropomorphic) arm and an RPP (cylindrical) arm: the iterative `ikine` solver from Peter Corke's Robotics Toolbox, and a feed-forward neural network that maps end-effector position directly to joint variables.

The network was trained for 20,000 epochs on an 80:20 train/test split and scored about 75% on the held-out set. For the target position \[1, 4, 9\], both methods returned joint angles of approximately \[78.7°, 4.3°, 120.1°\], and the animated arms were visually indistinguishable. Across ten test observations, predicted joint values tracked the solver's closely. A trained network answers in a single forward pass, while `ikine` iterates up to 1,000 times per query and sometimes fails near singularities.

## 1. Introduction

Inverse kinematics asks which joint variables place a robot's end effector at a desired pose. Forward kinematics is a direct composition of transforms, but its inverse is a nonlinear mapping that can have zero, one, several or infinitely many solutions. Most real-world manipulation tasks are specified in task space, so nearly every deployed arm needs an efficient IK method.

Conventional numerical solvers iterate on the Jacobian until the pose error converges. They are general, but their cost grows with robot complexity, they can stall near singularities, and their run time per query is not fixed. Neural networks offer an alternative: once trained, they approximate the inverse mapping in a single, constant-time forward pass.

### 1.1 Problem statement

Solve inverse kinematics for two 3-DOF manipulators with a neural network, and compare its joint-space solutions against the Robotics Toolbox `ikine` solver.

### 1.2 Objectives

1. Generate a training dataset of end-effector positions and joint variables using `ikine`.
2. Train a neural network to predict joint variables from end-effector position.
3. Compare predicted and solver joint values on test poses for the RRR and RPP arms.
4. Animate both arms from each method's solution to check that the poses agree.

## 2. Robot models

Both arms have three degrees of freedom and a 3D position target, so each IK problem maps three task-space coordinates to three joint variables. The frame assignments follow the standard textbook anthropomorphic and cylindrical arms shown in the project figures; the numeric link lengths used in the code are not recorded in the presentation and are kept symbolic here.

### 2.1 RRR (anthropomorphic) arm

Three revolute joints: a base yaw joint (θ1) followed by two parallel pitch joints (θ2, θ3) separated by links of length a2 and a3.

| Link i | a\_i | α\_i | d\_i | θ\_i |
| --- | --- | --- | --- | --- |
| 1 | 0 | π/2 | 0 | θ1\* |
| 2 | a2 | 0 | 0 | θ2\* |
| 3 | a3 | 0 | 0 | θ3\* |

Forward kinematics of the end-effector position (c\_i = cos θ\_i, s\_i = sin θ\_i, c23 = cos(θ2 + θ3)):

```latex
\begin{aligned}
p_x &= c_1\,(a_2 c_2 + a_3 c_{23})\\
p_y &= s_1\,(a_2 c_2 + a_3 c_{23})\\
p_z &= a_2 s_2 + a_3 s_{23}
\end{aligned}
```

### 2.2 RPP (cylindrical) arm

One revolute base joint (θ1) and two prismatic joints: a vertical lift (d2) and a horizontal reach (d3), mounted on a fixed column of height d1.

| Link i | a\_i | α\_i | d\_i | θ\_i |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | d1 | θ1\* |
| 2 | 0 | −π/2 | d2\* | 0 |
| 3 | 0 | 0 | d3\* | 0 |

Forward kinematics of the end-effector position:

```latex
\begin{aligned}
p_x &= -s_1\, d_3\\
p_y &= c_1\, d_3\\
p_z &= d_1 + d_2
\end{aligned}
```

Starred entries are joint variables. Reference figures of both arms appear on slide 7 of the project presentation.

## 3. Methodology

The solver produces both the ground truth and the training data; the network learns to imitate it, and both outputs drive the same animation for comparison.

&#91;embedded content: IK pipeline · offline training and online comparison\]

The top row runs once to build the model; the bottom row runs per query, with the solver and the network solving the same target in parallel.

### 3.1 Baseline: `ikine` numerical solver

1. The user enters a target end-effector position \[x, y, z\].
2. The target is passed to the Robotics Toolbox `ikine` solver, capped at 1,000 iterations per query.
3. The returned joint variables drive the RRR or RPP model through the toolbox `animate()` function, from the initial pose to the target.

The 1,000-iteration cap means some complex poses return no solution. It was judged sufficient to demonstrate the method.

### 3.2 Dataset generation

- 10,000 random end-effector positions were sampled, and `ikine` was run on each to obtain joint variables (`gen.m` for RRR, `gen_rpp.m` for RPP).
- Each sample pairs an input position (x, y, z) with a target joint vector (q1, q2, q3).
- Samples where the solver failed — either by hitting the iteration limit or because the pose was at or near a singularity — were excluded. This filtering is the only preprocessing step.
- Data moved between MATLAB and Python through intermediate files in the `Dataset` folder.

### 3.3 Neural network model

- **Input:** end-effector position, 3 values.
- **Output:** joint variables, 3 values (angles for RRR; θ1, d2, d3 for RPP).
- **Training:** `train.py`, 20,000 epochs, 80:20 train/test split of the 10,000 samples. Trained models are saved in the `Model` folder.
- **Inference:** `inference.py` loads the trained model and predicts joint variables for a user-entered position; the prediction then drives the same `animate()` call as the baseline.

The presentation does not record the network architecture (layer count and widths), activation functions, loss function, optimizer or learning rate. These should be added from `train.py` before the report is circulated.

## 4. Experimental setup

| Parameter | Value |
| --- | --- |
| Robots | RRR (3 revolute), RPP (1 revolute + 2 prismatic) |
| Baseline solver | Robotics Toolbox `ikine`, max 1,000 iterations |
| Dataset size | 10,000 samples per robot, solver failures removed |
| Train / test split | 80:20 (8,000 / 2,000) |
| Training epochs | 20,000 |
| Model input → output | position (x, y, z) → joint variables (q1, q2, q3) |
| Evaluation | test-set accuracy; per-joint comparison on 10 observations; single-pose animation check |
| Single-pose test target (RRR) | \[1, 4, 9\] |
| Software | MATLAB with Robotics Toolbox; Python for training and inference |

The presentation does not define the accuracy metric (for example, the per-joint error tolerance that counts a prediction as correct).

## 5. Results

The network's joint predictions matched the `ikine` solutions closely: about 75% test accuracy overall, and roughly 1° of difference on the single-pose RRR test.

### 5.1 Summary of measured results

| Measure | Result |
| --- | --- |
| Test-set accuracy (NN) | \~75% |
| RRR target \[1, 4, 9\] — `ikine` joint angles | ≈ \[78.7°, 4.3°, 120.1°\] |
| RRR target \[1, 4, 9\] — NN joint angles | ≈ \[78.7°, 4.3°, 120.1°\] |
| Difference in simulated pose | \~1° |

### 5.2 Single-pose comparison (RRR)

For the target \[1, 4, 9\], both methods returned near-identical joint vectors. Animating the RRR arm from each solution produced visually indistinguishable motions, with joint differences of about 1°.

### 5.3 Per-joint comparison over ten observations

The solver output q\_i and network prediction q\_i\_pred were plotted for each joint (i = 1, 2, 3) over ten test observations (slide 9 of the presentation).

- **q1:** the two curves overlap across all ten observations, spanning roughly −2.6 to 1.8.
- **q2:** close agreement throughout; the largest visible gap is at observation 6, where the value is near −2.4.
- **q3:** close agreement except a small offset at the observation-6 peak (around 16.5–17); other points overlap.

The values above are read from the plots and are approximate. The plots do not state which robot they come from or the units of q3; its range up to about 17 suggests a prismatic joint or unwrapped angles, which should be confirmed from the source data.

## 6. Discussion

The network learned a usable approximation of the solver's IK map for both arms, but the evidence supports "close agreement on the tested poses" rather than a general claim of superior reliability or speed.

### 6.1 Solver failures

Many sampled end-effector poses produced no `ikine` solution, for two reasons:

1. **Iteration limit.** The solver hit the 1,000-iteration cap before converging.
2. **Singularities.** The target sat at or near a singular configuration, where the Jacobian loses rank and the iterative update becomes ill-conditioned.

These samples were removed from the dataset. The network therefore never saw near-singular or hard-to-converge regions, and its behaviour there is untested.

### 6.2 Simulation tooling

The Robotics Toolbox `animate()` function made it straightforward to visualise both arms moving from the initial pose to each method's solution, which gave a quick visual check alongside the numeric comparison.

### 6.3 Limitations

- **Multiple IK solutions.** The RRR arm generally has up to four solutions for one position (shoulder left/right × elbow up/down). A network trained by regression on mixed branches can average them into an invalid pose. Agreement here likely depends on `ikine` converging to a consistent branch from its initial guess.
- **Closed-form IK exists for both arms.** Both are textbook 3-DOF geometries with analytical solutions (below), so this study demonstrates the learning approach rather than replacing a solver that is otherwise unavailable.
- **Accuracy metric.** "\~75% accuracy" is a classification-style figure applied to a regression task. Mean and maximum joint error, and Cartesian position error after forward kinematics, would be more informative.
- **Speed not measured.** The computational advantage of a forward pass is expected but no timing comparison was reported.
- **Small evaluation set.** Per-joint plots cover ten observations; the pose-level check covers one target.
- **Base-angle check.** With the RRR frames in Section 2.1, θ1 = atan2(p\_y, p\_x), which gives about 76.0° for \[1, 4, 9\], not the reported 78.7°. The code may use a base offset or a different frame convention; this should be reconciled.

Analytical IK, for reference. RPP:

```latex
\theta_1 = \operatorname{atan2}(-p_x,\, p_y),\qquad d_3 = \sqrt{p_x^2 + p_y^2},\qquad d_2 = p_z - d_1
```

RRR (θ3 sign selects elbow up or down; θ1 + π gives the second shoulder solution):

```latex
\begin{aligned}
\theta_1 &= \operatorname{atan2}(p_y,\, p_x)\\
c_3 &= \frac{p_x^2 + p_y^2 + p_z^2 - a_2^2 - a_3^2}{2 a_2 a_3},\qquad \theta_3 = \operatorname{atan2}\!\left(\pm\sqrt{1 - c_3^2},\, c_3\right)\\
\theta_2 &= \operatorname{atan2}\!\left((a_2 + a_3 c_3)\,p_z - a_3 s_3 r,\; (a_2 + a_3 c_3)\,r + a_3 s_3\,p_z\right),\qquad r = \sqrt{p_x^2 + p_y^2}
\end{aligned}
```

## 7. Future work

1. Report mean, maximum and per-joint error in degrees or millimetres, plus Cartesian position error after passing predictions through forward kinematics.
2. Benchmark inference time against `ikine` over the full test set.
3. Generate training data by sampling joint space and running forward kinematics, which yields exact labels, covers singular regions and avoids solver failures.
4. Handle multiple solutions explicitly — fix one branch, add a branch-selection input, or use a mixture-density or classification head.
5. Compare against the closed-form solutions as an exact baseline.
6. Extend to full 6-DOF pose (position and orientation) on a redundant or non-analytical arm, where a learned solver has the most to offer.

## 8. Conclusion

A neural network trained on solver-generated data reproduced the `ikine` inverse kinematics of RRR and RPP arms closely, matching joint angles to about 1° on the RRR test pose and tracking the solver across ten test observations. This supports neural networks as a fast, constant-time approximation for IK on these arms. Confirming the speed advantage, quantifying error properly and handling multiple solutions are the next steps toward a general claim.

## Appendix A. Implementation

### A.1 Codebase

| File / folder | Language | Purpose |
| --- | --- | --- |
| `gen.m`, `gen_rpp.m` | MATLAB | Generate training data with `ikine` for RRR and RPP |
| `train.py` | Python | Train the neural network |
| `inference.py` | Python | Run custom predictions with a trained model |
| `rrr_3_link_toolbox.m` | MATLAB | RRR inverse kinematics with the Robotics Toolbox, plus animation |
| `rrr_3_link_NN.m` | MATLAB | RRR inverse kinematics with the neural network (calls `py_rrr.py`), plus animation |
| `rpp_toolbox.m` | MATLAB | RPP inverse kinematics with the Robotics Toolbox, plus animation |
| `rpp_NN.m` | MATLAB | RPP inverse kinematics with the neural network, plus animation |
| `Model/` | — | Trained models |
| `Dataset/` | — | Temporary files for MATLAB–Python data exchange |

### A.2 Requirements

- MATLAB with Peter Corke's Robotics Toolbox.
- A Python environment callable from the command line. The NN scripts install required packages on first run.

### A.3 How to run

Extract the project archive and open the `run_project` folder.

**RRR arm**

1. Run `rrr_3_link_toolbox.m` and enter an end-effector position as \[x y z\]. The RRR arm animates from its initial pose to the target.
2. Run `rrr_3_link_NN.m`. It calls `py_rrr.py`, installs the Python dependencies, takes the target position and animates the arm using the network's prediction.

**RPP arm**

1. Run `rpp_toolbox.m` and enter \[x y z\]. The RPP arm animates to the target.
2. Run `rpp_NN.m`, enter \[x y z\], and the arm animates using the network's prediction.

Open points: the presentation lists the toolbox and NN scripts as `.py` in one place and `.m` in another, and states that `rpp_NN.m` calls `py_rrr.py`. Both should be checked against the repository.

## Appendix B. Team contributions

All three members contributed equally; the team's mixed disciplinary backgrounds split the work naturally across data, modelling and MATLAB integration.

| Member | Share | Core technical work | Testing | Results | Shared tasks |
| --- | --- | --- | --- | --- | --- |
| Durgesh Reddiyar | 33.34% | Neural network training; idea formation | Whole system | Results extraction | Brainstorming, literature review, presentation |
| Priyank Patel | 33.33% | MATLAB scripting; idea formation | Neural network | Results video | Brainstorming, literature review, presentation |
| Rumitkumar Patel | 33.33% | Data generation; team formation | MATLAB scripts | Results verification | Brainstorming, literature review, presentation |

**Contact:** dreddiya@asu.edu · ppatel65@asu.edu · rpatel71@asu.edu

## References

1. P. Corke, *Robotics, Vision and Control*, Springer — Robotics Toolbox (`ikine`, `animate`).
2. B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009 — anthropomorphic and cylindrical arm kinematics.
