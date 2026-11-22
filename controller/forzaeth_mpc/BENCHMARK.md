# Benchmark

## Conditions

Track THETRACK, 26.06 m. Car NUC5. Floor `dubi`. Raceline `iqp`. Control period 0.05 s, from
`MPC_freq: 20` in the configuration of each controller. Each run covers 20 laps, and starts at
`s = 20 m` with `v_x = 3 m/s`.

The plant and the MPC use the same model. Friction is the only difference that the runs
introduce.

```bash
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
```

These numbers come from runs made after the duplicate-key fix. STMPC and LB-MPC therefore use
the same lateral acceleration in the tire constraint. Before that fix, `correct_v_y_dot`
appeared twice in NUC5's configuration: STMPC used the kinematic value
`a_lat = v_x^2 * tan(delta) / L`, and LB-MPC used the true value `a_lat = v_y_dot`. The two
controllers solved different problems, so the older comparison was not valid.

Read each solve time as an indication only. The runs shared a machine, so the times depend on
the other work on it. Compare times inside one table, not across tables.

## Matched friction, plant and MPC both 1.0

| | lap best | lap mean | lap sigma | mean \|n\| | max \|n\| | mean v_x | solve ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| **KMPC** | 7.10 | 7.100 | 0.000 | 0.020 | 0.051 | 3.70 | 1.9 | 2.5 |
| **STMPC** | 7.40 | 7.448 | 0.025 | 0.078 | 0.294 | 3.64 | 5.3 | 7.1 |
| **LB-MPC** | 7.45 | 7.553 | 0.033 | 0.071 | 0.253 | 3.58 | 7.0 | 9.7 |

STMPC is the faster of the two single-track controllers when its model is correct: 7.448 s
against 7.553 s. This result is the expected one. A correct model leaves no error to learn, so
the Gaussian process only costs time in the horizon. LB-MPC still holds the line more closely,
at a mean `|n|` of 0.071 m against 0.078 m. The next table shows where it gains.

Do not read the first row as a result for KMPC. Its plant is its own kinematic model, which
has no tires and cannot lose grip. The car can therefore take any corner at any speed that the
box constraints permit. STMPC and LB-MPC drive against a Pacejka plant.

## Friction error, plant 0.8 and MPC 1.0

| | lap best | lap mean | mean \|n\| | max \|n\| | mean v_x | solve ms | p99 ms | GP points |
|---|---|---|---|---|---|---|---|---|
| **STMPC** | 8.10 | 8.160 | 0.185 | 0.489 | 3.53 | 5.8 | 8.0 | — |
| **LB-MPC**, no learning | 8.20 | 8.210 | 0.168 | 0.426 | 3.48 | 3.9 | 5.2 | 0 |
| **LB-MPC**, learning | **7.95** | **7.963** | **0.101** | **0.331** | 3.45 | 8.5 | 11.8 | 137 |

Learning gains 3.0 % in lap time and 40 % in mean lateral deviation against the same
controller without learning. Against nominal STMPC it gains 2.4 % and 45 %. The solve time
approximately doubles, from 3.9 ms to 8.5 ms, with a p99 of 11.8 ms. Both values stay far
below a budget of 50 ms.

The order therefore changes with the model error. STMPC wins when its model is correct, and
LB-MPC wins when it is not. This behaviour is the purpose of the learnt correction, and only
the equal tire constraint makes it visible.

These numbers need the fix on the branch `fix/balanced-dictionary-gate` of
`external/l4acados`. The earlier rule held the dictionary at 8 points, and the same run then
gave 8.05 s and 0.140 m. See "Dictionary gate" below.

KMPC does not appear in this table. Its model has no friction parameter.

## Solver results

In every 20-lap run, 100 % of the solves returned status 0 or status 2. Status 2 is not a failure.
`nlp_solver_max_iter` is 3, and the solver type is `SQP_WITH_FEASIBLE_QP`, so the design stops
the SQP before it converges. STMPC returned status 2 at every step, LB-MPC returned status 0 at
every step, and KMPC returned a mixture.

Do not compare the cost between controllers. KMPC reports a mean of 34.3 and STMPC reports 6.3,
because each one uses a different cost expression over a different state vector.

## Reference values

Both values below serve as the parity reference for a refactor. Each one comes from a single
solve with a fixed initial state, `x0 = (s=20, n=0, theta=0, v_x=3)`, and one fixed 22-entry
parameter vector on every stage.

STMPC:

```
cost 32.86543961175853    status 2    n_x 8    N 40
```

The value was `34.69739908591064` before the fix to `correct_v_y_dot` in NUC5. The
pre-extraction STMPC in race_stack returns the same number, and the largest difference over the
predicted trajectory is 0.0.

LB-MPC, with `stochastic_mode: disabled` and learning off:

```
cost 77.3159880919    status 0    n_x 8    n_p 22    N 40
```

## Research experiments of LB-MPC

`research/residual_validation/validation/experiments/sim_residual_cl_gpytorch.py` drives a
harder case: plant friction 0.8 against a model that assumes 1.0, an added forward-speed error
of 1.0 m/s^2, and state noise of 1e-5. The dictionary holds 50 points.

```
GP training MLL loss      0.727 -> -1.991 over 200 iterations
dictionary                50 kept / 600 dropped (m_max=50, min_dist=0.15)
nominal STMPC, 3 laps     8.277 / 8.290 / 8.290 s | lateral dev 0.268 ± 0.276 m
LB-MPC, 3 laps            8.505 / 8.337 / 8.415 s | lateral dev 0.148 ± 0.160 m  (-45 %)
```

This tree also reads the configuration of NUC5, so the fix to `correct_v_y_dot` changed its
numbers. They were 8.465 / 8.410 / 8.432 s and 0.279 m for nominal STMPC. The tree keeps the
earlier state order and the 14-entry parameter vector on purpose.

The next table gives the error of one prediction step, over 600 pairs that the dictionary never
held. The nominal model predicts an error of zero, so its value is the size of the target.

| channel | nominal | Gaussian process | change |
|---|---|---|---|
| v_x | 0.058883 | 0.012886 | **−78 %** |
| v_y | 0.073895 | 0.047387 | **−36 %** |
| yaw rate | 0.052255 | 0.068322 | **+31 %, worse** |
| norm | 0.107973 | 0.084139 | −22 % |

The Gaussian process therefore generalises, and does not only repeat its training data. The
gain on `v_x` recovers the added speed error.

The error on the yaw rate becomes worse. Under an error in friction alone, the yaw rate carries
little information, and one Gaussian process over five shared features fits it too closely.
This result is real, and the extraction did not cause it. The closed loop still improves.

One measurement above needs care: a throwaway script produced the table, from the
`dyn_from_ctrl`, `integrate_one_step` and `update_dictionary` functions of the experiment. The
other numbers come from repeated runs. Treat the sizes in the table as an indication.

Two further scripts, `validate_residual_mpc.py` and `simulate_residual_closed_loop.py`, do not
run. They call `build_multitask_gp`, `gp_predictor_from_model` and `save_gp_state`. race_stack
commit `e8e132245`, of 2025-11-24, removed those functions from `gp_interface.py` and left the
callers unchanged.

## Dictionary gate, fixed

The `"balanced"` rule of `OnlineLearningStrategy` compared two distances. The first was the
distance from the candidate to the nearest member. The second was the smallest distance inside
the dictionary. The rule admitted the candidate only if the first distance was larger. That
smallest distance cannot decrease when the rule admits a point, and only an eviction
recalculates it.
Evictions start only when the dictionary is full. Below that size, the limit therefore stayed
at the distance between the first two admitted samples.

The table below replays the rule over one 20-lap trajectory. Only the sample spacing changes.

| sample rate | gap of the first pair | final dictionary |
|---|---|---|
| 50 Hz | 0.263 | 137 |
| 25 Hz | 0.563 | 50 |
| 16.7 Hz | 0.832 | 30 |
| 10 Hz | 1.192 | 17 |

At one rate, the same data gave 30 points or 94 points, and only the first sample differed. At
20 Hz the controller held 8 points against a `max_num_data_points` of 200, because its first
two pairs fall in the transient at the start. The rule also measured the distance over raw
features, so the feature with the widest range decided each admission.

The fork now admits a candidate only if it is at least `min_dist` from every member. It
measures that distance per feature, in units of the standard deviation of the feature. The
standard deviation comes from a running estimate, or from `feature_scales`. Eviction from the
closest pair does not change. A `min_dist` of 0.15 gives 127 to 137 points on this track at
20 Hz.

`test/gpytorch_model/test_online_learning_gate.py` in the fork covers the sample rate, the
first sample, the units of the features, repeated points, and the limit at each admission.

## ZoRO, run for the first time

Every fixture in the repository sets `stochastic_mode: disabled`. No configuration had ever
called `add_zoro_description`. The table below drives each mode for 200 steps, with plant
friction 0.8 against a model that assumes 1.0.

| mode | `fdbk_K_mat` | 200 steps | result |
|---|---|---|---|
| `open_loop` | none | leaves the track at step 162, n = 1.66 m | no feedback, as designed |
| `lqr` | max \|K\| 13.06 | all 200 solves status 0, n = −0.004 m | works |
| `riccati` | **max \|K\| 0** | leaves the track at step 162, n = 1.66 m | **does not work** |

`riccati` fails in the same way as `open_loop`, at the same step and by the same distance, and
its feedback gain is zero. `add_zoro_description` sets `fdbk_K_mat` to `np.zeros((nu, nx))` and
sets `feedback_optimization_mode` to `"RICCATI_CONSTANT_COST"`. It expects acados to calculate
the gain from `riccati_Q_const` and `riccati_R_const`. acados does not do so, so `riccati`
becomes `open_loop`. **Use `lqr`. It is the only stochastic mode that works.**

The state permutation did not cause this. The block reads every index from the enums, and no
mode had ever run. The block did hold one error of that kind, which is now fixed: the code
wrote the regularisation to `R[InputIndex.X, InputIndex.X]` but read it from `R[0, 0]` and
`R[1, 1]`. Under the earlier input order, it therefore wrote `qjerk` into the slot of `ddelta`.

## How to check this repository

Run these commands from the root of the repository. `PYTHONPATH` and the path to the
l4acados test are both relative to it.

```bash
export ACADOS_SOURCE_DIR=/acados LD_LIBRARY_PATH=/acados/lib PYTHONPATH=$PWD MPLBACKEND=Agg
rm -rf c_generated_code acados_ocp*.json /tmp/lbmpc_sim

python3 -m forzaeth_mpc.stmpc.bicycle_model      # the state and input order
python3 -m forzaeth_mpc.stmpc.online_params      # the 22-entry parameter layout
python3 -m forzaeth_mpc.sim.core.controllers     # the three adapters
python3 -m forzaeth_mpc.sim.tracks               # the track loader
python3 -m forzaeth_mpc.config.paths             # the duplicate-key guard
python3 external/l4acados/test/gpytorch_model/test_online_learning_gate.py

python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
python3 -m forzaeth_mpc.sim.cli.run_laps --algo lbmpc --laps 20 --mu_track 0.8 --mu_mpc 1.0
python3 -m research.residual_validation.validation.experiments.sim_residual_cl_gpytorch
```

Delete `c_generated_code/` first. acados writes it to a path relative to the working
directory, so a run can otherwise load the compiled solver of an earlier run.

## Open points

- **LB-MPC calculates each correction against a zero input.** `lb_mpc.py` sets
  `_prev_input_for_residual` to `np.zeros(2)`, and `self.u0.copy()` stands beside it as a
  comment. The correction is therefore `r = x_k - f(x_{k-1}, 0)`, which charges the whole
  effect of the applied input to model error. race_stack commit `aeb5101d9`, of 2026-02-22,
  introduced this. The adapter in the simulator passes the true input instead. Decide the
  question before the next run on a car.
- **acados writes generated code to a path relative to the working directory.** The three
  controllers now use different model names: `Spatialbicycle_model`, `LBMPCbicycle_model` and
  `Kinematicbicycle_model`. They therefore cannot load each other's solver. One shared name
  made `run_laps --algo stmpc kmpc lbmpc` stop with `mismatching dimension for field "lbx"
  with dimension 8 (you have 5)`. It also made a single LB-MPC solve report `trying to set 22
  parameters ... has 14`. Two runs of the same controller from different directories can still
  collide.
