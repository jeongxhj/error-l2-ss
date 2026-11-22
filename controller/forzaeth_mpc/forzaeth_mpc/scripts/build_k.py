#!/usr/bin/env python3
import casadi as cs
import numpy as np
from forzaeth_mpc.utils.indices import InputIndex, StateIndex
from forzaeth_mpc.config import (CarConfig, LBMPCConfig, PacejkaTireConfig,
                        load_car_config_ros, load_LBMPC_config_ros,
                        load_pacejka_tire_config_ros)
from scipy.linalg import solve_continuous_are

# To execute it go to controller/ and run: python3 -m lbmpc.scripts.build_k


def build_straightline_dynamics(car_config: CarConfig, tire_config: PacejkaTireConfig):
    """
    Straight-line simplification: kapparef = 0 (no dependence on s).
    State:  x = [s, n, theta, vx, vy, yaw_rate, delta, accel]
    Input:  u = [derDelta, jerk]
    Continuous-time dynamics: xdot = f(x, u)
    """
    # Car parameters
    lr = car_config.lr
    lf = car_config.lf
    m = car_config.m
    Iz = car_config.Iz
    h_cg = car_config.h_cg

    # Tire parameters
    MU = tire_config.friction_coeff
    Bf = tire_config.Bf
    Cf = tire_config.Cf
    Df = tire_config.Df
    Ef = tire_config.Ef
    Br = tire_config.Br
    Cr = tire_config.Cr
    Dr = tire_config.Dr
    Er = tire_config.Er
    g = 9.81

    # Model constants
    v0 = 0.5  # to avoid divide-by-zero and smooth low-speed behavior

    # -----------------------------
    # SYMBOLS
    # -----------------------------
    s = cs.MX.sym("s")
    n = cs.MX.sym("n")
    th = cs.MX.sym("theta")
    vx = cs.MX.sym("v_x")
    vy = cs.MX.sym("v_y")
    r = cs.MX.sym("yaw_rate")
    delta = cs.MX.sym("delta")
    accel = cs.MX.sym("accel")
    x = cs.vertcat(s, n, th, vx, vy, r, delta, accel)

    derDelta = cs.MX.sym("derDelta")
    jerk = cs.MX.sym("jerk")
    u = cs.vertcat(derDelta, jerk)

    # -----------------------------
    # STRAIGHT-LINE KINEMATICS (curvature = 0)
    # -----------------------------
    s_dot = (vx * cs.cos(th)) - (vy * cs.sin(th))
    n_dot = (vx * cs.sin(th)) + (vy * cs.cos(th))
    th_dot = r  # since kappa = 0, theta_dot = yaw_rate

    # -----------------------------
    # SLIP ANGLES
    # -----------------------------
    alpha_f = cs.atan2((-vy - lf * r), (vx + v0)) + delta
    alpha_r = cs.atan2((-vy + lr * r), (vx + v0))

    # No load transfer simplification
    F_zf = m * g * lr / (lf + lr)
    F_zr = m * g * lf / (lf + lr)

    # Pacejka lateral forces
    Fy_f = MU * Df * F_zf * cs.sin(
        Cf * cs.atan(Bf * alpha_f - Ef * (Bf * alpha_f - cs.atan(Bf * alpha_f)))
    )
    Fy_r = MU * Dr * F_zr * cs.sin(
        Cr * cs.atan(Br * alpha_r - Er * (Br * alpha_r - cs.atan(Br * alpha_r)))
    )

    # Longitudinal motor force from accel state (same structure)
    F_motor = accel * m
    F_xf = 0.5 * F_motor
    F_xr = 0.5 * F_motor

    # -----------------------------
    # DYNAMICS
    # -----------------------------
    vx_dot = (1.0 / m) * (
        F_xr
        + F_xf * cs.cos(delta)
        - Fy_f * cs.sin(delta)
        + m * vy * r
    )
    vy_dot = (1.0 / m) * (
        Fy_r
        + F_xf * cs.sin(delta)
        + Fy_f * cs.cos(delta)
        - m * vx * r
    )
    r_dot = (1.0 / Iz) * (
        (-Fy_r * lr) + (Fy_f * lf) * cs.cos(delta)
    )

    delta_dot = derDelta
    accel_dot = jerk

    f = cs.vertcat(s_dot, n_dot, th_dot, vx_dot, vy_dot, r_dot, delta_dot, accel_dot)

    # Jacobians
    A = cs.jacobian(f, x)
    B = cs.jacobian(f, u)

    f_fun = cs.Function("f", [x, u], [f])
    A_fun = cs.Function("A", [x, u], [A])
    B_fun = cs.Function("B", [x, u], [B])

    return f_fun, A_fun, B_fun


def linearize_at_operating_point():
    racecar_version = "NUC6"
    floor = "dubi"
    lbmpc_config = load_LBMPC_config_ros(racecar_version)
    car_config = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)

    f_fun, A_fun, B_fun = build_straightline_dynamics(car_config, tire_config)

    # Operating point:
    # x = [s, n, theta, vx, vy, yaw_rate, delta, accel]
    x0 = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # u = [derDelta, jerk]
    u0 = np.array([0.0, 0.0], dtype=float)

    f0 = np.array(f_fun(x0, u0)).astype(float).reshape(-1)
    A0 = np.array(A_fun(x0, u0)).astype(float)
    B0 = np.array(B_fun(x0, u0)).astype(float)

    # Affine form: xdot ≈ A x + B u + c
    c0 = f0 - A0 @ x0 - B0 @ u0

    np.set_printoptions(precision=6, suppress=True)
    print("Operating point x0:", x0)
    print("Operating point u0:", u0)
    print("\nf(x0,u0) = f0:\n", f0)
    print("\nA = df/dx at (x0,u0):\n", A0)
    print("\nB = df/du at (x0,u0):\n", B0)
    print("\nc = f0 - A x0 - B u0:\n", c0)

    # -----------------------------
    # LQR (continuous-time Riccati) to compute constant K
    # -----------------------------
    nx = A0.shape[0]
    nu = B0.shape[1]

    # Build Q, R from your MPC weights (constant design around the operating point)
    Q = np.zeros((nx, nx), dtype=float)
    R = np.zeros((nu, nu), dtype=float)

    # Use your configured weights (these are numeric in lbmpc_config)
    Q[StateIndex.MIN_DIST_TO_CENTER_LINE_N, StateIndex.MIN_DIST_TO_CENTER_LINE_N] = float(lbmpc_config.qn)
    Q[StateIndex.ORIENTATION_THETA, StateIndex.ORIENTATION_THETA] = float(lbmpc_config.qalpha)
    Q[StateIndex.VELOCITY_V_X, StateIndex.VELOCITY_V_X] = float(lbmpc_config.qv)

    # Inputs are [derDelta, jerk]
    R[InputIndex.D_STEERING_ANGLE, InputIndex.D_STEERING_ANGLE] = max(float(lbmpc_config.qddelta), 1e-6)
    R[InputIndex.JERK, InputIndex.JERK] = max(float(lbmpc_config.qjerk), 1e-6)

    # Solve continuous-time algebraic Riccati equation:
    # A'P + P A - P B R^-1 B' P + Q = 0
    P = solve_continuous_are(A0, B0, Q, R)

    # Continuous-time LQR gain:
    # u = -K x   (for regulation about origin in deviation coordinates)
    K = np.linalg.solve(R, B0.T @ P)

    print("\nQ (LQR):\n", Q)
    print("\nR (LQR):\n", R)
    print("\nP (CARE solution):\n", P)
    print("\nK (continuous-time LQR, u = -K x):\n", K)

    return A0, B0, c0, f0


if __name__ == "__main__":
    linearize_at_operating_point()
