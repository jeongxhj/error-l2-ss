"""JET1 in f1tenth_gym: LTI models vs. the nonlinear sim, and frequency-domain passivity indices.

Longitudinal: v_cmd -> v is the gym's P speed loop, a = kp (v_cmd - v)  ->  1/(tau s + 1)
Lateral:      delta -> yaw rate r, linear single track with cornering stiffness C = mu*D*C*B*F_z
"""
import numpy as np
from scipy import signal
from forza_config_ros import load_car_config_ros, load_pacejka_tire_config_ros
from f110_gym.envs.dynamic_models import vehicle_dynamics_st_pacejka, pid

CAR, FLOOR, G = 'JET1', 'dubi', 9.81
car = load_car_config_ros(CAR)
tire = load_pacejka_tire_config_ros(CAR, FLOOR)
P = dict(mu=tire.friction_coeff, Bf=tire.Bf, Cf=tire.Cf, Df=tire.Df, Ef=tire.Ef,
         Br=tire.Br, Cr=tire.Cr, Dr=tire.Dr, Er=tire.Er,
         lf=car.lf, lr=car.lr, h=car.h_cg, m=car.m, I=car.Iz,
         s_min=-car.max_steering_angle, s_max=car.max_steering_angle,
         sv_min=-car.max_steering_velocity, sv_max=car.max_steering_velocity,
         v_switch=7.319, a_max=car.a_max, v_min=car.v_min, v_max=car.v_max)
DT, STEER_DELAY = 0.01, 2  # gym time step, steer_buffer_size


def f(x, u):
    p = P
    return vehicle_dynamics_st_pacejka(x, u, p['mu'], p['Bf'], p['Cf'], p['Df'], p['Ef'], p['Br'], p['Cr'], p['Dr'],
                                       p['Er'], p['lf'], p['lr'], p['h'], p['m'], p['I'], p['s_min'], p['s_max'],
                                       p['sv_min'], p['sv_max'], p['v_switch'], p['a_max'], p['v_min'], p['v_max'])


def sim(v0, cmd, T):
    """Replicates RaceCar.update_pose: steer delay buffer, pid, RK4."""
    x = np.array([0., 0., 0., v0, 0., 0., 0.])
    buf = [0.] * STEER_DELAY
    out = []
    for k in range(int(round(T / DT))):
        v_cmd, d_cmd = cmd(k * DT)
        steer = buf[-1]
        buf = [d_cmd] + buf[:-1]
        accl, sv = pid(v_cmd, steer, x[3], x[2], P['sv_max'], P['a_max'], P['v_max'], P['v_min'])
        u = np.array([sv, accl])
        k1 = f(x, u); k2 = f(x + DT * k1 / 2, u); k3 = f(x + DT * k2 / 2, u); k4 = f(x + DT * k3, u)
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.arange(1, len(out) + 1) * DT, np.array(out)


def tau_63(t, y):
    y = (y - y[0]) / (y[-1] - y[0])
    return t[np.argmax(y >= 1 - np.exp(-1))]


# ---------------- longitudinal ----------------
kp_acc = 10 * P['a_max'] / P['v_max']
kp_brk = 10 * P['a_max'] / (-P['v_min'])
print(f"== Longitudinal (v_cmd -> v), gym P loop: tau_acc = 1/kp = {1/kp_acc:.3f} s, tau_brake = {1/kp_brk:.3f} s")
print(f"   linear only while |v_cmd - v| < a_max/kp: accel {P['a_max']/kp_acc:.2f} m/s, brake {P['a_max']/kp_brk:.2f} m/s")
for v0, v1 in [(3.0, 3.5), (3.0, 2.8), (3.0, 5.0)]:
    t, X = sim(v0, lambda _t: (v1, 0.0), 3.0)
    tau = 1 / kp_acc if v1 > v0 else 1 / kp_brk
    lin = v0 + (v1 - v0) * (1 - np.exp(-t / tau))
    err = np.max(np.abs(X[:, 3] - lin)) / abs(v1 - v0) * 100
    print(f"   step {v0}->{v1} m/s: sim tau63 = {tau_63(np.r_[0, t], np.r_[v0, X[:, 3]]):.3f} s, "
          f"max |sim - 1/(tau s+1)| = {err:.1f} % of step")

# ---------------- lateral ----------------
L = P['lf'] + P['lr']
Fzf, Fzr = P['m'] * G * P['lr'] / L, P['m'] * G * P['lf'] / L
Caf = P['mu'] * P['Df'] * P['Cf'] * P['Bf'] * Fzf
Car = P['mu'] * P['Dr'] * P['Cr'] * P['Br'] * Fzr
print(f"\n== Lateral: cornering stiffness C_af = {Caf:.1f} N/rad, C_ar = {Car:.1f} N/rad")


def lat_ss(vx):
    """x = [v_y, r], u = delta, y = r (gym sign conventions, small angles)."""
    m, I, lf, lr = P['m'], P['I'], P['lf'], P['lr']
    A = np.array([[-(Caf + Car) / (m * vx), (-Caf * lf + Car * lr) / (m * vx) - vx],
                  [(-Caf * lf + Car * lr) / (I * vx), -(Caf * lf**2 + Car * lr**2) / (I * vx)]])
    B = np.array([[Caf / m], [Caf * lf / I]])
    C = np.array([[0., 1.]])
    return A, B, C, np.zeros((1, 1))


print(f"   gym steering: bang-bang sv = +-{P['sv_max']} rad/s, one step moves {P['sv_max']*DT:.3f} rad")
for vx, dcmd in [(3.0, 0.02), (3.0, 0.032), (3.0, 0.064), (5.0, 0.032)]:
    t, X = sim(vx, lambda _t: (vx, dcmd), 2.0)
    A, B, C, D = lat_ss(vx)
    tl = np.r_[0, t]
    d_act = np.r_[0, X[:, 2]]                      # actual steering angle of the sim as the input
    _, r_lin, _ = signal.lsim((A, B, C, D), d_act, tl)
    r_sim = np.r_[0, X[:, 5]]
    err = np.max(np.abs(r_sim - r_lin)) / np.max(np.abs(r_lin)) * 100
    print(f"   delta_cmd {dcmd:.3f} rad @ {vx} m/s: actual delta in last 0.1 s = {np.unique(np.round(X[-10:, 2], 4))}, "
          f"mean {X[-100:, 2].mean():.4f}; r sim {r_sim[-1]:.4f} vs lin(actual delta) {r_lin[-1]:.4f} rad/s, "
          f"max error {err:.1f} % of peak")


# ---------------- passivity indices (frequency domain) ----------------
W = np.logspace(-4, 6, 400000)


def indices(num, den, delay=0.0):
    Gj = np.polyval(num, 1j * W) / np.polyval(den, 1j * W) * np.exp(-1j * W * delay)
    nu = np.min(Gj.real)                    # IFP: Re G >= nu
    rho = np.min((1 / Gj).real)             # OFP: Re G >= rho |G|^2  <=>  Re(1/G) >= rho
    return nu, rho


def fmt(nu, rho):
    def s(v):
        return "-inf" if v < -1e3 else f"{v:+.4f}"
    tag = "passive" if nu >= -1e-9 else ("passivity-short (IFP)" if nu > -1e3 else "not IFP")
    return f"nu(IFP) = {s(nu)}, rho(OFP) = {s(rho)}  -> {tag}"


ta = 1 / kp_acc
print("\n== Passivity indices (nu < 0: input-passivity shortage, rho < 0: output-passivity shortage)")
print(f"   v_cmd -> v        1/({ta:.3f}s+1)          : {fmt(*indices([1], [ta, 1]))}")
print(f"   v_cmd -> p        1/(s({ta:.3f}s+1))       : {fmt(*indices([1], [ta, 1, 0]))}")
print(f"   v_cmd -> p+tau*v  (= 1/s)                 : {fmt(*indices([1], [1, 0]))}")
ts = car.tau_steer
for vx in [1.5, 3.0, 5.0, 7.0]:
    A, B, C, D = lat_ss(vx)
    num, den = signal.ss2tf(A, B, C, D)
    num = np.trim_zeros(np.round(num[0], 12), 'f')
    z, p = np.roots(num), np.roots(den)
    print(f"   delta -> r @ {vx} m/s: G(s) = ({num[0]:.2f}s + {num[1]:.2f})/(s^2 + {den[1]:.2f}s + {den[2]:.2f}), "
          f"poles {np.round(p, 2)}, zero {np.round(z, 2)}, DC {num[-1]/den[-1]:.3f}")
    print(f"        sim (delta=cmd, 0.02 s delay)     : {fmt(*indices(num, den, STEER_DELAY * DT))}")
    print(f"        no delay                          : {fmt(*indices(num, den))}")
    print(f"        real car servo 1/({ts:.3f}s+1)     : {fmt(*indices(num, np.polymul(den, [ts, 1])))}")
