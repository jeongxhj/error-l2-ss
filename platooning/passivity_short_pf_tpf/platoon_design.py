"""Paper (Jeong, Joo, Choi): passivity-short platoon, G(s) = m / (s (tau s^2 + s + m kv)), y = position.
1) eps of eq. (23) vs. the KYP / frequency condition eps = -2 min_w Re G(jw)
2) gain design table for the JET1 sim (m = 1, tau = 0.3 accel / 0.1 brake)
3) effect of a transport delay on eps
4) platoon on the gym's longitudinal dynamics (asymmetric P speed loop + saturation), 40 Hz control,
   speed command from a virtual integrator v_cmd = int a_d, PF / TPF, gains inside / outside the bounds
"""
import numpy as np

W = np.logspace(-3, 4, 400000)


def eps_23(m, tau, kv):
    a = min(m * kv * tau, 0.5)
    return 2 * m * tau**2 / (a * (2 * m * kv * tau - a))


def eps_freq(m, tau, kv, delay=0.0):
    s = 1j * W
    G = m * np.exp(-s * delay) / (s * (tau * s**2 + s + m * kv))
    return -2 * np.min(G.real)


print("== 1) eq. (23) vs -2 min Re G(jw)")
for m, tau, kv in [(1, 0.2, np.sqrt(5)), (1, 0.3, 1.667), (1, 0.1, 1.667), (1, 0.3, 4.0), (0.9, 0.25, 2.0)]:
    print(f"   m={m} tau={tau} kv={kv:.3f}:  eps(23) = {eps_23(m, tau, kv):.4f}   eps(freq) = {eps_freq(m, tau, kv):.4f}")

print("\n== 2) design table, JET1 sim: m = 1, tau_max = 0.3 s (accel), headway h = kv/k at k = k_max")
m, tau = 1.0, 0.3
print("   kv     zeta   eps    PF k<=1/eps  h_PF[s]   TPF k<=1/(2eps)  h_TPF[s]   (internal stab. PF k<kv/tau, TPF k<kv/(2tau))")
for kv in [1.0, 1.25, 1.667, 2.5, 4.0, 8.0]:
    e = eps_23(m, tau, kv)
    zeta = 1 / (2 * np.sqrt(tau * m * kv))
    print(f"   {kv:5.3f}  {zeta:5.2f}  {e:5.3f}   {1/e:8.3f}   {kv*e:6.2f}      {1/(2*e):8.3f}      {2*kv*e:6.2f}     "
          f"({kv/tau:.2f}, {kv/(2*tau):.2f})")

print("\n== 3) transport delay L (comm + actuation): eps(L), m=1, tau=0.3, kv=1.667")
for L in [0.0, 0.025, 0.05, 0.1, 0.2]:
    e = eps_freq(1, 0.3, 1.667, L)
    print(f"   L = {L:5.3f} s: eps = {e:.3f} -> PF k <= {1/e:.3f}, TPF k <= {1/(2*e):.3f}")

# ---------------- 4) platoon on the gym longitudinal dynamics ----------------
A_MAX, V_MAX, V_MIN = 5.0, 15.0, -5.0
KP_ACC, KP_BRK = 10 * A_MAX / V_MAX, 10 * A_MAX / (-V_MIN)     # gym pid(): 3.33 / 10 -> tau 0.3 / 0.1
DT, TC, T_END, D_GAP = 0.001, 0.025, 40.0, 1.0


def y0(t, vbar=2.0, A=0.3, w=3 * np.pi / 5):
    """paper eq. (24) scaled to 1/10 speed: cruise, sinusoidal perturbation, cruise (C^0 at t = 20)."""
    if t < 10:
        return vbar * t
    if t < 20:
        return vbar * t + A * np.sin(w * (t - 10))
    return 20 * vbar + vbar * (t - 20)


def run(topo, k, kv, N=6, delay_ticks=1):
    p, v, vcmd = -D_GAP * np.arange(1, N + 1), np.zeros(N), np.zeros(N)   # at rest, spacing d
    steps, ctrl_every = int(T_END / DT), int(round(TC / DT))
    hist = []                          # sampled at control rate: positions incl. leader
    E, amax = [], 0.0
    for n in range(steps):
        t = n * DT
        if n % ctrl_every == 0:
            hist.append(np.r_[y0(t), p])
            old = hist[max(0, len(hist) - 1 - delay_ticks)]               # predecessor info, delayed
            for i in range(N):
                yi = p[i] - (-(i + 1) * D_GAP)                              # own position minus offset
                pred = lambda j: old[j] - (-(j) * D_GAP)                    # j = 0 is the leader
                if topo == "PF" or i == 0:
                    u = k * (pred(i) - yi) if topo == "PF" or i == 0 else None
                if topo == "TPF" and i >= 1:
                    u = k * (pred(i - 1) + pred(i) - 2 * yi)
                ad = u - kv * v[i]
                vcmd[i] = np.clip(vcmd[i] + ad * TC, 0.0, V_MAX)
            # errors e_i = y_{i-1} - y_i - d (true, undelayed)
            full = np.r_[y0(t), p]
            E.append(full[:-1] - full[1:] - D_GAP)
        diff = vcmd - v
        a = np.where(diff > 0, KP_ACC, KP_BRK) * diff
        a = np.clip(a, -A_MAX, A_MAX)
        amax = max(amax, np.max(np.abs(a)))
        v += a * DT
        p += v * DT
    E = np.array(E)
    l2 = np.sqrt(np.sum(E**2, axis=0) * TC)
    gaps = E + D_GAP
    return l2 / l2[0], amax, gaps.min()


print("\n== 4) gym longitudinal dynamics, 6 followers, d = 1 m, vbar = 2 m/s, 40 Hz, comm delay 1 tick (25 ms)")
kv = 1.667
eps = eps_23(1, 0.3, kv)
cases = [("PF", 1.2, 1), ("PF", 1.2, 4), ("PF", 4.0, 1), ("TPF", 0.6, 1), ("TPF", 0.6, 4), ("TPF", 2.0, 1)]
print(f"   kv = {kv}, eps = {eps:.3f}: PF bound k <= {1/eps:.3f}, TPF bound k <= {1/(2*eps):.3f}")
for topo, k, dl in cases:
    r, amax, gmin = run(topo, k, kv, delay_ticks=dl)
    bound = 1 / eps if topo == "PF" else 1 / (2 * eps)
    tag = "inside" if k <= bound else "OUTSIDE"
    print(f"   {topo:3s} k={k:<4} ({tag} bound), delay {dl*25:3d} ms: ||e_i||/||e_1|| (i=2..6) = {np.round(r[1:], 3)}, "
          f"max|a| = {amax:.2f}, min gap = {gmin:.2f} m")
