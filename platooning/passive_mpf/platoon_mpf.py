"""Paper 2 (Jeong, Lee, Joo): passive platoon, mPF topology, y_i = velocity.
G_v(s) = 1/(s(tau s+1)), PD in series K(s) = alpha s + beta  ->  G~_v = (alpha s + beta)/(s(tau s + 1)),
passive iff alpha >= beta tau (Lemma 1).  u_i = k sum_j (y_j - y_i)  (eq. 7), gamma_i <= 1/m (Thm 2).

Implementation on a speed-commanded car (v = v_cmd/(tau s+1)):  v_cmd = alpha*u + beta*int(u)  (PI on u)
gives exactly u -> v = G~_v, no differentiation needed.
A) Lemma 1 and the passivity shortage a delay adds
B) replicate Table 1 of the paper (LTI, tau=0.2, alpha=1.4, beta=2.6, k=0.5, N=20)
C) the same design on the gym longitudinal dynamics (tau 0.3 accel / 0.1 brake, +-5 m/s^2), 40 Hz, delays
"""
import numpy as np

W = np.logspace(-3, 4, 300000)


def min_re(alpha, beta, tau, L=0.0):
    s = 1j * W
    G = (alpha * s + beta) / (s * (tau * s + 1)) * np.exp(-s * L)
    return np.min(G.real)


def preds(i, m):
    """0-based vehicle i (vehicle index i+1); leader is -1 -> returns indices into Y = [y0, y1..yN]."""
    I = i + 1
    return list(range(0, I)) if I <= m else list(range(I - m, I))


def ref(t, scale=1.0):
    if t < 0:
        return 0.0
    if t < 10:
        return 2.5 * t * scale
    if t < 20:
        return 30 * scale
    return 25 * scale


def simulate(N, m, k, alpha, beta, speed_loop, T=45.0, dt=1e-3, Tc=None, delay_ticks=0, scale=1.0, d=1.0):
    """speed_loop(vcmd, v) -> dv/dt.  Tc None: continuous control (every dt)."""
    v, I, p = np.zeros(N), np.zeros(N), -d * np.arange(1, N + 1)
    y0, p0 = 0.0, 0.0
    vcmd = np.zeros(N)
    every = 1 if Tc is None else int(round(Tc / dt))
    hist, e2 = [], np.zeros(N)
    P = [preds(i, m) for i in range(N)]
    min_gap = np.inf
    for n in range(int(T / dt)):
        t = n * dt
        if n % every == 0:
            Y = np.r_[y0, v]
            hist.append(Y.copy())
            Yd = hist[max(0, len(hist) - 1 - delay_ticks)]          # communicated (delayed) velocities
            h = dt * every
            u = np.array([k * sum(Yd[j] - v[i] for j in P[i]) for i in range(N)])
            I += u * h
            vcmd = np.maximum(alpha * u + beta * I, 0.0)
        Y = np.r_[y0, v]
        e2 += (Y[:-1] - Y[1:])**2 * dt
        v = v + speed_loop(vcmd, v) * dt
        p = p + v * dt
        y0 += (ref(t, scale) - y0) / 0.2 * dt                        # reference through 0.2 s low-pass
        p0 += y0 * dt
        min_gap = min(min_gap, np.min(p[:-1] - p[1:]))
    ratios = np.sqrt(e2 / e2[0])
    return ratios, min_gap, p[:-1] - p[1:]


# speed loop models: speed_loop(vcmd, v) -> dv/dt
lti = lambda tau: (lambda vcmd, v: (vcmd - v) / tau)                 # symmetric first order
A_MAX, KP_ACC, KP_BRK = 5.0, 10 * 5.0 / 15.0, 10 * 5.0 / 5.0          # gym pid(): tau 0.3 accel / 0.1 brake
gym = lambda vcmd, v: np.clip(np.where(vcmd > v, KP_ACC, KP_BRK) * (vcmd - v), -A_MAX, A_MAX)


def main():
    print("== A) Lemma 1: min Re G~(jw) (>= 0 passive; < 0 -> passivity-short with eps = -2 min Re)")
    for a, b, t in [(1.4, 2.6, 0.2), (1.4, 2.6, 0.3), (1.4, 2.6, 0.5), (1.4, 2.6, 0.6), (0.78, 2.6, 0.3), (0.7, 2.6, 0.3)]:
        print(f"   alpha={a} beta={b} tau={t}: alpha-beta*tau = {a-b*t:+.3f}, min Re = {min_re(a, b, t):+.4f}")
    print("   with a transport delay L (alpha=1.4, beta=2.6, tau=0.3): passivity is lost -> eps = -2 min Re")
    for L in [0.0125, 0.025, 0.05, 0.1]:
        m = min_re(1.4, 2.6, 0.3, L)
        print(f"   L = {L:.4f} s: min Re = {m:+.4f} -> eps = {-2*m:.3f}")

    print("\n== B) replicate paper Table 1 (LTI, tau=0.2, alpha=1.4, beta=2.6, k=0.5, N=20, continuous)")
    paper = {1: 0.9403, 2: 0.4838, 3: 0.3260, 5: 0.1974, 7: 0.1415, 10: 0.0994}
    for m in paper:
        r, _, _ = simulate(20, m, 0.5, 1.4, 2.6, lti(0.2))
        print(f"   m={m:2d}: gamma_max = {r[1:].max():.4f}   paper {paper[m]:.4f}   bound 1/m = {1/m:.4f}")

    print("\n== C) gym longitudinal dynamics (tau 0.3/0.1, +-5 m/s^2), N=10, 40 Hz, speeds scaled 1/12.5 "
          "(0.2t -> 2.4 -> 2.0 m/s), d=1 m")
    for m in [1, 2, 3, 5]:
        for dl in [1, 4]:
            r, gmin, gaps = simulate(10, m, 0.5, 1.4, 2.6, gym, T=45.0, Tc=0.025, delay_ticks=dl, scale=1 / 12.5)
            print(f"   m={m} delay {dl*25:3d} ms: gamma_max = {r[1:].max():.3f} (bound {1/m:.3f}), "
                  f"min gap {gmin:.2f} m, final gaps {np.round(gaps, 2)}")


if __name__ == "__main__":
    main()
