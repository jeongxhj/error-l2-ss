"""Why the gym platoon slightly exceeds gamma <= 1/m (Thm 2) for m >= 3: isolate the cause.

Same design as platoon_mpf.py section C (alpha=1.4, beta=2.6, k=0.5, N=10, speeds scaled 1/12.5), one effect at a time:
  - asymmetric tau (gym 0.3 accel / 0.1 brake) and +-5 m/s^2 saturation
  - 40 Hz zero-order-hold control
  - communication delay of the predecessors' velocities
  - a lower alpha (close to beta * tau_max) as an attempted fix
Result (2026-11-22): asymmetry and the 40 Hz ZOH keep gamma below 1/m; the communication delay alone pushes it over
(G~ with a delay is passivity-short, not passive, and the 1/m bound is nearly tight), lowering alpha does not help,
a faster loop with a shorter delay (100 Hz, 10 ms) brings it back to about 1/m.

Run:  python3 delay_cause_check.py   (numpy only, a few minutes)
"""
from platoon_mpf import simulate, lti, gym

CASES = [
    # name,                                              speed loop, alpha, Tc [s], delay [ticks]
    ("LTI tau=0.3 symmetric, continuous, no delay       ", lti(0.3), 1.4, None, 0),
    ("LTI tau=0.3 symmetric, 40 Hz, 25 ms delay         ", lti(0.3), 1.4, 0.025, 1),
    ("gym, continuous, no delay (asymmetry only)        ", gym, 1.4, None, 0),
    ("gym, 40 Hz, no comm delay (ZOH only)              ", gym, 1.4, 0.025, 0),
    ("gym, 100 Hz, 10 ms delay                          ", gym, 1.4, 0.010, 1),
    ("gym, 40 Hz, 25 ms delay, alpha=0.9 (~beta*tau_max)", gym, 0.9, 0.025, 1),
]


def main():
    print("gamma_max / bound 1/m  (N=10, k=0.5, beta=2.6, d=1 m)")
    for name, loop, alpha, Tc, dl in CASES:
        out = []
        for m in [2, 3, 5]:
            r, _, _ = simulate(10, m, 0.5, alpha, 2.6, loop, T=45.0, dt=1e-3, Tc=Tc, delay_ticks=dl, scale=1 / 12.5)
            flag = " !" if r[1:].max() > 1 / m else ""
            out.append(f"m={m}: {r[1:].max():.3f}/{1/m:.3f}{flag}")
        print(f"{name}: " + "   ".join(out))


if __name__ == "__main__":
    main()
