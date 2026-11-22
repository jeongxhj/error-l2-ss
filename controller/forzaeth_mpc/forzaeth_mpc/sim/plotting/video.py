"""Record a simulation run as a video.

The figure that `--plot` writes shows where the car went. This one shows the horizon that
the controller predicts at each step, which changes every step, together with the track
boundaries it has to respect.

It writes mp4 when ffmpeg is installed and an animated gif otherwise. Both work without a
display, because the figure is never shown.
"""
from __future__ import annotations

import os

import numpy as np

S, N, VX = 0, 1, 3          # state indices, the same for every model here


def pick_writer(fps: int):
    """The best writer available, and the file extension that goes with it.

    The container has no ffmpeg on PATH, but imageio-ffmpeg ships a binary, so point
    matplotlib at that one before falling back to a gif. A gif of a 20-lap run is about
    70 MB; the same run in mp4 is about 1 MB.
    """
    import matplotlib
    from matplotlib import animation
    if not animation.writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    if animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400), "mp4"
    return animation.PillowWriter(fps=fps), "gif"


if __name__ == "__main__":
    print("writer:", pick_writer(20)[1])


class SimVideo:
    """One figure, one frame per `add` call, streamed straight to the file."""

    def __init__(self, waypoints: np.ndarray, track_len: float, title: str,
                 path_without_ext: str, fps: int = 20, history_s: float = 4.0,
                 dpi: int = 110):
        import matplotlib
        matplotlib.use("Agg", force=True)      # never needs a display
        import matplotlib.pyplot as plt

        self.plt = plt
        self.track_len = float(track_len)
        self.waypoints = waypoints
        self.history_s = float(history_s)
        self.frames = 0

        writer, ext = pick_writer(fps)
        self.path = f"{path_without_ext}.{ext}"

        self.fig, (self.ax_xy, self.ax_v) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
        self.fig.suptitle(title)

        self._draw_track()
        self.path_line, = self.ax_xy.plot([], [], lw=1.2, color="tab:blue", label="driven")
        self.horizon, = self.ax_xy.plot([], [], lw=2.2, color="tab:orange",
                                        label="predicted horizon")
        self.car, = self.ax_xy.plot([], [], "o", ms=7, color="tab:red", label="car")
        self.ax_xy.set_aspect("equal")
        self.ax_xy.legend(loc="upper right", fontsize=8)
        self.ax_xy.set_xlabel("x [m]"), self.ax_xy.set_ylabel("y [m]")

        self.speed, = self.ax_v.plot([], [], lw=1.0, color="tab:blue")
        self.ax_v.set_xlabel("t [s]"), self.ax_v.set_ylabel("$v_x$ [m/s]")
        self.ax_v.grid(alpha=0.3)
        self.fig.tight_layout()

        self.writer = writer
        self.writer.setup(self.fig, self.path, dpi=dpi)

    def _draw_track(self) -> None:
        """The raceline and the two boundaries, from the d_left/d_right columns."""
        from forzaeth_mpc.sim.cli.run_laps import track_bounds_xy
        (lx, ly), (rx, ry) = track_bounds_xy(self.waypoints, self.track_len)
        self.ax_xy.plot(lx, ly, "-", lw=1.2, color="0.35", label="track bounds")
        self.ax_xy.plot(rx, ry, "-", lw=1.2, color="0.35")
        self.ax_xy.plot(self.waypoints[:, 0], self.waypoints[:, 1], "--", lw=0.6,
                        color="0.55", label="raceline")
        # the view never moves, so set it once from the boundaries
        pad = 0.3
        self.ax_xy.set_xlim(min(lx.min(), rx.min()) - pad, max(lx.max(), rx.max()) + pad)
        self.ax_xy.set_ylim(min(ly.min(), ry.min()) - pad, max(ly.max(), ry.max()) + pad)

    def add(self, X: np.ndarray, sd: np.ndarray, dt: float, laps: int = 0) -> None:
        """`X` is the state history, `sd` the predicted (s, n) over the horizon."""
        if X.size == 0:
            return
        from forzaeth_mpc.sim.cli.run_laps import frenet_to_xy

        # no wrap handling: frenet_to_xy maps s and s + track_len to the same point, so a
        # lap boundary is two adjacent samples and the line closes on itself
        keep = max(1, int(self.history_s / dt))
        tail = X[-keep:]
        px, py = frenet_to_xy(self.waypoints, tail[:, S], tail[:, N], self.track_len)
        self.car.set_data(px[-1:], py[-1:])
        self.path_line.set_data(px, py)

        if sd is not None and len(sd):
            sd = np.asarray(sd, dtype=float)
            hx, hy = frenet_to_xy(self.waypoints, sd[:, 0], sd[:, 1], self.track_len)
            self.horizon.set_data(hx, hy)

        t = np.arange(X.shape[0]) * dt
        self.speed.set_data(t[-keep:], tail[:, VX])
        self.ax_v.relim(), self.ax_v.autoscale_view()
        self.ax_v.set_title(f"lap {laps} | t = {t[-1]:.1f} s | "
                            f"$v_x$ = {tail[-1, VX]:.2f} m/s | n = {tail[-1, N]:+.3f} m")

        self.writer.grab_frame()
        self.frames += 1

    def close(self) -> str:
        self.writer.finish()
        self.plt.close(self.fig)
        size_mb = os.path.getsize(self.path) / 1e6
        print(f"[video] {self.frames} frames -> {self.path} ({size_mb:.1f} MB)")
        return self.path
