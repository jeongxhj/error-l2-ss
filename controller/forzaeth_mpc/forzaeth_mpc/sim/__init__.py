"""forzaeth_mpc.sim - ROS-free closed-loop simulation for the MPC controllers.

  core     controller adapters, plant, rollout, scenario
  tracks   read a global_waypoints.json into the canonical (N,7) array
  metrics  lap times, tracking error, solver success
  plotting reusable figures
  utils    integrators
  cli      runnable entry points
"""

__all__ = ["core", "tracks", "waypoints", "metrics", "utils", "plotting", "cli"]
