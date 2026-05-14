# Cube Tossing Sim2Real Sweep

This directory contains a CubeTossing sim2real sweep runner.
It rolls out Newton simulation from real-data initial states and ranks
physical parameter candidates by the gap to the reference real trajectories.

Default run:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/sweep/sweep_cube_tossing_sim2real.py --device cuda:0
```

The default spec is
[default_cube_tossing_physics_sweep.yaml](/D:/work/nerd/neural-robot-dynamics/mebam/sweep/default_cube_tossing_physics_sweep.yaml).

Default sweep axes:

- `physics_metric.inertia_kg_m2`
- `physics_metric.friction_coefficient`
- `physics_metric.restitution`

All three are treated as isotropic scalar values.

Default horizon: `32`

Quick dry-run:

```powershell
python mebam/sweep/sweep_cube_tossing_sim2real.py --dry-run
```

Short smoke test:

```powershell
python mebam/sweep/sweep_cube_tossing_sim2real.py --limit-candidates 4 --max-trajectories 8 --horizon 32 --device cpu
```

Outputs:

- `run_manifest.yaml`: selected trajectories, horizon, metric scales
- `results.jsonl`: per-candidate streaming results
- `leaderboard.csv`: sorted ranking table
- `summary.json`: best candidate and top results
- `best_contact_nets_cube.yaml`: best candidate applied to the cube config

Current gap metric:

- rollout from the same real initial state
- `position RMSE`
- `orientation RMSE` using quaternion geodesic angle
- `linear velocity RMSE`
- `angular velocity RMSE`
- final score = weighted normalized combination of the four RMSE terms

Default scales:

- position: `block_half_width_m`
- orientation: `1.0 rad`
- linear velocity: RMS over the selected real subset
- angular velocity: RMS over the selected real subset

Sweep spec format for isotropic scalar parameters:

```yaml
sweep:
  mode: grid   # grid | zip
  parameters:
    # isotropic scalar values
    physics_metric.inertia_kg_m2: [0.00065, 0.00081, 0.00097]
    physics_metric.friction_coefficient: [0.1, 0.18, 0.26]
    physics_metric.restitution: [0.0, 0.125, 0.25]

evaluation:
  trajectory_selection: strided   # first | strided | random
  max_trajectories: 16
  horizon: 32

metric:
  component_weights:
    position: 1.0
    orientation: 1.0
    linear_velocity: 1.0
    angular_velocity: 1.0
  scales:
    position_m: auto_half_extent  # or numeric
    orientation_rad: 1.0
    linear_velocity_mps: auto_rms # or numeric
    angular_velocity_rps: auto_rms
```

Any dotted `physics_metric.*` key that exists in the base cube config can be
used as a sweep parameter, as long as the value is a scalar. The current
CubeTossing setup does not support anisotropic vector-valued inertia, friction,
or restitution in this sweep runner.
