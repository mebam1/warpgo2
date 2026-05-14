# mebam/visualize

`CubeTossing`용 pretrained NeRD 모델을 로드해서 Newton viewer로 rollout을 확인하는 스크립트가 들어 있다.

## 파일

- [simulate_pretrained_cube_tossing.py](/D:/work/nerd/neural-robot-dynamics/mebam/visualize/simulate_pretrained_cube_tossing.py)

## 기본 실행

checkpoint를 자동 탐색해서 바로 시각화:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/visualize/simulate_pretrained_cube_tossing.py --camera-tracking
```

자동 탐색 순서:

1. `pretrained_models/NeRD_models/CubeTossing/model/nn/model.pt`
2. `runs/CubeTossingSimulation/**/nn/best_eval_model.pt`
3. `runs/CubeTossingSimulation/**/nn/best_valid_exp_trajectory_model.pt`
4. `runs/CubeTossingSimulation/**/nn/best_valid_passive_trajectory_model.pt`
5. `runs/CubeTossingSimulation/**/nn/final_model.pt`

## checkpoint 직접 지정

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt --camera-tracking
```

필요하면 `cfg.yaml`도 명시:

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt --cfg-path runs/CubeTossingSimulation/05-04-2026-17-43-28/cfg.yaml --camera-tracking
```

## HDF5 초기 상태에서 시작

trajectory 첫 프레임을 초기 상태로 사용:

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt --initial-state-hdf5 mebam/data/nerd/simulation/0.hdf5 --trajectory-index 0 --camera-tracking
```

real trajectory 기준으로 시작:

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt --initial-state-hdf5 mebam/data/nerd/real/0.hdf5 --trajectory-index 0 --camera-tracking
```

## 자주 쓰는 옵션

- `--device cpu`
- `--num-envs 1`
- `--num-rollouts 1`
- `--rollout-horizon 300`
- `--obs-type contact_nets`
- `--use-graph-capture`

CPU로 확인:

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt --device cpu --camera-tracking
```

도움말:

```powershell
python mebam/visualize/simulate_pretrained_cube_tossing.py --help
```

## Trajectory gap overlays

`compare_cube_tossing_trajectories.py` writes an SVG overlay of the cube COM
trajectory for the requested comparison. The rollout always starts from the
reference trajectory initial state.

`sim2real`: Newton Sim vs. real data

```powershell
python mebam/visualize/compare_cube_tossing_trajectories.py --comparison sim2real --real-hdf5 mebam/data/nerd/real/0.hdf5 --trajectory-index 0
```

`sim2nerd`: Newton Sim vs. NeRD Sim

```powershell
python mebam/visualize/compare_cube_tossing_trajectories.py --comparison sim2nerd --sim-hdf5 mebam/data/nerd/simulation/0.hdf5 --trajectory-index 0
```

`nerd2real`: NeRD Sim vs. real data

```powershell
python mebam/visualize/compare_cube_tossing_trajectories.py --comparison nerd2real --real-hdf5 mebam/data/nerd/real/0.hdf5 --trajectory-index 0
```

Useful options:

- `--horizon 50`
- `--device cpu`
- `--model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt`
- `--cfg-path runs/CubeTossingSimulation/05-04-2026-17-43-28/cfg.yaml`
- `--output-path figures/trajectory_gaps/custom_name.svg`

## Sim2NeRD Warp viewer

`visualize_sim2nerd_cube_tossing.py` replays Newton Sim and NeRD Sim together in
the Warp viewer. The two cubes are drawn with different colors and start from
the same live Newton-sim initial state. The Newton trajectory is collected on
the spot for the requested horizon; this script does not read a stored sim HDF5
trajectory anymore.

```powershell
python mebam/visualize/visualize_sim2nerd_cube_tossing.py --horizon 111 --camera-tracking
```

Useful options:

- `--model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt`
- `--cfg-path runs/CubeTossingSimulation/05-04-2026-17-43-28/cfg.yaml`
- `--horizon 80`
- `--solver-type mujoco`
- `--frame-stride 2`
- `--device cpu`
- `--hide-trails`

## Real2NeRD Warp viewer

`visualize_real2nerd_cube_tossing.py` replays a selected real trajectory and the
matching NeRD rollout together in the Warp viewer. The two cubes start from the
same real-data initial state, and NeRD rolls out for the requested horizon.

```powershell
python mebam/visualize/visualize_real2nerd_cube_tossing.py --real-hdf5 mebam/data/nerd/real/0.hdf5 --trajectory-index 0 --camera-tracking
```

Useful options:

- `--model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt`
- `--cfg-path runs/CubeTossingSimulation/05-04-2026-17-43-28/cfg.yaml`
- `--horizon 80`
- `--frame-stride 2`
- `--device cpu`
- `--hide-trails`

## Real-Sim-NeRD Warp viewer

`visualize_real_sim_nerd_cube_tossing.py` replays a selected real trajectory,
the matching Newton Sim rollout, and the matching NeRD rollout together in the
Warp viewer. The Sim and NeRD rollouts both start from the first state of the
selected real trajectory.

It also logs:

- `avg_position_error_real_sim_m`
- `avg_orientation_error_real_sim_rad`
- `avg_position_error_real_nerd_m`
- `avg_orientation_error_real_nerd_rad`

```powershell
python mebam/visualize/visualize_real_sim_nerd_cube_tossing.py --real-hdf5 mebam/data/nerd/real/0.hdf5 --trajectory-index 0 --camera-tracking
```

Useful options:

- `--model-path runs/CubeTossingSimulation/05-04-2026-17-43-28/nn/best_eval_model.pt`
- `--cfg-path runs/CubeTossingSimulation/05-04-2026-17-43-28/cfg.yaml`
- `--solver-type mujoco`
- `--horizon 80`
- `--frame-stride 2`
- `--device cpu`
- `--hide-trails`
