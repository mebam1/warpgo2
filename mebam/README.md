# mebam

이 폴더는 `contact-nets` real-world cube tossing 설정을 정리한 설정 파일과 샘플 데이터, 그리고 Newton 기반 `CubeTossingEnv`와 연결되는 보조 파일을 담고 있다.

## 1. Cube toss env 실행 명령어

권장 환경은 `nerd` conda 환경이다.

```powershell
conda activate nerd
cd D:\work\nerd\neural-robot-dynamics
python examples/example_newton_env.py --env-name CubeTossing --camera-tracking
```

자주 쓰는 변형:

```powershell
python examples/example_newton_env.py --env-name CubeTossing --device cpu
python examples/example_newton_env.py --env-name CubeTossing --solver-type xpbd --device cpu
python examples/example_newton_env.py --env-name CubeTossing --obs-type joint
```

관련 파일:

- 환경 구현: [env_cube.py](/D:/work/nerd/neural-robot-dynamics/envs/newton_envs/env_cube.py)
- 실행 예제: [example_newton_env.py](/D:/work/nerd/neural-robot-dynamics/examples/example_newton_env.py)
- 설정 파일: [contact_nets_cube.yaml](/D:/work/nerd/neural-robot-dynamics/mebam/config/contact_nets_cube.yaml)

### NeRD 훈련 명령어

simulation HDF5 여러 개를 묶어 train/valid/eval dataset을 만든 뒤 바로 학습:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/*.hdf5 --logdir runs/CubeTossingSimulation --device cuda:0
```

merged dataset만 만들고 실제 학습은 하지 않으려면:

```powershell
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/*.hdf5 --dataset-cache-dir mebam/data/nerd/prepared/cube_tossing_simulation --prepare-only --overwrite-cache
```

validation set을 별도 glob으로 지정하려면:

```powershell
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/train/*.hdf5 --valid-glob mebam/data/nerd/simulation/valid/*.hdf5 --logdir runs/CubeTossingSimulation
```

기존 `runs` 체크포인트를 real HDF5 (`mebam/data/nerd/real/0.hdf5`) 기준으로 이어서 학습하려면:

```powershell
python train/train.py --cfg mebam/config/nerd_cube_tossing.yaml --logdir runs/CubeTossingRealFromSim --device cuda:0 --seed 0 --skip-check-log-override --checkpoint runs/CubeTossingSimulation/05-06-2026-21-01-07/nn/best_eval_model.pt --no-time-stamp
```

## 2. 실행 흐름 설명

1. `example_newton_env.py`가 `--env-name CubeTossing` 인자를 받아 `CubeTossingEnv`를 생성한다.
2. `CubeTossingEnv`는 `mebam/config/contact_nets_cube.yaml`을 읽어서 큐브 크기, 질량, 관성, 마찰, restitution, 중력, reset 분포, 종료 조건을 로드한다.
3. `create_articulation()`에서 Newton `ModelBuilder`로 free rigid body 큐브 1개와 box collider를 만든다.
4. `reset_envs()`에서 toss 초기 상태를 샘플링한다.
   - 초기 위치는 `default_position_m` 기준
   - XY 위치 노이즈 추가
   - 임의 orientation 샘플링
   - 초기 선속도/각속도 노이즈 추가
   - Z 선속도는 항상 아래 방향이 되도록 보정
5. 메인 루프에서 `env.update()`가 Newton solver 한 프레임을 진행한다.
6. 매 프레임 `compute_cost_termination()`이 정지 여부와 범위 이탈 여부를 계산한다.
   - 충분히 바닥에 닿고
   - 선속도/각속도가 작으면 종료
   - 혹은 너무 낮게 떨어지거나 XY 반경을 벗어나도 종료
7. 종료되거나 `rollout-frames`에 도달하면 `env.reset()`으로 다음 toss를 시작한다.
8. `env.render()`가 OpenGL viewer에 현재 상태를 시각화한다.

## 3. 사용된 물리 파라미터 표

아래 표는 `CubeTossingEnv`가 실제로 읽어 사용하는 파라미터만 정리한 것이다.

| 분류 | 파라미터 | 값 | 단위 | 사용 위치 | 설명 |
| --- | --- | ---: | --- | --- | --- |
| geometry | `block_half_width_m` | `0.0524` | m | body shape | 큐브 반폭. `hx=hy=hz=0.0524`로 box collider 생성 |
| rigid body | `mass_kg` | `0.37` | kg | body | 큐브 질량 |
| rigid body | `inertia_kg_m2` | `0.00081` | kg m^2 | body | 등방 관성으로 `diag(I)` 사용 |
| contact | `friction_coefficient` | `0.18` | - | shape config | 접촉 마찰계수 |
| contact | `restitution` | `0.125` | - | shape config | 반발계수 |
| dynamics | `gravity_mps2` | `9.81` | m/s^2 | environment | 환경에서는 Z축 방향으로 `-9.81` 적용 |
| timing | `dt_s` | `0.006756756756756757` | s | environment | 프레임 시간 |
| timing | `sample_rate_hz` | `148.0` | Hz | environment | 시뮬레이션 FPS로 사용 |
| reset | `default_position_m` | `[0.0, 0.0, 0.11004]` | m | reset | 기본 COM 초기 위치 |
| reset | `position_noise_xy_m` | `0.09432` | m | reset | XY 위치 랜덤 범위 |
| reset | `default_linear_velocity_mps` | `[0.0, 1.048, 0.0]` | m/s | reset | 기본 초기 선속도 |
| reset | `linear_velocity_noise_mps` | `0.1048` | m/s | reset | 선속도 랜덤 범위 |
| reset | `enforce_downward_z_velocity` | `true` | bool | reset | Z 선속도를 항상 하향으로 보정 |
| reset | `default_angular_velocity_rps` | `[0.0, 0.0, 0.0]` | rad/s | reset | 기본 초기 각속도 |
| reset | `angular_velocity_noise_rps` | `4.0` | rad/s | reset | 각속도 랜덤 범위 |
| reset | `random_orientation` | `true` | bool | reset | 임의 orientation 샘플링 여부 |
| termination | `min_corner_height_threshold_m` | `0.007` | m | termination | 최소 corner 높이가 이 값 이하이면 바닥 접촉으로 간주 |
| termination | `linear_speed_threshold_mps` | `0.05` | m/s | termination | 정지 판정용 선속도 임계값 |
| termination | `angular_speed_threshold_rps` | `0.5` | rad/s | termination | 정지 판정용 각속도 임계값 |
| termination | `min_com_height_m` | `-0.2096` | m | termination | COM이 이 값 아래로 가면 범위 이탈 |
| termination | `max_xy_radius_m` | `2.0` | m | termination | XY 반경이 이 값보다 크면 범위 이탈 |

참고:

- `physics_normalized` 섹션은 `contact-nets` 원본 데이터 정규화 정보를 보존하기 위한 값이고, 현재 Newton `CubeTossingEnv` 생성에는 직접 쓰지 않는다.
- `toss_detection`, `toss_filter` 섹션은 원본 real-world 데이터 전처리 규칙 기록용이며, 현재 Newton 런타임 reset/termination에는 직접 쓰지 않는다.
