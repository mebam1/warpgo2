# NeRD Datasets

이 디렉토리는 `CubeTossing`용 NeRD HDF5 데이터셋을 저장한다.

## Real dataset 생성

`contact-nets` 실측 `.pt`를 NeRD HDF5로 변환:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/scripts/convert_contact_nets_pt_to_nerd_hdf5.py --output-path mebam/data/nerd/real/0.hdf5
```

기본 입력은 `mebam/data/contect_nets/*.pt` 이다.

## Simulation dataset 생성

Newton `CubeTossingEnv` rollout을 `mebam/data/nerd/simulation/<index>.hdf5` 로 저장:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/scripts/generate_cube_tossing_sim_nerd_hdf5.py --output-dir mebam/data/nerd/simulation --num-trajectories 128 --device cuda:0 --solver-type mujoco
```

CPU에서 생성:

```powershell
python mebam/scripts/generate_cube_tossing_sim_nerd_hdf5.py --output-dir mebam/data/nerd/simulation --num-trajectories 128 --device cpu --solver-type mujoco
```

기존 파일 덮어쓰기:

```powershell
python mebam/scripts/generate_cube_tossing_sim_nerd_hdf5.py --output-dir mebam/data/nerd/simulation --num-trajectories 128 --overwrite
```

## NeRD 학습

여러 개의 simulation HDF5를 묶어 train/valid/eval dataset을 만든 뒤 `train/train.py`를 호출:

```powershell
cd D:\work\nerd\neural-robot-dynamics
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/*.hdf5 --logdir runs/CubeTossingSimulation --device cuda:0
```

우선 merged dataset만 만들고 실제 학습은 하지 않으려면:

```powershell
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/*.hdf5 --dataset-cache-dir mebam/data/nerd/prepared/cube_tossing_simulation --prepare-only --overwrite-cache
```

validation set을 별도 glob으로 주고 싶으면:

```powershell
python mebam/scripts/train_nerd_cube_tossing.py --input-glob mebam/data/nerd/simulation/train/*.hdf5 --valid-glob mebam/data/nerd/simulation/valid/*.hdf5 --logdir runs/CubeTossingSimulation
```

## 생성 결과 확인

저장된 HDF5 trajectory replay:

```powershell
python mebam/scripts/visualize_nerd_hdf5_cube_tossing.py --input-path mebam/data/nerd/simulation/0.hdf5 --camera-tracking
```
