# mebam/data

## Real data training

Working directory:

```powershell
conda activate nerd
cd D:\work\nerd\neural-robot-dynamics
```

Convert `mebam/data/contact_nets/*.pt` to the NeRD HDF5 dataset:

```powershell
python mebam/scripts/convert_contact_nets_pt_to_nerd_hdf5.py --input-glob mebam/data/contact_nets/*.pt --output-path mebam/data/nerd/real/0.hdf5
```

Start training with the real dataset:

```powershell
python train/train.py --cfg mebam/config/nerd_cube_tossing.yaml --logdir runs/CubeTossingReal --device cuda:0 --seed 0 --skip-check-log-override
```

Resume the latest run under the same log root:

```powershell
python train/train.py --cfg mebam/config/nerd_cube_tossing.yaml --logdir runs/CubeTossingReal --device cuda:0 --seed 0 --skip-check-log-override --resume
```

Run on CPU:

```powershell
python train/train.py --cfg mebam/config/nerd_cube_tossing.yaml --logdir runs/CubeTossingReal --device cpu --seed 0 --skip-check-log-override
```

Notes:

- `mebam/config/nerd_cube_tossing.yaml` currently uses `mebam/data/nerd/real/0.hdf5` for train, valid, and eval.
- Training will run, but the validation metrics are not a reliable generalization measure.
- Resume checkpoints are written to `runs/.../nn/latest_checkpoint.pt`.
