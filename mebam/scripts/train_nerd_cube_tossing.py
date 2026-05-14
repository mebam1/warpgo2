from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mebam.scripts.convert_contact_nets_pt_to_nerd_hdf5 import create_dataset


DEFAULT_CFG = REPO_ROOT / "mebam" / "config" / "nerd_cube_tossing.yaml"
DEFAULT_INPUT_GLOB = "mebam/data/nerd/simulation/*.hdf5"
DEFAULT_CACHE_DIR = REPO_ROOT / "mebam" / "data" / "nerd" / "prepared" / "cube_tossing_simulation"
DEFAULT_LOGDIR = REPO_ROOT / "runs" / "CubeTossing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare NeRD CubeTossing training datasets from one or more HDF5 "
            "trajectory files and optionally launch train/train.py."
        )
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for training HDF5 trajectory files.",
    )
    parser.add_argument(
        "--valid-glob",
        default=None,
        help=(
            "Optional glob for validation HDF5 files. If omitted, input-glob files "
            "are split by ratio into train/valid."
        ),
    )
    parser.add_argument(
        "--eval-glob",
        default=None,
        help=(
            "Optional glob for rollout evaluation HDF5 files. If omitted, the valid "
            "dataset is reused for evaluation."
        ),
    )
    parser.add_argument(
        "--cfg",
        default=str(DEFAULT_CFG.relative_to(REPO_ROOT)),
        help="Base NeRD config YAML path.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=str(DEFAULT_CACHE_DIR.relative_to(REPO_ROOT)),
        help="Directory where merged train/valid/passive_valid HDF5 files are written.",
    )
    parser.add_argument(
        "--logdir",
        default=str(DEFAULT_LOGDIR.relative_to(REPO_ROOT)),
        help="Training log directory passed to train/train.py.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=80.0,
        help="Train split percent used when --valid-glob is not provided.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=20.0,
        help="Valid split percent used when --valid-glob is not provided.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="Compression passed to merged HDF5 writers. Use 'none' to disable.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch train/train.py.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to train/train.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to train/train.py.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Optional env.num_envs override for train/train.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional algorithm.batch_size override.",
    )
    parser.add_argument(
        "--sample-sequence-length",
        type=int,
        default=None,
        help=(
            "Optional override for algorithm.sample_sequence_length and "
            "env.neural_solver_cfg.num_states_history."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional algorithm.num_epochs override.",
    )
    parser.add_argument(
        "--num-iters-per-epoch",
        type=int,
        default=None,
        help="Optional algorithm.num_iters_per_epoch override.",
    )
    parser.add_argument(
        "--num-valid-batches",
        type=int,
        default=None,
        help="Optional algorithm.num_valid_batches override.",
    )
    parser.add_argument(
        "--eval-rollout-horizon",
        type=int,
        default=None,
        help="Optional algorithm.eval.rollout_horizon override.",
    )
    parser.add_argument(
        "--eval-num-rollouts",
        type=int,
        default=None,
        help="Optional algorithm.eval.num_rollouts override.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path passed to train/train.py.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing training checkpoint / run directory.",
    )
    parser.add_argument(
        "--cfg-overrides",
        default="",
        help="Additional config overrides appended after dataset path overrides.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build merged HDF5 datasets and print the train command.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Rewrite merged datasets even if the cache directory already exists.",
    )
    parser.add_argument(
        "--no-time-stamp",
        action="store_true",
        help="Pass --no-time-stamp to train/train.py.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Pass --render to train/train.py for evaluation-time visualization.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative_str(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def numeric_sort_key(path: Path) -> tuple[int, Any]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def list_hdf5_paths(pattern: str) -> list[Path]:
    resolved = str(resolve_repo_path(pattern))
    paths = [Path(path) for path in glob.glob(resolved)]
    paths = [path for path in paths if path.is_file() and path.suffix == ".hdf5"]
    return sorted(paths, key=numeric_sort_key)


def split_paths_by_ratio(
    paths: list[Path],
    train_ratio: float,
    valid_ratio: float,
) -> tuple[list[Path], list[Path], bool]:
    if not paths:
        return [], [], False
    if train_ratio <= 0.0 or valid_ratio < 0.0:
        raise ValueError("train_ratio must be > 0 and valid_ratio must be >= 0.")

    total = len(paths)
    train_end = int(round(total * train_ratio / 100.0))
    train_end = min(total, max(1, train_end))
    valid_end = int(round(total * (train_ratio + valid_ratio) / 100.0))
    valid_end = min(total, max(train_end, valid_end))

    train_paths = paths[:train_end]
    valid_paths = paths[train_end:valid_end]

    duplicated_valid = False
    if not valid_paths:
        valid_paths = list(train_paths)
        duplicated_valid = True

    return train_paths, valid_paths, duplicated_valid


def decode_attr_strings(values: Any, fallback: list[str]) -> list[str]:
    if values is None:
        return fallback
    decoded = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    if not decoded:
        return fallback
    return decoded


def load_hdf5_trajectories(path: Path) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as h5_file:
        data_group = h5_file["data"]
        mode = data_group.attrs.get("mode", None)
        if mode != "trajectory":
            raise ValueError(f"{path} is not a trajectory dataset. mode={mode!r}")

        if "states" not in data_group:
            raise KeyError(f"{path} does not contain /data/states.")

        states_shape = data_group["states"].shape
        if len(states_shape) < 3:
            raise ValueError(f"{path} states must have shape (T, B, ...), got {states_shape}.")

        max_frames, num_trajectories = states_shape[0], states_shape[1]
        raw_traj_lengths = np.asarray(
            data_group.attrs.get(
                "raw_traj_lengths",
                np.full(num_trajectories, max_frames, dtype=np.int32),
            ),
            dtype=np.int32,
        ).reshape(-1)
        if raw_traj_lengths.shape[0] != num_trajectories:
            raw_traj_lengths = np.full(num_trajectories, max_frames, dtype=np.int32)

        if "traj_lengths" in data_group:
            traj_lengths = np.asarray(data_group["traj_lengths"][()], dtype=np.int32).reshape(-1)
        else:
            traj_lengths = np.maximum(raw_traj_lengths - 1, 1)
        if traj_lengths.shape[0] != num_trajectories:
            traj_lengths = np.maximum(raw_traj_lengths - 1, 1)

        source_files = decode_attr_strings(
            data_group.attrs.get("source_files", None),
            fallback=[f"{path.as_posix()}#traj={traj_idx}" for traj_idx in range(num_trajectories)],
        )
        if len(source_files) != num_trajectories:
            source_files = [f"{path.as_posix()}#traj={traj_idx}" for traj_idx in range(num_trajectories)]

        dataset_keys = [key for key in data_group.keys() if key != "traj_lengths"]
        arrays = {key: np.asarray(data_group[key][()], dtype=np.float32) for key in dataset_keys}

    trajectories = []
    for traj_idx in range(num_trajectories):
        raw_num_frames = int(raw_traj_lengths[traj_idx])
        raw_num_frames = max(1, min(raw_num_frames, max_frames))
        effective_num_frames = int(traj_lengths[traj_idx])
        effective_num_frames = max(1, min(effective_num_frames, raw_num_frames))

        trajectory = {
            key: arrays[key][:raw_num_frames, traj_idx, ...]
            for key in dataset_keys
        }
        trajectory["source_file"] = source_files[traj_idx]
        trajectory["raw_num_frames"] = raw_num_frames
        trajectory["effective_num_frames"] = effective_num_frames
        trajectories.append(trajectory)

    return trajectories


def load_trajectories_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in paths:
        trajectories.extend(load_hdf5_trajectories(path))
    return trajectories


def ensure_cache_dir(cache_dir: Path, overwrite_cache: bool) -> None:
    expected_outputs = [
        cache_dir / "train.hdf5",
        cache_dir / "valid.hdf5",
        cache_dir / "passive_valid.hdf5",
        cache_dir / "manifest.yaml",
    ]
    if overwrite_cache and cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_file():
                child.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite_cache and all(path.exists() for path in expected_outputs):
        return


def build_dataset_cache(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    input_paths = list_hdf5_paths(args.input_glob)
    if not input_paths:
        raise FileNotFoundError(
            f"No HDF5 files matched input glob: {resolve_repo_path(args.input_glob)}"
        )

    if args.valid_glob is not None:
        train_paths = input_paths
        valid_paths = list_hdf5_paths(args.valid_glob)
        duplicated_valid = False
        if not valid_paths:
            raise FileNotFoundError(
                f"No HDF5 files matched valid glob: {resolve_repo_path(args.valid_glob)}"
            )
    else:
        train_paths, valid_paths, duplicated_valid = split_paths_by_ratio(
            input_paths,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )

    if args.eval_glob is not None:
        eval_paths = list_hdf5_paths(args.eval_glob)
        if not eval_paths:
            raise FileNotFoundError(
                f"No HDF5 files matched eval glob: {resolve_repo_path(args.eval_glob)}"
            )
    else:
        eval_paths = valid_paths

    train_trajectories = load_trajectories_from_paths(train_paths)
    valid_trajectories = load_trajectories_from_paths(valid_paths)
    eval_trajectories = load_trajectories_from_paths(eval_paths)

    cache_dir = resolve_repo_path(args.dataset_cache_dir)
    ensure_cache_dir(cache_dir, overwrite_cache=args.overwrite_cache)

    train_path = cache_dir / "train.hdf5"
    valid_path = cache_dir / "valid.hdf5"
    passive_valid_path = cache_dir / "passive_valid.hdf5"
    eval_path = cache_dir / "eval.hdf5"

    create_dataset(train_trajectories, train_path, args.compression, split_name="train")
    create_dataset(valid_trajectories, valid_path, args.compression, split_name="valid")
    create_dataset(
        valid_trajectories,
        passive_valid_path,
        args.compression,
        split_name="passive_valid",
    )
    create_dataset(eval_trajectories, eval_path, args.compression, split_name="eval")

    manifest = {
        "input_glob": args.input_glob,
        "valid_glob": args.valid_glob,
        "eval_glob": args.eval_glob,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "duplicated_valid_from_train": duplicated_valid,
        "train_files": [repo_relative_str(path) for path in train_paths],
        "valid_files": [repo_relative_str(path) for path in valid_paths],
        "eval_files": [repo_relative_str(path) for path in eval_paths],
        "outputs": {
            "train_dataset_path": repo_relative_str(train_path),
            "valid_dataset_path": repo_relative_str(valid_path),
            "passive_valid_dataset_path": repo_relative_str(passive_valid_path),
            "eval_dataset_path": repo_relative_str(eval_path),
        },
    }
    with (cache_dir / "manifest.yaml").open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(manifest, file_obj, sort_keys=False, allow_unicode=True)

    print(
        "Prepared merged datasets: "
        f"train={len(train_trajectories)}, valid={len(valid_trajectories)}, eval={len(eval_trajectories)}"
    )
    print(f"Dataset cache: {cache_dir}")
    return train_path, valid_path, passive_valid_path, eval_path


def build_cfg_overrides(
    args: argparse.Namespace,
    train_path: Path,
    valid_path: Path,
    passive_valid_path: Path,
    eval_path: Path,
) -> str:
    override_items = [
        ("algorithm.dataset.train_dataset_path", repo_relative_str(train_path)),
        ("algorithm.dataset.valid_datasets.exp_trajectory", repo_relative_str(valid_path)),
        (
            "algorithm.dataset.valid_datasets.passive_trajectory",
            repo_relative_str(passive_valid_path),
        ),
        ("algorithm.eval.dataset_path", repo_relative_str(eval_path)),
    ]

    if args.batch_size is not None:
        override_items.append(("algorithm.batch_size", str(args.batch_size)))
    if args.sample_sequence_length is not None:
        override_items.append(
            ("algorithm.sample_sequence_length", str(args.sample_sequence_length))
        )
        override_items.append(
            (
                "env.neural_solver_cfg.num_states_history",
                str(args.sample_sequence_length),
            )
        )
    if args.num_epochs is not None:
        override_items.append(("algorithm.num_epochs", str(args.num_epochs)))
    if args.num_iters_per_epoch is not None:
        override_items.append(
            ("algorithm.num_iters_per_epoch", str(args.num_iters_per_epoch))
        )
    if args.num_valid_batches is not None:
        override_items.append(("algorithm.num_valid_batches", str(args.num_valid_batches)))
    if args.eval_rollout_horizon is not None:
        override_items.append(
            ("algorithm.eval.rollout_horizon", str(args.eval_rollout_horizon))
        )
    if args.eval_num_rollouts is not None:
        override_items.append(("algorithm.eval.num_rollouts", str(args.eval_num_rollouts)))
    if args.num_envs is not None:
        override_items.append(("env.num_envs", str(args.num_envs)))

    parts = []
    for key, value in override_items:
        parts.extend([key, value])

    user_overrides = args.cfg_overrides.strip()
    if user_overrides:
        parts.extend(user_overrides.split())

    return " ".join(parts)


def build_train_command(args: argparse.Namespace, cfg_overrides: str) -> list[str]:
    cmd = [
        args.python_exe,
        str(REPO_ROOT / "train" / "train.py"),
        "--cfg",
        repo_relative_str(resolve_repo_path(args.cfg)),
        "--logdir",
        repo_relative_str(resolve_repo_path(args.logdir)),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--skip-check-log-override",
        "--cfg-overrides",
        cfg_overrides,
    ]

    if args.no_time_stamp:
        cmd.append("--no-time-stamp")
    if args.render:
        cmd.append("--render")
    if args.resume:
        cmd.append("--resume")
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])

    return cmd


def main() -> None:
    args = parse_args()
    train_path, valid_path, passive_valid_path, eval_path = build_dataset_cache(args)
    cfg_overrides = build_cfg_overrides(
        args,
        train_path=train_path,
        valid_path=valid_path,
        passive_valid_path=passive_valid_path,
        eval_path=eval_path,
    )
    train_cmd = build_train_command(args, cfg_overrides)

    print("Train command:")
    print(" ".join(train_cmd))

    if args.prepare_only:
        return

    subprocess.run(train_cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
