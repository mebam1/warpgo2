from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import state_convention, torch_utils
from utils.commons import (
    CONTACT_DEPTH_UPPER_RATIO,
    get_min_contact_event_threshold,
)


DEFAULT_CUBE_CONFIG = REPO_ROOT / "mebam" / "config" / "contact_nets_cube.yaml"
DEFAULT_NERD_CONFIG = REPO_ROOT / "mebam" / "config" / "nerd_cube_tossing.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ContactNets cube tossing .pt trajectories into NeRD-compatible "
            "HDF5 trajectory datasets."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="mebam/data/contact_nets/*.pt",
        help="Glob pattern for input .pt files, relative to repo root by default.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Single output HDF5 path. If omitted, the script writes train/valid/"
            "passive_valid splits using mebam/config/nerd_cube_tossing.yaml."
        ),
    )
    parser.add_argument(
        "--test-output-path",
        default=None,
        help="Optional test split output path when writing cfg-based splits.",
    )
    parser.add_argument(
        "--cube-config",
        default=str(DEFAULT_CUBE_CONFIG.relative_to(REPO_ROOT)),
        help="Cube/task config YAML path.",
    )
    parser.add_argument(
        "--nerd-config",
        default=str(DEFAULT_NERD_CONFIG.relative_to(REPO_ROOT)),
        help="NeRD model config YAML path.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="HDF5 compression to use for datasets. Use 'none' to disable.",
    )
    parser.add_argument(
        "--no-derived-fields",
        action="store_true",
        help=(
            "Do not store convenience fields such as next_states, states_body, "
            "states_embedding, contact_points_1_body, and gravity_dir_body."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def numeric_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def list_input_paths(pattern: str) -> list[Path]:
    abs_pattern = str(resolve_repo_path(pattern))
    paths = [Path(p) for p in glob.glob(abs_pattern)]
    paths = [p for p in paths if p.is_file() and p.suffix == ".pt"]
    return sorted(paths, key=numeric_sort_key)


def canonicalize_quaternion_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = torch_utils.normalize(quat_xyzw)
    flip_mask = (quat_xyzw[:, 3] < 0.0).unsqueeze(-1).expand_as(quat_xyzw)
    quat_xyzw = quat_xyzw.clone()
    quat_xyzw[flip_mask] *= -1.0
    return quat_xyzw


def build_box_contact_points(half_extent: float) -> np.ndarray:
    points = []
    for point_id in range(8):
        sign_x = float(point_id % 2) * 2.0 - 1.0
        sign_y = float((point_id // 2) % 2) * 2.0 - 1.0
        sign_z = float((point_id // 4) % 2) * 2.0 - 1.0
        points.append(
            [sign_x * half_extent, sign_y * half_extent, sign_z * half_extent]
        )
    points.extend(
        [
            [half_extent, 0.0, 0.0],
            [-half_extent, 0.0, 0.0],
            [0.0, half_extent, 0.0],
            [0.0, -half_extent, 0.0],
            [0.0, 0.0, half_extent],
            [0.0, 0.0, -half_extent],
        ]
    )
    return np.asarray(points, dtype=np.float32)


def load_contact_nets_tensor(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path} must contain a torch.Tensor, got {type(obj)!r}")

    if obj.ndim == 3 and obj.shape[0] == 1:
        obj = obj.squeeze(0)
    if obj.ndim != 2 or obj.shape[1] != 19:
        raise ValueError(
            f"{path} must have shape [1, T, 19] or [T, 19], got {tuple(obj.shape)}"
        )

    return obj.detach().cpu().numpy()


def build_contact_fields(
    pos: torch.Tensor,
    quat_xyzw: torch.Tensor,
    cube_half_extent: float,
) -> dict[str, np.ndarray]:
    traj_len = pos.shape[0]
    local_contact_points = torch.from_numpy(build_box_contact_points(cube_half_extent))
    contact_count = int(local_contact_points.shape[0])
    local_contact_points = local_contact_points.unsqueeze(0).expand(traj_len, -1, -1)

    quat_expanded = quat_xyzw.unsqueeze(1).expand(traj_len, contact_count, 4).reshape(-1, 4)
    corners_world = torch_utils.quat_rotate(
        quat_expanded, local_contact_points.reshape(-1, 3)
    ).reshape(traj_len, contact_count, 3) + pos.unsqueeze(1)

    contact_points_0 = local_contact_points.clone()
    contact_points_1 = corners_world.clone()
    contact_points_1[..., 2] = 0.0

    contact_normals = torch.zeros_like(corners_world)
    contact_normals[..., 2] = 1.0

    contact_depths = corners_world[..., 2]
    contact_thicknesses_0 = torch.zeros_like(contact_depths)
    contact_thicknesses_1 = torch.zeros_like(contact_depths)
    min_contact_event_threshold = get_min_contact_event_threshold(cube_half_extent)
    contact_event_threshold = CONTACT_DEPTH_UPPER_RATIO * (
        contact_thicknesses_0 + contact_thicknesses_1
    )
    contact_event_threshold = torch.where(
        contact_event_threshold < min_contact_event_threshold,
        torch.full_like(contact_event_threshold, min_contact_event_threshold),
        contact_event_threshold,
    )
    contact_masks = (contact_depths < contact_event_threshold).to(torch.float32)

    return {
        "contact_points_0": contact_points_0.numpy().astype(np.float32),
        "contact_points_1": contact_points_1.numpy().astype(np.float32),
        "contact_normals": contact_normals.numpy().astype(np.float32),
        "contact_depths": contact_depths.numpy().astype(np.float32),
        "contact_thicknesses_0": contact_thicknesses_0.numpy().astype(np.float32),
        "contact_thicknesses_1": contact_thicknesses_1.numpy().astype(np.float32),
        "contact_masks": contact_masks.numpy().astype(np.float32),
    }


def build_derived_fields(
    states: torch.Tensor,
    next_states: torch.Tensor,
    root_body_q: torch.Tensor,
    gravity_dir: torch.Tensor,
    contact_points_1: torch.Tensor,
    contact_normals: torch.Tensor,
) -> dict[str, np.ndarray]:
    pos = states[:, 0:3]
    quat = states[:, 3:7]
    lin_vel = states[:, 7:10]
    ang_vel = states[:, 10:13]

    next_pos = next_states[:, 0:3]
    next_quat = next_states[:, 3:7]
    next_lin_vel = next_states[:, 7:10]
    next_ang_vel = next_states[:, 10:13]

    p_body, quat_body, lin_vel_body, ang_vel_body = (
        torch_utils.convert_free_states_com_w2b(
            pos, quat, pos, quat, lin_vel, ang_vel
        )
    )
    next_p_body, next_quat_body, next_lin_vel_body, next_ang_vel_body = (
        torch_utils.convert_free_states_com_w2b(
            pos,
            quat,
            next_pos,
            next_quat,
            next_lin_vel,
            next_ang_vel,
        )
    )

    quat_body = canonicalize_quaternion_xyzw(quat_body)
    next_quat_body = canonicalize_quaternion_xyzw(next_quat_body)

    states_body = torch.cat([p_body, quat_body, lin_vel_body, ang_vel_body], dim=-1)
    next_states_body = torch.cat(
        [next_p_body, next_quat_body, next_lin_vel_body, next_ang_vel_body], dim=-1
    )

    traj_len = states.shape[0]
    contact_count = contact_points_1.shape[1]

    frame_pos = pos.unsqueeze(1).expand(traj_len, contact_count, 3).reshape(-1, 3)
    frame_quat = quat.unsqueeze(1).expand(traj_len, contact_count, 4).reshape(-1, 4)

    contact_points_1_body = torch_utils.transform_point_inverse(
        frame_pos, frame_quat, contact_points_1.reshape(-1, 3)
    ).reshape(traj_len, contact_count, 3)
    contact_normals_body = torch_utils.quat_rotate_inverse(
        frame_quat, contact_normals.reshape(-1, 3)
    ).reshape(traj_len, contact_count, 3)
    gravity_dir_body = torch_utils.quat_rotate_inverse(quat, gravity_dir)

    return {
        "states_body": states_body.numpy().astype(np.float32),
        "next_states_body": next_states_body.numpy().astype(np.float32),
        "states_embedding": states_body.numpy().astype(np.float32),
        "contact_points_1_body": contact_points_1_body.numpy().astype(np.float32),
        "contact_normals_body": contact_normals_body.numpy().astype(np.float32),
        "gravity_dir_body": gravity_dir_body.numpy().astype(np.float32),
        "root_body_q_body_anchor": root_body_q.numpy().astype(np.float32),
    }


def convert_trajectory(
    path: Path,
    cube_cfg: dict,
    include_derived_fields: bool,
) -> dict:
    raw = load_contact_nets_tensor(path)

    unit_cfg = cube_cfg["data_format"]["units"]
    scale = float(unit_cfg["recover_metric_position_velocity_multiplier"])
    cube_half_extent = float(unit_cfg["block_half_width_m"])

    pos = torch.tensor(raw[:, 0:3] * scale, dtype=torch.float32)
    quat_wxyz = torch.tensor(raw[:, 3:7], dtype=torch.float32)
    quat_xyzw = torch.cat([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], dim=-1)
    quat_xyzw = canonicalize_quaternion_xyzw(quat_xyzw)

    lin_vel = torch.tensor(raw[:, 7:10] * scale, dtype=torch.float32)
    ang_vel_body = torch.tensor(raw[:, 10:13], dtype=torch.float32)
    ang_vel_world = torch_utils.quat_rotate(quat_xyzw, ang_vel_body)

    states = torch.cat([pos, quat_xyzw, lin_vel, ang_vel_world], dim=-1)
    root_body_q = torch.cat([pos, quat_xyzw], dim=-1)

    next_states = torch.cat([states[1:], states[-1:]], dim=0)
    gravity_dir = torch.zeros((states.shape[0], 3), dtype=torch.float32)
    gravity_dir[:, 2] = -1.0

    contact_fields = build_contact_fields(pos, quat_xyzw, cube_half_extent)

    trajectory = {
        "states": states.numpy().astype(np.float32),
        "root_body_q": root_body_q.numpy().astype(np.float32),
        "next_states": next_states.numpy().astype(np.float32),
        "gravity_dir": gravity_dir.numpy().astype(np.float32),
        **contact_fields,
        "source_file": str(path),
        "raw_num_frames": int(states.shape[0]),
        "effective_num_frames": int(max(states.shape[0] - 1, 1)),
    }

    if include_derived_fields:
        derived_fields = build_derived_fields(
            states=states,
            next_states=next_states,
            root_body_q=root_body_q,
            gravity_dir=gravity_dir,
            contact_points_1=torch.from_numpy(contact_fields["contact_points_1"]),
            contact_normals=torch.from_numpy(contact_fields["contact_normals"]),
        )
        trajectory.update(derived_fields)

    return trajectory


def pad_last_frame(array: np.ndarray, target_len: int) -> np.ndarray:
    if array.shape[0] == target_len:
        return array
    if array.shape[0] > target_len:
        return array[:target_len]
    pad_len = target_len - array.shape[0]
    pad = np.repeat(array[-1:], pad_len, axis=0)
    return np.concatenate([array, pad], axis=0)


def stack_trajectories(trajectories: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not trajectories:
        raise ValueError("No trajectories to stack.")

    data_keys = [
        key
        for key in trajectories[0].keys()
        if key not in {"source_file", "raw_num_frames", "effective_num_frames"}
    ]
    max_len = max(traj["raw_num_frames"] for traj in trajectories)

    stacked = {}
    for key in data_keys:
        stacked[key] = np.stack(
            [pad_last_frame(traj[key], max_len) for traj in trajectories],
            axis=1,
        )

    traj_lengths = np.asarray(
        [traj["effective_num_frames"] for traj in trajectories], dtype=np.int32
    )
    return stacked, traj_lengths


def create_dataset(
    trajectories: list[dict],
    output_path: Path,
    compression: str,
    split_name: str,
    dataset_attrs: dict | None = None,
) -> None:
    if not trajectories:
        raise ValueError(f"Cannot write empty trajectory set to {output_path}.")

    stacked, traj_lengths = stack_trajectories(trajectories)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h5_compression = None if compression.lower() == "none" else compression
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        attrs = {
            "mode": "trajectory",
            "split_name": split_name,
            "total_transitions": int(traj_lengths.sum()),
            "num_trajectories": len(trajectories),
            "max_trajectory_length": int(stacked["states"].shape[0]),
            "effective_length_rule": (
                "traj_lengths = max(num_frames - 1, 1); next_states tail repeats last frame"
            ),
            "state_convention": state_convention.NEWTON_FREE_JOINT_STATE_CONVENTION,
            "qd_layout": "lin_world_then_ang_world",
            "linear_velocity_frame": "world",
            "angular_velocity_frame": "world",
            "source_files": np.asarray(
                [traj["source_file"] for traj in trajectories], dtype=string_dtype
            ),
            "raw_traj_lengths": np.asarray(
                [traj["raw_num_frames"] for traj in trajectories], dtype=np.int32
            ),
        }
        if dataset_attrs is not None:
            attrs.update(dataset_attrs)
        for key, value in attrs.items():
            data_group.attrs[key] = value

        for key, value in stacked.items():
            data_group.create_dataset(
                key,
                data=value,
                compression=h5_compression,
            )
        data_group.create_dataset(
            "traj_lengths",
            data=traj_lengths,
            compression=h5_compression,
        )


def split_paths_by_ratio(
    paths: list[Path],
    train_pct: float,
    valid_pct: float,
) -> tuple[list[Path], list[Path], list[Path], bool]:
    total = len(paths)
    if total == 0:
        return [], [], [], False

    train_end = int(round(total * train_pct / 100.0))
    train_end = min(total, max(1, train_end))
    valid_end = int(round(total * (train_pct + valid_pct) / 100.0))
    valid_end = min(total, max(train_end, valid_end))

    train_paths = paths[:train_end]
    valid_paths = paths[train_end:valid_end]
    test_paths = paths[valid_end:]

    duplicated_valid = False
    if not valid_paths:
        valid_paths = list(train_paths)
        duplicated_valid = True

    return train_paths, valid_paths, test_paths, duplicated_valid


def convert_paths(
    paths: list[Path],
    cube_cfg: dict,
    include_derived_fields: bool,
) -> list[dict]:
    return [
        convert_trajectory(
            path=path,
            cube_cfg=cube_cfg,
            include_derived_fields=include_derived_fields,
        )
        for path in paths
    ]


def main() -> None:
    args = parse_args()

    cube_cfg = load_yaml(resolve_repo_path(args.cube_config))
    nerd_cfg = load_yaml(resolve_repo_path(args.nerd_config))
    include_derived_fields = not args.no_derived_fields

    input_paths = list_input_paths(args.input_glob)
    if not input_paths:
        raise FileNotFoundError(
            f"No .pt files matched pattern: {resolve_repo_path(args.input_glob)}"
        )

    if args.output_path:
        output_path = resolve_repo_path(args.output_path)
        trajectories = convert_paths(input_paths, cube_cfg, include_derived_fields)
        create_dataset(
            trajectories=trajectories,
            output_path=output_path,
            compression=args.compression,
            split_name="all",
        )
        print(
            f"Wrote {len(trajectories)} trajectories to {output_path} "
            f"(max_len={max(t['raw_num_frames'] for t in trajectories)})."
        )
        return

    split_cfg = cube_cfg.get("training_defaults", {}).get("split_percent", {})
    train_pct = float(split_cfg.get("train", 50.0))
    valid_pct = float(split_cfg.get("valid", 30.0))

    train_paths, valid_paths, test_paths, duplicated_valid = split_paths_by_ratio(
        input_paths, train_pct=train_pct, valid_pct=valid_pct
    )

    train_out = resolve_repo_path(
        nerd_cfg["algorithm"]["dataset"]["train_dataset_path"]
    )
    valid_out = resolve_repo_path(
        nerd_cfg["algorithm"]["dataset"]["valid_datasets"]["exp_trajectory"]
    )
    passive_valid_out = resolve_repo_path(
        nerd_cfg["algorithm"]["dataset"]["valid_datasets"]["passive_trajectory"]
    )

    train_trajectories = convert_paths(train_paths, cube_cfg, include_derived_fields)
    valid_trajectories = convert_paths(valid_paths, cube_cfg, include_derived_fields)

    create_dataset(
        trajectories=train_trajectories,
        output_path=train_out,
        compression=args.compression,
        split_name="train",
    )
    create_dataset(
        trajectories=valid_trajectories,
        output_path=valid_out,
        compression=args.compression,
        split_name="valid",
    )
    create_dataset(
        trajectories=valid_trajectories,
        output_path=passive_valid_out,
        compression=args.compression,
        split_name="passive_valid",
    )

    if args.test_output_path and test_paths:
        test_trajectories = convert_paths(test_paths, cube_cfg, include_derived_fields)
        create_dataset(
            trajectories=test_trajectories,
            output_path=resolve_repo_path(args.test_output_path),
            compression=args.compression,
            split_name="test",
        )

    print(
        f"Wrote train={len(train_paths)}, valid={len(valid_paths)}, test={len(test_paths)} "
        f"from {len(input_paths)} input trajectories."
    )
    if duplicated_valid:
        print("Validation split was empty after ratio split; duplicated train split for valid/passive_valid.")
    if test_paths and not args.test_output_path:
        print("Test trajectories were not written because --test-output-path was not provided.")
    print(f"Train output: {train_out}")
    print(f"Valid output: {valid_out}")
    print(f"Passive valid output: {passive_valid_out}")


if __name__ == "__main__":
    main()
