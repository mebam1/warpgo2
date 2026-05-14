from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import SolverType
from utils import state_convention, torch_utils, warp_utils
from utils.env_utils import create_fixed_contact_env
from utils.python_utils import set_random_seed


DEFAULT_REAL_HDF5 = "mebam/data/nerd/real/0.hdf5"
DEFAULT_CUBE_CONFIG = "mebam/config/contact_nets_cube.yaml"
DEFAULT_SWEEP_CONFIG = "mebam/sweep/default_cube_tossing_physics_sweep.yaml"
DEFAULT_OUTPUT_DIR = "runs/CubeTossingSim2RealSweep"

SOLVER_TYPES = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


@dataclass
class Candidate:
    candidate_id: int
    overrides: dict[str, float]


@dataclass
class MetricScales:
    position_m: float
    orientation_rad: float
    linear_velocity_mps: float
    angular_velocity_rps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep CubeTossing physical parameters and rank candidates by the "
            "sim2real gap against mebam real trajectories."
        )
    )
    parser.add_argument(
        "--real-hdf5",
        type=str,
        default=DEFAULT_REAL_HDF5,
        help="Real-data trajectory HDF5 used as the reference rollout set.",
    )
    parser.add_argument(
        "--cube-config",
        type=str,
        default=DEFAULT_CUBE_CONFIG,
        help="Base CubeTossing YAML used before applying sweep overrides.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=DEFAULT_SWEEP_CONFIG,
        help="YAML file describing sweep parameters, evaluation, and metric options.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write JSON/CSV summaries and the best config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Newton/Warp device string.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="mujoco",
        choices=list(SOLVER_TYPES.keys()),
        help="Ground-truth Newton solver used for the sim rollout.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint"],
        help="Observation mode for CubeTossingEnv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic trajectory subset selection.",
    )
    parser.add_argument(
        "--trajectory-indices",
        type=str,
        default=None,
        help="Comma-separated explicit real-trajectory indices to score.",
    )
    parser.add_argument(
        "--trajectory-selection",
        type=str,
        default=None,
        choices=["first", "strided", "random"],
        help="Subset policy when --trajectory-indices is omitted.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of real trajectories to evaluate.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Common transition horizon. Clipped to the shortest selected real trajectory.",
    )
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Optional cap on how many candidates to evaluate after candidate generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected trajectories and candidate count, then exit.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)


def parse_indices(indices_arg: str | None) -> list[int] | None:
    if indices_arg is None:
        return None
    items = [item.strip() for item in indices_arg.split(",") if item.strip()]
    if not items:
        return []
    return [int(item) for item in items]


def select_trajectory_indices(
    total_trajectories: int,
    explicit_indices: list[int] | None,
    max_trajectories: int | None,
    selection_mode: str,
    seed: int,
) -> list[int]:
    if explicit_indices is not None:
        indices = explicit_indices
    else:
        if max_trajectories is None or max_trajectories >= total_trajectories:
            return list(range(total_trajectories))
        if selection_mode == "first":
            indices = list(range(max_trajectories))
        elif selection_mode == "random":
            rng = np.random.default_rng(seed)
            indices = sorted(
                int(idx)
                for idx in rng.choice(
                    total_trajectories,
                    size=max_trajectories,
                    replace=False,
                )
            )
        else:
            linspace = np.linspace(
                0,
                total_trajectories - 1,
                num=max_trajectories,
                endpoint=True,
            )
            indices = sorted({int(round(idx)) for idx in linspace})
            while len(indices) < max_trajectories:
                for idx in range(total_trajectories):
                    if idx not in indices:
                        indices.append(idx)
                    if len(indices) == max_trajectories:
                        break
            indices = sorted(indices[:max_trajectories])

    if not indices:
        raise ValueError("No trajectory indices were selected.")
    for idx in indices:
        if idx < 0 or idx >= total_trajectories:
            raise IndexError(
                f"Selected trajectory index {idx} is outside [0, {total_trajectories - 1}]."
            )
    return indices


def canonicalize_quaternion_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = torch_utils.normalize(quat_xyzw)
    flip_mask = quat_xyzw[..., 3:4] < 0.0
    return torch.where(flip_mask, -quat_xyzw, quat_xyzw)


def load_reference_rollouts(
    real_hdf5_path: Path,
    selected_indices: list[int],
    requested_horizon: int | None,
) -> tuple[torch.Tensor, int, str]:
    with h5py.File(real_hdf5_path, "r") as h5_file:
        data_group = h5_file["data"]
        states_key = "states_world" if "states_world" in data_group else "states"
        states = torch.from_numpy(
            data_group[states_key][:, selected_indices, :].astype(np.float32)
        )
        if "traj_lengths" in data_group:
            traj_lengths = [int(data_group["traj_lengths"][idx]) for idx in selected_indices]
        else:
            traj_lengths = [int(states.shape[0] - 1)] * len(selected_indices)
        convention = (
            state_convention.NEWTON_FREE_JOINT_STATE_CONVENTION
            if states_key == "states_world"
            else state_convention.infer_free_joint_state_convention_from_attrs(
                data_group.attrs
            )
        )

    normalized_states, _ = state_convention.normalize_free_joint_states(states, convention)
    normalized_states = normalized_states.clone()
    normalized_states[..., 3:7] = canonicalize_quaternion_xyzw(
        normalized_states[..., 3:7]
    )

    common_horizon = min(traj_lengths)
    if requested_horizon is not None:
        if requested_horizon <= 0:
            raise ValueError("--horizon must be positive when provided.")
        common_horizon = min(common_horizon, requested_horizon)
    if common_horizon <= 0:
        raise ValueError("The selected trajectories do not contain any transitions.")

    return normalized_states[: common_horizon + 1], common_horizon, convention


def compute_metric_scales(
    reference_states: torch.Tensor,
    cube_cfg: dict[str, Any],
    metric_cfg: dict[str, Any],
) -> MetricScales:
    scales_cfg = metric_cfg.get("scales", {})

    def _auto_linear_scale(values: torch.Tensor, minimum: float) -> float:
        rms = float(torch.sqrt(torch.mean(values * values)).item())
        return max(rms, minimum)

    half_extent = float(cube_cfg["data_format"]["units"]["block_half_width_m"])
    pos_scale_raw = scales_cfg.get("position_m", "auto_half_extent")
    if pos_scale_raw == "auto_half_extent":
        position_scale = half_extent
    else:
        position_scale = float(pos_scale_raw)

    orient_scale_raw = scales_cfg.get("orientation_rad", 1.0)
    orientation_scale = float(orient_scale_raw)

    lin_scale_raw = scales_cfg.get("linear_velocity_mps", "auto_rms")
    if lin_scale_raw == "auto_rms":
        linear_velocity_scale = _auto_linear_scale(reference_states[..., 7:10], 0.1)
    else:
        linear_velocity_scale = float(lin_scale_raw)

    ang_scale_raw = scales_cfg.get("angular_velocity_rps", "auto_rms")
    if ang_scale_raw == "auto_rms":
        angular_velocity_scale = _auto_linear_scale(reference_states[..., 10:13], 0.1)
    else:
        angular_velocity_scale = float(ang_scale_raw)

    return MetricScales(
        position_m=position_scale,
        orientation_rad=orientation_scale,
        linear_velocity_mps=linear_velocity_scale,
        angular_velocity_rps=angular_velocity_scale,
    )


def quaternion_angle_error(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    quat_a = torch_utils.normalize(quat_a)
    quat_b = torch_utils.normalize(quat_b)
    dot = torch.sum(quat_a * quat_b, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def score_rollout_batch(
    simulated_states: torch.Tensor,
    reference_states: torch.Tensor,
    scales: MetricScales,
    metric_cfg: dict[str, Any],
) -> dict[str, float]:
    position_error = torch.linalg.vector_norm(
        simulated_states[..., 0:3] - reference_states[..., 0:3], dim=-1
    )
    orientation_error = quaternion_angle_error(
        simulated_states[..., 3:7],
        reference_states[..., 3:7],
    )
    linear_velocity_error = torch.linalg.vector_norm(
        simulated_states[..., 7:10] - reference_states[..., 7:10], dim=-1
    )
    angular_velocity_error = torch.linalg.vector_norm(
        simulated_states[..., 10:13] - reference_states[..., 10:13], dim=-1
    )

    position_rmse = float(torch.sqrt(torch.mean(position_error.square())).item())
    orientation_rmse = float(torch.sqrt(torch.mean(orientation_error.square())).item())
    linear_velocity_rmse = float(
        torch.sqrt(torch.mean(linear_velocity_error.square())).item()
    )
    angular_velocity_rmse = float(
        torch.sqrt(torch.mean(angular_velocity_error.square())).item()
    )

    weights = metric_cfg.get("component_weights", {})
    weighted_terms = [
        float(weights.get("position", 1.0)) * (position_rmse / scales.position_m) ** 2,
        float(weights.get("orientation", 1.0))
        * (orientation_rmse / scales.orientation_rad) ** 2,
        float(weights.get("linear_velocity", 1.0))
        * (linear_velocity_rmse / scales.linear_velocity_mps) ** 2,
        float(weights.get("angular_velocity", 1.0))
        * (angular_velocity_rmse / scales.angular_velocity_rps) ** 2,
    ]
    aggregate_score = float(math.sqrt(sum(weighted_terms)))

    terminal_position_rmse = float(
        torch.sqrt(torch.mean(position_error[-1].square())).item()
    )
    terminal_orientation_rmse = float(
        torch.sqrt(torch.mean(orientation_error[-1].square())).item()
    )

    return {
        "aggregate_score": aggregate_score,
        "position_rmse_m": position_rmse,
        "orientation_rmse_rad": orientation_rmse,
        "linear_velocity_rmse_mps": linear_velocity_rmse,
        "angular_velocity_rmse_rps": angular_velocity_rmse,
        "terminal_position_rmse_m": terminal_position_rmse,
        "terminal_orientation_rmse_rad": terminal_orientation_rmse,
    }


def set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = payload
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Cannot assign override {dotted_key!r}: missing dict key {key!r}.")
        current = current[key]
    leaf_key = keys[-1]
    if leaf_key not in current:
        raise KeyError(f"Cannot assign override {dotted_key!r}: missing leaf key {leaf_key!r}.")
    current[leaf_key] = value


def build_candidate_cube_config(
    base_cube_cfg: dict[str, Any],
    overrides: dict[str, float],
) -> dict[str, Any]:
    candidate_cfg = copy.deepcopy(base_cube_cfg)
    for dotted_key, value in overrides.items():
        set_nested_value(candidate_cfg, dotted_key, value)
    return candidate_cfg


def capture_env_states(env) -> torch.Tensor:
    torch_device = warp_utils.device_to_torch(env.device)
    state_dim = env.dof_q_per_env + env.dof_qd_per_env
    states = torch.zeros(
        (env.num_envs, state_dim),
        dtype=torch.float32,
        device=torch_device,
    )
    warp_utils.acquire_states_to_torch(env, states)
    states[:, 3:7] = canonicalize_quaternion_xyzw(states[:, 3:7])
    return states


def rollout_newton_batch_from_initial_states(
    initial_states: torch.Tensor,
    horizon: int,
    config_path: Path,
    device: str,
    solver_type: str,
    obs_type: str,
    seed: int,
) -> torch.Tensor:
    env = create_fixed_contact_env(
        env_name="CubeTossing",
        num_envs=int(initial_states.shape[0]),
        device=device,
        use_graph_capture=False,
        render=False,
        seed=seed,
        random_reset=False,
        solver_type=SOLVER_TYPES[solver_type],
        obs_type=obs_type,
        camera_tracking=False,
        config_path=str(config_path),
    )

    torch_device = warp_utils.device_to_torch(env.device)
    rollout_states = torch.zeros(
        (horizon + 1, env.num_envs, env.dof_q_per_env + env.dof_qd_per_env),
        dtype=torch.float32,
        device=torch_device,
    )

    try:
        env.reset()
        initial_states = initial_states.to(torch_device)
        warp_utils.assign_states_from_torch(env, initial_states)
        warp_utils.eval_fk(env.model, env.state)

        rollout_states[0].copy_(capture_env_states(env))
        for step in range(horizon):
            env.update()
            rollout_states[step + 1].copy_(capture_env_states(env))
    finally:
        if hasattr(env, "close"):
            env.close()

    return rollout_states.cpu()


def build_candidates(
    sweep_cfg: dict[str, Any],
    limit_candidates: int | None,
) -> list[Candidate]:
    sweep_section = sweep_cfg.get("sweep", {})
    parameters = sweep_section.get("parameters", {})
    if not parameters:
        raise ValueError("Sweep config must define sweep.parameters.")

    sweep_mode = sweep_section.get("mode", "grid")
    ordered_items = list(parameters.items())
    candidate_dicts: list[dict[str, float]] = []

    if sweep_mode == "grid":
        values_product = itertools.product(*(values for _, values in ordered_items))
        for combo in values_product:
            candidate_dicts.append(
                {
                    dotted_key: float(value)
                    for (dotted_key, _), value in zip(ordered_items, combo)
                }
            )
    elif sweep_mode == "zip":
        lengths = {len(values) for _, values in ordered_items}
        if len(lengths) != 1:
            raise ValueError("sweep.mode=zip requires every parameter list to have the same length.")
        for combo in zip(*(values for _, values in ordered_items)):
            candidate_dicts.append(
                {
                    dotted_key: float(value)
                    for (dotted_key, _), value in zip(ordered_items, combo)
                }
            )
    else:
        raise ValueError(f"Unsupported sweep.mode: {sweep_mode!r}")

    if limit_candidates is not None:
        candidate_dicts = candidate_dicts[:limit_candidates]

    return [
        Candidate(candidate_id=candidate_id, overrides=overrides)
        for candidate_id, overrides in enumerate(candidate_dicts)
    ]


def coerce_result_record(record: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.floating, np.integer)):
            coerced[key] = value.item()
        elif isinstance(value, dict):
            coerced[key] = coerce_result_record(value)
        else:
            coerced[key] = value
    return coerced


def write_leaderboard_csv(
    csv_path: Path,
    sorted_results: list[dict[str, Any]],
    parameter_keys: list[str],
) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "aggregate_score",
        "position_rmse_m",
        "orientation_rmse_rad",
        "linear_velocity_rmse_mps",
        "angular_velocity_rmse_rps",
        "terminal_position_rmse_m",
        "terminal_orientation_rmse_rad",
        "runtime_sec",
    ] + parameter_keys

    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(sorted_results, start=1):
            row = {key: result.get(key) for key in fieldnames}
            row["rank"] = rank
            for parameter_key in parameter_keys:
                row[parameter_key] = result["overrides"].get(parameter_key)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube_config_path = resolve_repo_path(args.cube_config)
    sweep_config_path = resolve_repo_path(args.sweep_config)
    real_hdf5_path = resolve_repo_path(args.real_hdf5)

    base_cube_cfg = load_yaml(cube_config_path)
    sweep_cfg = load_yaml(sweep_config_path)

    with h5py.File(real_hdf5_path, "r") as h5_file:
        total_trajectories = int(h5_file["data"]["states"].shape[1])

    evaluation_cfg = sweep_cfg.get("evaluation", {})
    metric_cfg = sweep_cfg.get("metric", {})

    explicit_indices = parse_indices(args.trajectory_indices)
    selection_mode = (
        args.trajectory_selection
        or evaluation_cfg.get("trajectory_selection", "strided")
    )
    max_trajectories = (
        args.max_trajectories
        if args.max_trajectories is not None
        else evaluation_cfg.get("max_trajectories", 16)
    )
    requested_horizon = (
        args.horizon if args.horizon is not None else evaluation_cfg.get("horizon", 96)
    )

    selected_indices = select_trajectory_indices(
        total_trajectories=total_trajectories,
        explicit_indices=explicit_indices,
        max_trajectories=max_trajectories,
        selection_mode=selection_mode,
        seed=args.seed,
    )
    reference_states, common_horizon, reference_convention = load_reference_rollouts(
        real_hdf5_path=real_hdf5_path,
        selected_indices=selected_indices,
        requested_horizon=requested_horizon,
    )
    metric_scales = compute_metric_scales(
        reference_states=reference_states,
        cube_cfg=base_cube_cfg,
        metric_cfg=metric_cfg,
    )

    candidates = build_candidates(sweep_cfg, args.limit_candidates)
    parameter_keys = list(sweep_cfg["sweep"]["parameters"].keys())

    run_manifest = {
        "real_hdf5": str(real_hdf5_path),
        "cube_config": str(cube_config_path),
        "sweep_config": str(sweep_config_path),
        "solver_type": args.solver_type,
        "device": args.device,
        "seed": int(args.seed),
        "selected_trajectory_indices": selected_indices,
        "num_selected_trajectories": len(selected_indices),
        "common_horizon": int(common_horizon),
        "reference_state_convention": reference_convention,
        "metric_scales": {
            "position_m": metric_scales.position_m,
            "orientation_rad": metric_scales.orientation_rad,
            "linear_velocity_mps": metric_scales.linear_velocity_mps,
            "angular_velocity_rps": metric_scales.angular_velocity_rps,
        },
        "num_candidates": len(candidates),
    }
    save_yaml(output_dir / "run_manifest.yaml", run_manifest)

    print(
        f"[sim2real-sweep] trajectories={len(selected_indices)} "
        f"horizon={common_horizon} candidates={len(candidates)}"
    )
    print(f"[sim2real-sweep] trajectory_indices={selected_indices}")
    print(f"[sim2real-sweep] metric_scales={run_manifest['metric_scales']}")

    if args.dry_run:
        return

    reference_initial_states = reference_states[0]
    results_jsonl_path = output_dir / "results.jsonl"
    all_results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_result: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="cube_tossing_sweep_", dir=str(output_dir)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with results_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_cfg = build_candidate_cube_config(base_cube_cfg, candidate.overrides)
                candidate_cfg_path = temp_dir / f"candidate_{candidate.candidate_id:05d}.yaml"
                save_yaml(candidate_cfg_path, candidate_cfg)

                start_time = time.perf_counter()
                try:
                    simulated_states = rollout_newton_batch_from_initial_states(
                        initial_states=reference_initial_states,
                        horizon=common_horizon,
                        config_path=candidate_cfg_path,
                        device=args.device,
                        solver_type=args.solver_type,
                        obs_type=args.obs_type,
                        seed=args.seed,
                    )
                    metrics = score_rollout_batch(
                        simulated_states=simulated_states,
                        reference_states=reference_states,
                        scales=metric_scales,
                        metric_cfg=metric_cfg,
                    )
                    result = {
                        "candidate_id": candidate.candidate_id,
                        "runtime_sec": float(time.perf_counter() - start_time),
                        "overrides": dict(candidate.overrides),
                        **metrics,
                    }
                except Exception as exc:
                    result = {
                        "candidate_id": candidate.candidate_id,
                        "runtime_sec": float(time.perf_counter() - start_time),
                        "overrides": dict(candidate.overrides),
                        "error": f"{type(exc).__name__}: {exc}",
                        "aggregate_score": float("inf"),
                    }

                all_results.append(result)
                jsonl_file.write(json.dumps(coerce_result_record(result)) + "\n")
                jsonl_file.flush()

                if result.get("aggregate_score", float("inf")) < best_score:
                    best_score = float(result["aggregate_score"])
                    best_result = result

                print(
                    f"[sim2real-sweep] {candidate_index}/{len(candidates)} "
                    f"id={candidate.candidate_id} score={result.get('aggregate_score')} "
                    f"best={best_score}"
                )

    sorted_results = sorted(
        all_results,
        key=lambda item: (
            float(item.get("aggregate_score", float("inf"))),
            int(item.get("candidate_id", 0)),
        ),
    )
    write_leaderboard_csv(output_dir / "leaderboard.csv", sorted_results, parameter_keys)

    summary = {
        "run_manifest": run_manifest,
        "best_result": coerce_result_record(best_result) if best_result is not None else None,
        "top_10": [coerce_result_record(item) for item in sorted_results[:10]],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    if best_result is not None and math.isfinite(float(best_result["aggregate_score"])):
        best_cfg = build_candidate_cube_config(base_cube_cfg, best_result["overrides"])
        save_yaml(output_dir / "best_contact_nets_cube.yaml", best_cfg)

    print(f"[sim2real-sweep] wrote {output_dir / 'leaderboard.csv'}")
    print(f"[sim2real-sweep] wrote {output_dir / 'summary.json'}")
    if best_result is not None:
        print(
            f"[sim2real-sweep] best candidate id={best_result['candidate_id']} "
            f"score={best_result['aggregate_score']}"
        )


if __name__ == "__main__":
    main()
