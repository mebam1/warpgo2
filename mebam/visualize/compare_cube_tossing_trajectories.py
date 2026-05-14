from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import h5py
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.neural_environment import NeuralEnvironment
from envs.newton_envs import SolverType
from utils import state_convention, torch_utils, warp_utils
from utils.checkpoint_utils import load_neural_model_checkpoint
from utils.env_utils import create_fixed_contact_env
from utils.python_utils import set_random_seed


SOLVER_TYPES = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}

DEFAULT_REAL_HDF5 = "mebam/data/nerd/real/0.hdf5"
DEFAULT_SIM_HDF5 = "mebam/data/nerd/simulation/0.hdf5"
DEFAULT_OUTPUT_DIR = "figures/trajectory_gaps"

TRAJ_A_COLOR = "#2563eb"
TRAJ_B_COLOR = "#d97706"
GRID_COLOR = "#d1d5db"
AXIS_COLOR = "#6b7280"
TEXT_COLOR = "#111827"
PANEL_BG = "#f8fafc"


@dataclass
class LoadedTrajectory:
    path: Path
    trajectory_index: int
    states: torch.Tensor
    effective_horizon: int
    state_convention: str = "unknown"

    @property
    def raw_num_frames(self) -> int:
        return int(self.states.shape[0])

    @property
    def initial_state(self) -> torch.Tensor:
        return self.states[0]


@dataclass
class ComparisonResult:
    title: str
    subtitle: str
    initial_state_source: str
    reference_state_convention: str
    trajectory_a_label: str
    trajectory_a_states: torch.Tensor
    trajectory_b_label: str
    trajectory_b_states: torch.Tensor
    reference_path: Path
    model_path: Path | None
    horizon: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize CubeTossing trajectory gaps as SVG overlays. The rollout "
            "starts from the reference trajectory initial state, then compares the "
            "resulting COM position trajectory against the reference data."
        )
    )
    parser.add_argument(
        "--comparison",
        required=True,
        choices=["sim2real", "sim2nerd", "nerd2real"],
        help="Which pair to compare.",
    )
    parser.add_argument(
        "--real-hdf5",
        type=str,
        default=DEFAULT_REAL_HDF5,
        help="Reference real-data HDF5 path for sim2real and nerd2real.",
    )
    parser.add_argument(
        "--sim-hdf5",
        type=str,
        default=DEFAULT_SIM_HDF5,
        help="Reference Newton-sim HDF5 path for sim2nerd.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Trajectory index inside the selected HDF5 file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Number of transitions to compare. Defaults to the full effective "
            "length of the reference trajectory."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "NeRD checkpoint (*.pt). Required for sim2nerd and nerd2real unless a "
            "default CubeTossing checkpoint can be auto-discovered."
        ),
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=None,
        help="Optional cfg.yaml path paired with --model-path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch/Newton device string.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="mujoco",
        choices=list(SOLVER_TYPES.keys()),
        help="Newton solver used when sim2real rolls out from real initial state.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint"],
        help="Observation type passed to CubeTossingEnv / NeuralEnvironment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed. Random reset stays disabled for these comparisons.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Output SVG path. Defaults to "
            "figures/trajectory_gaps/<comparison>_traj<trajectory-index>.svg."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title written into the SVG header.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def find_latest_checkpoint(pattern: str) -> Path | None:
    candidates = sorted(
        REPO_ROOT.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_default_model_path() -> Path:
    preferred = (
        REPO_ROOT
        / "pretrained_models"
        / "NeRD_models"
        / "CubeTossing"
        / "model"
        / "nn"
        / "model.pt"
    )
    if preferred.exists():
        return preferred

    search_patterns = [
        "runs/CubeTossingSimulation/**/nn/best_eval_model.pt",
        "runs/CubeTossingSimulation/**/nn/best_valid_exp_trajectory_model.pt",
        "runs/CubeTossingSimulation/**/nn/best_valid_passive_trajectory_model.pt",
        "runs/CubeTossingSimulation/**/nn/final_model.pt",
    ]
    for pattern in search_patterns:
        match = find_latest_checkpoint(pattern)
        if match is not None:
            return match

    raise FileNotFoundError(
        "Could not find a CubeTossing checkpoint. Pass --model-path explicitly."
    )


def resolve_cfg_path(model_path: Path, cfg_path_arg: str | None) -> Path:
    if cfg_path_arg is not None:
        cfg_path = resolve_repo_path(cfg_path_arg)
    else:
        cfg_path = model_path.parents[1] / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def get_neural_solver_cfg(cfg: dict) -> dict:
    env_cfg = cfg.get("env", {})
    if "neural_solver_cfg" in env_cfg:
        return dict(env_cfg["neural_solver_cfg"])
    if "neural_integrator_cfg" in env_cfg:
        return dict(env_cfg["neural_integrator_cfg"])
    raise KeyError("Config must contain env.neural_solver_cfg or env.neural_integrator_cfg.")


def load_trajectory(path: Path, trajectory_index: int) -> LoadedTrajectory:
    with h5py.File(path, "r") as h5_file:
        data_group = h5_file["data"]
        states_key = "states_world" if "states_world" in data_group else "states"
        states = data_group[states_key]
        if trajectory_index < 0 or trajectory_index >= states.shape[1]:
            raise IndexError(
                f"trajectory_index {trajectory_index} is out of range for B={states.shape[1]}."
            )

        if "traj_lengths" in h5_file["data"]:
            effective_horizon = int(h5_file["data"]["traj_lengths"][trajectory_index])
            raw_num_frames = effective_horizon + 1
        else:
            raw_num_frames = int(states.shape[0])
            effective_horizon = max(raw_num_frames - 1, 0)

        clipped_states = torch.from_numpy(
            states[:raw_num_frames, trajectory_index, :].astype("float32")
        )
        if states_key == "states_world":
            attr_convention = state_convention.NEWTON_FREE_JOINT_STATE_CONVENTION
        else:
            attr_convention = state_convention.infer_free_joint_state_convention_from_attrs(
                data_group.attrs
            )

    return LoadedTrajectory(
        path=path,
        trajectory_index=trajectory_index,
        states=clipped_states,
        effective_horizon=effective_horizon,
        state_convention=attr_convention,
    )


def normalize_trajectory_states(trajectory: LoadedTrajectory) -> LoadedTrajectory:
    normalized_states, convention = state_convention.normalize_free_joint_states(
        trajectory.states,
        trajectory.state_convention,
    )
    if convention == state_convention.LEGACY_FREE_JOINT_STATE_CONVENTION:
        return LoadedTrajectory(
            path=trajectory.path,
            trajectory_index=trajectory.trajectory_index,
            states=normalized_states,
            effective_horizon=trajectory.effective_horizon,
            state_convention=convention,
        )

    return LoadedTrajectory(
        path=trajectory.path,
        trajectory_index=trajectory.trajectory_index,
        states=normalized_states,
        effective_horizon=trajectory.effective_horizon,
        state_convention=convention,
    )


def capture_env_state(env) -> torch.Tensor:
    torch_device = warp_utils.device_to_torch(env.device)
    state_dim = env.dof_q_per_env + env.dof_qd_per_env
    out = torch.zeros((env.num_envs, state_dim), dtype=torch.float32, device=torch_device)
    warp_utils.acquire_states_to_torch(env, out)
    return out


def rollout_newton_from_initial_state(
    initial_state: torch.Tensor,
    horizon: int,
    device: str,
    solver_type: str,
    obs_type: str,
    seed: int,
) -> torch.Tensor:
    env = create_fixed_contact_env(
        env_name="CubeTossing",
        num_envs=1,
        device=device,
        use_graph_capture=False,
        render=False,
        seed=seed,
        random_reset=False,
        solver_type=SOLVER_TYPES[solver_type],
        obs_type=obs_type,
        camera_tracking=False,
    )

    torch_device = warp_utils.device_to_torch(env.device)
    rollout_states = torch.zeros(
        (horizon + 1, env.dof_q_per_env + env.dof_qd_per_env),
        dtype=torch.float32,
        device=torch_device,
    )

    try:
        env.reset()
        initial_state_batch = initial_state.unsqueeze(0).to(torch_device)
        warp_utils.assign_states_from_torch(env, initial_state_batch)
        warp_utils.eval_fk(env.model, env.state)

        rollout_states[0].copy_(capture_env_state(env)[0])
        for step in range(horizon):
            env.update()
            rollout_states[step + 1].copy_(capture_env_state(env)[0])
    finally:
        env.close()

    return rollout_states.cpu()


def rollout_nerd_from_initial_state(
    initial_state: torch.Tensor,
    horizon: int,
    device: str,
    obs_type: str,
    seed: int,
    model_path_arg: str | None,
    cfg_path_arg: str | None,
) -> tuple[torch.Tensor, Path]:
    model_path = (
        resolve_repo_path(model_path_arg)
        if model_path_arg is not None
        else resolve_default_model_path()
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    cfg_path = resolve_cfg_path(model_path, cfg_path_arg)
    cfg = load_yaml(cfg_path)
    neural_solver_cfg = get_neural_solver_cfg(cfg)

    env_name = cfg.get("env", {}).get("env_name", "CubeTossing")
    if env_name != "CubeTossing":
        raise ValueError(
            f"This visualizer is dedicated to CubeTossing, but cfg env_name={env_name!r}."
        )

    neural_model, robot_name, _ = load_neural_model_checkpoint(
        model_path,
        map_location=device,
    )
    neural_model.to(device)
    neural_model.eval()
    if hasattr(neural_model, "fix_input_names"):
        neural_model.fix_input_names()

    newton_env_cfg = dict(cfg.get("env", {}).get("newton_env_cfg", {}))
    newton_env_cfg["obs_type"] = obs_type
    newton_env_cfg["camera_tracking"] = False
    newton_env_cfg["random_reset"] = False
    newton_env_cfg["seed"] = seed

    neural_env = NeuralEnvironment(
        env_name="CubeTossing",
        num_envs=1,
        newton_env_cfg=newton_env_cfg,
        neural_solver_cfg=neural_solver_cfg,
        neural_model=neural_model,
        default_env_mode="neural",
        use_graph_capture=False,
        device=device,
        render=False,
    )

    if neural_env.robot_name != robot_name:
        raise ValueError(
            f"Checkpoint robot_name={robot_name!r} does not match env robot_name={neural_env.robot_name!r}."
        )

    torch_device = torch.device(device)
    zero_actions = torch.zeros((1, neural_env.action_dim), device=torch_device)
    rollout_states = torch.zeros(
        (horizon + 1, neural_env.state_dim),
        dtype=torch.float32,
        device=torch_device,
    )

    try:
        with torch.no_grad():
            neural_env.reset(initial_state.unsqueeze(0).to(torch_device))
            neural_env.init_rnn(neural_env.num_envs)
            rollout_states[0].copy_(neural_env.states[0])
            for step in range(horizon):
                rollout_states[step + 1].copy_(neural_env.step(zero_actions)[0])
    finally:
        neural_env.close()

    return rollout_states.cpu(), model_path


def build_comparison(args: argparse.Namespace) -> ComparisonResult:
    if args.comparison in ("sim2real", "nerd2real"):
        reference = normalize_trajectory_states(
            load_trajectory(
                resolve_repo_path(args.real_hdf5),
                trajectory_index=args.trajectory_index,
            )
        )
    else:
        reference = normalize_trajectory_states(
            load_trajectory(
                resolve_repo_path(args.sim_hdf5),
                trajectory_index=args.trajectory_index,
            )
        )

    horizon = reference.effective_horizon if args.horizon is None else args.horizon
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    if horizon > reference.effective_horizon:
        raise ValueError(
            f"Requested horizon {horizon} exceeds reference effective horizon "
            f"{reference.effective_horizon} for {reference.path}."
        )

    reference_states = reference.states[: horizon + 1].clone()

    if args.comparison == "sim2real":
        simulated_states = rollout_newton_from_initial_state(
            initial_state=reference.initial_state,
            horizon=horizon,
            device=args.device,
            solver_type=args.solver_type,
            obs_type=args.obs_type,
            seed=args.seed,
        )
        return ComparisonResult(
            title=args.title or "sim2real gap",
            subtitle="Newton Sim vs. Real trajectory",
            initial_state_source=f"real trajectory: {reference.path.name}[traj={reference.trajectory_index}]",
            reference_state_convention=reference.state_convention,
            trajectory_a_label="Newton Sim",
            trajectory_a_states=simulated_states,
            trajectory_b_label="Real data",
            trajectory_b_states=reference_states,
            reference_path=reference.path,
            model_path=None,
            horizon=horizon,
        )

    if args.comparison == "sim2nerd":
        nerd_states, model_path = rollout_nerd_from_initial_state(
            initial_state=reference.initial_state,
            horizon=horizon,
            device=args.device,
            obs_type=args.obs_type,
            seed=args.seed,
            model_path_arg=args.model_path,
            cfg_path_arg=args.cfg_path,
        )
        return ComparisonResult(
            title=args.title or "sim2nerd gap",
            subtitle="Newton Sim vs. NeRD Sim trajectory",
            initial_state_source=f"Newton sim trajectory: {reference.path.name}[traj={reference.trajectory_index}]",
            reference_state_convention=reference.state_convention,
            trajectory_a_label="Newton Sim",
            trajectory_a_states=reference_states,
            trajectory_b_label="NeRD Sim",
            trajectory_b_states=nerd_states,
            reference_path=reference.path,
            model_path=model_path,
            horizon=horizon,
        )

    nerd_states, model_path = rollout_nerd_from_initial_state(
        initial_state=reference.initial_state,
        horizon=horizon,
        device=args.device,
        obs_type=args.obs_type,
        seed=args.seed,
        model_path_arg=args.model_path,
        cfg_path_arg=args.cfg_path,
    )
    return ComparisonResult(
        title=args.title or "nerd2real gap",
        subtitle="NeRD Sim vs. Real trajectory",
        initial_state_source=f"real trajectory: {reference.path.name}[traj={reference.trajectory_index}]",
        reference_state_convention=reference.state_convention,
        trajectory_a_label="NeRD Sim",
        trajectory_a_states=nerd_states,
        trajectory_b_label="Real data",
        trajectory_b_states=reference_states,
        reference_path=reference.path,
        model_path=model_path,
        horizon=horizon,
    )


def compute_metrics(states_a: torch.Tensor, states_b: torch.Tensor) -> dict[str, float]:
    if states_a.shape != states_b.shape:
        raise ValueError(
            f"State tensors must have the same shape, got {tuple(states_a.shape)} and {tuple(states_b.shape)}."
        )

    pos_a = states_a[:, 0:3]
    pos_b = states_b[:, 0:3]
    diff = pos_a - pos_b
    dist = torch.linalg.norm(diff, dim=-1)

    def path_length(pos: torch.Tensor) -> float:
        if pos.shape[0] < 2:
            return 0.0
        return float(torch.linalg.norm(pos[1:] - pos[:-1], dim=-1).sum().item())

    return {
        "num_frames": float(states_a.shape[0]),
        "mean_pos_error_m": float(dist.mean().item()),
        "max_pos_error_m": float(dist.max().item()),
        "final_pos_error_m": float(dist[-1].item()),
        "path_length_a_m": path_length(pos_a),
        "path_length_b_m": path_length(pos_b),
        "max_height_gap_m": float(torch.abs(pos_a[:, 2] - pos_b[:, 2]).max().item()),
    }


def compute_bounds(series_list: list[torch.Tensor]) -> tuple[float, float, float, float]:
    points = torch.cat(series_list, dim=0)
    min_x = float(points[:, 0].min().item())
    max_x = float(points[:, 0].max().item())
    min_y = float(points[:, 1].min().item())
    max_y = float(points[:, 1].max().item())

    span_x = max(max_x - min_x, 1.0e-6)
    span_y = max(max_y - min_y, 1.0e-6)
    pad_x = max(span_x * 0.08, 0.01)
    pad_y = max(span_y * 0.08, 0.01)
    return (
        min_x - pad_x,
        max_x + pad_x,
        min_y - pad_y,
        max_y + pad_y,
    )


def fmt_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 10.0:
        return f"{value:.2f}"
    if abs_value >= 1.0:
        return f"{value:.3f}"
    return f"{value:.4f}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def project_isometric(points_xyz: torch.Tensor) -> torch.Tensor:
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    u = (x - y) * math.cos(math.pi / 6.0)
    v = z + (x + y) * math.sin(math.pi / 6.0)
    return torch.stack([u, v], dim=-1)


def polyline_points_str(
    points_2d: torch.Tensor,
    bounds: tuple[float, float, float, float],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> str:
    min_x, max_x, min_y, max_y = bounds
    data_w = max(max_x - min_x, 1.0e-6)
    data_h = max(max_y - min_y, 1.0e-6)

    margin_left = 58.0
    margin_right = 18.0
    margin_top = 24.0
    margin_bottom = 42.0

    inner_x = panel_x + margin_left
    inner_y = panel_y + margin_top
    inner_w = panel_w - margin_left - margin_right
    inner_h = panel_h - margin_top - margin_bottom

    scale = min(inner_w / data_w, inner_h / data_h)
    used_w = data_w * scale
    used_h = data_h * scale
    offset_x = (inner_w - used_w) * 0.5
    offset_y = (inner_h - used_h) * 0.5

    xy = []
    for point in points_2d:
        sx = inner_x + offset_x + (float(point[0].item()) - min_x) * scale
        sy = inner_y + inner_h - offset_y - (float(point[1].item()) - min_y) * scale
        xy.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(xy)


def map_point_to_panel(
    point_2d: torch.Tensor,
    bounds: tuple[float, float, float, float],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> tuple[float, float]:
    encoded = polyline_points_str(
        point_2d.view(1, 2),
        bounds,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
    )
    x_str, y_str = encoded.split(" ")[0].split(",")
    return float(x_str), float(y_str)


def render_panel(
    title: str,
    x_label: str,
    y_label: str,
    traj_a_points: torch.Tensor,
    traj_b_points: torch.Tensor,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> str:
    bounds = compute_bounds([traj_a_points, traj_b_points])
    min_x, max_x, min_y, max_y = bounds

    margin_left = 58.0
    margin_right = 18.0
    margin_top = 24.0
    margin_bottom = 42.0

    inner_x = panel_x + margin_left
    inner_y = panel_y + margin_top
    inner_w = panel_w - margin_left - margin_right
    inner_h = panel_h - margin_top - margin_bottom

    parts = [
        f'<g transform="translate(0,0)">',
        (
            f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" '
            f'height="{panel_h:.1f}" rx="6" fill="{PANEL_BG}" stroke="#e5e7eb" />'
        ),
        (
            f'<text x="{panel_x + panel_w / 2.0:.1f}" y="{panel_y + 16.0:.1f}" '
            f'font-size="14" text-anchor="middle" fill="{TEXT_COLOR}" '
            f'font-family="Segoe UI, Arial, sans-serif">{escape(title)}</text>'
        ),
    ]

    for tick_idx in range(5):
        tx = tick_idx / 4.0
        x_value = min_x + (max_x - min_x) * tx
        sx = inner_x + inner_w * tx
        parts.append(
            f'<line x1="{sx:.2f}" y1="{inner_y:.2f}" x2="{sx:.2f}" '
            f'y2="{inner_y + inner_h:.2f}" stroke="{GRID_COLOR}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{sx:.2f}" y="{panel_y + panel_h - 10.0:.2f}" font-size="11" '
            f'text-anchor="middle" fill="{AXIS_COLOR}" '
            f'font-family="Consolas, Menlo, monospace">{fmt_value(x_value)}</text>'
        )

        ty = tick_idx / 4.0
        y_value = min_y + (max_y - min_y) * ty
        sy = inner_y + inner_h - inner_h * ty
        parts.append(
            f'<line x1="{inner_x:.2f}" y1="{sy:.2f}" x2="{inner_x + inner_w:.2f}" '
            f'y2="{sy:.2f}" stroke="{GRID_COLOR}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{panel_x + 46.0:.2f}" y="{sy + 4.0:.2f}" font-size="11" '
            f'text-anchor="end" fill="{AXIS_COLOR}" '
            f'font-family="Consolas, Menlo, monospace">{fmt_value(y_value)}</text>'
        )

    if min_x <= 0.0 <= max_x:
        zero_x = map_point_to_panel(
            torch.tensor([0.0, min_y]),
            bounds,
            panel_x,
            panel_y,
            panel_w,
            panel_h,
        )[0]
        parts.append(
            f'<line x1="{zero_x:.2f}" y1="{inner_y:.2f}" x2="{zero_x:.2f}" '
            f'y2="{inner_y + inner_h:.2f}" stroke="{AXIS_COLOR}" stroke-width="1.2" />'
        )

    if min_y <= 0.0 <= max_y:
        zero_y = map_point_to_panel(
            torch.tensor([min_x, 0.0]),
            bounds,
            panel_x,
            panel_y,
            panel_w,
            panel_h,
        )[1]
        parts.append(
            f'<line x1="{inner_x:.2f}" y1="{zero_y:.2f}" x2="{inner_x + inner_w:.2f}" '
            f'y2="{zero_y:.2f}" stroke="{AXIS_COLOR}" stroke-width="1.2" />'
        )

    points_a = polyline_points_str(traj_a_points, bounds, panel_x, panel_y, panel_w, panel_h)
    points_b = polyline_points_str(traj_b_points, bounds, panel_x, panel_y, panel_w, panel_h)

    parts.append(
        f'<polyline fill="none" stroke="{TRAJ_A_COLOR}" stroke-width="2.6" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points_a}" />'
    )
    parts.append(
        f'<polyline fill="none" stroke="{TRAJ_B_COLOR}" stroke-width="2.6" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points_b}" />'
    )

    for color, points in ((TRAJ_A_COLOR, traj_a_points), (TRAJ_B_COLOR, traj_b_points)):
        start_x, start_y = map_point_to_panel(points[0], bounds, panel_x, panel_y, panel_w, panel_h)
        end_x, end_y = map_point_to_panel(points[-1], bounds, panel_x, panel_y, panel_w, panel_h)
        parts.append(
            f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="4.5" fill="white" '
            f'stroke="{color}" stroke-width="2" />'
        )
        parts.append(
            f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="3.8" fill="{color}" '
            f'stroke="white" stroke-width="1.2" />'
        )

    parts.append(
        f'<text x="{panel_x + panel_w / 2.0:.1f}" y="{panel_y + panel_h - 10.0:.1f}" '
        f'font-size="12" text-anchor="middle" fill="{TEXT_COLOR}" '
        f'font-family="Segoe UI, Arial, sans-serif">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="{panel_x + 18.0:.1f}" y="{panel_y + panel_h / 2.0:.1f}" '
        f'font-size="12" text-anchor="middle" fill="{TEXT_COLOR}" '
        f'transform="rotate(-90 {panel_x + 18.0:.1f} {panel_y + panel_h / 2.0:.1f})" '
        f'font-family="Segoe UI, Arial, sans-serif">{escape(y_label)}</text>'
    )
    parts.append("</g>")
    return "\n".join(parts)


def render_svg(result: ComparisonResult, metrics: dict[str, float]) -> str:
    width = 1440
    height = 1010

    traj_a_xyz = result.trajectory_a_states[:, 0:3]
    traj_b_xyz = result.trajectory_b_states[:, 0:3]

    panels = [
        (
            "Isometric COM trajectory",
            "iso(x, y)",
            "iso(z)",
            project_isometric(traj_a_xyz),
            project_isometric(traj_b_xyz),
            40.0,
            190.0,
            660.0,
            340.0,
        ),
        (
            "XY projection",
            "x (m)",
            "y (m)",
            traj_a_xyz[:, [0, 1]],
            traj_b_xyz[:, [0, 1]],
            740.0,
            190.0,
            660.0,
            340.0,
        ),
        (
            "XZ projection",
            "x (m)",
            "z (m)",
            traj_a_xyz[:, [0, 2]],
            traj_b_xyz[:, [0, 2]],
            40.0,
            580.0,
            660.0,
            340.0,
        ),
        (
            "YZ projection",
            "y (m)",
            "z (m)",
            traj_a_xyz[:, [1, 2]],
            traj_b_xyz[:, [1, 2]],
            740.0,
            580.0,
            660.0,
            340.0,
        ),
    ]

    metric_lines = [
        f"frames={int(metrics['num_frames'])}",
        f"mean_pos_error={metrics['mean_pos_error_m']:.4f} m",
        f"max_pos_error={metrics['max_pos_error_m']:.4f} m",
        f"final_pos_error={metrics['final_pos_error_m']:.4f} m",
        f"path_length[{result.trajectory_a_label}]={metrics['path_length_a_m']:.4f} m",
        f"path_length[{result.trajectory_b_label}]={metrics['path_length_b_m']:.4f} m",
        f"max_height_gap={metrics['max_height_gap_m']:.4f} m",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        (
            f'<text x="40" y="52" font-size="28" fill="{TEXT_COLOR}" '
            f'font-family="Segoe UI Semibold, Segoe UI, Arial, sans-serif">{escape(result.title)}</text>'
        ),
        (
            f'<text x="40" y="84" font-size="16" fill="#374151" '
            f'font-family="Segoe UI, Arial, sans-serif">{escape(result.subtitle)}</text>'
        ),
        (
            f'<text x="40" y="110" font-size="13" fill="#4b5563" '
            f'font-family="Segoe UI, Arial, sans-serif">Initial state source: '
            f'{escape(result.initial_state_source)}</text>'
        ),
        (
            f'<text x="40" y="130" font-size="13" fill="#4b5563" '
            f'font-family="Segoe UI, Arial, sans-serif">Reference file: '
            f'{escape(display_path(result.reference_path))}</text>'
        ),
        (
            f'<text x="40" y="150" font-size="13" fill="#4b5563" '
            f'font-family="Segoe UI, Arial, sans-serif">Reference state convention: '
            f'{escape(result.reference_state_convention)}</text>'
        ),
        (
            f'<line x1="40" y1="166" x2="94" y2="166" stroke="{TRAJ_A_COLOR}" stroke-width="4" />'
            f'<text x="104" y="170" font-size="13" fill="{TEXT_COLOR}" '
            f'font-family="Segoe UI, Arial, sans-serif">{escape(result.trajectory_a_label)}</text>'
        ),
        (
            f'<line x1="250" y1="166" x2="304" y2="166" stroke="{TRAJ_B_COLOR}" stroke-width="4" />'
            f'<text x="314" y="170" font-size="13" fill="{TEXT_COLOR}" '
            f'font-family="Segoe UI, Arial, sans-serif">{escape(result.trajectory_b_label)}</text>'
        ),
    ]

    if result.model_path is not None:
        parts.append(
            f'<text x="510" y="170" font-size="13" fill="#4b5563" '
            f'font-family="Segoe UI, Arial, sans-serif">NeRD checkpoint: '
            f'{escape(display_path(result.model_path))}</text>'
        )

    parts.append(
        (
            f'<text x="1010" y="52" font-size="13" fill="{TEXT_COLOR}" '
            f'font-family="Consolas, Menlo, monospace">'
            f'{escape(metric_lines[0])}</text>'
        )
    )
    for idx, line in enumerate(metric_lines[1:], start=1):
        parts.append(
            (
                f'<text x="1010" y="{52 + idx * 20}" font-size="13" fill="{TEXT_COLOR}" '
                f'font-family="Consolas, Menlo, monospace">{escape(line)}</text>'
            )
        )

    for panel in panels:
        parts.append(render_panel(*panel))

    parts.append(
        (
            f'<text x="40" y="974" font-size="12" fill="#6b7280" '
            f'font-family="Segoe UI, Arial, sans-serif">'
            f'All panels overlay the cube COM trajectory from state[:, 0:3]. '
            f'Start markers are hollow; end markers are filled.</text>'
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def default_output_path(args: argparse.Namespace) -> Path:
    filename = f"{args.comparison}_traj{args.trajectory_index}.svg"
    return REPO_ROOT / DEFAULT_OUTPUT_DIR / filename


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    result = build_comparison(args)
    metrics = compute_metrics(result.trajectory_a_states, result.trajectory_b_states)
    svg_text = render_svg(result, metrics)

    output_path = (
        resolve_repo_path(args.output_path)
        if args.output_path is not None
        else default_output_path(args)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_text, encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"comparison={args.comparison}")
    print(f"reference_state_convention={result.reference_state_convention}")
    print(
        f"{result.trajectory_a_label} vs {result.trajectory_b_label} "
        f"(frames={int(metrics['num_frames'])})"
    )
    print(f"mean_pos_error_m={metrics['mean_pos_error_m']:.6f}")
    print(f"max_pos_error_m={metrics['max_pos_error_m']:.6f}")
    print(f"final_pos_error_m={metrics['final_pos_error_m']:.6f}")


if __name__ == "__main__":
    main()
