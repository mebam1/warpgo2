from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import RenderMode
from mebam.visualize.compare_cube_tossing_trajectories import (
    DEFAULT_REAL_HDF5,
    load_trajectory,
    normalize_trajectory_states,
    rollout_nerd_from_initial_state,
    rollout_newton_from_initial_state,
)
from mebam.visualize.visualize_sim2nerd_cube_tossing import (
    NERD_COLOR,
    SIM_COLOR,
    SOLVER_TYPES,
    Sim2NeRDComparisonEnv,
    build_frame_ids,
    show_viewer_window,
)
from utils.python_utils import set_random_seed


REAL_COLOR = (0.145, 0.659, 0.302)


@dataclass
class RealSimNeRDTrajectories:
    real_states: torch.Tensor
    sim_states: torch.Tensor
    nerd_states: torch.Tensor
    model_path: Path
    reference_label: str
    horizon: int
    state_convention: str


class RealSimNeRDComparisonEnv(Sim2NeRDComparisonEnv):
    def _init_overlay_buffers(self) -> None:
        torch_device = self._get_overlay_torch_device()
        self._comparison_xforms_torch = torch.zeros(
            (3, 7), dtype=torch.float32, device=torch_device
        )
        self._comparison_xforms_torch[:, 6] = 1.0
        self._comparison_colors_torch = torch.tensor(
            [REAL_COLOR, SIM_COLOR, NERD_COLOR],
            dtype=torch.float32,
            device=torch_device,
        )
        self._comparison_xforms_wp = wp.from_torch(
            self._comparison_xforms_torch, dtype=wp.transform
        )
        self._comparison_colors_wp = wp.from_torch(
            self._comparison_colors_torch, dtype=wp.vec3
        )

    def set_cube_states(
        self,
        real_state: torch.Tensor,
        sim_state: torch.Tensor,
        nerd_state: torch.Tensor,
    ) -> None:
        if self._comparison_xforms_torch is None:
            self._init_overlay_buffers()

        comparison_xforms = torch.stack(
            [real_state[:7], sim_state[:7], nerd_state[:7]], dim=0
        ).to(self._comparison_xforms_torch.device)
        self._comparison_xforms_torch.copy_(comparison_xforms)
        self._camera_target = comparison_xforms[:, 0:3].mean(dim=0).detach().cpu()

    def _clear_trails(self) -> None:
        self._trail_starts_torch = None
        self._trail_starts_wp = None
        self._trail_ends_torch = None
        self._trail_ends_wp = None
        self._trail_colors_torch = None
        self._trail_colors_wp = None

    def set_trails(
        self,
        real_states_prefix: torch.Tensor,
        sim_states_prefix: torch.Tensor,
        nerd_states_prefix: torch.Tensor,
    ) -> None:
        if not self.show_trails or real_states_prefix.shape[0] < 2:
            self._clear_trails()
            return

        torch_device = self._get_overlay_torch_device()
        real_positions = real_states_prefix[:, 0:3].to(torch_device)
        sim_positions = sim_states_prefix[:, 0:3].to(torch_device)
        nerd_positions = nerd_states_prefix[:, 0:3].to(torch_device)

        self._trail_starts_torch = torch.cat(
            [real_positions[:-1], sim_positions[:-1], nerd_positions[:-1]], dim=0
        )
        self._trail_ends_torch = torch.cat(
            [real_positions[1:], sim_positions[1:], nerd_positions[1:]], dim=0
        )

        def repeat_color(color: tuple[float, float, float], count: int) -> torch.Tensor:
            return torch.tensor(
                [color],
                dtype=torch.float32,
                device=torch_device,
            ).repeat(count, 1)

        self._trail_colors_torch = torch.cat(
            [
                repeat_color(REAL_COLOR, real_positions.shape[0] - 1),
                repeat_color(SIM_COLOR, sim_positions.shape[0] - 1),
                repeat_color(NERD_COLOR, nerd_positions.shape[0] - 1),
            ],
            dim=0,
        )

        self._trail_starts_wp = wp.from_torch(self._trail_starts_torch, dtype=wp.vec3)
        self._trail_ends_wp = wp.from_torch(self._trail_ends_torch, dtype=wp.vec3)
        self._trail_colors_wp = wp.from_torch(self._trail_colors_torch, dtype=wp.vec3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize real-data, Newton Sim, and NeRD cube-tossing trajectories "
            "together in the Warp viewer. The Sim and NeRD rollouts both start "
            "from the selected real trajectory initial state."
        )
    )
    parser.add_argument(
        "--real-hdf5",
        type=str,
        default=DEFAULT_REAL_HDF5,
        help="Reference real-data HDF5 path.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Trajectory index inside the real-data HDF5 file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Number of transitions to replay. Defaults to the reference "
            "trajectory effective horizon."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional NeRD checkpoint path.",
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=None,
        help="Optional cfg.yaml paired with --model-path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch/Newton device string for the rollouts and viewer env.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint"],
        help="Observation type passed to the rollouts and viewer env.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="mujoco",
        choices=list(SOLVER_TYPES.keys()),
        help="Newton solver used for the Sim rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS. Defaults to the env sample rate.",
    )
    parser.add_argument(
        "--camera-tracking",
        action="store_true",
        help="Track the mean position of the three cubes with the viewer camera.",
    )
    parser.add_argument(
        "--hide-trails",
        action="store_true",
        help="Do not draw trajectory trails.",
    )
    parser.add_argument(
        "--trail-width",
        type=float,
        default=0.01,
        help="Line width used for trajectory trails.",
    )
    parser.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        help="Loop playback after the last frame. Enabled by default.",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Stop playback at the last frame instead of looping.",
    )
    parser.add_argument(
        "--close-on-finish",
        action="store_true",
        help="Close the viewer when the final frame is reached.",
    )
    parser.set_defaults(loop=True)
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_real_sim_nerd_trajectories(
    args: argparse.Namespace,
) -> RealSimNeRDTrajectories:
    reference = normalize_trajectory_states(
        load_trajectory(
            resolve_repo_path(args.real_hdf5),
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

    real_states = reference.states[: horizon + 1].detach().cpu()
    sim_states = rollout_newton_from_initial_state(
        reference.initial_state,
        horizon=horizon,
        device=args.device,
        solver_type=args.solver_type,
        obs_type=args.obs_type,
        seed=args.seed,
    )
    sim_states = sim_states[: horizon + 1].detach().cpu()

    nerd_states, model_path = rollout_nerd_from_initial_state(
        reference.initial_state,
        horizon=horizon,
        device=args.device,
        obs_type=args.obs_type,
        seed=args.seed,
        model_path_arg=args.model_path,
        cfg_path_arg=args.cfg_path,
    )
    nerd_states = nerd_states[: horizon + 1].detach().cpu()

    return RealSimNeRDTrajectories(
        real_states=real_states,
        sim_states=sim_states,
        nerd_states=nerd_states,
        model_path=model_path,
        reference_label=f"{reference.path}[traj={reference.trajectory_index}]",
        horizon=horizon,
        state_convention=reference.state_convention,
    )


def average_position_error(reference_states: torch.Tensor, other_states: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(reference_states[:, 0:3] - other_states[:, 0:3], dim=-1)
        .mean()
        .item()
    )


def average_orientation_error_radians(
    reference_states: torch.Tensor,
    other_states: torch.Tensor,
) -> float:
    reference_quats = reference_states[:, 3:7]
    other_quats = other_states[:, 3:7]

    reference_quats = reference_quats / reference_quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    other_quats = other_quats / other_quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    alignment = torch.sum(reference_quats * other_quats, dim=-1).abs().clamp(0.0, 1.0)
    return float((2.0 * torch.acos(alignment)).mean().item())


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    trajectories = build_real_sim_nerd_trajectories(args)

    real_sim_position_error = average_position_error(
        trajectories.real_states, trajectories.sim_states
    )
    real_sim_orientation_error = average_orientation_error_radians(
        trajectories.real_states, trajectories.sim_states
    )
    real_nerd_position_error = average_position_error(
        trajectories.real_states, trajectories.nerd_states
    )
    real_nerd_orientation_error = average_orientation_error_radians(
        trajectories.real_states, trajectories.nerd_states
    )

    env = RealSimNeRDComparisonEnv(
        num_envs=1,
        seed=args.seed,
        random_reset=False,
        render_mode=RenderMode.OPENGL,
        device=args.device,
        camera_tracking=args.camera_tracking,
        obs_type=args.obs_type,
        use_graph_capture=False,
        show_trails=not args.hide_trails,
        trail_width=args.trail_width,
    )

    frame_ids = build_frame_ids(
        num_frames=trajectories.horizon + 1,
        frame_stride=args.frame_stride,
    )
    sleep_dt = 1.0 / (args.fps if args.fps is not None else env.fps)

    print(f"reference={trajectories.reference_label}")
    print(f"checkpoint={trajectories.model_path}")
    print(f"trajectory_index={args.trajectory_index}")
    print(f"frames={len(frame_ids)} / {trajectories.horizon + 1}")
    print(f"reference_state_convention={trajectories.state_convention}")
    print(f"avg_position_error_real_sim_m={real_sim_position_error:.6f}")
    print(f"avg_orientation_error_real_sim_rad={real_sim_orientation_error:.6f}")
    print(f"avg_position_error_real_nerd_m={real_nerd_position_error:.6f}")
    print(f"avg_orientation_error_real_nerd_rad={real_nerd_orientation_error:.6f}")

    env.reset()
    show_viewer_window(env)
    frame_cursor = 1

    try:
        initial_frame_id = frame_ids[0]
        env.set_cube_states(
            trajectories.real_states[initial_frame_id],
            trajectories.sim_states[initial_frame_id],
            trajectories.nerd_states[initial_frame_id],
        )
        env.set_trails(
            trajectories.real_states[: initial_frame_id + 1],
            trajectories.sim_states[: initial_frame_id + 1],
            trajectories.nerd_states[: initial_frame_id + 1],
        )
        env.sim_time = initial_frame_id * env.frame_dt
        env.render()

        while env.viewer.is_running():
            if not env.viewer.is_paused():
                if frame_cursor >= len(frame_ids):
                    if args.loop:
                        frame_cursor = 0
                    else:
                        if args.close_on_finish:
                            break
                        env.render()
                        time.sleep(sleep_dt)
                        continue

                frame_id = frame_ids[frame_cursor]
                env.set_cube_states(
                    trajectories.real_states[frame_id],
                    trajectories.sim_states[frame_id],
                    trajectories.nerd_states[frame_id],
                )
                env.set_trails(
                    trajectories.real_states[: frame_id + 1],
                    trajectories.sim_states[: frame_id + 1],
                    trajectories.nerd_states[: frame_id + 1],
                )
                env.sim_time = frame_id * env.frame_dt
                env.render()
                frame_cursor += 1
                time.sleep(sleep_dt)
            else:
                env.render()
                time.sleep(sleep_dt)
    finally:
        if env.viewer is not None:
            env.viewer.close()
        env.close()


if __name__ == "__main__":
    main()
