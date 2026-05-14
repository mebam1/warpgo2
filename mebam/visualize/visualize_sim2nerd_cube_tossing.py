from __future__ import annotations

import argparse
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp
import newton

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import RenderMode, SolverType
from envs.newton_envs.env_cube import (
    CubeTossingEnv,
    _add_body_compat,
    _add_free_joint_articulation_compat,
    _add_shape_box_compat,
    _diag_mat33,
)
from mebam.visualize.compare_cube_tossing_trajectories import (
    capture_env_state,
    rollout_nerd_from_initial_state,
)
from utils.env_utils import create_fixed_contact_env
from utils.python_utils import set_random_seed


SIM_COLOR = (0.145, 0.388, 0.922)
NERD_COLOR = (0.851, 0.467, 0.054)
SOLVER_TYPES = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


@dataclass
class Sim2NeRDTrajectories:
    sim_states: torch.Tensor
    nerd_states: torch.Tensor
    model_path: Path
    reference_label: str
    horizon: int
    state_convention: str


class Sim2NeRDComparisonEnv(CubeTossingEnv):
    def __init__(self, *args, show_trails: bool = True, trail_width: float = 0.01, **kwargs):
        self.show_trails = show_trails
        self.trail_width = trail_width
        self._comparison_xforms_torch: torch.Tensor | None = None
        self._comparison_xforms_wp = None
        self._comparison_colors_torch: torch.Tensor | None = None
        self._comparison_colors_wp = None
        self._trail_starts_torch: torch.Tensor | None = None
        self._trail_starts_wp = None
        self._trail_ends_torch: torch.Tensor | None = None
        self._trail_ends_wp = None
        self._trail_colors_torch: torch.Tensor | None = None
        self._trail_colors_wp = None
        self._camera_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        super().__init__(*args, **kwargs)
        self._init_overlay_buffers()

    def create_articulation(self, builder: newton.ModelBuilder):
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=self.friction_coefficient,
        )
        if hasattr(shape_cfg, "restitution"):
            shape_cfg.restitution = self.restitution
        if hasattr(shape_cfg, "is_visible"):
            shape_cfg.is_visible = False

        body = _add_body_compat(
            builder,
            xform=wp.transform(
                p=wp.vec3(*self._initial_position),
                q=wp.quat_identity(),
            ),
            mass=self.mass,
            inertia=_diag_mat33(self.inertia),
            name="anchor_cube",
        )
        _add_shape_box_compat(
            builder,
            body=body,
            hx=self.cube_half_extent,
            hy=self.cube_half_extent,
            hz=self.cube_half_extent,
            cfg=shape_cfg,
            name="anchor_cube_shape",
        )
        _add_free_joint_articulation_compat(builder, body=body, name="anchor_cube")

    def _get_overlay_torch_device(self) -> torch.device:
        if self.viewer is not None and hasattr(self.viewer, "renderer"):
            renderer = self.viewer.renderer
            renderer_device = getattr(renderer, "device", None)
            if renderer_device is None:
                renderer_device = getattr(renderer, "_device", None)
            if renderer_device is not None:
                return wp.device_to_torch(wp.get_device(renderer_device))
        return wp.device_to_torch(self.device)

    def _init_overlay_buffers(self) -> None:
        torch_device = self._get_overlay_torch_device()
        self._comparison_xforms_torch = torch.zeros(
            (2, 7), dtype=torch.float32, device=torch_device
        )
        self._comparison_xforms_torch[:, 6] = 1.0
        self._comparison_colors_torch = torch.tensor(
            [SIM_COLOR, NERD_COLOR],
            dtype=torch.float32,
            device=torch_device,
        )
        self._comparison_xforms_wp = wp.from_torch(
            self._comparison_xforms_torch, dtype=wp.transform
        )
        self._comparison_colors_wp = wp.from_torch(
            self._comparison_colors_torch, dtype=wp.vec3
        )

    def set_cube_states(self, sim_state: torch.Tensor, nerd_state: torch.Tensor) -> None:
        if self._comparison_xforms_torch is None:
            self._init_overlay_buffers()

        comparison_xforms = torch.stack([sim_state[:7], nerd_state[:7]], dim=0).to(
            self._comparison_xforms_torch.device
        )
        self._comparison_xforms_torch.copy_(comparison_xforms)
        self._camera_target = comparison_xforms[:, 0:3].mean(dim=0).detach().cpu()

    def set_trails(
        self,
        sim_states_prefix: torch.Tensor,
        nerd_states_prefix: torch.Tensor,
    ) -> None:
        if not self.show_trails or sim_states_prefix.shape[0] < 2:
            self._trail_starts_torch = None
            self._trail_starts_wp = None
            self._trail_ends_torch = None
            self._trail_ends_wp = None
            self._trail_colors_torch = None
            self._trail_colors_wp = None
            return

        torch_device = self._get_overlay_torch_device()
        sim_positions = sim_states_prefix[:, 0:3].to(torch_device)
        nerd_positions = nerd_states_prefix[:, 0:3].to(torch_device)

        self._trail_starts_torch = torch.cat(
            [sim_positions[:-1], nerd_positions[:-1]], dim=0
        )
        self._trail_ends_torch = torch.cat(
            [sim_positions[1:], nerd_positions[1:]], dim=0
        )
        sim_colors = torch.tensor(
            [SIM_COLOR],
            dtype=torch.float32,
            device=torch_device,
        ).repeat(sim_positions.shape[0] - 1, 1)
        nerd_colors = torch.tensor(
            [NERD_COLOR],
            dtype=torch.float32,
            device=torch_device,
        ).repeat(nerd_positions.shape[0] - 1, 1)
        self._trail_colors_torch = torch.cat([sim_colors, nerd_colors], dim=0)

        self._trail_starts_wp = wp.from_torch(self._trail_starts_torch, dtype=wp.vec3)
        self._trail_ends_wp = wp.from_torch(self._trail_ends_torch, dtype=wp.vec3)
        self._trail_colors_wp = wp.from_torch(self._trail_colors_torch, dtype=wp.vec3)

    def custom_render(self, render_state, viewer):
        viewer.log_shapes(
            "/comparison/cubes",
            int(newton.GeoType.BOX),
            (self.cube_half_extent, self.cube_half_extent, self.cube_half_extent),
            self._comparison_xforms_wp,
            colors=self._comparison_colors_wp,
        )
        if self._trail_starts_wp is not None:
            viewer.log_lines(
                "/comparison/trails",
                self._trail_starts_wp,
                self._trail_ends_wp,
                self._trail_colors_wp,
                width=self.trail_width,
            )

        if self.camera_tracking and hasattr(viewer, "_scaling"):
            cam_pos = wp.vec3(
                float(self._camera_target[0]),
                float(self._camera_target[1] - 1.0),
                float(self._camera_target[2] + 0.8),
            )
            cam_pos = cam_pos * viewer._scaling
            viewer.update_view_matrix(cam_pos=cam_pos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Newton Sim and NeRD Sim cube-tossing trajectories together "
            "in the Warp viewer using two differently colored cubes. The Newton "
            "Sim trajectory is collected live for the requested horizon, then "
            "NeRD rolls out from the same initial state."
        )
    )
    parser.add_argument(
        "--sim-hdf5",
        type=str,
        default=None,
        help="Compatibility option only. Ignored by this script.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Compatibility option only. Ignored by this script.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=111,
        help="Number of Newton-sim transitions to collect and replay.",
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
        help="Torch/Newton device string for the NeRD rollout and viewer env.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint"],
        help="Observation type passed to the NeRD rollout and viewer env.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="mujoco",
        choices=list(SOLVER_TYPES.keys()),
        help="Newton solver used to collect the live sim trajectory.",
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
        help="Track the mean position of the two cubes with the viewer camera.",
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


def show_viewer_window(env: CubeTossingEnv) -> None:
    if env.viewer is None or not hasattr(env.viewer, "renderer"):
        return
    renderer = env.viewer.renderer
    window = getattr(renderer, "window", None)
    if window is None:
        return
    try:
        window.set_visible(True)
    except Exception:
        pass
    try:
        window.activate()
    except Exception:
        pass
    try:
        window.set_location(80, 80)
    except Exception:
        pass

    original_dispatch_event = window.dispatch_event

    def safe_dispatch_event(self, event_type, *args):
        try:
            return original_dispatch_event(event_type, *args)
        except AssertionError:
            if event_type == "on_deactivate":
                return False
            raise

    window.dispatch_event = types.MethodType(safe_dispatch_event, window)


def collect_newton_sim_trajectory(
    horizon: int,
    device: str,
    solver_type: str,
    obs_type: str,
    seed: int,
) -> torch.Tensor:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    env = create_fixed_contact_env(
        env_name="CubeTossing",
        num_envs=1,
        device=device,
        use_graph_capture=False,
        render=False,
        seed=seed,
        random_reset=True,
        solver_type=SOLVER_TYPES[solver_type],
        obs_type=obs_type,
        camera_tracking=False,
    )

    try:
        env.reset()
        rollout_states = torch.zeros(
            (horizon + 1, env.dof_q_per_env + env.dof_qd_per_env),
            dtype=torch.float32,
            device=wp.device_to_torch(env.device),
        )
        rollout_states[0].copy_(capture_env_state(env)[0])
        for step in range(horizon):
            env.update()
            rollout_states[step + 1].copy_(capture_env_state(env)[0])
        return rollout_states.detach().cpu()
    finally:
        env.close()


def build_sim2nerd_trajectories(args: argparse.Namespace) -> Sim2NeRDTrajectories:
    sim_states = collect_newton_sim_trajectory(
        horizon=args.horizon,
        device=args.device,
        solver_type=args.solver_type,
        obs_type=args.obs_type,
        seed=args.seed,
    )
    nerd_states, model_path = rollout_nerd_from_initial_state(
        sim_states[0],
        horizon=args.horizon,
        device=args.device,
        obs_type=args.obs_type,
        seed=args.seed,
        model_path_arg=args.model_path,
        cfg_path_arg=args.cfg_path,
    )
    nerd_states = nerd_states[: args.horizon + 1].detach().cpu()

    return Sim2NeRDTrajectories(
        sim_states=sim_states,
        nerd_states=nerd_states,
        model_path=model_path,
        reference_label=f"live_newton_rollout://solver={args.solver_type}/seed={args.seed}",
        horizon=args.horizon,
        state_convention="newton_free_joint_qd=[lin_world,ang_world]",
    )


def build_frame_ids(num_frames: int, frame_stride: int) -> list[int]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    frame_ids = list(range(0, num_frames, frame_stride))
    if frame_ids[-1] != num_frames - 1:
        frame_ids.append(num_frames - 1)
    return frame_ids


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    if args.sim_hdf5 is not None:
        print(
            "Ignoring --sim-hdf5. This script now collects a live Newton Sim "
            "trajectory for the requested --horizon."
        )
    if args.trajectory_index != 0:
        print("Ignoring --trajectory-index. Live sim collection does not use HDF5 indices.")

    trajectories = build_sim2nerd_trajectories(args)

    env = Sim2NeRDComparisonEnv(
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

    env.reset()
    show_viewer_window(env)
    frame_cursor = 1

    try:
        initial_frame_id = frame_ids[0]
        env.set_cube_states(
            trajectories.sim_states[initial_frame_id],
            trajectories.nerd_states[initial_frame_id],
        )
        env.set_trails(
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
                    trajectories.sim_states[frame_id],
                    trajectories.nerd_states[frame_id],
                )
                env.set_trails(
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
