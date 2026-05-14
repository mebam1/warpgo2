from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from convert_contact_nets_pt_to_nerd_hdf5 import (
    build_derived_fields,
    canonicalize_quaternion_xyzw,
    create_dataset,
)
from utils import state_convention
from utils import warp_utils
from utils.commons import CONTACT_DEPTH_UPPER_RATIO, get_min_contact_event_threshold
from utils.env_utils import create_fixed_contact_env
from utils.python_utils import set_random_seed
from envs.newton_envs import SolverType


SOLVER_TYPES = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated CubeTossing trajectories and write them as "
            "NeRD-compatible single-trajectory HDF5 files under "
            "mebam/data/nerd/simulation."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="mebam/data/nerd/simulation",
        help="Directory to write numbered HDF5 trajectories into.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=128,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting numeric file index. Trajectories are written as <index>.hdf5.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=111,
        help="Maximum number of frames per trajectory, including the initial frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for environment resets.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Newton device string, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="mujoco",
        choices=list(SOLVER_TYPES.keys()),
        help="Ground-truth Newton solver used for rollout generation.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="HDF5 compression to use. Pass none to disable compression.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing HDF5 files in the output directory.",
    )
    return parser.parse_args()


def compute_contact_masks(
    contact_depths: torch.Tensor,
    contact_thicknesses_0: torch.Tensor,
    contact_thicknesses_1: torch.Tensor,
    min_contact_event_threshold: float,
) -> torch.Tensor:
    threshold = CONTACT_DEPTH_UPPER_RATIO * (
        contact_thicknesses_0 + contact_thicknesses_1
    )
    threshold = torch.where(
        threshold < min_contact_event_threshold,
        torch.full_like(threshold, min_contact_event_threshold),
        threshold,
    )
    return (contact_depths < threshold).to(torch.float32)


def capture_frame(env) -> dict[str, torch.Tensor]:
    env.collision_detection(env.model, env.state)

    torch_device = wp.device_to_torch(env.device)
    states = torch.zeros(
        (env.num_envs, env.dof_q_per_env + env.dof_qd_per_env),
        dtype=torch.float32,
        device=torch_device,
    )
    warp_utils.acquire_states_to_torch(env, states)
    states = states[:1].clone()
    states[:, 3:7] = canonicalize_quaternion_xyzw(states[:, 3:7])

    root_body_q = (
        wp.to_torch(env.state.body_q)[0 :: env.bodies_per_env, :]
        .view(env.num_envs, 7)[:1]
        .clone()
    )
    root_body_q[:, 3:7] = canonicalize_quaternion_xyzw(root_body_q[:, 3:7])

    gravity_dir = torch.zeros((1, 3), dtype=torch.float32, device=torch_device)
    gravity_dir[:, env.model.up_axis] = -1.0

    num_contacts = env.num_contacts_per_env
    contact_points_0 = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_point0
    ).view(env.num_envs, num_contacts, 3)[:1].clone()
    contact_points_1 = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_point1
    ).view(env.num_envs, num_contacts, 3)[:1].clone()
    contact_normals = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_normal
    ).view(env.num_envs, num_contacts, 3)[:1].clone()
    contact_depths = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_depth
    ).view(env.num_envs, num_contacts)[:1].clone()
    contact_thicknesses_0 = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_thickness0
    ).view(env.num_envs, num_contacts)[:1].clone()
    contact_thicknesses_1 = wp.to_torch(
        env.contacts_neural_solver.rigid_contact_thickness1
    ).view(env.num_envs, num_contacts)[:1].clone()
    min_contact_event_threshold = get_min_contact_event_threshold(env.cube_half_extent)
    contact_masks = compute_contact_masks(
        contact_depths,
        contact_thicknesses_0,
        contact_thicknesses_1,
        min_contact_event_threshold,
    )

    return {
        "states": states.cpu(),
        "root_body_q": root_body_q.cpu(),
        "gravity_dir": gravity_dir.cpu(),
        "contact_points_0": contact_points_0.cpu(),
        "contact_points_1": contact_points_1.cpu(),
        "contact_normals": contact_normals.cpu(),
        "contact_depths": contact_depths.cpu(),
        "contact_thicknesses_0": contact_thicknesses_0.cpu(),
        "contact_thicknesses_1": contact_thicknesses_1.cpu(),
        "contact_masks": contact_masks.cpu(),
    }


def build_trajectory_dict(env, max_frames: int, source_name: str) -> dict:
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")

    frames: list[dict[str, torch.Tensor]] = []
    frames.append(capture_frame(env))

    done_buf = wp.zeros(env.num_envs, dtype=wp.bool, device=env.device)
    cost_buf = wp.zeros(env.num_envs, dtype=wp.float32, device=env.device)

    for frame_idx in range(1, max_frames):
        env.update()
        frames.append(capture_frame(env))

        done_buf.zero_()
        cost_buf.zero_()
        env.compute_cost_termination(
            env.state,
            env.control,
            frame_idx,
            max_frames - 1,
            cost_buf,
            done_buf,
        )
        if wp.to_torch(done_buf).any().item():
            break

    states = torch.cat([frame["states"] for frame in frames], dim=0)
    root_body_q = torch.cat([frame["root_body_q"] for frame in frames], dim=0)
    gravity_dir = torch.cat([frame["gravity_dir"] for frame in frames], dim=0)
    contact_points_0 = torch.cat([frame["contact_points_0"] for frame in frames], dim=0)
    contact_points_1 = torch.cat([frame["contact_points_1"] for frame in frames], dim=0)
    contact_normals = torch.cat([frame["contact_normals"] for frame in frames], dim=0)
    contact_depths = torch.cat([frame["contact_depths"] for frame in frames], dim=0)
    contact_thicknesses_0 = torch.cat(
        [frame["contact_thicknesses_0"] for frame in frames], dim=0
    )
    contact_thicknesses_1 = torch.cat(
        [frame["contact_thicknesses_1"] for frame in frames], dim=0
    )
    contact_masks = torch.cat([frame["contact_masks"] for frame in frames], dim=0)
    next_states = torch.cat([states[1:], states[-1:]], dim=0)

    derived_fields = build_derived_fields(
        states=states,
        next_states=next_states,
        root_body_q=root_body_q,
        gravity_dir=gravity_dir,
        contact_points_1=contact_points_1,
        contact_normals=contact_normals,
    )

    local_frame_flag = np.ones((states.shape[0], 1), dtype="float32")

    return {
        "states": derived_fields["states_body"],
        "states_world": states.numpy().astype("float32"),
        "root_body_q": root_body_q.numpy().astype("float32"),
        "next_states": derived_fields["next_states_body"],
        "next_states_world": next_states.numpy().astype("float32"),
        "gravity_dir": derived_fields["gravity_dir_body"],
        "gravity_dir_world": gravity_dir.numpy().astype("float32"),
        "contact_points_0": contact_points_0.numpy().astype("float32"),
        "contact_points_1": derived_fields["contact_points_1_body"],
        "contact_points_1_world": contact_points_1.numpy().astype("float32"),
        "contact_normals": derived_fields["contact_normals_body"],
        "contact_normals_world": contact_normals.numpy().astype("float32"),
        "contact_depths": contact_depths.numpy().astype("float32"),
        "contact_thicknesses_0": contact_thicknesses_0.numpy().astype("float32"),
        "contact_thicknesses_1": contact_thicknesses_1.numpy().astype("float32"),
        "contact_masks": contact_masks.numpy().astype("float32"),
        "_inputs_already_in_model_frame": local_frame_flag,
        **derived_fields,
        "source_file": source_name,
        "raw_num_frames": int(states.shape[0]),
        "effective_num_frames": int(max(states.shape[0] - 1, 1)),
    }


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite to replace existing files."
        )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_fixed_contact_env(
        env_name="CubeTossing",
        num_envs=1,
        device=args.device,
        use_graph_capture=False,
        render=False,
        seed=args.seed,
        random_reset=True,
        solver_type=SOLVER_TYPES[args.solver_type],
        obs_type="contact_nets",
        camera_tracking=False,
    )

    try:
        for traj_offset in range(args.num_trajectories):
            traj_index = args.start_index + traj_offset
            output_path = output_dir / f"{traj_index}.hdf5"
            ensure_output_path(output_path, overwrite=args.overwrite)

            env.reset()
            source_name = (
                f"simulation://CubeTossingEnv/solver={args.solver_type}/traj={traj_index}"
            )
            trajectory = build_trajectory_dict(
                env=env,
                max_frames=args.max_frames,
                source_name=source_name,
            )
            create_dataset(
                trajectories=[trajectory],
                output_path=output_path,
                compression=args.compression,
                split_name="all",
                dataset_attrs={
                    "state_convention": (
                        state_convention.BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION
                    ),
                    "qd_layout": "lin_body_then_ang_body",
                    "position_frame": "body",
                    "orientation_frame": "body",
                    "linear_velocity_frame": "body",
                    "angular_velocity_frame": "body",
                    "states_frame": "body",
                    "next_states_frame": "body",
                    "contact_points_0_frame": "body",
                    "contact_points_1_frame": "body",
                    "contact_normals_frame": "body",
                    "gravity_dir_frame": "body",
                    "root_body_q_frame": "world",
                    "body_anchor_step": "every",
                    "world_state_backup_key": "states_world",
                    "world_next_state_backup_key": "next_states_world",
                },
            )
            print(
                f"Wrote {output_path} with {trajectory['raw_num_frames']} frames "
                f"({trajectory['effective_num_frames']} transitions)."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
