import argparse
import os
import sys

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if base_dir not in sys.path:
    sys.path.append(base_dir)

import torch
import warp as wp

from envs.newton_envs import Go2Environment, RenderMode, SolverType
from utils import warp_utils
from utils.python_utils import set_random_seed


SOLVER_CLS = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Go2 state.joint_qd[0:3] behaves like COM linear "
            "velocity or world-origin twist linear velocity."
        )
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-type",
        type=str,
        default=None,
        choices=list(SOLVER_CLS.keys()),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="How many zero-action frame steps to roll out after the override.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=1.0,
        help="Initial base z position. Uses [0, 10, base_height].",
    )
    parser.add_argument(
        "--keep-initial-base-quat",
        action="store_true",
        help="Keep the reset quaternion instead of forcing identity [0, 0, 0, 1].",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable OpenGL rendering during the rollout.",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> Go2Environment:
    env_kwargs = dict(
        num_envs=1,
        seed=args.seed,
        random_reset=False,
        device=args.device,
        obs_type="joint",
        render_mode=RenderMode.OPENGL if args.render else RenderMode.NONE,
        setup_viewer=args.render,
        use_graph_capture=False,
    )
    if args.solver_type is not None:
        env_kwargs["solver_type"] = SOLVER_CLS[args.solver_type]
    return Go2Environment(**env_kwargs)


def acquire_state_matrix(env: Go2Environment) -> torch.Tensor:
    torch_device = torch.device(wp.device_to_torch(env.device))
    states = torch.zeros(
        (env.num_envs, env.dof_q_per_env + env.dof_qd_per_env),
        dtype=torch.float32,
        device=torch_device,
    )
    warp_utils.acquire_states_to_torch(env, states)
    return states


def read_joint_q(env: Go2Environment) -> torch.Tensor:
    return wp.to_torch(env.state.joint_q).view(env.num_envs, env.dof_q_per_env)


def read_body_q(env: Go2Environment) -> torch.Tensor:
    return wp.to_torch(env.state.body_q).view(env.num_envs, env.bodies_per_env, 7)


def read_base_positions(env: Go2Environment) -> tuple[torch.Tensor, torch.Tensor]:
    joint_q = read_joint_q(env)[0, 0:3].detach().cpu().clone()
    body_q = read_body_q(env)[0, 0, 0:3].detach().cpu().clone()
    return joint_q, body_q


def format_vec(vec: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:+.6f}" for value in vec.tolist()) + "]"


def quat_rotate_xyzw(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat_xyzw[0:3]
    q_w = quat_xyzw[3]
    two_q_cross_v = 2.0 * torch.cross(q_xyz, vec, dim=0)
    return vec + q_w * two_q_cross_v + torch.cross(q_xyz, two_q_cross_v, dim=0)


def classify_velocity(
    measured_vel: torch.Tensor,
    expected_com_vel: torch.Tensor,
    expected_twist_vel: torch.Tensor,
) -> tuple[str, float, float]:
    measured_xy = measured_vel[0:2]
    expected_com_xy = expected_com_vel[0:2]
    expected_twist_xy = expected_twist_vel[0:2]
    dist_to_com = torch.linalg.vector_norm(measured_xy - expected_com_xy).item()
    dist_to_twist = torch.linalg.vector_norm(measured_xy - expected_twist_xy).item()
    if dist_to_com <= dist_to_twist:
        label = "closer to COM linear velocity"
    else:
        label = "closer to world-origin twist linear velocity"
    return label, dist_to_com, dist_to_twist


def print_step_result(
    label: str,
    pos: torch.Tensor,
    measured_vel: torch.Tensor,
    expected_com_vel: torch.Tensor,
    expected_twist_vel: torch.Tensor,
):
    verdict, dist_to_com, dist_to_twist = classify_velocity(
        measured_vel, expected_com_vel, expected_twist_vel
    )
    print(
        f"  {label}: pos={format_vec(pos)} "
        f"measured_vel={format_vec(measured_vel)} "
        f"xy_dist_to_com={dist_to_com:.6f} "
        f"xy_dist_to_twist={dist_to_twist:.6f} "
        f"-> {verdict}"
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)

    env = build_env(args)
    env.reset()

    try:
        warp_utils.eval_fk(env.model, env.state)
        print("Called eval_fk immediately after reset.")

        states = acquire_state_matrix(env)
        q_count = env.dof_q_per_env

        initial_quat = states[0, 3:7].detach().cpu().clone()
        if args.keep_initial_base_quat:
            base_quat_xyzw = states[0, 3:7].clone()
            quat_mode = "kept reset quaternion"
        else:
            base_quat_xyzw = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=torch.float32,
                device=states.device,
            )
            quat_mode = "forced identity quaternion"

        states[0, 0:3] = torch.tensor(
            [0.0, 10.0, args.base_height],
            dtype=torch.float32,
            device=states.device,
        )
        states[0, 3:7] = base_quat_xyzw
        states[0, q_count:] = 0.0
        states[0, q_count + 0 : q_count + 3] = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=states.device,
        )
        states[0, q_count + 3 : q_count + 6] = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=states.device,
        )

        warp_utils.assign_states_from_torch(env, states)
        warp_utils.eval_fk(env.model, env.state)
        print("Applied manual free-joint override and called eval_fk again.")

        configured_joint_q = read_joint_q(env)[0].detach().cpu().clone()
        configured_joint_qd = (
            wp.to_torch(env.state.joint_qd)
            .view(env.num_envs, env.dof_qd_per_env)[0]
            .detach()
            .cpu()
            .clone()
        )

        base_pos = configured_joint_q[0:3]
        base_quat_xyzw_cpu = configured_joint_q[3:7]
        commanded_lin_vel = configured_joint_qd[0:3]
        commanded_ang_vel_local = configured_joint_qd[3:6]
        commanded_ang_vel_world = quat_rotate_xyzw(
            base_quat_xyzw_cpu, commanded_ang_vel_local
        )
        expected_com_vel = commanded_lin_vel.clone()
        expected_twist_vel = commanded_lin_vel - torch.cross(
            base_pos, commanded_ang_vel_world, dim=0
        )

        print("Go2 free-joint qd semantics experiment")
        print(
            f"solver={env.solver_type} device={env.device} frame_dt={env.frame_dt:.6f} "
            f"sim_substeps={env.sim_substeps} use_graph_capture={env.use_graph_capture}"
        )
        print("measured_vel primary source: state.joint_q[0:3]")
        print("cross-check source: state.body_q base body position (body index 0)")
        print(f"base_quat_mode={quat_mode}")
        print(f"reset base quaternion={format_vec(initial_quat)}")
        print(f"configured base position={format_vec(base_pos)}")
        print(f"configured base quaternion={format_vec(base_quat_xyzw_cpu)}")
        print(f"configured free joint_qd[0:6]={format_vec(configured_joint_qd[0:6])}")
        print(f"expected if qd[0:3] is COM linear vel={format_vec(expected_com_vel)}")
        print(
            f"expected if qd[0:3] is world-origin twist linear="
            f"{format_vec(expected_twist_vel)}"
        )

        torch_device = torch.device(wp.device_to_torch(env.device))
        zero_actions = torch.zeros(
            (env.num_envs, env.control_dim),
            dtype=torch.float32,
            device=torch_device,
        )

        prev_joint_pos, prev_body_pos = read_base_positions(env)
        print(
            f"step_0: joint_q_pos={format_vec(prev_joint_pos)} "
            f"body_q_base_pos={format_vec(prev_body_pos)}"
        )

        joint_measured_vels = []
        body_measured_vels = []

        for step_idx in range(args.steps):
            env.assign_control(
                wp.from_torch(zero_actions),
                env.control,
                env.state,
            )
            env.update()
            if args.render:
                env.render()

            joint_pos, body_pos = read_base_positions(env)
            joint_measured_vel = (joint_pos - prev_joint_pos) / env.frame_dt
            body_measured_vel = (body_pos - prev_body_pos) / env.frame_dt
            joint_measured_vels.append(joint_measured_vel)
            body_measured_vels.append(body_measured_vel)

            print(f"step_{step_idx + 1}:")
            print_step_result(
                "joint_q[0:3]",
                joint_pos,
                joint_measured_vel,
                expected_com_vel,
                expected_twist_vel,
            )
            print_step_result(
                "body_q[base=0]",
                body_pos,
                body_measured_vel,
                expected_com_vel,
                expected_twist_vel,
            )

            prev_joint_pos = joint_pos
            prev_body_pos = body_pos

        if joint_measured_vels:
            compare_steps = min(3, len(joint_measured_vels))
            joint_mean_vel = torch.stack(joint_measured_vels[:compare_steps], dim=0).mean(
                dim=0
            )
            body_mean_vel = torch.stack(body_measured_vels[:compare_steps], dim=0).mean(
                dim=0
            )

            print(f"summary_first_{compare_steps}_steps:")
            print_step_result(
                "joint_q[0:3] mean",
                prev_joint_pos,
                joint_mean_vel,
                expected_com_vel,
                expected_twist_vel,
            )
            print_step_result(
                "body_q[base=0] mean",
                prev_body_pos,
                body_mean_vel,
                expected_com_vel,
                expected_twist_vel,
            )
    finally:
        if getattr(env, "viewer", None) is not None:
            env.viewer.close()
        env.close()


if __name__ == "__main__":
    main()
