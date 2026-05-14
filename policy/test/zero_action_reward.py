import argparse
import sys
from pathlib import Path

import torch
import warp as wp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import Go2Environment, RenderMode, SolverType
from utils.python_utils import set_random_seed


DEFAULT_RL_CFG_PATH = Path(__file__).resolve().parents[1] / "go2_ppo.yaml"
SOLVER_CLS = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Go2 reward under zero action starting from the default pose "
            "(random_reset=False)."
        )
    )
    parser.add_argument(
        "--rl-cfg",
        type=str,
        default=str(DEFAULT_RL_CFG_PATH),
        help="Path to the PPO YAML config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Warp/Torch device string.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="How many zero-action simulation steps to roll out after reset.",
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default=None,
        choices=list(SOLVER_CLS.keys()),
        help="Optional solver override.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default=None,
        choices=["policy", "joint"],
        help="Optional newton_env_cfg.obs_type override.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable OpenGL rendering during the rollout.",
    )
    parser.add_argument(
        "--disable-graph-capture",
        action="store_true",
        help="Deprecated: graph capture is currently disabled for this test.",
    )
    parser.add_argument(
        "--print-up-vec",
        action="store_true",
        help="Print the base up vector used by Go2 reward computation.",
    )
    return parser.parse_args()


def load_rl_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(args: argparse.Namespace, rl_cfg: dict) -> Go2Environment:
    env_cfg = dict(rl_cfg["env"].get("newton_env_cfg", {}))
    env_cfg["seed"] = args.seed
    env_cfg["random_reset"] = False
    env_cfg["num_envs"] = args.num_envs
    env_cfg["device"] = args.device
    env_cfg["render_mode"] = RenderMode.OPENGL if args.render else RenderMode.NONE
    env_cfg["setup_viewer"] = args.render
    env_cfg["use_graph_capture"] = False

    if args.obs_type is not None:
        env_cfg["obs_type"] = args.obs_type
    if args.solver_type is not None:
        env_cfg["solver_type"] = SOLVER_CLS[args.solver_type]

    return Go2Environment(**env_cfg)


def compute_rewards(
    env: Go2Environment,
    step: int,
    traj_length: int,
    reward_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cost_buf = wp.zeros(env.num_envs, dtype=wp.float32, device=env.device)
    done_buf = wp.zeros(env.num_envs, dtype=wp.bool, device=env.device)
    env.compute_cost_termination(
        env.state,
        env.control,
        step,
        traj_length,
        cost_buf,
        done_buf,
    )
    cost = wp.to_torch(cost_buf).clone()
    done = wp.to_torch(done_buf).clone()
    reward = -cost + reward_bias
    return reward, cost, done


def quat_rotate_xyzw(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat_xyzw[:, 0:3]
    quat_w = quat_xyzw[:, 3:4]
    two_q_cross_v = 2.0 * torch.cross(quat_xyz, vec, dim=1)
    return vec + quat_w * two_q_cross_v + torch.cross(quat_xyz, two_q_cross_v, dim=1)


def compute_base_up_vec(env: Go2Environment) -> torch.Tensor:
    joint_q = wp.to_torch(env.state.joint_q).view(env.num_envs, env.dof_q_per_env)
    quat_xyzw = joint_q[:, 3:7]
    quat_xyzw = quat_xyzw / torch.clamp(
        torch.linalg.vector_norm(quat_xyzw, dim=1, keepdim=True),
        min=1.0e-8,
    )
    local_up = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=quat_xyzw.device)
    local_up[:, 2] = 1.0
    return quat_rotate_xyzw(quat_xyzw, local_up)


def format_vec(vec: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:+.6f}" for value in vec.detach().cpu().tolist()) + "]"


def print_stats(
    label: str,
    reward: torch.Tensor,
    cost: torch.Tensor,
    done: torch.Tensor,
    up_vec: torch.Tensor | None = None,
):
    reward_cpu = reward.detach().cpu()
    cost_cpu = cost.detach().cpu()
    done_cpu = done.detach().cpu()
    msg = (
        f"{label}: "
        f"reward_mean={reward_cpu.mean().item():+.6f} "
        f"reward_min={reward_cpu.min().item():+.6f} "
        f"reward_max={reward_cpu.max().item():+.6f} "
        f"cost_mean={cost_cpu.mean().item():+.6f} "
        f"done={done_cpu.tolist()}"
    )
    if up_vec is not None:
        up_cpu = up_vec.detach().cpu()
        msg += (
            f" up_z_mean={up_cpu[:, 2].mean().item():+.6f} "
            f"up_z_min={up_cpu[:, 2].min().item():+.6f} "
            f"up_z_max={up_cpu[:, 2].max().item():+.6f} "
            f"up_vec_env0={format_vec(up_cpu[0])}"
        )
    print(msg)


def main():
    args = parse_args()
    rl_cfg = load_rl_config(Path(args.rl_cfg).resolve())
    reward_bias = float(rl_cfg["env"].get("reward_bias", 0.0))

    set_random_seed(args.seed)
    env = build_env(args, rl_cfg)
    env.reset()

    torch_device = torch.device(wp.device_to_torch(env.device))
    zero_actions = torch.zeros(
        (env.num_envs, env.control_dim),
        dtype=torch.float32,
        device=torch_device,
    )

    print("Zero-action reward from default pose")
    print(f"device={env.device} num_envs={env.num_envs} steps={args.steps}")
    print(f"reward_bias={reward_bias:+.6f}")

    try:
        reward, cost, done = compute_rewards(
            env,
            step=0,
            traj_length=max(1, args.steps),
            reward_bias=reward_bias,
        )
        up_vec = compute_base_up_vec(env) if args.print_up_vec else None
        print_stats("initial_state", reward, cost, done, up_vec)

        cumulative_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=torch_device)
        for step_idx in range(args.steps):
            env.assign_control(
                wp.from_torch(zero_actions),
                env.control,
                env.state,
            )
            env.update()
            if args.render:
                env.render()

            reward, cost, done = compute_rewards(
                env,
                step=step_idx + 1,
                traj_length=max(1, args.steps),
                reward_bias=reward_bias,
            )
            cumulative_reward += reward
            up_vec = compute_base_up_vec(env) if args.print_up_vec else None
            print_stats(f"step_{step_idx + 1}", reward, cost, done, up_vec)

        print(
            "rollout_summary: "
            f"cumulative_reward_mean={cumulative_reward.mean().item():+.6f} "
            f"cumulative_reward_min={cumulative_reward.min().item():+.6f} "
            f"cumulative_reward_max={cumulative_reward.max().item():+.6f}"
        )
    finally:
        if getattr(env, "viewer", None) is not None:
            env.viewer.close()
        env.close()


if __name__ == "__main__":
    main()
