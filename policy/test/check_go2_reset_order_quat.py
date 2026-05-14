import argparse
import math
import sys
from pathlib import Path

import torch
import warp as wp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import Go2Environment, RenderMode, SolverType
from envs.rlgames_env_wrapper import RlgamesEnvironment, eval_timeout
from utils import warp_utils
from utils.python_utils import set_random_seed


DEFAULT_RL_CFG_PATH = Path(__file__).resolve().parents[1] / "go2_ppo.yaml"
SOLVER_CLS = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


class Go2RlgAdapter:
    def __init__(
        self,
        num_envs: int,
        device: str,
        render: bool,
        newton_env_cfg: dict | None = None,
        solver_type: SolverType | None = None,
    ):
        env_cfg = dict(newton_env_cfg or {})
        env_cfg["num_envs"] = num_envs
        env_cfg["device"] = device
        env_cfg["setup_viewer"] = False
        env_cfg["use_graph_capture"] = False
        env_cfg["render_mode"] = RenderMode.OPENGL if render else RenderMode.NONE
        env_cfg["random_reset"] = False
        env_cfg["obs_type"] = "policy"
        if solver_type is not None:
            env_cfg["solver_type"] = solver_type

        self.env = Go2Environment(**env_cfg)
        self._torch_device = torch.device(wp.device_to_torch(self.env.device))

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def render_mode(self):
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, value):
        self.env.render_mode = value

    @property
    def action_dim(self):
        return self.env.control_dim

    @property
    def action_limits(self):
        return self.env.control_limits

    @property
    def control_limits(self):
        return self.env.control_limits

    @property
    def torch_device(self):
        return self._torch_device

    def setup_viewer(self):
        self.env.setup_viewer()

    def compute_observations(
        self,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        self.env.compute_observations(
            self.env.state,
            self.env.control,
            observations,
            step,
            horizon_length,
        )

    def compute_cost_termination(
        self,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        self.env.compute_cost_termination(
            self.env.state,
            self.env.control,
            step,
            traj_length,
            cost,
            terminated,
        )

    def step(self, actions: torch.Tensor):
        if actions.device != self.torch_device:
            actions = actions.to(self.torch_device)
        actions = actions.contiguous()
        self.env.assign_control(
            wp.from_torch(actions),
            self.env.control,
            self.env.state,
        )
        self.env.update()

    def reset(self):
        self.env.reset()

    def reset_envs(self, env_ids=None):
        self.env.reset_envs(env_ids)

    def get_extras(self, extras: dict):
        self.env.get_extras(extras)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Go2 RL env reset ordering around terminal observations, "
            "with a quaternion-focused check."
        )
    )
    parser.add_argument(
        "--rl-cfg",
        type=str,
        default=str(DEFAULT_RL_CFG_PATH),
        help="Path to the Go2 PPO YAML config.",
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
        "--solver-type",
        type=str,
        default=None,
        choices=list(SOLVER_CLS.keys()),
        help="Optional solver override.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="Action repeat inside the RL wrapper. Use 1 to isolate reset ordering.",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=1,
        help="Episode length used by the RL wrapper. Default 1 forces a done on the next step.",
    )
    parser.add_argument(
        "--test-yaw-deg",
        type=float,
        default=90.0,
        help="Yaw angle used to overwrite the current base quaternion before the terminal step.",
    )
    parser.add_argument(
        "--max-allowed-angle-deg",
        type=float,
        default=5.0,
        help="Allowed angle gap between the final returned obs quaternion and the pre-reset quaternion.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable OpenGL rendering during the diagnostic step.",
    )
    return parser.parse_args()


def load_rl_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonicalize_quaternion_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = quat_xyzw / torch.clamp(torch.linalg.vector_norm(quat_xyzw), min=1.0e-8)
    if quat_xyzw[3].item() < 0.0:
        quat_xyzw = -quat_xyzw
    return quat_xyzw


def quaternion_angle_deg(q0_xyzw: torch.Tensor, q1_xyzw: torch.Tensor) -> float:
    q0 = q0_xyzw / torch.clamp(torch.linalg.vector_norm(q0_xyzw), min=1.0e-8)
    q1 = q1_xyzw / torch.clamp(torch.linalg.vector_norm(q1_xyzw), min=1.0e-8)
    alignment = torch.clamp(torch.abs(torch.dot(q0, q1)), max=1.0)
    angle_rad = 2.0 * torch.acos(alignment).item()
    return math.degrees(angle_rad)


def quaternion_from_yaw_deg(yaw_deg: float, device: torch.device) -> torch.Tensor:
    yaw_rad = math.radians(yaw_deg)
    half = 0.5 * yaw_rad
    return torch.tensor(
        [0.0, 0.0, math.sin(half), math.cos(half)],
        dtype=torch.float32,
        device=device,
    )


def format_quat(quat_xyzw: torch.Tensor) -> str:
    quat_xyzw = canonicalize_quaternion_xyzw(quat_xyzw.detach().cpu())
    return "[" + ", ".join(f"{value:+.6f}" for value in quat_xyzw.tolist()) + "]"


def read_joint_quat_xyzw(go2_env: Go2Environment) -> torch.Tensor:
    joint_q = wp.to_torch(go2_env.state.joint_q).view(go2_env.num_envs, go2_env.dof_q_per_env)
    return joint_q[0, 3:7].detach().cpu().clone()


def read_body_quat_xyzw(go2_env: Go2Environment) -> torch.Tensor:
    body_q = wp.to_torch(go2_env.state.body_q).view(go2_env.num_envs, go2_env.bodies_per_env, 7)
    return body_q[0, 0, 3:7].detach().cpu().clone()


def read_policy_obs_quat_xyzw(wrapper: RlgamesEnvironment) -> torch.Tensor:
    obs = wrapper.get_observations()
    return obs[0, 1:5].detach().cpu().clone()


def overwrite_base_quaternion(
    go2_env: Go2Environment,
    quat_xyzw: torch.Tensor,
):
    torch_device = torch.device(wp.device_to_torch(go2_env.device))
    states = torch.zeros(
        (go2_env.num_envs, go2_env.dof_q_per_env + go2_env.dof_qd_per_env),
        dtype=torch.float32,
        device=torch_device,
    )
    warp_utils.acquire_states_to_torch(go2_env, states)
    q_count = go2_env.dof_q_per_env
    states[0, 3:7] = quat_xyzw.to(torch_device)
    states[0, q_count : q_count + 6] = 0.0
    warp_utils.assign_states_from_torch(go2_env, states)
    warp_utils.eval_fk(go2_env.model, go2_env.state)


def build_wrapper(args: argparse.Namespace, rl_cfg: dict) -> RlgamesEnvironment:
    env_cfg = dict(rl_cfg["env"])
    newton_env_cfg = dict(env_cfg.get("newton_env_cfg", {}))
    solver_type = SOLVER_CLS[args.solver_type] if args.solver_type is not None else None
    adapter = Go2RlgAdapter(
        num_envs=1,
        device=args.device,
        render=args.render,
        newton_env_cfg=newton_env_cfg,
        solver_type=solver_type,
    )
    return RlgamesEnvironment(
        env=adapter,
        render_mode="human" if args.render else None,
        max_episode_length=args.max_episode_length,
        reward_bias=env_cfg.get("reward_bias", 0.0),
        control_steps=args.control_steps,
        image_width=64,
        image_height=64,
    )


def run_reset_order_check(
    wrapper: RlgamesEnvironment,
    args: argparse.Namespace,
) -> int:
    go2_env = wrapper.neural_env.env
    wrapper.reset()

    initial_reset_body_quat = read_body_quat_xyzw(go2_env)
    initial_reset_obs_quat = read_policy_obs_quat_xyzw(wrapper)

    override_quat = quaternion_from_yaw_deg(
        args.test_yaw_deg,
        device=torch.device(wp.device_to_torch(go2_env.device)),
    )
    overwrite_base_quaternion(go2_env, override_quat)

    pre_step_body_quat = read_body_quat_xyzw(go2_env)
    pre_step_joint_quat = read_joint_quat_xyzw(go2_env)
    pre_step_obs_quat = read_policy_obs_quat_xyzw(wrapper)

    zero_actions = torch.zeros(
        (1, wrapper.num_actions),
        dtype=torch.float32,
        device=wrapper.neural_env.torch_device,
    )
    captured = {}
    original_reset_envs = wrapper.neural_env.reset_envs

    def instrumented_reset_envs(env_ids=None):
        captured["pre_reset_body_quat"] = read_body_quat_xyzw(go2_env)
        captured["pre_reset_joint_quat"] = read_joint_quat_xyzw(go2_env)
        captured["pre_reset_obs_quat"] = read_policy_obs_quat_xyzw(wrapper)
        return original_reset_envs(env_ids)

    wrapper.neural_env.reset_envs = instrumented_reset_envs
    try:
        obs_dict, _, dones, extras = wrapper.step(zero_actions)
    finally:
        wrapper.neural_env.reset_envs = original_reset_envs

    dones = dones.detach().cpu().clone()
    time_outs = extras["time_outs"].detach().cpu().clone()

    pre_reset_body_quat = captured.get("pre_reset_body_quat")
    pre_reset_joint_quat = captured.get("pre_reset_joint_quat")
    pre_reset_obs_quat = captured.get("pre_reset_obs_quat")
    post_reset_body_quat = read_body_quat_xyzw(go2_env)
    post_reset_joint_quat = read_joint_quat_xyzw(go2_env)
    returned_obs_quat = obs_dict["obs"][0, 1:5].detach().cpu().clone()

    if pre_reset_body_quat is None:
        pre_reset_body_quat = post_reset_body_quat.clone()
    if pre_reset_joint_quat is None:
        pre_reset_joint_quat = post_reset_joint_quat.clone()
    if pre_reset_obs_quat is None:
        pre_reset_obs_quat = returned_obs_quat.clone()

    angle_obs_to_pre_reset_body = quaternion_angle_deg(returned_obs_quat, pre_reset_body_quat)
    angle_obs_to_post_reset_body = quaternion_angle_deg(returned_obs_quat, post_reset_body_quat)
    angle_pre_reset_obs_to_pre_reset_body = quaternion_angle_deg(
        pre_reset_obs_quat, pre_reset_body_quat
    )
    angle_post_reset_body_to_initial_reset = quaternion_angle_deg(
        post_reset_body_quat, initial_reset_body_quat
    )

    print("Go2 reset-order quaternion diagnostic")
    print(
        f"device={go2_env.device} solver={go2_env.solver_type} "
        f"control_steps={wrapper.control_steps} max_episode_length={wrapper.max_episode_length}"
    )
    print(f"done={dones.tolist()} time_outs={time_outs.tolist()}")
    print("quaternion convention=xyzw, printed after canonicalization (w >= 0)")
    print(f"initial_reset_body_quat={format_quat(initial_reset_body_quat)}")
    print(f"initial_reset_obs_quat={format_quat(initial_reset_obs_quat)}")
    print(f"override_target_quat={format_quat(override_quat.detach().cpu())}")
    print(f"pre_step_body_quat={format_quat(pre_step_body_quat)}")
    print(f"pre_step_joint_quat={format_quat(pre_step_joint_quat)}")
    print(f"pre_step_obs_quat={format_quat(pre_step_obs_quat)}")
    print(f"pre_reset_body_quat={format_quat(pre_reset_body_quat)}")
    print(f"pre_reset_joint_quat={format_quat(pre_reset_joint_quat)}")
    print(f"pre_reset_obs_quat={format_quat(pre_reset_obs_quat)}")
    print(f"post_reset_body_quat={format_quat(post_reset_body_quat)}")
    print(f"post_reset_joint_quat={format_quat(post_reset_joint_quat)}")
    print(f"returned_obs_quat={format_quat(returned_obs_quat)}")
    print(
        f"angle(pre_reset_obs_quat, pre_reset_body_quat)={angle_pre_reset_obs_to_pre_reset_body:.3f} deg"
    )
    print(
        f"angle(returned_obs_quat, pre_reset_body_quat)={angle_obs_to_pre_reset_body:.3f} deg"
    )
    print(
        f"angle(returned_obs_quat, post_reset_body_quat)={angle_obs_to_post_reset_body:.3f} deg"
    )
    print(
        f"angle(post_reset_body_quat, initial_reset_body_quat)={angle_post_reset_body_to_initial_reset:.3f} deg"
    )

    if not dones[0].item():
        print("FAIL: the diagnostic step did not terminate, so reset ordering was not exercised.")
        return 2

    if angle_obs_to_pre_reset_body <= args.max_allowed_angle_deg:
        print("PASS: the final returned obs quaternion stays close to the pre-reset body quaternion.")
        return 0

    if angle_obs_to_post_reset_body + 1.0e-4 < angle_obs_to_pre_reset_body:
        print(
            "FAIL: the final returned obs quaternion is much closer to the post-reset quaternion "
            "than to the pre-reset body quaternion."
        )
        return 1

    print(
        "FAIL: the final returned obs quaternion is not within the allowed angle threshold "
        "from the pre-reset body quaternion."
    )
    return 1


def main():
    args = parse_args()
    set_random_seed(args.seed)
    rl_cfg = load_rl_config(Path(args.rl_cfg).resolve())
    wrapper = build_wrapper(args, rl_cfg)
    try:
        exit_code = run_reset_order_check(wrapper, args)
    finally:
        if getattr(wrapper.neural_env.env, "viewer", None) is not None:
            wrapper.neural_env.env.viewer.close()
        wrapper.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
