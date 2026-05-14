import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
from rl_games.torch_runner import Runner

import torch._dynamo
torch._dynamo.config.suppress_errors = True


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    import gymnasium as gym

    sys.modules["gym"] = gym

from envs.newton_envs import Go2Environment, RenderMode
from envs.rlgames_env_wrapper import RLGPUAlgoObserver, register_env
from utils.python_utils import get_time_stamp, set_random_seed


DEFAULT_RL_CFG_PATH = Path(__file__).resolve().with_name("go2_ppo.yaml")


class Go2PolicyEnvironment:
    def __init__(
        self,
        num_envs: int,
        device: str,
        render: bool,
        use_graph_capture: bool,
        newton_env_cfg: dict | None = None,
    ):
        env_cfg = dict(newton_env_cfg or {})
        env_cfg["num_envs"] = num_envs
        env_cfg["device"] = device
        env_cfg["setup_viewer"] = False

        # Rendering and selective per-env resets do not stay in sync reliably when
        # the simulation graph is captured ahead of time.
        env_cfg["use_graph_capture"] = use_graph_capture and not render

        if not render:
            env_cfg.setdefault("render_mode", RenderMode.NONE)
            env_cfg.setdefault("env_offset", (0.0, 0.0, 0.0))

        self.env = Go2Environment(**env_cfg)
        self._torch_device = torch.device(wp.device_to_torch(self.env.device))
        self.policy_action_limits = np.asarray(self.env.control_limits, dtype=np.float32)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-cfg",
        type=str,
        default=str(DEFAULT_RL_CFG_PATH),
        help="Path to the PPO YAML config.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Override experiment name under runs/.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not append a timestamp to --exp-name.",
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
        default=None,
        help="Override env.num_envs.",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=None,
        help="Override env.max_episode_length.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override rl.config.max_epochs.",
    )
    parser.add_argument(
        "--horizon-length",
        type=int,
        default=None,
        help="Override rl.config.horizon_length.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Override rl.config.minibatch_size.",
    )
    parser.add_argument(
        "--mini-epochs",
        type=int,
        default=None,
        help="Override rl.config.mini_epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override rl.config.learning_rate.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Override rl.config.player.games_num.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=None,
        help="Override env.control_steps.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        choices=["policy", "joint"],
        default=None,
        help="Override newton_env_cfg.obs_type.",
    )
    parser.add_argument(
        "--playback",
        type=str,
        default=None,
        help="Path to an rl_games checkpoint (*.pth) for evaluation.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable OpenGL rendering.",
    )
    parser.add_argument(
        "--disable-graph-capture",
        action="store_true",
        help="Disable Warp graph capture.",
    )
    return parser.parse_args()


def resolve_rl_cfg_path(args: argparse.Namespace) -> Path:
    if args.playback is None:
        return Path(args.rl_cfg).resolve()

    checkpoint_path = Path(args.playback).resolve()
    run_dir = checkpoint_path.parent.parent
    rl_cfg_path = run_dir / "rl_cfg.yaml"
    if not rl_cfg_path.exists():
        raise FileNotFoundError(f"Could not find rl_cfg.yaml near checkpoint: {checkpoint_path}")
    return rl_cfg_path


def apply_cli_overrides(rl_cfg: dict, args: argparse.Namespace):
    env_cfg = rl_cfg["env"]
    algo_cfg = rl_cfg["rl"]["config"]
    player_cfg = algo_cfg.setdefault("player", {})
    newton_env_cfg = env_cfg.setdefault("newton_env_cfg", {})

    if args.playback is None:
        if args.exp_name is not None:
            if args.no_timestamp:
                algo_cfg["full_experiment_name"] = args.exp_name
            else:
                algo_cfg["full_experiment_name"] = f"{args.exp_name}/{get_time_stamp()}"
        elif "full_experiment_name" not in algo_cfg:
            algo_cfg["full_experiment_name"] = f"{env_cfg['env_name']}PPO/{get_time_stamp()}"

    if args.num_envs is not None:
        env_cfg["num_envs"] = args.num_envs
    if args.max_episode_length is not None:
        env_cfg["max_episode_length"] = args.max_episode_length
    if args.max_epochs is not None:
        algo_cfg["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        algo_cfg["horizon_length"] = args.horizon_length
    if args.minibatch_size is not None:
        algo_cfg["minibatch_size"] = args.minibatch_size
    if args.mini_epochs is not None:
        algo_cfg["mini_epochs"] = args.mini_epochs
    if args.learning_rate is not None:
        algo_cfg["learning_rate"] = args.learning_rate
    if args.num_games is not None:
        player_cfg["games_num"] = args.num_games
    if args.control_steps is not None:
        env_cfg["control_steps"] = args.control_steps
    if args.obs_type is not None:
        newton_env_cfg["obs_type"] = args.obs_type

    rl_cfg["seed"] = args.seed


def load_rl_config(args: argparse.Namespace) -> dict:
    rl_cfg_path = resolve_rl_cfg_path(args)
    with rl_cfg_path.open("r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)
    apply_cli_overrides(rl_cfg, args)
    return rl_cfg


def construct_env(rl_cfg: dict, args: argparse.Namespace) -> Go2PolicyEnvironment:
    env_cfg = deepcopy(rl_cfg["env"])
    env_name = env_cfg.get("env_name")
    if env_name != "Go2":
        raise ValueError(f"policy/train_go2_ppo.py only supports env_name='Go2', got {env_name!r}.")

    newton_env_cfg = dict(env_cfg.get("newton_env_cfg", {}))
    env = Go2PolicyEnvironment(
        num_envs=env_cfg["num_envs"],
        device=args.device,
        render=args.render,
        use_graph_capture=not args.disable_graph_capture,
        newton_env_cfg=newton_env_cfg,
    )

    register_env(
        env,
        render_mode="human" if args.render else None,
        max_episode_length=env_cfg["max_episode_length"],
        reward_bias=env_cfg.get("reward_bias", 0.0),
        control_steps=env_cfg.get("control_steps", 1),
        image_width=64,
        image_height=64,
    )
    return env


def construct_rlg_config(rl_cfg: dict) -> dict:
    other_params = rl_cfg["rl"].get("other_params", {})
    algo_config = deepcopy(rl_cfg["rl"]["config"])
    player_config = dict(algo_config.get("player", {}))
    legacy_deterministic = player_config.pop("determenistic", None)
    if legacy_deterministic is not None and "deterministic" not in player_config:
        player_config["deterministic"] = legacy_deterministic
    if "player" in algo_config or player_config:
        algo_config["player"] = player_config

    return {
        "params": {
            "seed": rl_cfg["seed"],
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {
                            "name": "const_initializer",
                            "val": other_params.get("sigma_init_val", -1.0),
                        },
                        "fixed_sigma": other_params.get("fixed_sigma", True),
                    }
                },
                **rl_cfg["rl"]["network"],
            },
            "load_checkpoint": False,
            "load_path": "",
            "config": {
                **algo_config,
                "env_name": "warp",
                "multi_gpu": False,
                "ppo": True,
                "torch_compile": algo_config.get("torch_compile", False),
                "mixed_precision": True,
                "value_bootstrap": True,
                "num_actors": rl_cfg["env"]["num_envs"],
            },
        }
    }


def save_run_config(rl_cfg: dict) -> Path:
    run_dir = REPO_ROOT / "runs" / rl_cfg["rl"]["config"]["full_experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "rl_cfg.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(rl_cfg, f, sort_keys=False)
    return run_dir


def train_policy(runner: Runner):
    runner.run({"train": True})


def evaluate_policy(runner: Runner, policy_path: str):
    runner.run(
        {
            "train": False,
            "play": True,
            "checkpoint": policy_path,
        }
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)

    rl_cfg = load_rl_config(args)
    env = construct_env(rl_cfg, args)
    rlg_config = construct_rlg_config(rl_cfg)

    if args.playback is None:
        save_run_config(rl_cfg)

    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config)
    runner.reset()

    try:
        if args.playback is None:
            train_policy(runner)
        else:
            evaluate_policy(runner, args.playback)
    finally:
        env.close()


if __name__ == "__main__":
    main()
