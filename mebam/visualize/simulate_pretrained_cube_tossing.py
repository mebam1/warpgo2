from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import h5py
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.neural_environment import NeuralEnvironment
from utils.checkpoint_utils import load_neural_model_checkpoint
from utils.python_utils import set_random_seed
from utils.torch_utils import num_params_torch_model
from utils import state_convention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize CubeTossing rollouts using a pretrained NeRD model checkpoint."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to a trained NeRD model checkpoint (*.pt). If omitted, the script "
            "tries pretrained_models/NeRD_models/CubeTossing first and then "
            "runs/CubeTossingSimulation."
        ),
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=None,
        help=(
            "Optional cfg.yaml path. If omitted, the script looks for cfg.yaml in the "
            "checkpoint parent training directory."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch/Newton device string.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to render.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of passive toss rollouts to play.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=300,
        help="Maximum number of steps per rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for the environment reset.",
    )
    parser.add_argument(
        "--camera-tracking",
        action="store_true",
        help="Enable camera tracking in CubeTossing viewer.",
    )
    parser.add_argument(
        "--use-graph-capture",
        action="store_true",
        help="Enable graph capture for the wrapped Newton environment.",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint"],
        help="Observation type passed to CubeTossingEnv.",
    )
    parser.add_argument(
        "--initial-state-hdf5",
        type=str,
        default=None,
        help=(
            "Optional HDF5 trajectory file. If provided, rollout resets from its first "
            "state instead of using random reset."
        ),
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Trajectory index used with --initial-state-hdf5.",
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
    candidates = sorted(REPO_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_default_model_path() -> Path:
    preferred = REPO_ROOT / "pretrained_models" / "NeRD_models" / "CubeTossing" / "model" / "nn" / "model.pt"
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


def load_initial_states(
    hdf5_path: Path,
    trajectory_index: int,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    with h5py.File(hdf5_path, "r") as h5_file:
        data_group = h5_file["data"]
        states_key = "states_world" if "states_world" in data_group else "states"
        states = data_group[states_key]
        if trajectory_index < 0 or trajectory_index >= states.shape[1]:
            raise IndexError(
                f"trajectory_index {trajectory_index} is out of range for B={states.shape[1]}."
            )
        trajectory_states = torch.from_numpy(
            states[:, trajectory_index, :].astype("float32")
        )
        if states_key == "states_world":
            return trajectory_states[0:1].repeat(num_envs, 1).to(device)
        trajectory_states, _ = state_convention.normalize_free_joint_states(
            trajectory_states,
            state_convention.infer_free_joint_state_convention_from_attrs(
                data_group.attrs
            ),
        )
    return trajectory_states[0:1].repeat(num_envs, 1).to(device)


def stabilize_viewer_window(neural_env: NeuralEnvironment) -> None:
    viewer = neural_env.env.viewer
    if viewer is None or not hasattr(viewer, "renderer"):
        return

    renderer = viewer.renderer
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


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_path = resolve_repo_path(args.model_path) if args.model_path is not None else resolve_default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    cfg_path = resolve_cfg_path(model_path, args.cfg_path)
    cfg = load_yaml(cfg_path)
    neural_solver_cfg = get_neural_solver_cfg(cfg)

    env_name = cfg.get("env", {}).get("env_name", "CubeTossing")
    if env_name != "CubeTossing":
        raise ValueError(
            f"This visualizer is dedicated to CubeTossing, but cfg env_name={env_name!r}."
        )

    neural_model, robot_name, _ = load_neural_model_checkpoint(
        model_path,
        map_location=args.device,
    )
    neural_model.to(args.device)
    if hasattr(neural_model, "fix_input_names"):
        neural_model.fix_input_names()

    newton_env_cfg = dict(cfg.get("env", {}).get("newton_env_cfg", {}))
    newton_env_cfg["obs_type"] = args.obs_type
    newton_env_cfg["camera_tracking"] = args.camera_tracking
    newton_env_cfg["random_reset"] = args.initial_state_hdf5 is None
    newton_env_cfg["seed"] = args.seed

    neural_env = NeuralEnvironment(
        env_name="CubeTossing",
        num_envs=args.num_envs,
        newton_env_cfg=newton_env_cfg,
        neural_solver_cfg=neural_solver_cfg,
        neural_model=neural_model,
        default_env_mode="neural",
        use_graph_capture=args.use_graph_capture,
        device=args.device,
        render=True,
    )
    stabilize_viewer_window(neural_env)

    if neural_env.robot_name != robot_name:
        raise ValueError(
            f"Checkpoint robot_name={robot_name!r} does not match env robot_name={neural_env.robot_name!r}."
        )

    torch_device = torch.device(args.device)
    initial_states = None
    if args.initial_state_hdf5 is not None:
        initial_states = load_initial_states(
            hdf5_path=resolve_repo_path(args.initial_state_hdf5),
            trajectory_index=args.trajectory_index,
            num_envs=args.num_envs,
            device=torch_device,
        )

    zero_actions = torch.zeros(
        (args.num_envs, neural_env.action_dim),
        device=torch_device,
    )

    print(f"Checkpoint: {model_path}")
    print(f"Config:      {cfg_path}")
    print(f"Parameters:  {num_params_torch_model(neural_model)}")
    print(f"num_envs={args.num_envs}, rollouts={args.num_rollouts}, horizon={args.rollout_horizon}")

    viewer = neural_env.env.viewer
    num_rounds = int(math.ceil(args.num_rollouts / args.num_envs))

    try:
        for _ in range(num_rounds):
            if initial_states is not None:
                neural_env.reset(initial_states.clone())
            else:
                neural_env.reset()
            neural_env.init_rnn(neural_env.num_envs)
            neural_env.render()

            for _ in range(args.rollout_horizon):
                if viewer is not None and not viewer.is_running():
                    return
                neural_env.step(zero_actions)
                neural_env.render()
    finally:
        neural_env.close()


if __name__ == "__main__":
    main()
