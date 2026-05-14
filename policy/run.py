import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_CFG_PATH = Path(__file__).resolve().with_name("go2_ppo.yaml")
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "Go2PPO"
DEFAULT_CHECKPOINT_NAME = "Go2PPO.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a trained Go2 PPO policy in the Newton environment."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific rl_games checkpoint (*.pth).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory under runs/Go2PPO to load from. Uses nn/Go2PPO.pth by default.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint filename to look for inside <run-dir>/nn/.",
    )
    parser.add_argument(
        "--rl-cfg",
        type=str,
        default=str(DEFAULT_RL_CFG_PATH),
        help="Fallback PPO YAML config path. Ignored when --checkpoint or --run-dir resolves to a run.",
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
        help="Number of environments to visualize. Defaults to 1 for a clean viewer.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Override rl.config.player.games_num.",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=None,
        help="Override env.max_episode_length.",
    )
    parser.add_argument(
        "--horizon-length",
        type=int,
        default=None,
        help="Override rl.config.horizon_length.",
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
        "--headless",
        action="store_true",
        help="Disable OpenGL rendering.",
    )
    parser.add_argument(
        "--disable-graph-capture",
        action="store_true",
        help="Disable Warp graph capture.",
    )
    return parser.parse_args()


def _latest_checkpoint_in_dir(nn_dir: Path) -> Path | None:
    if not nn_dir.exists():
        return None
    candidates = sorted(
        (path for path in nn_dir.glob("*.pth") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        return None
    return candidates[-1]


def _checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    nn_dir = run_dir / "nn"
    named_checkpoint = nn_dir / checkpoint_name
    if named_checkpoint.exists():
        return named_checkpoint.resolve()

    latest_checkpoint = _latest_checkpoint_in_dir(nn_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint.resolve()

    raise FileNotFoundError(f"No checkpoint (*.pth) found under {nn_dir}")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint and args.run_dir:
        raise ValueError("Use either --checkpoint or --run-dir, not both.")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if args.run_dir:
        return _checkpoint_from_run_dir(Path(args.run_dir).resolve(), args.checkpoint_name)

    if not DEFAULT_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"Could not find default run directory: {DEFAULT_RUNS_DIR}"
        )

    run_dirs = sorted(path for path in DEFAULT_RUNS_DIR.iterdir() if path.is_dir())
    for run_dir in reversed(run_dirs):
        try:
            return _checkpoint_from_run_dir(run_dir, args.checkpoint_name)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"No checkpoints found under {DEFAULT_RUNS_DIR}"
    )


def build_playback_args(
    cli_args: argparse.Namespace,
    checkpoint_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        rl_cfg=cli_args.rl_cfg,
        exp_name=None,
        no_timestamp=False,
        device=cli_args.device,
        seed=cli_args.seed,
        num_envs=cli_args.num_envs,
        max_episode_length=cli_args.max_episode_length,
        max_epochs=None,
        horizon_length=cli_args.horizon_length,
        minibatch_size=None,
        mini_epochs=None,
        learning_rate=None,
        num_games=cli_args.num_games,
        control_steps=cli_args.control_steps,
        obs_type=cli_args.obs_type,
        playback=str(checkpoint_path),
        render=not cli_args.headless,
        disable_graph_capture=cli_args.disable_graph_capture,
    )


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args)
    playback_args = build_playback_args(args, checkpoint_path)

    from train_go2_ppo import (
        RLGPUAlgoObserver,
        Runner,
        construct_env,
        construct_rlg_config,
        evaluate_policy,
        load_rl_config,
        set_random_seed,
    )

    print(f"Using checkpoint: {checkpoint_path}")

    set_random_seed(playback_args.seed)
    rl_cfg = load_rl_config(playback_args)
    env = construct_env(rl_cfg, playback_args)
    rlg_config = construct_rlg_config(rl_cfg)

    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config)
    runner.reset()

    try:
        evaluate_policy(runner, str(checkpoint_path))
    finally:
        env.close()


if __name__ == "__main__":
    main()
