# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.append(base_dir)

import warp as wp

import envs.newton_envs as newton_envs
from envs.newton_envs import RenderMode, SolverType
from utils.python_utils import set_random_seed


ENV_CLS = {
    "Cartpole": getattr(newton_envs, "CartpoleEnvironment", None),
    "Ant": getattr(newton_envs, "AntEnvironment", None),
    "CubeTossing": getattr(newton_envs, "CubeTossingEnv", None),
    "Go2": getattr(newton_envs, "Go2Environment", None),
}

SOLVER_CLS = {
    "euler": SolverType.EULER,
    "featherstone": SolverType.FEATHERSTONE,
    "mujoco": SolverType.MUJOCO,
    "xpbd": SolverType.XPBD,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        default="CubeTossing",
        choices=list(ENV_CLS.keys()),
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default=None,
        choices=list(SOLVER_CLS.keys()),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rollout-frames",
        type=int,
        default=300,
        help="Reset the environment after this many rendered frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="contact_nets",
        choices=["contact_nets", "joint", "policy"],
    )
    parser.add_argument(
        "--camera-tracking",
        action="store_true",
    )
    parser.add_argument(
        "--use-graph-capture",
        action="store_const",
        const=True,
        default=None,
        help="Override the environment default and enable Warp graph capture.",
    )
    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help=(
            "Check environment termination each frame and reset immediately. "
            "Disabled by default to avoid per-frame GPU/CPU sync in the viewer loop."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    env_cls = ENV_CLS[args.env_name]
    env_kwargs = dict(
        num_envs=args.num_envs,
        seed=args.seed,
        random_reset=True,
        render_mode=RenderMode.OPENGL,
        device=args.device,
    )
    if args.use_graph_capture is not None:
        env_kwargs["use_graph_capture"] = args.use_graph_capture

    if args.solver_type is not None:
        env_kwargs["solver_type"] = SOLVER_CLS[args.solver_type]
    if args.env_name in ("Ant", "CubeTossing", "Go2"):
        env_kwargs["camera_tracking"] = args.camera_tracking
    if args.env_name == "CubeTossing":
        env_kwargs["obs_type"] = args.obs_type
    elif args.env_name == "Go2":
        env_kwargs["obs_type"] = (
            "policy" if args.obs_type == "contact_nets" else args.obs_type
        )

    env = env_cls(**env_kwargs)
    env.reset()

    cost_buf = None
    done_buf = None
    if args.reset_on_done:
        cost_buf = wp.zeros(args.num_envs, dtype=wp.float32, device=env.device)
        done_buf = wp.zeros(args.num_envs, dtype=wp.bool, device=env.device)

    frame = 0
    try:
        while env.viewer.is_running():
            if not env.viewer.is_paused():
                env.update()
                frame += 1

                reset_now = frame >= args.rollout_frames
                if args.reset_on_done:
                    cost_buf.zero_()
                    done_buf.zero_()
                    env.compute_cost_termination(
                        env.state,
                        env.control,
                        frame,
                        args.rollout_frames,
                        cost_buf,
                        done_buf,
                    )
                    reset_now = reset_now or wp.to_torch(done_buf).any().item()

                if reset_now:
                    env.reset()
                    frame = 0

            env.render()
    finally:
        if env.viewer is not None:
            env.viewer.close()
        env.close()


if __name__ == "__main__":
    main()
