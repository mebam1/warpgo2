from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import torch
import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.newton_envs import RenderMode
from envs.newton_envs.env_cube import CubeTossingEnv
from utils import state_convention, warp_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a CubeTossing NeRD HDF5 trajectory by assigning the saved "
            "generalized states directly to the Newton environment and rendering it."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a NeRD HDF5 file, for example mebam/data/nerd/real/0.hdf5.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Trajectory index along the dataset B dimension.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth frame from the HDF5 trajectory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Playback frames per second.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Newton device string, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--camera-tracking",
        action="store_true",
        help="Enable the CubeTossing camera tracking view.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback after the last frame.",
    )
    parser.add_argument(
        "--close-on-finish",
        action="store_true",
        help=(
            "Exit when the final frame is reached. By default the viewer stays open "
            "and keeps showing the last frame."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_states(path: Path, trajectory_index: int) -> torch.Tensor:
    with h5py.File(path, "r") as h5_file:
        data_group = h5_file["data"]
        states_key = "states_world" if "states_world" in data_group else "states"
        states = data_group[states_key]
        if trajectory_index < 0 or trajectory_index >= states.shape[1]:
            raise IndexError(
                f"trajectory_index {trajectory_index} is out of range for B={states.shape[1]}."
            )
        loaded_states = torch.from_numpy(states[:, trajectory_index, :].astype("float32"))
        if states_key == "states_world":
            return loaded_states
        normalized_states, _ = state_convention.normalize_free_joint_states(
            loaded_states,
            state_convention.infer_free_joint_state_convention_from_attrs(
                data_group.attrs
            ),
        )
        return normalized_states


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


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input_path)
    states = load_states(input_path, trajectory_index=args.trajectory_index)
    if args.frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    env = CubeTossingEnv(
        num_envs=1,
        seed=0,
        random_reset=False,
        render_mode=RenderMode.OPENGL,
        device=args.device,
        camera_tracking=args.camera_tracking,
        obs_type="contact_nets",
        use_graph_capture=False,
    )

    torch_device = wp.device_to_torch(env.device)
    frame_ids = list(range(0, states.shape[0], args.frame_stride))
    if not frame_ids:
        raise ValueError(f"No frames available in {input_path}.")

    env.reset()
    show_viewer_window(env)
    frame_cursor = 0
    sleep_dt = 1.0 / args.fps

    first_state_batch = states[frame_ids[0] : frame_ids[0] + 1].to(torch_device)
    warp_utils.assign_states_from_torch(env, first_state_batch)
    env.render()

    try:
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
                state_batch = states[frame_id : frame_id + 1].to(torch_device)
                warp_utils.assign_states_from_torch(env, state_batch)
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
