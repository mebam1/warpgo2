from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def unpack_neural_model_checkpoint(
    checkpoint: Any,
) -> tuple[torch.nn.Module, str, dict[str, Any] | None]:
    if isinstance(checkpoint, dict):
        if "model" not in checkpoint or "robot_name" not in checkpoint:
            raise KeyError(
                "Checkpoint dict must contain 'model' and 'robot_name' keys."
            )
        trainer_state = checkpoint.get("trainer_state")
        return checkpoint["model"], checkpoint["robot_name"], trainer_state

    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2:
        return checkpoint[0], checkpoint[1], None

    raise TypeError(
        "Unsupported checkpoint format. Expected dict or [model, robot_name]."
    )


def load_neural_model_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device,
) -> tuple[torch.nn.Module, str, dict[str, Any] | None]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    return unpack_neural_model_checkpoint(checkpoint)
