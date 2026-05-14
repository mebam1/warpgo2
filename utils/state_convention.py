from __future__ import annotations

from typing import Any, Mapping

import torch

from utils import torch_utils


LEGACY_FREE_JOINT_STATE_CONVENTION = "legacy_contact_nets_qd=[ang_body,twist]"
NEWTON_FREE_JOINT_STATE_CONVENTION = "newton_free_joint_qd=[lin_world,ang_world]"
BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION = (
    "body_anchor_free_joint_qd=[lin_body,ang_body]"
)
UNKNOWN_STATE_CONVENTION = "unknown"


def _decode_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _get_attr(attrs: Mapping[str, Any] | None, key: str) -> Any:
    if attrs is None:
        return None
    if key in attrs:
        return _decode_attr_value(attrs[key])
    return None


def canonicalize_state_convention_label(label: str | None) -> str:
    if label is None:
        return UNKNOWN_STATE_CONVENTION
    if label in (
        LEGACY_FREE_JOINT_STATE_CONVENTION,
        "legacy_contact_nets_qd=[ang,twist]",
    ):
        return LEGACY_FREE_JOINT_STATE_CONVENTION
    if label in (
        NEWTON_FREE_JOINT_STATE_CONVENTION,
        "newton_qd=[lin,ang]",
    ):
        return NEWTON_FREE_JOINT_STATE_CONVENTION
    if label in (
        BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION,
        "body_free_joint_qd=[lin_body,ang_body]",
        "body_qd=[lin,ang]",
    ):
        return BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION
    return label


def infer_free_joint_state_convention_from_attrs(
    attrs: Mapping[str, Any] | None,
) -> str:
    state_convention = canonicalize_state_convention_label(
        _get_attr(attrs, "state_convention")
    )
    if state_convention != UNKNOWN_STATE_CONVENTION:
        return state_convention

    qd_layout = _get_attr(attrs, "qd_layout")
    if qd_layout == "lin_world_then_ang_world":
        return NEWTON_FREE_JOINT_STATE_CONVENTION
    if qd_layout == "ang_body_then_twist":
        return LEGACY_FREE_JOINT_STATE_CONVENTION
    if qd_layout == "lin_body_then_ang_body":
        return BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION

    linear_velocity_frame = _get_attr(attrs, "linear_velocity_frame")
    angular_velocity_frame = _get_attr(attrs, "angular_velocity_frame")
    if linear_velocity_frame == "world" and angular_velocity_frame == "world":
        return NEWTON_FREE_JOINT_STATE_CONVENTION
    if linear_velocity_frame == "body" and angular_velocity_frame == "body":
        return BODY_ANCHOR_FREE_JOINT_STATE_CONVENTION
    if angular_velocity_frame == "body":
        return LEGACY_FREE_JOINT_STATE_CONVENTION

    return UNKNOWN_STATE_CONVENTION


def recover_linear_velocity_from_legacy_states(states: torch.Tensor) -> torch.Tensor:
    shape = states.shape
    flat_states = states.reshape(-1, shape[-1])
    pos = flat_states[:, 0:3]
    ang_body = flat_states[:, 7:10]
    twist = flat_states[:, 10:13]
    lin_world = twist - torch.cross(pos, ang_body, dim=-1)
    return lin_world.view(*shape[:-1], 3)


def convert_legacy_states_to_newton(states: torch.Tensor) -> torch.Tensor:
    shape = states.shape
    flat_states = states.reshape(-1, shape[-1])
    pos = flat_states[:, 0:3]
    quat = flat_states[:, 3:7]
    ang_body = flat_states[:, 7:10]
    lin_world = recover_linear_velocity_from_legacy_states(flat_states)
    ang_world = torch_utils.quat_rotate(quat, ang_body)
    converted = torch.cat([pos, quat, lin_world, ang_world], dim=-1)
    return converted.view(*shape[:-1], shape[-1])


def _fit_displacement_residual(
    displacements: torch.Tensor,
    candidate_velocity: torch.Tensor,
) -> float:
    denom = float((candidate_velocity * candidate_velocity).sum().item())
    if denom < 1.0e-12:
        return float("inf")
    alpha = float((displacements * candidate_velocity).sum().item()) / denom
    residual = displacements - alpha * candidate_velocity
    return float(torch.linalg.norm(residual, dim=-1).mean().item())


def infer_free_joint_state_convention(states: torch.Tensor) -> str:
    if states.shape[-1] < 13:
        return UNKNOWN_STATE_CONVENTION

    if states.ndim == 2:
        sequence = states
    elif states.ndim >= 3:
        sequence = states[:, 0, :]
    else:
        return UNKNOWN_STATE_CONVENTION

    if sequence.shape[0] < 2:
        return UNKNOWN_STATE_CONVENTION

    num_samples = min(int(sequence.shape[0]) - 1, 8)
    displacements = sequence[1 : num_samples + 1, 0:3] - sequence[:num_samples, 0:3]
    newton_linear = sequence[:num_samples, 7:10]
    legacy_linear = recover_linear_velocity_from_legacy_states(sequence[:num_samples])

    newton_residual = _fit_displacement_residual(displacements, newton_linear)
    legacy_residual = _fit_displacement_residual(displacements, legacy_linear)
    if legacy_residual < newton_residual:
        return LEGACY_FREE_JOINT_STATE_CONVENTION
    return NEWTON_FREE_JOINT_STATE_CONVENTION


def normalize_free_joint_states(
    states: torch.Tensor,
    state_convention: str | None = None,
) -> tuple[torch.Tensor, str]:
    convention = canonicalize_state_convention_label(state_convention)
    if convention == UNKNOWN_STATE_CONVENTION:
        convention = infer_free_joint_state_convention(states)

    if convention == LEGACY_FREE_JOINT_STATE_CONVENTION:
        return convert_legacy_states_to_newton(states), convention

    return states, convention
