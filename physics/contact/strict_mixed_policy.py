#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime policy derived from strict mixed inner-solver diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import tensorflow as tf


def _stat_as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        if isinstance(value, tf.Tensor):
            if value.shape.rank == 0:
                value = value.numpy()
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class StrictMixedRuntimePolicy:
    phase_hold: bool = False
    continuation_backoff: bool = False
    force_detach: bool = False
    traction_scale: float = 1.0
    reasons: Tuple[str, ...] = tuple()

    def as_stats(self, *, include_text: bool = True) -> Dict[str, object]:
        stats: Dict[str, object] = {
            "strict_phase_hold": float(self.phase_hold),
            "strict_continuation_backoff": float(self.continuation_backoff),
            "continuation_backoff_applied": float(self.continuation_backoff),
            "strict_force_detach": float(self.force_detach),
            "strict_traction_scale": float(self.traction_scale),
        }
        if include_text:
            stats["phase_hold_reason"] = ",".join(self.reasons)
        return stats


def resolve_strict_mixed_runtime_policy(
    diagnostics: Mapping[str, Any] | None,
    *,
    route_mode: str = "legacy",
) -> StrictMixedRuntimePolicy:
    """Convert inner-solver diagnostics into trainer/loss runtime controls."""

    route = str(route_mode or "legacy").strip().lower()
    if route == "legacy":
        return StrictMixedRuntimePolicy()

    diagnostics = diagnostics or {}
    fallback_used = _stat_as_float(diagnostics.get("fallback_used", diagnostics.get("inner_fallback_used", 0.0))) > 0.5
    skip_batch = _stat_as_float(diagnostics.get("skip_batch", diagnostics.get("inner_skip_batch", 0.0))) > 0.5
    converged = _stat_as_float(diagnostics.get("converged", diagnostics.get("inner_converged", 1.0))) > 0.5
    not_converged = (not converged) or (_stat_as_float(diagnostics.get("not_converged", 0.0)) > 0.5)
    max_penetration = _stat_as_float(diagnostics.get("max_penetration", diagnostics.get("inner_max_penetration", 0.0)))
    cone_violation = _stat_as_float(diagnostics.get("cone_violation", diagnostics.get("inner_cone_violation", 0.0)))
    fb_residual_norm = _stat_as_float(
        diagnostics.get("fb_residual_norm", diagnostics.get("inner_fb_residual_norm", 0.0))
    )
    normal_step_norm = _stat_as_float(
        diagnostics.get("normal_step_norm", diagnostics.get("inner_normal_step_norm", 0.0))
    )
    tangential_step_norm = _stat_as_float(
        diagnostics.get("tangential_step_norm", diagnostics.get("inner_tangential_step_norm", 0.0))
    )

    reasons = []
    if fallback_used:
        reasons.append("fallback")
    if skip_batch:
        reasons.append("skip_batch")
    if not_converged:
        reasons.append("not_converged")
    if max_penetration > 1.0e-3:
        reasons.append("penetration")
    if cone_violation > 1.0e-3:
        reasons.append("cone")
    if fb_residual_norm > 1.0e-2:
        reasons.append("fb_residual")
    if normal_step_norm > 1.0e-2:
        reasons.append("normal_step")
    if tangential_step_norm > 1.0e-2:
        reasons.append("tangential_step")

    severe = fallback_used or skip_batch
    unstable = severe or not_converged or max_penetration > 1.0e-3 or cone_violation > 1.0e-3 or fb_residual_norm > 1.0e-2
    caution = unstable or normal_step_norm > 1.0e-3 or tangential_step_norm > 1.0e-3

    if severe:
        traction_scale = 0.25
    elif caution:
        traction_scale = 0.5
    else:
        traction_scale = 1.0

    return StrictMixedRuntimePolicy(
        phase_hold=unstable,
        continuation_backoff=unstable,
        force_detach=unstable,
        traction_scale=traction_scale,
        reasons=tuple(reasons),
    )
