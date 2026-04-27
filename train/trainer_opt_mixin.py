# -*- coding: utf-8 -*-
"""Optimization/loss-step mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from model.loss_energy import TotalEnergy
from physics.contact.strict_mixed_policy import resolve_strict_mixed_runtime_policy
from train.loss_weights import combine_loss, update_loss_weights
from train import normal_contact_training_protocol


def _compute_uncertainty_proxy_sigma(
    u_pred: tf.Tensor,
    residual_scalar: tf.Tensor,
    *,
    proxy_scale: float = 1.0,
    eps: float = 1.0e-6,
) -> tf.Tensor:
    """Build residual-driven sigma proxy from predicted displacement magnitude."""

    u_pred = tf.cast(u_pred, tf.float32)
    residual_scalar = tf.cast(residual_scalar, tf.float32)
    umag = tf.sqrt(tf.reduce_sum(tf.square(u_pred), axis=1, keepdims=True) + tf.cast(eps, tf.float32))
    umag_mean = tf.reduce_mean(umag) + tf.cast(eps, tf.float32)
    sigma = tf.cast(proxy_scale, tf.float32) * residual_scalar * (umag / umag_mean) + tf.cast(eps, tf.float32)
    return sigma


def capped_continuation_update(
    eps_n: float,
    k_t: float,
    *,
    eps_factor: float = 0.7,
    k_t_factor: float = 1.3,
):
    """Apply continuation update with hard per-step caps."""

    eps_scale = max(0.7, float(eps_factor))
    kt_scale = max(0.0, min(float(k_t_factor), 1.3))
    return float(eps_n) * eps_scale, float(k_t) * kt_scale


_TANGENTIAL_TRAINING_SCALES = {
    "off": 0.0,
    "soft": 0.5,
    "full": 1.0,
}
_RISK_GUARD_EPS = 1.0e-12


def _diagnostic_as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        if isinstance(value, tf.Tensor):
            if value.shape.rank == 0:
                value = value.numpy()
        return float(value)
    except Exception:
        return float(default)


def _diagnostic_as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.string:
                return value.numpy().decode("utf-8")
            if value.shape.rank == 0:
                return str(value.numpy())
        return str(value)
    except Exception:
        return default


def _normalize_risk_guard_allowed_buckets(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset({"A", "B"})
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    normalized = [str(item).strip().upper() for item in items if str(item).strip()]
    if not normalized:
        return frozenset({"A", "B"})
    return frozenset(normalized)


def resolve_tangential_training_scale(mode: Any) -> float:
    normalized = str(mode or "full").strip().lower().replace("-", "_")
    if normalized not in _TANGENTIAL_TRAINING_SCALES:
        valid = ", ".join(sorted(_TANGENTIAL_TRAINING_SCALES))
        raise ValueError(f"Unsupported tangential_training_mode '{mode}'. Expected one of: {valid}.")
    return float(_TANGENTIAL_TRAINING_SCALES[normalized])


def classify_strict_risk_bucket(
    stats: Optional[Mapping[str, Any]],
    *,
    tol_t_default: float = 1.0e-5,
) -> str:
    diagnostics = stats or {}
    route_mode = _diagnostic_as_text(diagnostics.get("strict_route_mode"), "legacy").strip().lower()
    if route_mode == "legacy":
        return ""
    fallback_used = _diagnostic_as_float(
        diagnostics.get("fallback_used", diagnostics.get("inner_fallback_used")),
        0.0,
    )
    if fallback_used <= 0.5:
        return ""
    effective_alpha_scale = _diagnostic_as_float(diagnostics.get("effective_alpha_scale"), 0.0)
    tail_has_effective_step = _diagnostic_as_float(diagnostics.get("tail_has_effective_step"), 0.0)
    ft_residual_norm = _diagnostic_as_float(
        diagnostics.get("ft_residual_norm", diagnostics.get("inner_ft_norm")),
        0.0,
    )
    tol_t = _diagnostic_as_float(diagnostics.get("tol_t"), tol_t_default)
    if ft_residual_norm <= tol_t:
        return ""
    if abs(effective_alpha_scale) <= _RISK_GUARD_EPS and tail_has_effective_step <= 0.5:
        return "A"
    if effective_alpha_scale > _RISK_GUARD_EPS and ft_residual_norm > tol_t:
        return "B"
    return ""


def classify_strict_step_class(
    stats: Optional[Mapping[str, Any]],
    *,
    tol_t_default: float = 1.0e-5,
) -> str:
    diagnostics = stats or {}
    route_mode = _diagnostic_as_text(diagnostics.get("strict_route_mode"), "legacy").strip().lower()
    if route_mode == "legacy":
        return ""
    bucket = classify_strict_risk_bucket(diagnostics, tol_t_default=tol_t_default)
    if bucket:
        return bucket
    return "C"


def inject_bilevel_diagnostics(stats: Dict[str, Any], diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    """Inject strict-bilevel diagnostics into trainer stats with canonical keys."""

    if stats is None:
        stats = {}
    diagnostics = diagnostics or {}
    key_map = {
        "inner_fn_norm": "fn_norm",
        "inner_ft_norm": "ft_norm",
        "inner_cone_violation": "cone_violation",
        "inner_max_penetration": "max_penetration",
        "inner_fb_residual_norm": "fb_residual_norm",
        "inner_normal_step_norm": "normal_step_norm",
        "inner_tangential_step_norm": "tangential_step_norm",
        "inner_fallback_used": "fallback_used",
        "inner_converged": "converged",
        "inner_skip_batch": "skip_batch",
        "inner_convergence_rate": "inner_convergence_rate",
        "inner_fallback_rate": "inner_fallback_rate",
        "inner_skip_rate": "inner_skip_rate",
        "continuation_frozen": "continuation_frozen",
        "continuation_freeze_events": "continuation_freeze_events",
        "ift_linear_residual": "ift_linear_residual",
        "normal_ift_ready": "normal_ift_ready",
        "normal_ift_consumed": "normal_ift_consumed",
        "normal_ift_condition_metric": "normal_ift_condition_metric",
        "normal_ift_inputs_present": "normal_ift_inputs_present",
        "normal_ift_core_valid_ratio": "normal_ift_core_valid_ratio",
        "grad_u_norm": "grad_u_norm",
        "grad_sigma_norm": "grad_sigma_norm",
        "strict_phase_hold": "strict_phase_hold",
        "strict_continuation_backoff": "strict_continuation_backoff",
        "continuation_backoff_applied": "continuation_backoff_applied",
        "strict_force_detach": "strict_force_detach",
        "strict_traction_scale": "strict_traction_scale",
        "phase_hold_reason": "phase_hold_reason",
        "inner_solver_not_stable_count": "inner_solver_not_stable_count",
        "tol_t": "tol_t",
        "fallback_reason_code": "fallback_reason_code",
        "effective_alpha_scale": "effective_alpha_scale",
        "tail_has_effective_step": "tail_has_effective_step",
        "strict_step_update_scale": "strict_step_update_scale",
        "total_step_update_scale": "total_step_update_scale",
    }
    for out_key, in_key in key_map.items():
        if in_key not in diagnostics:
            continue
        value = diagnostics[in_key]
        try:
            if isinstance(value, tf.Tensor):
                if value.shape.rank == 0:
                    stats[out_key] = float(tf.cast(value, tf.float32).numpy())
                else:
                    stats[out_key] = value
            else:
                stats[out_key] = float(value)
        except Exception:
            stats[out_key] = value
    if "strict_step_update_scale" not in stats and "step_update_scale" in diagnostics:
        value = diagnostics["step_update_scale"]
        try:
            if isinstance(value, tf.Tensor):
                stats["strict_step_update_scale"] = float(tf.cast(value, tf.float32).numpy())
            else:
                stats["strict_step_update_scale"] = float(value)
        except Exception:
            stats["strict_step_update_scale"] = value
    if "ft_residual_norm" not in stats and "inner_ft_norm" in stats:
        stats["ft_residual_norm"] = stats["inner_ft_norm"]
    trace = diagnostics.get("iteration_trace")
    if isinstance(trace, Mapping):
        fallback_reason = trace.get("fallback_trigger_reason")
        if fallback_reason is not None:
            stats["fallback_trigger_reason"] = str(fallback_reason)
        iterations = trace.get("iterations")
        if isinstance(iterations, Sequence) and len(iterations) > 0 and isinstance(iterations[-1], Mapping):
            last_iter = iterations[-1]
            trace_key_map = {
                "tangential_step_mode": "tangential_step_mode",
                "effective_alpha_scale": "effective_alpha_scale",
                "tail_has_effective_step": "tail_has_effective_step",
                "ft_residual_norm": "ft_residual_after",
            }
            for out_key, in_key in trace_key_map.items():
                if in_key not in last_iter:
                    continue
                value = last_iter[in_key]
                try:
                    if isinstance(value, tf.Tensor):
                        if value.dtype == tf.string:
                            stats[out_key] = value.numpy().decode("utf-8")
                        elif value.shape.rank == 0:
                            stats[out_key] = float(tf.cast(value, tf.float32).numpy())
                        else:
                            stats[out_key] = value
                    elif isinstance(value, str):
                        stats[out_key] = value
                    else:
                        stats[out_key] = float(value)
                except Exception:
                    stats[out_key] = value
    return stats


_NORMAL_CONTACT_TIGHTENING_PROTOCOL = "progressive_normal_contact"
_NORMAL_CONTACT_TIGHTENING_STAGE_LABELS = (
    "weak_coupling_warmup",
    "transition_normal_coupling",
    "strict_normal_coupling",
)
_NORMAL_CONTACT_TIGHTENING_PHASE_INDEX = {
    "phase1": 0,
    "phase1a": 0,
    "phase1b": 0,
    "phase2": 1,
    "phase2a": 1,
    "phase2b": 1,
    "phase3": 2,
    "phase3a": 2,
    "phase3b": 2,
}


class TrainerOptMixin:
    _STRICT_MIXED_EXPERIMENTAL_PROFILES = frozenset(
        {
            "strict_mixed_experimental",
            "strict_mixed_experimental_post_reentry",
            "normal_contact_first_mainline",
        }
    )
    _STRICT_MIXED_FULL_IFT_PROFILES = frozenset({"strict_mixed_experimental_post_reentry"})
    _STRICT_MIXED_ALLOWED_KEYS = frozenset(
        {
            "R_const",
            "R_eq",
            "R_u",
            "R_t",
            "R_tr",
            "E_tight",
            "E_data",
            "E_smooth",
            "E_unc",
            "E_reg",
        }
    )
    _STRICT_MIXED_DISABLED_KEYS = frozenset(
        {
            "E_int",
            "E_cn",
            "E_ct",
            "E_sigma",
            "E_eq",
            "E_bc",
            "E_bi",
            "E_ed",
            "path_penalty_total",
            "fric_path_penalty_total",
            "R_contact_comp",
            "R_fric_comp",
        }
    )
    # Guard/soft controls are intended to damp the strict traction channel only.
    _STRICT_UPDATE_KEYS = frozenset(
        {
            "R_t",
            "R_tr",
        }
    )
    _CONTACT_BACKEND_AUTO = "auto"
    _CONTACT_BACKEND_LEGACY = "legacy_alm"
    _CONTACT_BACKEND_INNER = "inner_solver"
    _VALID_CONTACT_BACKENDS = frozenset(
        {
            _CONTACT_BACKEND_AUTO,
            _CONTACT_BACKEND_LEGACY,
            _CONTACT_BACKEND_INNER,
        }
    )

    @staticmethod
    def _stat_as_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    value = value.numpy().decode("utf-8")
                elif value.shape.rank == 0:
                    value = value.numpy()
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _stat_as_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        try:
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    return value.numpy().decode("utf-8")
                if value.shape.rank == 0:
                    return str(value.numpy())
            return str(value)
        except Exception:
            return default

    def _is_experimental_training_profile(self) -> bool:
        cfg = getattr(self, "cfg", None)
        raw = getattr(cfg, "training_profile", "locked") if cfg is not None else "locked"
        profile = str(raw or "locked").strip().lower().replace("-", "_")
        return profile in self._STRICT_MIXED_EXPERIMENTAL_PROFILES

    def _supports_full_ift_training_profile(self) -> bool:
        cfg = getattr(self, "cfg", None)
        raw = getattr(cfg, "training_profile", "locked") if cfg is not None else "locked"
        profile = str(raw or "locked").strip().lower().replace("-", "_")
        return profile in self._STRICT_MIXED_FULL_IFT_PROFILES

    def _resolve_bilevel_objective_route(self) -> str:
        if not self._is_experimental_training_profile():
            return "legacy"
        flags = getattr(self, "_mixed_phase_flags", {}) or {}
        phase_name = str(flags.get("phase_name", "phase0") or "phase0").strip().lower()
        normal_ift = bool(flags.get("normal_ift_enabled", False))
        tangential_ift = bool(flags.get("tangential_ift_enabled", False))
        allow_full_ift_warmstart = bool(flags.get("allow_full_ift_warmstart", False))
        if phase_name in {"", "phase0"}:
            return "legacy"
        if tangential_ift:
            if allow_full_ift_warmstart or self._supports_full_ift_training_profile():
                return "full_ift"
            raise NotImplementedError(
                "Strict mixed only supports forward-only or normal-ready routes; tangential/full IFT remains disabled."
            )
        if normal_ift:
            return "normal_ready"
        return "forward_only"

    def _resolve_contact_backend(self) -> str:
        route_mode = self._resolve_bilevel_objective_route()
        requested = str(getattr(getattr(self, "cfg", None), "contact_backend", self._CONTACT_BACKEND_AUTO) or "")
        requested = requested.strip().lower() or self._CONTACT_BACKEND_AUTO
        if requested not in self._VALID_CONTACT_BACKENDS:
            valid = ", ".join(sorted(self._VALID_CONTACT_BACKENDS))
            raise ValueError(f"Unsupported contact_backend '{requested}'. Expected one of: {valid}.")

        default_backend = (
            self._CONTACT_BACKEND_LEGACY
            if route_mode == "legacy"
            else self._CONTACT_BACKEND_INNER
        )
        if requested == self._CONTACT_BACKEND_AUTO:
            return default_backend
        if requested != default_backend:
            raise ValueError(
                f"contact_backend='{requested}' is incompatible with strict_route_mode='{route_mode}' "
                f"(expected '{default_backend}')."
            )
        return requested

    def _resolve_coupling_tightening_runtime_settings(
        self,
        *,
        route_mode: str,
    ) -> Dict[str, Any]:
        cfg = getattr(self, "cfg", None)
        base_refinement_steps = max(1, int(getattr(cfg, "stage_inner_steps", 1) or 1))
        base_tail_qn_iters = max(0, int(getattr(cfg, "max_tail_qn_iters", 0) or 0))
        profile = str(getattr(cfg, "training_profile", "") or "").strip().lower().replace("-", "_")
        default = {
            "protocol_enabled": False,
            "coupling_tightening_protocol": "",
            "coupling_tightening_stage_label": "",
            "coupling_tightening_stage_index": 0,
            "coupling_tightening_stage_count": 0,
            "coupling_tightening_strength": 1.0,
            "coupling_phase_traction_scale": 1.0,
            "effective_stage_inner_steps": base_refinement_steps,
            "effective_max_tail_qn_iters": base_tail_qn_iters,
        }
        if profile != "normal_contact_first_mainline":
            return default

        flags = getattr(self, "_mixed_phase_flags", {}) or {}
        if route_mode == "full_ift" and bool(flags.get("allow_full_ift_warmstart", False)):
            return default

        protocol = str(
            getattr(cfg, "coupling_tightening_protocol", _NORMAL_CONTACT_TIGHTENING_PROTOCOL)
            or _NORMAL_CONTACT_TIGHTENING_PROTOCOL
        ).strip().lower().replace("-", "_")
        if not protocol:
            protocol = _NORMAL_CONTACT_TIGHTENING_PROTOCOL

        raw_labels = getattr(cfg, "coupling_tightening_stage_labels", _NORMAL_CONTACT_TIGHTENING_STAGE_LABELS)
        labels = tuple(str(x or "").strip() for x in (raw_labels or ()))
        if len(labels) != 3 or any(not label for label in labels):
            labels = _NORMAL_CONTACT_TIGHTENING_STAGE_LABELS

        phase_name = str(flags.get("phase_name", "phase1") or "phase1").strip().lower().replace("-", "_")
        default_index = 2 if route_mode == "normal_ready" else 0
        stage_index = int(_NORMAL_CONTACT_TIGHTENING_PHASE_INDEX.get(phase_name, default_index))
        stage_index = max(0, min(stage_index, len(labels) - 1))
        stage_count = len(labels)
        strength = float(stage_index + 1) / float(stage_count) if stage_count > 0 else 1.0
        effective_stage_inner_steps = max(1, int(np.ceil(float(base_refinement_steps) * strength)))
        if base_tail_qn_iters > 0:
            effective_max_tail_qn_iters = max(1, int(np.ceil(float(base_tail_qn_iters) * strength)))
        else:
            effective_max_tail_qn_iters = 0
        return {
            "protocol_enabled": True,
            "coupling_tightening_protocol": protocol,
            "coupling_tightening_stage_label": labels[stage_index],
            "coupling_tightening_stage_index": stage_index,
            "coupling_tightening_stage_count": stage_count,
            "coupling_tightening_strength": float(strength),
            "coupling_phase_traction_scale": float(strength),
            "effective_stage_inner_steps": effective_stage_inner_steps,
            "effective_max_tail_qn_iters": effective_max_tail_qn_iters,
        }

    def _resolve_coupling_tightening_stats(
        self,
        *,
        route_mode: str,
        refinement_steps: int,
    ) -> Dict[str, Any]:
        settings = normal_contact_training_protocol.resolve_normal_contact_runtime_settings(
            self,
            route_mode=route_mode,
        )
        if not settings.get("protocol_enabled", False):
            return {}
        return {
            "coupling_tightening_protocol": str(settings["coupling_tightening_protocol"]),
            "coupling_tightening_stage_label": str(settings["coupling_tightening_stage_label"]),
            "coupling_tightening_stage_index": float(settings["coupling_tightening_stage_index"]),
            "coupling_tightening_stage_count": float(settings["coupling_tightening_stage_count"]),
            "coupling_tightening_strength": float(settings["coupling_tightening_strength"]),
            "coupling_apply_schedule": "apply_each_refinement",
            "coupling_refinement_steps": float(max(1, int(refinement_steps))),
            "coupling_phase_traction_scale": float(settings["coupling_phase_traction_scale"]),
            "coupling_tail_qn_budget": float(settings["effective_max_tail_qn_iters"]),
        }

    def _apply_route_weight_overrides(self, route_mode: str) -> Dict[str, float]:
        if route_mode == "legacy":
            self._active_weight_overrides = {}
            return self._active_weight_overrides
        overrides = {key: 0.0 for key in self._STRICT_MIXED_DISABLED_KEYS}
        self._active_weight_overrides = overrides
        return overrides

    def _evaluate_total_objective(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        stress_fn=None,
        tape=None,
    ):
        route_mode = self._resolve_bilevel_objective_route()
        self._apply_route_weight_overrides(route_mode)
        if route_mode == "legacy":
            Pi, parts, stats = total.energy(self.model.u_fn, params=params, tape=tape, stress_fn=stress_fn)
        else:
            if not hasattr(total, "assemble_strict_mixed_outer_loss"):
                raise RuntimeError(
                    "TotalEnergy.assemble_strict_mixed_outer_loss() is required for strict mixed bilevel mode."
                )
            Pi, parts, stats = total.assemble_strict_mixed_outer_loss(
                self.model.u_fn,
                params=params,
                tape=tape,
                stress_fn=stress_fn,
            )
        if stats is None:
            stats = {}
        else:
            stats = dict(stats)
        if tf.inside_function():
            stats["strict_route_mode"] = tf.constant(route_mode, dtype=tf.string)
        else:
            stats["strict_route_mode"] = route_mode
        return Pi, parts, stats

    def _accumulate_strict_bilevel_stats(
        self,
        stats: Optional[Mapping[str, Any]],
        *,
        route_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {} if stats is None else dict(stats)
        if route_mode is None:
            route_mode = self._stat_as_text(out.get("strict_route_mode"), "legacy")
        if route_mode == "legacy":
            out["strict_route_mode"] = route_mode
            out["continuation_frozen"] = float(bool(getattr(self, "_contact_hardening_frozen", False)))
            out["continuation_freeze_events"] = float(int(getattr(self, "_continuation_freeze_events", 0) or 0))
            out["continuation_backoff_applied"] = 0.0
            out["phase_hold_reason"] = ""
            out["inner_solver_not_stable_count"] = float(
                int(getattr(self, "_inner_solver_not_stable_count", 0) or 0)
            )
            return out

        counters = getattr(self, "_strict_bilevel_stats", None)
        if not isinstance(counters, dict):
            counters = {"total": 0, "converged": 0, "fallback": 0, "skipped": 0}
            self._strict_bilevel_stats = counters

        converged = self._stat_as_float(
            out.get("inner_converged", out.get("converged", 0.0)),
            0.0,
        ) > 0.5
        fallback = self._stat_as_float(
            out.get("inner_fallback_used", out.get("fallback_used", 0.0)),
            0.0,
        ) > 0.5
        skipped = self._stat_as_float(
            out.get("inner_skip_batch", out.get("mixed_strict_skipped", 0.0)),
            0.0,
        ) > 0.5

        counters["total"] = int(counters.get("total", 0)) + 1
        counters["converged"] = int(counters.get("converged", 0)) + int(converged)
        counters["fallback"] = int(counters.get("fallback", 0)) + int(fallback)
        counters["skipped"] = int(counters.get("skipped", 0)) + int(skipped)
        total_count = max(1, int(counters["total"]))

        policy = resolve_strict_mixed_runtime_policy(out, route_mode=route_mode)
        if (not converged) and (not policy.phase_hold):
            policy = resolve_strict_mixed_runtime_policy(
                {
                    **out,
                    "skip_batch": 1.0 if skipped else out.get("skip_batch", 0.0),
                    "fallback_used": 1.0 if fallback else out.get("fallback_used", 0.0),
                    "inner_skip_batch": 1.0 if skipped else out.get("inner_skip_batch", 0.0),
                    "inner_fallback_used": 1.0 if fallback else out.get("inner_fallback_used", 0.0),
                    "not_converged": 1.0,
                },
                route_mode=route_mode,
            )

        self._strict_bilevel_freeze_requested = bool(policy.phase_hold)
        self._strict_bilevel_backoff_requested = bool(policy.continuation_backoff)
        self._strict_bilevel_force_detach = bool(policy.force_detach)
        self._strict_bilevel_traction_scale = float(policy.traction_scale)
        if policy.phase_hold:
            self._inner_solver_not_stable_count = int(
                getattr(self, "_inner_solver_not_stable_count", 0) or 0
            ) + 1

        out["inner_convergence_rate"] = float(counters["converged"]) / float(total_count)
        out["inner_fallback_rate"] = float(counters["fallback"]) / float(total_count)
        out["inner_skip_rate"] = float(counters["skipped"]) / float(total_count)
        out.update(policy.as_stats())
        out["inner_solver_not_stable_count"] = float(
            int(getattr(self, "_inner_solver_not_stable_count", 0) or 0)
        )
        out["strict_route_mode"] = route_mode
        out["continuation_frozen"] = float(bool(getattr(self, "_contact_hardening_frozen", False)))
        out["continuation_freeze_events"] = float(int(getattr(self, "_continuation_freeze_events", 0) or 0))
        return out

    def _resolve_step_update_control(
        self,
        stats: Optional[Mapping[str, Any]],
        *,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        diagnostics = {} if stats is None else dict(stats)
        route_mode = self._stat_as_text(diagnostics.get("strict_route_mode"), "legacy").strip().lower()
        configured_tangential_mode = str(getattr(self.cfg, "tangential_training_mode", "full") or "full")
        normalized_tangential_mode = configured_tangential_mode.strip().lower().replace("-", "_")
        strict_step_class = classify_strict_step_class(diagnostics)
        risk_bucket_raw = classify_strict_risk_bucket(diagnostics)
        guard_bucket = risk_bucket_raw
        if risk_bucket_raw == "B":
            tail_has_effective_step = _diagnostic_as_float(
                diagnostics.get("tail_has_effective_step"),
                0.0,
            )
            if tail_has_effective_step <= 0.5:
                guard_bucket = ""
        allowed_guard_buckets = _normalize_risk_guard_allowed_buckets(
            getattr(self.cfg, "risk_guard_allowed_buckets", ("A", "B"))
        )
        if guard_bucket and guard_bucket not in allowed_guard_buckets:
            guard_bucket = ""
        if step is not None and int(step) <= 1:
            self._risk_guard_first_c_event_seen = False
        first_c_event_seen_before_step = bool(getattr(self, "_risk_guard_first_c_event_seen", False))
        protect_prefix_enabled = bool(getattr(self.cfg, "protect_prefix_enabled", False))
        protect_first_n_steps = max(0, int(getattr(self.cfg, "protect_first_n_steps", 0) or 0))
        guard_activate_after_first_c_event = bool(
            getattr(self.cfg, "guard_activate_after_first_c_event", False)
        )
        protect_by_step_count = (
            protect_prefix_enabled
            and step is not None
            and int(step) <= protect_first_n_steps
        )
        protect_by_first_c = (
            protect_prefix_enabled
            and guard_activate_after_first_c_event
            and not first_c_event_seen_before_step
        )
        protect_prefix_active = bool(protect_by_step_count or protect_by_first_c)
        protect_prefix_reason = "none"
        if protect_by_step_count:
            protect_prefix_reason = "step_count"
        elif protect_by_first_c:
            protect_prefix_reason = "legacy_first_c_event"

        tangential_mode = normalized_tangential_mode
        tangential_scale = 1.0
        if route_mode != "legacy":
            tangential_scale = resolve_tangential_training_scale(tangential_mode)
        if protect_prefix_active and route_mode != "legacy":
            tangential_mode = "full"
            tangential_scale = resolve_tangential_training_scale("full")

        guard_scale = 1.0
        guard_applied = False
        delayed_guard_ready = not protect_prefix_active
        if (
            delayed_guard_ready
            and guard_bucket
            and bool(getattr(self.cfg, "risk_guard_enabled", False))
        ):
            guard_scale = max(0.0, float(getattr(self.cfg, "risk_guard_scale", 1.0) or 0.0))
            guard_applied = guard_scale < (1.0 - _RISK_GUARD_EPS)

        first_c_event_seen_after_step = first_c_event_seen_before_step or (strict_step_class == "C")
        self._risk_guard_first_c_event_seen = bool(first_c_event_seen_after_step)
        step_scale = float(tangential_scale) * float(guard_scale)
        return {
            "tangential_training_mode": str(tangential_mode).strip().lower().replace("-", "_"),
            "tangential_update_scale": float(tangential_scale),
            "risk_bucket_raw": str(risk_bucket_raw or ""),
            "risk_bucket": guard_bucket,
            "risk_guard_applied": guard_applied,
            "risk_guard_scale": float(guard_scale),
            "step_update_scale": float(step_scale),
            "strict_step_class": str(strict_step_class or ""),
            "protect_prefix_active": bool(protect_prefix_active),
            "protect_prefix_reason": str(protect_prefix_reason),
            "first_c_event_seen_before_step": bool(first_c_event_seen_before_step),
            "first_c_event_seen_after_step": bool(first_c_event_seen_after_step),
            "delayed_guard_ready": bool(delayed_guard_ready),
        }

    def _collect_trainable_variables(self):
        m = self.model

        if hasattr(m, "trainable_variables") and m.trainable_variables:
            return self._apply_trainable_scope(m.trainable_variables)

        vars_list = []
        common_attrs = [
            "field",
            "net",
            "model",
            "encoder",
            "cond_encoder",
            "cond_enc",
            "embed",
            "embedding",
            "backbone",
            "trunk",
            "head",
            "blocks",
            "layers",
        ]
        for name in common_attrs:
            sub = getattr(m, name, None)
            if sub is None:
                continue
            if hasattr(sub, "trainable_variables"):
                vars_list += list(sub.trainable_variables)
            elif isinstance(sub, (list, tuple)):
                for layer in sub:
                    if hasattr(layer, "trainable_variables"):
                        vars_list += list(layer.trainable_variables)

        seen, out = set(), []
        for v in vars_list:
            if v is None:
                continue
            vid = id(v)
            if vid in seen:
                continue
            seen.add(vid)
            out.append(v)

        if not out:
            try:
                out = list(tf.compat.v1.trainable_variables())
            except Exception:
                out = []
        if not out:
            raise RuntimeError(
                "[trainer] Could not find trainable variables. Ensure model submodules are built."
            )
        return self._apply_trainable_scope(out)

    def _apply_trainable_scope(self, variables):
        scope = str(getattr(getattr(self, "cfg", None), "trainable_scope", "all") or "all").strip().lower()
        vars_list = list(variables or [])
        if scope in {"", "all"}:
            return vars_list
        if scope == "encoder_only":
            return list(getattr(self.model, "encoder", None).trainable_variables or [])
        if scope == "ase_adapter_only":
            selected = []
            for var in vars_list:
                name = str(getattr(var, "name", "") or "")
                if (
                    name.startswith("assembly_state_evolution_encoder/mlp_")
                    or name.startswith("assembly_state_evolution_encoder/dense_")
                ):
                    selected.append(var)
            if not selected:
                raise RuntimeError(
                    "[trainer] trainable_scope='ase_adapter_only' found no ASE adapter variables."
                )
            return selected
        raise ValueError(f"Unsupported trainable_scope='{scope}'.")

    def _uncertainty_enabled(self) -> bool:
        if float(getattr(self.cfg, "uncertainty_loss_weight", 0.0) or 0.0) <= 0.0:
            return False
        if int(getattr(self.cfg, "uncertainty_sample_points", 0) or 0) <= 0:
            return False
        try:
            out_dim = int(getattr(self.model.field.cfg, "uncertainty_out_dim", 0) or 0)
        except Exception:
            out_dim = 0
        return out_dim > 0 and hasattr(self.model, "uvar_fn") and self.elasticity is not None

    def _compute_uncertainty_proxy_loss_tf(
        self,
        params: Dict[str, Any],
        parts: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        if not self._uncertainty_enabled():
            z = tf.cast(0.0, tf.float32)
            return z, {"unc_sigma_mean": z, "unc_proxy_mean": z}

        n_req = int(getattr(self.cfg, "uncertainty_sample_points", 0) or 0)
        X_all = tf.cast(self.elasticity.X_vol_tf, tf.float32)
        n = tf.minimum(tf.shape(X_all)[0], tf.cast(n_req, tf.int32))
        X = X_all[:n]

        u_mean, log_var = self.model.uvar_fn(X, params)
        lv_min = float(getattr(self.cfg, "uncertainty_logvar_min", -8.0))
        lv_max = float(getattr(self.cfg, "uncertainty_logvar_max", 6.0))
        log_var = tf.clip_by_value(tf.cast(log_var, tf.float32), lv_min, lv_max)
        sigma_pred = tf.exp(0.5 * log_var)

        e_cn = tf.cast(parts.get("E_cn", tf.cast(0.0, tf.float32)), tf.float32)
        e_ct = tf.cast(parts.get("E_ct", tf.cast(0.0, tf.float32)), tf.float32)
        e_eq = tf.cast(parts.get("E_eq", tf.cast(0.0, tf.float32)), tf.float32)
        residual_scalar = tf.sqrt(tf.maximum(e_cn + e_ct + e_eq, 0.0) + 1.0e-12)
        residual_scalar = residual_scalar / (1.0 + residual_scalar)

        sigma_proxy = _compute_uncertainty_proxy_sigma(
            tf.cast(u_mean, tf.float32),
            residual_scalar,
            proxy_scale=float(getattr(self.cfg, "uncertainty_proxy_scale", 1.0)),
        )
        sigma_proxy = tf.broadcast_to(sigma_proxy, tf.shape(sigma_pred))
        loss_main = tf.reduce_mean(tf.square(sigma_pred - sigma_proxy))
        loss_reg = tf.cast(1.0e-3, tf.float32) * tf.reduce_mean(tf.square(log_var))
        loss_unc = loss_main + loss_reg
        stats = {
            "unc_sigma_mean": tf.reduce_mean(sigma_pred),
            "unc_proxy_mean": tf.reduce_mean(sigma_proxy),
            "unc_residual_scalar": residual_scalar,
        }
        return loss_unc, stats

    def _compute_total_loss(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        adaptive: bool = True,
    ):
        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False

        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        Pi_raw, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
        route_mode = self._stat_as_text(stats.get("strict_route_mode"), "legacy")
        stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)
        Pi = Pi_raw
        if self.loss_state is not None and adaptive:
            update_loss_weights(self.loss_state, parts, stats)
        weights = self._build_weight_vector()
        weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
        stats.update(floor_diag)
        Pi = self._loss_from_parts_and_weights(parts, weights)
        reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
        loss = Pi + reg
        return loss, Pi, parts, stats

    def _compute_total_loss_incremental(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        locked_deltas: Optional[tf.Tensor] = None,
        force_then_lock: bool = False,
        adaptive: bool = True,
    ):
        """Compute loss for a single stage with optional lock penalty."""

        del locked_deltas, force_then_lock
        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False

        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
        route_mode = self._stat_as_text(stats.get("strict_route_mode"), "legacy")
        stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)

        if self.loss_state is not None and adaptive:
            update_loss_weights(self.loss_state, parts, stats)
        weights = self._build_weight_vector()
        weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
        stats.update(floor_diag)
        Pi = self._loss_from_parts_and_weights(parts, weights)
        reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
        loss = Pi + reg
        return loss, Pi, parts, stats

    def _loss_from_parts_and_weights(self, parts: Dict[str, tf.Tensor], weights: tf.Tensor) -> tf.Tensor:
        """Combine scalar parts with a fixed weight vector (order follows self._loss_keys)."""

        loss = tf.constant(0.0, dtype=tf.float32)
        for idx, key in enumerate(getattr(self, "_loss_keys", [])):
            if key not in parts:
                continue
            val = parts[key]
            if not isinstance(val, tf.Tensor):
                continue
            if val.shape.rank != 0:
                continue
            loss = loss + tf.cast(weights[idx], tf.float32) * tf.cast(val, tf.float32)
        return loss

    def _split_loss_from_parts_and_weights(
        self,
        parts: Dict[str, tf.Tensor],
        weights: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split weighted loss into baseline-compatible and strict traction channels."""

        base_loss = tf.constant(0.0, dtype=tf.float32)
        strict_loss = tf.constant(0.0, dtype=tf.float32)
        for idx, key in enumerate(getattr(self, "_loss_keys", [])):
            if key not in parts:
                continue
            val = parts[key]
            if not isinstance(val, tf.Tensor):
                continue
            if val.shape.rank != 0:
                continue
            term = tf.cast(weights[idx], tf.float32) * tf.cast(val, tf.float32)
            if key in self._STRICT_UPDATE_KEYS:
                strict_loss = strict_loss + term
            else:
                base_loss = base_loss + term
        return base_loss, strict_loss

    @staticmethod
    def _compute_apply_gradients_kwargs(optimizer: Any) -> Dict[str, Any]:
        """Detect optimizer kwargs once to avoid per-step reflection overhead."""

        if optimizer is None:
            return {}
        try:
            sig = inspect.signature(optimizer.apply_gradients)
        except (TypeError, ValueError, AttributeError):
            return {}
        if "experimental_aggregate_gradients" in sig.parameters:
            return {"experimental_aggregate_gradients": False}
        return {}

    def _build_weight_vector_from_maps(
        self,
        weight_map: Mapping[str, Any],
        sign_map: Optional[Mapping[str, Any]] = None,
    ) -> tf.Tensor:
        keys = getattr(self, "_loss_keys", [])
        if not keys:
            return tf.zeros((0,), dtype=tf.float32)
        sign_map = sign_map or {}
        overrides = getattr(self, "_active_weight_overrides", {}) or {}
        weights = []
        for key in keys:
            if key in overrides:
                w = float(overrides.get(key, 0.0) or 0.0)
            else:
                w = float(weight_map.get(key, 0.0) or 0.0)
            sign = float(sign_map.get(key, 1.0))
            weights.append(w * sign)
        return tf.convert_to_tensor(weights, dtype=tf.float32)

    def _refresh_static_weight_vector(self):
        """Refresh cached weight vector for non-adaptive training."""

        if self.loss_state is not None:
            self._static_weight_vector = None
            return
        self._static_weight_vector = self._build_weight_vector_from_maps(getattr(self, "_base_weights", {}), {})

    def _build_weight_vector(self) -> tf.Tensor:
        """Build a weight vector aligned with self._loss_keys (sign applied)."""

        keys = getattr(self, "_loss_keys", [])
        if not keys:
            return tf.zeros((0,), dtype=tf.float32)

        route_mode = self._resolve_bilevel_objective_route()
        self._apply_route_weight_overrides(route_mode)

        if self.loss_state is not None:
            return self._build_weight_vector_from_maps(
                self.loss_state.current,
                self.loss_state.sign_overrides,
            )

        if getattr(self, "_active_weight_overrides", None):
            return self._build_weight_vector_from_maps(getattr(self, "_base_weights", {}), {})

        cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            self._refresh_static_weight_vector()
            cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            return tf.zeros((0,), dtype=tf.float32)
        return cached

    def _loss_key_index(self, key: str) -> Optional[int]:
        try:
            return list(getattr(self, "_loss_keys", [])).index(key)
        except ValueError:
            return None

    def _apply_supervision_contribution_floor(
        self,
        parts: Mapping[str, tf.Tensor],
        weights: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = tf.float32
        zero = tf.cast(0.0, dtype)
        idx_data = self._loss_key_index("E_data")
        if idx_data is None:
            return weights, {}

        data_weight = tf.cast(weights[idx_data], dtype)
        data_loss = tf.cast(parts.get("E_data", zero), dtype)
        ratio = tf.cast(
            float(getattr(self.cfg, "supervision_contribution_floor_ratio", 0.0) or 0.0),
            dtype,
        )
        enabled = bool(getattr(self.cfg, "supervision_contribution_floor_enabled", False))

        diag = {
            "data_eff_w": data_weight,
            "data_floor_active": zero,
            "data_floor_target": zero,
            "data_phys_contrib": zero,
            "data_eff_contrib": data_weight * data_loss,
        }
        if (not enabled) or float(getattr(self.cfg, "supervision_contribution_floor_ratio", 0.0) or 0.0) <= 0.0:
            return weights, diag

        phys_contrib = zero
        if any(key in parts for key in ("R_const", "R_t", "R_tr")):
            phys_keys = ("R_const", "R_t", "R_tr")
        else:
            phys_keys = ("E_sigma", "E_ct")
        for key in phys_keys:
            idx = self._loss_key_index(key)
            if idx is None:
                continue
            phys_contrib = phys_contrib + tf.abs(tf.cast(weights[idx], dtype)) * tf.abs(
                tf.cast(parts.get(key, zero), dtype)
            )

        target = ratio * phys_contrib
        safe_data_loss = tf.maximum(tf.abs(data_loss), tf.cast(1.0e-12, dtype))
        has_data = tf.abs(data_loss) > tf.cast(1.0e-12, dtype)
        required_weight = tf.where(has_data, target / safe_data_loss, data_weight)
        eff_weight = tf.where(has_data, tf.maximum(data_weight, required_weight), data_weight)
        floor_active = tf.cast(
            tf.logical_and(
                has_data,
                eff_weight > data_weight + tf.cast(1.0e-12, dtype),
            ),
            dtype,
        )
        weights = tf.tensor_scatter_nd_update(
            tf.cast(weights, dtype),
            indices=tf.constant([[idx_data]], dtype=tf.int32),
            updates=tf.reshape(eff_weight, (1,)),
        )
        diag = {
            "data_eff_w": eff_weight,
            "data_floor_active": floor_active,
            "data_floor_target": target,
            "data_phys_contrib": phys_contrib,
            "data_eff_contrib": eff_weight * data_loss,
        }
        return weights, diag

    @staticmethod
    def _is_stress_trainable_variable(var: tf.Variable) -> bool:
        name = str(getattr(var, "name", "") or "").strip().lower()
        return ("stress" in name) or ("sigma" in name)

    def _split_gradient_norm_stats(
        self,
        grads: Sequence[Optional[tf.Tensor]],
        train_vars: Sequence[tf.Variable],
    ) -> Dict[str, tf.Tensor]:
        u_grads: List[tf.Tensor] = []
        sigma_grads: List[tf.Tensor] = []
        for grad, var in zip(grads, train_vars):
            if grad is None:
                continue
            if self._is_stress_trainable_variable(var):
                sigma_grads.append(grad)
            else:
                u_grads.append(grad)
        return {
            "grad_u_norm": self._safe_global_norm(u_grads),
            "grad_sigma_norm": self._safe_global_norm(sigma_grads),
        }

    @tf.function(reduce_retracing=True)
    def _compiled_step(self, params: Dict[str, Any], weights: tf.Tensor):
        """Compiled forward+backward for the standard (non-incremental) path."""

        train_vars = self._train_vars
        opt = self.optimizer
        total = self._total_ref

        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False
        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        with tf.GradientTape() as tape:
            _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)
            eff_weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
            stats.update(floor_diag)
            loss_no_reg = self._loss_from_parts_and_weights(parts, eff_weights)
            reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
            loss_total = loss_no_reg + reg

            use_loss_scale = hasattr(opt, "get_scaled_loss")
            if use_loss_scale:
                scaled_loss = opt.get_scaled_loss(loss_total)

        if use_loss_scale:
            scaled_grads = tape.gradient(scaled_loss, train_vars)
            grads = opt.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss_total, train_vars)
        stats.update(self._split_gradient_norm_stats(grads, train_vars))

        return loss_total, loss_no_reg, parts, stats, grads

    @tf.function(reduce_retracing=True)
    def _compiled_stage_step(
        self,
        params: Dict[str, Any],
        weights: tf.Tensor,
    ):
        """Compiled forward+backward for one incremental stage."""

        train_vars = self._train_vars
        opt = self.optimizer
        total = self._total_ref

        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False
        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        with tf.GradientTape(persistent=True) as tape:
            _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)

            eff_weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
            stats.update(floor_diag)
            base_loss_no_reg, strict_loss_no_reg = self._split_loss_from_parts_and_weights(parts, eff_weights)
            loss_no_reg = base_loss_no_reg + strict_loss_no_reg
            reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
            base_loss_total = base_loss_no_reg + reg
            loss_total = base_loss_total + strict_loss_no_reg

            use_loss_scale = hasattr(opt, "get_scaled_loss")
            if use_loss_scale:
                scaled_base_loss = opt.get_scaled_loss(base_loss_total)
                scaled_strict_loss = opt.get_scaled_loss(strict_loss_no_reg)

        if use_loss_scale:
            scaled_base_grads = tape.gradient(scaled_base_loss, train_vars)
            scaled_strict_grads = tape.gradient(scaled_strict_loss, train_vars)
            base_grads = opt.get_unscaled_gradients(scaled_base_grads)
            strict_grads = opt.get_unscaled_gradients(scaled_strict_grads)
        else:
            base_grads = tape.gradient(base_loss_total, train_vars)
            strict_grads = tape.gradient(strict_loss_no_reg, train_vars)
        del tape
        grads = self._add_gradients(base_grads, strict_grads)
        stats.update(self._split_gradient_norm_stats(grads, train_vars))

        return loss_total, loss_no_reg, parts, stats, base_grads, strict_grads

    def _train_step(
        self,
        total,
        preload_case: Dict[str, np.ndarray],
        *,
        step: Optional[int] = None,
    ):
        return self._train_step_incremental(total, preload_case, step=step)

    def _train_step_incremental(
        self,
        total,
        preload_case: Dict[str, np.ndarray],
        *,
        step: Optional[int] = None,
    ):
        """Incremental Mode A: solve stages sequentially with per-stage updates."""

        opt = self.optimizer
        train_vars = self._train_vars or self._collect_trainable_variables()
        if self._total_ref is None:
            self._total_ref = total

        params_full = self._make_preload_params(preload_case)
        stage_count = self._get_stage_count(params_full)
        active_count = self._active_stage_count(step, stage_count)

        stage_mode = str(getattr(self.cfg.total_cfg, "preload_stage_mode", "") or "")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        force_then_lock = stage_mode == "force_then_lock"

        if self.contact is not None:
            self.contact.reset_multipliers(reset_reference=True)

        protocol_route_mode = self._resolve_bilevel_objective_route()
        protocol_settings = normal_contact_training_protocol.resolve_normal_contact_runtime_settings(
            self,
            route_mode=protocol_route_mode,
        )
        stage_inner_steps = max(1, int(protocol_settings.get("effective_stage_inner_steps", getattr(self.cfg, "stage_inner_steps", 1))))
        stage_alm_every = max(1, int(getattr(self.cfg, "stage_alm_every", 1)))
        use_delta_st = bool(getattr(self.contact.friction.cfg, "use_delta_st", False)) if self.contact else False

        Pi = tf.constant(0.0, dtype=tf.float32)
        parts: Dict[str, tf.Tensor] = {}
        stats: Dict[str, Any] = {}
        grad_norm = tf.constant(0.0, dtype=tf.float32)

        for stage_idx in range(active_count):
            stage_params = self._extract_stage_params(params_full, stage_idx, keep_context=True)
            if force_then_lock:
                stage_last = stage_params.get("stage_last")
                if stage_last is not None and "P" in stage_params:
                    P_cum = tf.convert_to_tensor(stage_params["P"], dtype=tf.float32)
                    stage_params = dict(stage_params)
                    stage_params["P_cumulative"] = P_cum
                    stage_params["P"] = P_cum * tf.cast(stage_last, P_cum.dtype)

            prev_params = None
            if self.contact is not None and use_delta_st and stage_idx > 0:
                prev_params = self._extract_stage_params(params_full, stage_idx - 1, keep_context=True)
                if force_then_lock:
                    prev_last = prev_params.get("stage_last")
                    if prev_last is not None and "P" in prev_params:
                        P_cum_prev = tf.convert_to_tensor(prev_params["P"], dtype=tf.float32)
                        prev_params = dict(prev_params)
                        prev_params["P_cumulative"] = P_cum_prev
                        prev_params["P"] = P_cum_prev * tf.cast(prev_last, P_cum_prev.dtype)

            if prev_params is not None:
                u_nodes = None
                if self.elasticity is not None:
                    u_nodes = self.elasticity._eval_u_on_nodes(self.model.u_fn, prev_params)
                self.contact.friction.capture_reference(self.model.u_fn, prev_params, u_nodes=u_nodes)

            for _ in range(stage_inner_steps):
                weight_vec = self._build_weight_vector()
                loss, loss_no_reg, parts, stats, base_grads, strict_grads = self._compiled_stage_step(
                    stage_params,
                    weight_vec,
                )
                route_mode = self._stat_as_text(
                    stats.get("strict_route_mode") if isinstance(stats, Mapping) else None,
                    "legacy",
                )
                stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)

                stats.update(
                    normal_contact_training_protocol.resolve_normal_contact_protocol_stats(
                        self,
                        route_mode=route_mode,
                        refinement_steps=stage_inner_steps,
                    )
                )

                if self.loss_state is not None:
                    update_loss_weights(self.loss_state, parts, stats)
                    Pi = combine_loss(parts, self.loss_state)
                else:
                    Pi = loss_no_reg

                grads = self._add_gradients(base_grads, strict_grads)
                if not any(g is not None for g in grads):
                    raise RuntimeError("[trainer] All gradients are None in incremental step.")

                update_control = self._resolve_step_update_control(stats, step=step)
                stats["tangential_training_mode"] = update_control["tangential_training_mode"]
                stats["tangential_update_scale"] = float(update_control["tangential_update_scale"])
                stats["risk_guard_bucket_raw"] = str(update_control["risk_bucket_raw"] or "")
                stats["risk_guard_bucket"] = str(update_control["risk_bucket"] or "")
                stats["risk_guard_applied"] = float(update_control["risk_guard_applied"])
                stats["risk_guard_scale"] = float(update_control["risk_guard_scale"])
                stats["strict_step_update_scale"] = float(update_control["step_update_scale"])
                stats["total_step_update_scale"] = float(
                    self._compute_total_step_update_scale(
                        base_grads,
                        strict_grads,
                        update_control["step_update_scale"],
                    )
                )
                stats["strict_step_class"] = str(update_control["strict_step_class"] or "")
                stats["protect_prefix_active"] = float(update_control["protect_prefix_active"])
                stats["protect_prefix_reason"] = str(update_control["protect_prefix_reason"] or "none")
                stats["first_c_event_seen_before_step"] = float(
                    update_control["first_c_event_seen_before_step"]
                )
                stats["first_c_event_seen_after_step"] = float(
                    update_control["first_c_event_seen_after_step"]
                )
                stats["delayed_guard_ready"] = float(update_control["delayed_guard_ready"])
                controlled_grads = self._compose_controlled_gradients(
                    base_grads,
                    strict_grads,
                    update_control["step_update_scale"],
                )
                non_none = [(g, v) for g, v in zip(controlled_grads, train_vars) if g is not None]
                g_list, v_list = zip(*non_none)
                grad_norm = self._safe_global_norm(g_list)
                grad_norm_val = float(grad_norm.numpy())
                grad_norm_finite = bool(np.isfinite(grad_norm_val))

                clip_norm = (
                    getattr(self, "clip_grad_norm", None)
                    or getattr(self, "grad_clip_norm", None)
                    or getattr(self.cfg, "clip_grad_norm", None)
                    or getattr(self.cfg, "grad_clip_norm", None)
                )
                if clip_norm is not None and float(clip_norm) > 0.0 and grad_norm_finite:
                    g_list = self._safe_clip_by_global_norm(g_list, clip_norm, grad_norm)

                loss_val = float(tf.cast(loss, tf.float32).numpy())
                if not (np.isfinite(loss_val) and grad_norm_finite):
                    continue

                opt.apply_gradients(zip(g_list, v_list), **getattr(self, "_apply_gradients_kwargs", {}))

            if stage_alm_every > 0 and ((stage_idx + 1) % stage_alm_every == 0):
                total.update_multipliers(self.model.u_fn, params=stage_params)

            if use_delta_st and self.contact is not None:
                self.contact.friction.commit_reference()

        return Pi, parts, stats, grad_norm

    def _flatten_tensor_list(
        self, tensors: Sequence[Optional[tf.Tensor]], sizes: Sequence[int]
    ) -> tf.Tensor:
        flats: List[tf.Tensor] = []
        for tensor, size in zip(tensors, sizes):
            if tensor is None:
                flats.append(tf.zeros((size,), dtype=tf.float32))
            else:
                flats.append(tf.reshape(tf.cast(tensor, tf.float32), (-1,)))
        if not flats:
            return tf.zeros((0,), dtype=tf.float32)
        return tf.concat(flats, axis=0)

    def _safe_global_norm(self, grads: Sequence[tf.Tensor]) -> tf.Tensor:
        """Compute global norm without densifying IndexedSlices."""

        def _squared_norm(g: tf.Tensor) -> tf.Tensor:
            if isinstance(g, tf.IndexedSlices):
                values = tf.cast(g.values, tf.float32)
                return tf.reduce_sum(tf.square(values))
            values = tf.cast(g, tf.float32)
            return tf.reduce_sum(tf.square(values))

        squared = [_squared_norm(g) for g in grads]
        if not squared:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.sqrt(tf.add_n(squared))

    @staticmethod
    def _scale_gradients(
        grads: Sequence[Optional[tf.Tensor]],
        scale: float,
    ) -> List[Optional[tf.Tensor]]:
        scale_tensor = tf.cast(scale, tf.float32)
        if float(scale) == 1.0:
            return list(grads)
        scaled: List[Optional[tf.Tensor]] = []
        for grad in grads:
            if grad is None:
                scaled.append(None)
                continue
            if isinstance(grad, tf.IndexedSlices):
                scaled.append(
                    tf.IndexedSlices(
                        grad.values * scale_tensor,
                        grad.indices,
                        grad.dense_shape,
                    )
                )
            else:
                scaled.append(grad * tf.cast(scale_tensor, grad.dtype))
        return scaled

    @staticmethod
    def _add_gradients(
        lhs_grads: Sequence[Optional[tf.Tensor]],
        rhs_grads: Sequence[Optional[tf.Tensor]],
    ) -> List[Optional[tf.Tensor]]:
        combined: List[Optional[tf.Tensor]] = []
        for lhs, rhs in zip(lhs_grads, rhs_grads):
            if lhs is None:
                combined.append(rhs)
                continue
            if rhs is None:
                combined.append(lhs)
                continue
            if isinstance(lhs, tf.IndexedSlices) or isinstance(rhs, tf.IndexedSlices):
                combined.append(tf.convert_to_tensor(lhs) + tf.convert_to_tensor(rhs))
            else:
                combined.append(lhs + rhs)
        return combined

    def _compose_controlled_gradients(
        self,
        base_grads: Sequence[Optional[tf.Tensor]],
        strict_grads: Sequence[Optional[tf.Tensor]],
        strict_scale: float,
    ) -> List[Optional[tf.Tensor]]:
        scaled_strict_grads = self._scale_gradients(strict_grads, strict_scale)
        return self._add_gradients(base_grads, scaled_strict_grads)

    def _compute_total_step_update_scale(
        self,
        base_grads: Sequence[Optional[tf.Tensor]],
        strict_grads: Sequence[Optional[tf.Tensor]],
        strict_scale: float,
    ) -> float:
        raw_grads = [grad for grad in self._add_gradients(base_grads, strict_grads) if grad is not None]
        if not raw_grads:
            return 1.0
        controlled_grads = [
            grad
            for grad in self._compose_controlled_gradients(base_grads, strict_grads, strict_scale)
            if grad is not None
        ]
        raw_norm = float(self._safe_global_norm(raw_grads).numpy())
        if raw_norm <= 1.0e-12:
            return 1.0
        controlled_norm = float(self._safe_global_norm(controlled_grads).numpy())
        return controlled_norm / raw_norm

    def _safe_clip_by_global_norm(
        self, grads: Sequence[tf.Tensor], clip_norm: float, global_norm: tf.Tensor
    ) -> List[tf.Tensor]:
        """Clip gradients using a precomputed global norm while keeping IndexedSlices sparse."""

        clip_norm = tf.cast(clip_norm, tf.float32)
        global_norm = tf.cast(global_norm, tf.float32)
        safe_norm = tf.maximum(global_norm, tf.constant(1e-12, dtype=tf.float32))
        scale = tf.minimum(1.0, clip_norm / safe_norm)

        clipped: List[tf.Tensor] = []
        for g in grads:
            if isinstance(g, tf.IndexedSlices):
                clipped.append(tf.IndexedSlices(g.values * scale, g.indices, g.dense_shape))
            else:
                clipped.append(g * scale)
        return clipped

    def _assign_from_flat(
        self, flat_tensor: tf.Tensor, variables: Sequence[tf.Variable], sizes: Sequence[int]
    ):
        offset = 0
        for var, size in zip(variables, sizes):
            next_offset = offset + size
            slice_tensor = tf.reshape(flat_tensor[offset:next_offset], var.shape)
            var.assign(tf.cast(slice_tensor, var.dtype))
            offset = next_offset
