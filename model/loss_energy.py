#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Total energy assembly for PINN with contact + tightening."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

from model.pinn_model import (
    CONTACT_SURFACE_NORMALS_KEY,
    CONTACT_SURFACE_T1_KEY,
    CONTACT_SURFACE_T2_KEY,
    INNER_CONTACT_GAP_N_KEY,
    INNER_CONTACT_LAMBDA_N_KEY,
    INNER_CONTACT_NORMALS_KEY,
    INNER_CONTACT_WEIGHTS_KEY,
)
from model import mixed_residual_coupling
from physics.elasticity_residual import ElasticityResidual
from physics.elasticity_config import ElasticityConfig
from physics.contact.contact_operator import (
    ContactOperator,
    ContactOperatorConfig,
    traction_matching_terms,
)
from physics.contact.strict_mixed_policy import resolve_strict_mixed_runtime_policy
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig, traction_bc_residual
from physics.tightening_model import NutTighteningPenalty, TighteningConfig
from physics.traction_utils import normal_tangential_components
from train.trainer_supervision_features import RingFeatureConfig, compute_ring_coordinate_components_tf


def compute_incremental_ed_penalty(
    delta_elastic: tf.Tensor,
    friction_dissipation: tf.Tensor,
    external_work_proxy: tf.Tensor,
    *,
    margin: tf.Tensor,
    use_relu: bool = True,
    squared: bool = True,
) -> tf.Tensor:
    """Incremental energy-dissipation mismatch penalty.

    raw = delta_elastic + friction_dissipation - external_work_proxy - margin
    penalty = relu(raw)^2 (default)
    """

    raw = delta_elastic + friction_dissipation - external_work_proxy - margin
    if use_relu:
        pen = tf.nn.relu(raw)
    else:
        pen = tf.abs(raw)
    if squared:
        pen = pen * pen
    return pen


def traction_bc_residual_from_model(model, X, params, normals, target_t):
    """Compute mixed traction BC residual using the model stress head only."""

    sigma_vec = model.sigma_fn(X, params)
    return traction_bc_residual(sigma_vec, normals, target_t)


def traction_matching_residual(sigma_s, sigma_m, n, t1, t2, inner_result):
    """Mixed contact residual via traction matching against inner solve result."""

    return traction_matching_terms(sigma_s, sigma_m, n, t1, t2, inner_result)


@dataclass
class TotalConfig:
    loss_mode: str = "energy"  # "energy" | "residual"
    w_int: float = 1.0
    w_cn: float = 1.0
    w_ct: float = 1.0
    w_bc: float = 1.0
    w_tight: float = 1.0
    w_sigma: float = 1.0
    w_eq: float = 0.0
    w_reg: float = 0.0
    w_bi: float = 0.0
    w_ed: float = 0.0
    w_unc: float = 0.0
    w_data: float = 0.0
    w_delta_data: float = 0.0
    w_optical_modal: float = 0.0
    w_smooth: float = 0.0
    sigma_ref: float = 1.0
    data_smoothing_k: int = 0
    data_weight_enabled: bool = False
    data_weight_blend: float = 1.0
    data_weight_power: float = 1.0
    optical_modal_enabled: bool = False
    optical_modal_center: Tuple[float, float] = (0.0, 0.0)
    optical_modal_r_in: float = 0.0
    optical_modal_r_out: float = 1.0
    optical_modal_radial_order: int = 2
    optical_modal_fourier_order: int = 6
    optical_modal_target_component: int = 2
    path_penalty_weight: float = 1.0
    fric_path_penalty_weight: float = 1.0
    ed_enabled: bool = False
    ed_external_scale: float = 1.0
    ed_margin: float = 0.0
    ed_use_relu: bool = True
    ed_square: bool = True
    adaptive_scheme: str = "contact_only"
    update_every_steps: int = 150
    dtype: str = "float32"


class TotalEnergy:
    STRICT_MIXED_ACTIVE_KEYS = (
        "R_const",
        "R_eq",
        "R_u",
        "R_t",
        "R_tr",
        "E_tight",
        "E_data",
        "E_delta_data",
        "E_optical_modal",
        "E_smooth",
        "E_unc",
        "E_reg",
    )
    STRICT_MIXED_ZERO_KEYS = (
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
    )

    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64
        self.elasticity: Optional[ElasticityResidual] = None
        self.contact: Optional[ContactOperator] = None
        self.bcs: List[BoundaryPenalty] = []
        self.tightening: Optional[NutTighteningPenalty] = None
        self._ensure_weight_vars()
        self._built = False
        self.mixed_bilevel_flags = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
            "allow_full_ift_warmstart": False,
        }
        self._strict_mixed_last_active = False

    def _ensure_weight_vars(self):
        if not hasattr(self, "w_int"):
            self.w_int = tf.Variable(self.cfg.w_int, dtype=self.dtype, trainable=False, name="w_int")
        if not hasattr(self, "w_cn"):
            self.w_cn = tf.Variable(self.cfg.w_cn, dtype=self.dtype, trainable=False, name="w_cn")
        if not hasattr(self, "w_ct"):
            self.w_ct = tf.Variable(self.cfg.w_ct, dtype=self.dtype, trainable=False, name="w_ct")
        if not hasattr(self, "w_bc"):
            self.w_bc = tf.Variable(self.cfg.w_bc, dtype=self.dtype, trainable=False, name="w_bc")
        if not hasattr(self, "w_tight"):
            self.w_tight = tf.Variable(self.cfg.w_tight, dtype=self.dtype, trainable=False, name="w_tight")
        if not hasattr(self, "w_sigma"):
            self.w_sigma = tf.Variable(self.cfg.w_sigma, dtype=self.dtype, trainable=False, name="w_sigma")
        if not hasattr(self, "w_eq"):
            self.w_eq = tf.Variable(self.cfg.w_eq, dtype=self.dtype, trainable=False, name="w_eq")
        if not hasattr(self, "w_reg"):
            self.w_reg = tf.Variable(self.cfg.w_reg, dtype=self.dtype, trainable=False, name="w_reg")
        if not hasattr(self, "w_bi"):
            self.w_bi = tf.Variable(getattr(self.cfg, "w_bi", 0.0), dtype=self.dtype, trainable=False, name="w_bi")
        if not hasattr(self, "w_ed"):
            self.w_ed = tf.Variable(getattr(self.cfg, "w_ed", 0.0), dtype=self.dtype, trainable=False, name="w_ed")
        if not hasattr(self, "w_unc"):
            self.w_unc = tf.Variable(getattr(self.cfg, "w_unc", 0.0), dtype=self.dtype, trainable=False, name="w_unc")
        if not hasattr(self, "w_data"):
            self.w_data = tf.Variable(getattr(self.cfg, "w_data", 0.0), dtype=self.dtype, trainable=False, name="w_data")
        if not hasattr(self, "w_delta_data"):
            self.w_delta_data = tf.Variable(
                getattr(self.cfg, "w_delta_data", 0.0),
                dtype=self.dtype,
                trainable=False,
                name="w_delta_data",
            )
        if not hasattr(self, "w_optical_modal"):
            self.w_optical_modal = tf.Variable(
                getattr(self.cfg, "w_optical_modal", 0.0),
                dtype=self.dtype,
                trainable=False,
                name="w_optical_modal",
            )
        if not hasattr(self, "w_smooth"):
            self.w_smooth = tf.Variable(getattr(self.cfg, "w_smooth", 0.0), dtype=self.dtype, trainable=False, name="w_smooth")

    def _loss_mode(self) -> str:
        mode = str(getattr(self.cfg, "loss_mode", "energy") or "energy").strip().lower()
        if mode in {"residual", "residual_only", "res"}:
            return "residual"
        return "energy"

    @staticmethod
    def _resolve_bound_variant(fn, method_name: str):
        """Use an alternate bound method when available (e.g. pointwise forward)."""

        if fn is None:
            return None
        owner = getattr(fn, "__self__", None)
        if owner is None:
            return fn
        alt = getattr(owner, method_name, None)
        if callable(alt):
            return alt
        return fn

    def set_mixed_bilevel_flags(self, flags: Optional[Dict[str, object]] = None):
        """Attach trainer-resolved mixed-bilevel phase flags to this total-energy assembly."""

        merged = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
            "allow_full_ift_warmstart": False,
            "tangential_training_mode": "full",
            "risk_guard_enabled": False,
            "risk_guard_scale": 1.0,
        }
        if isinstance(flags, dict):
            merged.update(flags)
        self.mixed_bilevel_flags = merged
        self._strict_mixed_last_active = False

    def _strict_mixed_requested(self) -> bool:
        phase_name = str(self.mixed_bilevel_flags.get("phase_name", "phase0") or "phase0").strip().lower()
        return phase_name not in {"", "phase0"}

    def _strict_mixed_route_mode(self) -> str:
        phase_name = str(self.mixed_bilevel_flags.get("phase_name", "phase0") or "phase0").strip().lower()
        if phase_name in {"", "phase0"}:
            return "legacy"
        if bool(self.mixed_bilevel_flags.get("tangential_ift_enabled", False)):
            return "full_ift"
        if bool(self.mixed_bilevel_flags.get("normal_ift_enabled", False)):
            return "normal_ready"
        return "forward_only"

    def _strict_mixed_skip_stats(self, reason: str) -> Dict[str, tf.Tensor]:
        return {
            "mixed_strict_active": tf.cast(0.0, self.dtype),
            "mixed_strict_skipped": tf.cast(1.0, self.dtype),
            "inner_converged": tf.cast(0.0, self.dtype),
            "inner_skip_batch": tf.cast(1.0, self.dtype),
            "skip_batch": tf.cast(1.0, self.dtype),
            "inner_fallback_used": tf.cast(0.0, self.dtype),
            "inner_fb_residual_norm": tf.cast(0.0, self.dtype),
            "inner_normal_step_norm": tf.cast(0.0, self.dtype),
            "inner_tangential_step_norm": tf.cast(0.0, self.dtype),
            "ift_linear_residual": tf.cast(0.0, self.dtype),
            "normal_ift_ready": tf.cast(0.0, self.dtype),
            "normal_ift_consumed": tf.cast(0.0, self.dtype),
            "normal_ift_condition_metric": tf.cast(0.0, self.dtype),
            "normal_ift_inputs_present": tf.cast(0.0, self.dtype),
            "normal_ift_core_valid_ratio": tf.cast(0.0, self.dtype),
            "mixed_strict_skip_reason": tf.constant(str(reason), dtype=tf.string),
        }

    @staticmethod
    def _strict_mixed_contact_stress_params(params, strict_inputs, inner_result=None):
        if isinstance(params, dict):
            stress_params = dict(params)
        elif params is None:
            stress_params = {}
        else:
            return params
        stress_params[CONTACT_SURFACE_NORMALS_KEY] = strict_inputs["normals"]
        stress_params[CONTACT_SURFACE_T1_KEY] = strict_inputs["t1"]
        stress_params[CONTACT_SURFACE_T2_KEY] = strict_inputs["t2"]
        if inner_result is not None and getattr(inner_result, "state", None) is not None:
            stress_params[INNER_CONTACT_GAP_N_KEY] = strict_inputs["g_n"]
            stress_params[INNER_CONTACT_LAMBDA_N_KEY] = inner_result.state.lambda_n
            stress_params[INNER_CONTACT_NORMALS_KEY] = strict_inputs["normals"]
            stress_params[INNER_CONTACT_WEIGHTS_KEY] = strict_inputs["weights"]
        return stress_params

    def _strict_mixed_normal_ift_stats(
        self,
        linearization,
        *,
        route_mode: str,
    ) -> Dict[str, tf.Tensor]:
        zero = tf.cast(0.0, self.dtype)
        one = tf.cast(1.0, self.dtype)
        stats = {
            "normal_ift_ready": zero,
            "normal_ift_consumed": zero,
            "normal_ift_condition_metric": zero,
            "normal_ift_inputs_present": zero,
            "normal_ift_core_valid_ratio": zero,
        }
        if route_mode != "normal_ready" or not isinstance(linearization, dict):
            return stats

        has_jac_inputs = "jac_inputs" in linearization
        stats["normal_ift_inputs_present"] = tf.cast(1.0 if has_jac_inputs else 0.0, self.dtype)
        if not self._strict_mixed_linearization_contract_is_valid(linearization, route_mode=route_mode):
            return stats

        required_core = ("residual", "jac_z")
        if any(key not in linearization for key in required_core):
            return stats
        if not has_jac_inputs:
            return stats

        residual = tf.cast(linearization["residual"], self.dtype)
        jac_z_payload = linearization["jac_z"]
        if isinstance(jac_z_payload, tf.SparseTensor):
            jac_z = tf.SparseTensor(
                indices=jac_z_payload.indices,
                values=tf.cast(jac_z_payload.values, self.dtype),
                dense_shape=jac_z_payload.dense_shape,
            )
        else:
            jac_z = tf.cast(jac_z_payload, self.dtype)
        stats_chunk_elems = tf.constant(1 << 20, dtype=tf.int32)

        def _finite_and_total_count(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            if isinstance(tensor, tf.SparseTensor):
                values = tf.reshape(tensor.values, (-1,))
                total_size = tf.reduce_prod(tf.cast(tensor.dense_shape, tf.int64))
                total_count = tf.cast(total_size, self.dtype)
                stored_count = tf.cast(tf.size(values), self.dtype)
                finite_stored = tf.reduce_sum(tf.cast(tf.math.is_finite(values), self.dtype))
                nonfinite_stored = stored_count - finite_stored
                return total_count - nonfinite_stored, total_count
            flat = tf.reshape(tensor, (-1,))
            total_size = tf.size(flat)
            total_count = tf.cast(total_size, self.dtype)

            def _cond(offset, finite_count):
                del finite_count
                return offset < total_size

            def _body(offset, finite_count):
                chunk_end = tf.minimum(offset + stats_chunk_elems, total_size)
                chunk = flat[offset:chunk_end]
                chunk_finite = tf.reduce_sum(tf.cast(tf.math.is_finite(chunk), self.dtype))
                return chunk_end, finite_count + chunk_finite

            _, finite_count = tf.while_loop(
                _cond,
                _body,
                loop_vars=(tf.constant(0, dtype=tf.int32), zero),
                parallel_iterations=1,
                swap_memory=True,
            )
            return finite_count, total_count

        residual_finite, residual_total = _finite_and_total_count(residual)
        jac_z_finite, jac_z_total = _finite_and_total_count(jac_z)
        # The input Jacobian can dwarf the state blocks; keep the readiness summary
        # on the residual/state-Jacobian core tensors to avoid a second huge scan.
        core_valid_ratio = (
            residual_finite + jac_z_finite
        ) / tf.maximum(
            residual_total + jac_z_total,
            tf.cast(1.0, self.dtype),
        )

        sentinel = tf.cast(1.0e12, self.dtype)

        def _finite_abs_extrema(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            if isinstance(tensor, tf.SparseTensor):
                finite_chunk = tf.boolean_mask(tensor.values, tf.math.is_finite(tensor.values))
                abs_chunk = tf.abs(tf.cast(finite_chunk, self.dtype))
                chunk_max = tf.cond(
                    tf.size(abs_chunk) > 0,
                    lambda: tf.reduce_max(abs_chunk),
                    lambda: zero,
                )
                positive_chunk = tf.boolean_mask(abs_chunk, abs_chunk > tf.cast(0.0, self.dtype))
                chunk_min = tf.cond(
                    tf.size(positive_chunk) > 0,
                    lambda: tf.reduce_min(positive_chunk),
                    lambda: sentinel,
                )
                return chunk_max, chunk_min
            flat = tf.reshape(tensor, (-1,))
            total_size = tf.size(flat)

            def _cond(offset, max_abs, min_pos_abs):
                del max_abs, min_pos_abs
                return offset < total_size

            def _body(offset, max_abs, min_pos_abs):
                chunk_end = tf.minimum(offset + stats_chunk_elems, total_size)
                chunk = flat[offset:chunk_end]
                finite_chunk = tf.boolean_mask(chunk, tf.math.is_finite(chunk))
                abs_chunk = tf.abs(finite_chunk)
                chunk_max = tf.cond(
                    tf.size(abs_chunk) > 0,
                    lambda: tf.reduce_max(abs_chunk),
                    lambda: zero,
                )
                positive_chunk = tf.boolean_mask(abs_chunk, abs_chunk > tf.cast(0.0, self.dtype))
                chunk_min = tf.cond(
                    tf.size(positive_chunk) > 0,
                    lambda: tf.reduce_min(positive_chunk),
                    lambda: sentinel,
                )
                return chunk_end, tf.maximum(max_abs, chunk_max), tf.minimum(min_pos_abs, chunk_min)

            _, max_abs, min_pos_abs = tf.while_loop(
                _cond,
                _body,
                loop_vars=(tf.constant(0, dtype=tf.int32), zero, sentinel),
                parallel_iterations=1,
                swap_memory=True,
            )
            return max_abs, min_pos_abs

        max_abs, min_abs = _finite_abs_extrema(jac_z)
        min_abs = tf.where(min_abs >= sentinel, tf.cast(1.0e-12, self.dtype), min_abs)
        condition_metric = tf.where(
            max_abs > tf.cast(0.0, self.dtype),
            max_abs / tf.maximum(min_abs, tf.cast(1.0e-12, self.dtype)),
            zero,
        )
        consumed = tf.cast(core_valid_ratio > tf.cast(0.0, self.dtype), self.dtype)

        stats.update(
            {
                "normal_ift_ready": one,
                "normal_ift_consumed": consumed,
                "normal_ift_condition_metric": condition_metric,
                "normal_ift_core_valid_ratio": core_valid_ratio,
            }
        )
        return stats

    @staticmethod
    def _strict_mixed_linearization_contract_is_valid(
        linearization,
        *,
        route_mode: str,
    ) -> bool:
        if route_mode != "normal_ready" or not isinstance(linearization, dict):
            return False
        if str(linearization.get("schema_version", "") or "") != "strict_mixed_v2":
            return False
        if str(linearization.get("route_mode", "") or "") != "normal_ready":
            return False
        if str(linearization.get("contract_mode", "") or "") != "normal_contact_first":
            return False
        if str(linearization.get("primary_state", "") or "") != "lambda_n":
            return False
        if str(linearization.get("tangential_contract_role", "") or "") != "auxiliary_friction_fixed_point":
            return False
        state_layout = linearization.get("state_layout")
        input_layout = linearization.get("input_layout")
        if not isinstance(state_layout, dict) or not isinstance(input_layout, dict):
            return False
        if list(state_layout.get("order") or []) != ["lambda_n", "lambda_t"]:
            return False
        if list(input_layout.get("order") or []) != ["g_n", "ds_t"]:
            return False
        return True

    def _strict_mixed_contact_terms(
        self,
        u_fn,
        params,
        *,
        u_nodes: Optional[tf.Tensor] = None,
        stress_fn=None,
    ) -> Tuple[bool, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)

        if self.contact is None or not self._strict_mixed_requested():
            self._strict_mixed_last_active = False
            return False, {}, {}

        stress_fn_contact = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        if stress_fn_contact is None:
            self._strict_mixed_last_active = False
            return False, {}, self._strict_mixed_skip_stats("stress_fn_missing")
        if not hasattr(self.contact, "strict_mixed_inputs") or not hasattr(self.contact, "solve_strict_inner"):
            self._strict_mixed_last_active = False
            return False, {}, self._strict_mixed_skip_stats("strict_contact_adapter_missing")

        route_mode = self._strict_mixed_route_mode()
        strict_inputs = self.contact.strict_mixed_inputs(u_fn, params, u_nodes=u_nodes)
        solve_kwargs = {
            "u_nodes": u_nodes,
            "strict_inputs": strict_inputs,
        }
        max_tail_qn_iters = int(self.mixed_bilevel_flags.get("max_tail_qn_iters", 0) or 0)
        if max_tail_qn_iters > 0:
            solve_kwargs["max_tail_qn_iters"] = max_tail_qn_iters
        signature_gate = str(self.mixed_bilevel_flags.get("max_inner_iters_signature_gate", "") or "").strip()
        signature_gated_max_inner_iters = int(
            self.mixed_bilevel_flags.get("signature_gated_max_inner_iters", 0) or 0
        )
        if route_mode == "normal_ready" and signature_gate and signature_gated_max_inner_iters > 0:
            solve_kwargs["return_iteration_trace"] = True
            solve_kwargs["max_inner_iters_signature_gate"] = signature_gate
            solve_kwargs["signature_gated_max_inner_iters"] = signature_gated_max_inner_iters
        if route_mode == "normal_ready":
            solve_kwargs["return_linearization"] = True
        while True:
            try:
                inner_result = self.contact.solve_strict_inner(
                    u_fn,
                    params,
                    **solve_kwargs,
                )
                break
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in (
                    "return_linearization",
                    "max_tail_qn_iters",
                    "return_iteration_trace",
                    "max_inner_iters_signature_gate",
                    "signature_gated_max_inner_iters",
                ):
                    if key in message and key in solve_kwargs:
                        solve_kwargs.pop(key, None)
                        removed = True
                if not removed:
                    raise
        diagnostics = dict(inner_result.diagnostics)
        policy = resolve_strict_mixed_runtime_policy(diagnostics, route_mode=route_mode)
        detach_inner = bool(self.mixed_bilevel_flags.get("detach_inner_solution", True)) or bool(policy.force_detach)
        parts, stats = mixed_residual_coupling.assemble_mixed_residual_coupling(
            inner_result=inner_result,
            strict_inputs=strict_inputs,
            route_mode=route_mode,
            detach_inner_solution=detach_inner,
            policy=policy,
            stress_fn_contact=stress_fn_contact,
            stress_params=self._strict_mixed_contact_stress_params(params, strict_inputs, inner_result),
            protocol_traction_scale=float(self.mixed_bilevel_flags.get("coupling_phase_traction_scale", 1.0) or 1.0),
            dtype=dtype,
            linearization_contract_is_valid=lambda lin, current_route: self._strict_mixed_linearization_contract_is_valid(
                lin,
                route_mode=current_route,
            ),
            normal_ift_stats_fn=lambda lin, current_route: self._strict_mixed_normal_ift_stats(
                lin,
                route_mode=current_route,
            ),
        )
        self._strict_mixed_last_active = True
        return True, parts, stats

    def _compute_data_smoothing_terms(
        self,
        X_obs: tf.Tensor,
        U_pred: tf.Tensor,
        data_ref_rms: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        k = int(getattr(self.cfg, "data_smoothing_k", 0) or 0)
        if k <= 0 or float(getattr(self.cfg, "w_smooth", 0.0) or 0.0) <= 0.0:
            return zero, {}

        X_obs = tf.cast(tf.reshape(X_obs, (-1, tf.shape(X_obs)[-1])), dtype)
        U_pred = tf.cast(tf.reshape(U_pred, (-1, tf.shape(U_pred)[-1])), dtype)
        n_static = X_obs.shape[0]
        if n_static is not None and int(n_static) <= 1:
            return zero, {}
        if n_static is not None:
            k = min(k, max(int(n_static) - 1, 1))
        if k <= 0:
            return zero, {}

        x2 = tf.reduce_sum(tf.square(X_obs), axis=1, keepdims=True)
        dist2 = x2 - 2.0 * tf.matmul(X_obs, X_obs, transpose_b=True) + tf.transpose(x2)
        dist2 = tf.maximum(dist2, tf.cast(0.0, dtype))
        n = tf.shape(dist2)[0]
        dist2 = dist2 + tf.eye(n, dtype=dtype) * tf.cast(1.0e30, dtype)
        _, nbr_idx = tf.math.top_k(-dist2, k=k)
        nbr_u = tf.gather(U_pred, nbr_idx)
        nbr_mean = tf.reduce_mean(nbr_u, axis=1)
        smooth_res = U_pred - nbr_mean
        smooth_rel = smooth_res / tf.maximum(data_ref_rms, tf.cast(1.0e-12, dtype))
        loss = tf.reduce_mean(tf.square(smooth_rel))
        stats = {
            "data_smooth_rms": tf.sqrt(tf.reduce_mean(tf.square(smooth_res)) + tf.cast(1.0e-20, dtype)),
            "data_smooth_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(smooth_rel)) + tf.cast(1.0e-20, dtype)),
        }
        return loss, stats

    def _resolve_supervision_u_fn(self, u_fn, params):
        target_frame = str(params.get("supervision_target_frame", "cartesian") or "cartesian").strip().lower()
        if target_frame == "cylindrical":
            u_primary = self._resolve_bound_variant(u_fn, "u_primary_fn")
            if callable(u_primary):
                return u_primary
        return u_fn

    def _compute_data_supervision_terms(self, u_fn, params) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        if not isinstance(params, dict):
            return zero, zero, {}

        X_obs = params.get("X_obs")
        U_obs = params.get("U_obs")
        if X_obs is None or U_obs is None:
            return zero, zero, {}

        X_obs = tf.cast(tf.convert_to_tensor(X_obs), dtype)
        U_obs = tf.cast(tf.convert_to_tensor(U_obs), dtype)
        u_supervision = self._resolve_supervision_u_fn(u_fn, params)

        U_pred = tf.cast(u_supervision(X_obs, params), dtype)
        diff = U_pred - U_obs
        data_ref_rms = tf.sqrt(tf.reduce_mean(tf.square(U_obs)) + tf.cast(1.0e-20, dtype))
        data_ref_rms = tf.maximum(data_ref_rms, tf.cast(1.0e-12, dtype))
        diff_rel = diff / data_ref_rms
        weight_stats: Dict[str, tf.Tensor] = {}
        if bool(getattr(self.cfg, "data_weight_enabled", False)) and params.get("data_weight") is not None:
            data_weight = tf.cast(tf.convert_to_tensor(params["data_weight"]), dtype)
            data_weight = tf.reshape(data_weight, (-1, 1))
            data_weight = tf.maximum(data_weight, tf.cast(0.0, dtype))
            mean_weight = tf.maximum(tf.reduce_mean(data_weight), tf.cast(1.0e-12, dtype))
            data_weight = data_weight / mean_weight
            power = max(0.0, float(getattr(self.cfg, "data_weight_power", 1.0) or 1.0))
            if abs(power - 1.0) > 1.0e-12:
                data_weight = tf.pow(tf.maximum(data_weight, tf.cast(1.0e-12, dtype)), tf.cast(power, dtype))
                data_weight = data_weight / tf.maximum(tf.reduce_mean(data_weight), tf.cast(1.0e-12, dtype))
            blend = min(max(float(getattr(self.cfg, "data_weight_blend", 1.0) or 0.0), 0.0), 1.0)
            data_weight = (tf.cast(1.0 - blend, dtype) + tf.cast(blend, dtype) * data_weight)
            data_weight = data_weight / tf.maximum(tf.reduce_mean(data_weight), tf.cast(1.0e-12, dtype))
            loss = tf.reduce_mean(tf.square(diff_rel) * data_weight)
            weight_stats = {
                "data_weight_mean": tf.reduce_mean(data_weight),
                "data_weight_max": tf.reduce_max(data_weight),
            }
        else:
            loss = tf.reduce_mean(tf.square(diff_rel))
        loss_smooth, smooth_stats = self._compute_data_smoothing_terms(X_obs, U_pred, data_ref_rms)
        stats = {
            "data_rms": tf.sqrt(tf.reduce_mean(tf.square(diff)) + tf.cast(1.0e-20, dtype)),
            "data_mae": tf.reduce_mean(tf.abs(diff)),
            "data_ref_rms": data_ref_rms,
            "data_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(diff_rel)) + tf.cast(1.0e-20, dtype)),
            "data_rel_mae": tf.reduce_mean(tf.abs(diff_rel)),
            "data_n_obs": tf.cast(tf.shape(X_obs)[0], dtype),
        }
        stats.update(weight_stats)
        stats.update(smooth_stats)
        return loss, loss_smooth, stats

    def _compute_stage_delta_supervision_terms(
        self,
        u_fn,
        prev_params,
        cur_params,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        if not isinstance(prev_params, dict) or not isinstance(cur_params, dict):
            return zero, {}

        X_obs = cur_params.get("X_obs")
        U_obs_delta = cur_params.get("U_obs_delta")
        if X_obs is None or U_obs_delta is None:
            return zero, {}

        X_obs = tf.cast(tf.convert_to_tensor(X_obs), dtype)
        U_obs_delta = tf.cast(tf.convert_to_tensor(U_obs_delta), dtype)

        prev_eval_params = dict(prev_params)
        if "supervision_target_frame" in cur_params and "supervision_target_frame" not in prev_eval_params:
            prev_eval_params["supervision_target_frame"] = cur_params["supervision_target_frame"]

        u_cur = self._resolve_supervision_u_fn(u_fn, cur_params)
        u_prev = self._resolve_supervision_u_fn(u_fn, prev_eval_params)
        U_cur = tf.cast(u_cur(X_obs, cur_params), dtype)
        U_prev = tf.cast(u_prev(X_obs, prev_eval_params), dtype)
        delta_pred = U_cur - U_prev
        diff = delta_pred - U_obs_delta

        delta_ref_rms = tf.sqrt(tf.reduce_mean(tf.square(U_obs_delta)) + tf.cast(1.0e-20, dtype))
        delta_ref_rms = tf.maximum(delta_ref_rms, tf.cast(1.0e-12, dtype))
        diff_rel = diff / delta_ref_rms
        if bool(getattr(self.cfg, "data_weight_enabled", False)) and cur_params.get("data_weight") is not None:
            data_weight = tf.cast(tf.convert_to_tensor(cur_params["data_weight"]), dtype)
            data_weight = tf.reshape(data_weight, (-1, 1))
            data_weight = tf.maximum(data_weight, tf.cast(0.0, dtype))
            data_weight = data_weight / tf.maximum(tf.reduce_mean(data_weight), tf.cast(1.0e-12, dtype))
            loss = tf.reduce_mean(tf.square(diff_rel) * data_weight)
        else:
            loss = tf.reduce_mean(tf.square(diff_rel))

        stats = {
            "delta_data_rms": tf.sqrt(tf.reduce_mean(tf.square(diff)) + tf.cast(1.0e-20, dtype)),
            "delta_data_mae": tf.reduce_mean(tf.abs(diff)),
            "delta_data_ref_rms": delta_ref_rms,
            "delta_data_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(diff_rel)) + tf.cast(1.0e-20, dtype)),
            "delta_data_rel_mae": tf.reduce_mean(tf.abs(diff_rel)),
        }
        return loss, stats

    def _optical_modal_basis(self, X_obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dtype = self.dtype
        center = getattr(self.cfg, "optical_modal_center", (0.0, 0.0)) or (0.0, 0.0)
        r_in = float(getattr(self.cfg, "optical_modal_r_in", 0.0) or 0.0)
        r_out = float(getattr(self.cfg, "optical_modal_r_out", 1.0) or 1.0)
        ring_cfg = RingFeatureConfig(
            center_x=float(center[0]),
            center_y=float(center[1]),
            r_in=r_in,
            r_out=r_out,
            fourier_order=max(1, int(getattr(self.cfg, "optical_modal_fourier_order", 1) or 1)),
        )
        comps = compute_ring_coordinate_components_tf(tf.cast(X_obs, tf.float32), ring_cfg)
        rho = tf.clip_by_value(comps["rho"], 0.0, 1.0)
        theta = comps["theta"]
        r = comps["r"]
        mask = tf.cast(
            tf.logical_and(r >= tf.cast(ring_cfg.r_in, tf.float32), r <= tf.cast(ring_cfg.r_out, tf.float32)),
            dtype,
        )

        radial_terms = []
        ones = tf.ones_like(rho, dtype=tf.float32)
        for radial_order in range(max(0, int(getattr(self.cfg, "optical_modal_radial_order", 0) or 0)) + 1):
            if radial_order == 0:
                radial_terms.append(ones)
            else:
                radial_terms.append(tf.pow(rho, tf.cast(radial_order, tf.float32)))

        basis_terms = []
        for radial in radial_terms:
            basis_terms.append(radial)
        for angular_order in range(1, max(1, int(getattr(self.cfg, "optical_modal_fourier_order", 1) or 1)) + 1):
            angular = tf.cast(angular_order, tf.float32)
            sin_t = tf.sin(angular * theta)
            cos_t = tf.cos(angular * theta)
            for radial in radial_terms:
                basis_terms.append(radial * sin_t)
                basis_terms.append(radial * cos_t)

        basis = tf.cast(tf.stack(basis_terms, axis=-1), dtype)
        basis = basis * mask[:, None]
        denom = tf.sqrt(tf.reduce_mean(tf.square(basis), axis=0, keepdims=True) + tf.cast(1.0e-12, dtype))
        basis = basis / denom
        return basis, mask[:, None]

    def _compute_optical_modal_supervision_terms(self, u_fn, params) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        enabled = bool(getattr(self.cfg, "optical_modal_enabled", False))
        if not enabled and float(getattr(self.cfg, "w_optical_modal", 0.0) or 0.0) <= 0.0:
            return zero, {}
        if not isinstance(params, dict):
            return zero, {}

        X_obs = params.get("X_obs")
        U_obs = params.get("U_obs")
        if X_obs is None or U_obs is None:
            return zero, {}

        X_obs = tf.cast(tf.convert_to_tensor(X_obs), dtype)
        U_obs = tf.cast(tf.convert_to_tensor(U_obs), dtype)
        u_supervision = self._resolve_supervision_u_fn(u_fn, params)
        U_pred = tf.cast(u_supervision(X_obs, params), dtype)

        component_count = tf.shape(U_obs)[-1]
        component_index = tf.minimum(
            tf.cast(max(0, int(getattr(self.cfg, "optical_modal_target_component", 2) or 2)), tf.int32),
            component_count - 1,
        )
        pred_comp = tf.gather(U_pred, component_index, axis=-1)[:, None]
        obs_comp = tf.gather(U_obs, component_index, axis=-1)[:, None]

        basis, mask = self._optical_modal_basis(X_obs)
        denom = tf.maximum(tf.reduce_sum(mask), tf.cast(1.0, dtype))
        pred_coeff = tf.reduce_sum(basis * pred_comp, axis=0) / denom
        obs_coeff = tf.reduce_sum(basis * obs_comp, axis=0) / denom
        diff = pred_coeff - obs_coeff
        ref = tf.sqrt(tf.reduce_mean(tf.square(obs_coeff)) + tf.cast(1.0e-20, dtype))
        ref = tf.maximum(ref, tf.cast(1.0e-12, dtype))
        diff_rel = diff / ref
        loss = tf.reduce_mean(tf.square(diff_rel))

        stats = {
            "optical_modal_rms": tf.sqrt(tf.reduce_mean(tf.square(diff)) + tf.cast(1.0e-20, dtype)),
            "optical_modal_ref_rms": ref,
            "optical_modal_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(diff_rel)) + tf.cast(1.0e-20, dtype)),
            "optical_modal_n_modes": tf.cast(tf.shape(basis)[-1], dtype),
            "optical_modal_n_obs": tf.reduce_sum(mask),
        }
        return loss, stats

    def attach(self, elasticity: Optional[ElasticityResidual] = None,
               contact: Optional[ContactOperator] = None,
               tightening: Optional[NutTighteningPenalty] = None,
               bcs: Optional[List[BoundaryPenalty]] = None):
        if elasticity is not None:
            self.elasticity = elasticity
        if contact is not None:
            self.contact = contact
        if tightening is not None:
            self.tightening = tightening
        if bcs is not None:
            self.bcs = list(bcs)
        self._built = True

    def reset(self):
        self.elasticity = None
        self.contact = None
        self.tightening = None
        self.bcs = []
        self._built = False

    def scale_volume_weights(self, factor: float):
        if getattr(self.elasticity, "w_vol_tf", None) is None:
            return
        try:
            self.elasticity.w_vol_tf = self.elasticity.w_vol_tf * tf.cast(factor, self.dtype)
        except Exception:
            pass

    def energy(self, u_fn, params=None, tape=None, stress_fn=None):
        self._ensure_weight_vars()
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")
        if self._loss_mode() == "residual":
            if isinstance(params, dict) and params.get("stages"):
                Pi, parts, stats = self._residual_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
            else:
                parts, stats = self._compute_parts_residual(u_fn, params or {}, tape, stress_fn=stress_fn)
                Pi = self._combine_parts(parts)
            return Pi, parts, stats
        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._energy_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
            return Pi, parts, stats
        parts, stats = self._compute_parts(u_fn, params or {}, tape, stress_fn=stress_fn)
        Pi = self._combine_parts(parts)
        return Pi, parts, stats

    def assemble_strict_mixed_outer_loss(self, u_fn, params=None, tape=None, stress_fn=None):
        """Explicit strict-mixed outer-loss assembly entrypoint for trainer dispatch."""

        return self.strict_mixed_objective(u_fn, params=params, tape=tape, stress_fn=stress_fn)

    def strict_mixed_objective(self, u_fn, params=None, tape=None, stress_fn=None):
        self._ensure_weight_vars()
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before strict_mixed_objective().")
        params = params or {}
        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._residual_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
        else:
            parts, stats = self._compute_parts_residual(
                u_fn,
                params,
                tape,
                stress_fn=stress_fn,
                allow_legacy_contact_fallback=False,
            )
            Pi = self._combine_parts_with_keys(parts, self.STRICT_MIXED_ACTIVE_KEYS)
        for key in self.STRICT_MIXED_ZERO_KEYS:
            if key in parts:
                parts[key] = tf.cast(0.0, self.dtype)
        return Pi, parts, stats

    def _compute_parts(self, u_fn, params, tape=None, stress_fn=None, *, allow_legacy_contact_fallback: bool = True):
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_bc": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
            "E_reg": zero,
            "E_bi": zero,
            "E_ed": zero,
            "E_unc": zero,
            "E_data": zero,
            "E_delta_data": zero,
            "E_optical_modal": zero,
            "E_smooth": zero,
            "R_const": zero,
            "R_eq": zero,
            "R_u": zero,
            "R_t": zero,
            "R_tr": zero,
            "R_contact_comp": zero,
            "R_fric_comp": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        u_nodes = None
        elastic_cache = None
        u_fn_elastic = self._resolve_bound_variant(u_fn, "u_fn_pointwise")
        stress_fn_elastic = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            E_int_res = self.elasticity.energy(
                u_fn_elastic,
                params,
                tape=tape,
                return_cache=bool(stress_fn_elastic),
                u_nodes=u_nodes,
            )
            if bool(stress_fn_elastic):
                E_int, estates, elastic_cache = E_int_res  # type: ignore[misc]
            else:
                E_int, estates = E_int_res  # type: ignore[misc]
            parts["E_int"] = tf.cast(E_int, dtype)
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            strict_requested = self._strict_mixed_requested()
            strict_active, strict_parts, strict_stats = self._strict_mixed_contact_terms(
                u_fn,
                params,
                u_nodes=u_nodes,
                stress_fn=stress_fn,
            )
            stats.update(strict_stats)
            if strict_active:
                for key in ("E_cn", "E_ct", "E_bi", "R_t", "R_tr", "R_fric_comp", "R_contact_comp"):
                    if key in strict_parts:
                        parts[key] = tf.cast(strict_parts[key], dtype)
            elif (not strict_requested) or allow_legacy_contact_fallback:
                _, cparts, stats_cn, stats_ct = self.contact.energy(u_fn, params, u_nodes=u_nodes)
                if "E_cn" in cparts:
                    parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
                elif "E_n" in cparts:
                    parts["E_cn"] = tf.cast(cparts["E_n"], dtype)
                if "E_ct" in cparts:
                    parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
                elif "E_t" in cparts:
                    parts["E_ct"] = tf.cast(cparts["E_t"], dtype)
                parts["R_t"] = tf.cast(parts["E_cn"], dtype)
                parts["R_tr"] = tf.cast(parts["E_ct"], dtype)
                stats.update(stats_cn)
                stats.update(stats_ct)
                if "R_fric_comp" in stats_ct:
                    parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
                if "R_contact_comp" in stats_cn:
                    parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
                if "E_bi" in stats_ct:
                    parts["E_bi"] = tf.cast(stats_ct["E_bi"], dtype)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                E_bc_i, si = b.energy(u_fn, params)
                bc_terms.append(tf.cast(E_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)
                parts["R_u"] = tf.cast(parts["E_bc"], dtype)

        if self.tightening is not None:
            E_tight, tstats = self.tightening.energy(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        E_data, E_smooth, data_stats = self._compute_data_supervision_terms(u_fn, params)
        parts["E_data"] = tf.cast(E_data, dtype)
        parts["E_smooth"] = tf.cast(E_smooth, dtype)
        stats.update(data_stats)
        E_optical_modal, optical_modal_stats = self._compute_optical_modal_supervision_terms(u_fn, params)
        parts["E_optical_modal"] = tf.cast(E_optical_modal, dtype)
        stats.update(optical_modal_stats)

        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        use_stress = stress_fn_elastic is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and getattr(self.elasticity.cfg, "stress_loss_weight", 0.0) > 0.0
        use_eq = use_stress and w_eq > 1e-15

        if use_sigma or use_eq:
            eps_vec = tf.cast(elastic_cache["eps_vec"], dtype)
            lam = tf.cast(elastic_cache.get("lam", 0.0), dtype)
            mu = tf.cast(elastic_cache.get("mu", 0.0), dtype)
            dof_idx = tf.cast(elastic_cache.get("dof_idx", 0), tf.int32)

            sigma_phys = elastic_cache.get("sigma_phys")
            if sigma_phys is not None:
                sigma_phys = tf.cast(sigma_phys, dtype)
            else:
                eps_tensor = elastic_cache.get("eps_tensor")
                if eps_tensor is None:
                    eps_tensor = tf.stack(
                        [
                            eps_vec[:, 0],
                            eps_vec[:, 1],
                            eps_vec[:, 2],
                            0.5 * eps_vec[:, 3],
                            0.5 * eps_vec[:, 4],
                            0.5 * eps_vec[:, 5],
                        ],
                        axis=1,
                    )
                else:
                    eps_tensor = tf.cast(eps_tensor, dtype)
                tr_eps = eps_tensor[:, 0] + eps_tensor[:, 1] + eps_tensor[:, 2]
                eye_vec = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
                sigma_phys = lam[:, None] * tr_eps[:, None] * eye_vec + 2.0 * mu[:, None] * eps_tensor

            sigma_phys = sigma_phys[:, :6]
            sigma_phys = tf.stack(
                [
                    sigma_phys[:, 0],
                    sigma_phys[:, 1],
                    sigma_phys[:, 2],
                    sigma_phys[:, 5],
                    sigma_phys[:, 3],
                    sigma_phys[:, 4],
                ],
                axis=1,
            )

            node_ids = tf.reshape(dof_idx // 3, (-1,))
            unique_nodes, rev = tf.unique(node_ids)
            X_nodes = tf.cast(tf.gather(self.elasticity.X_nodes_tf, unique_nodes), dtype)
            _, sigma_pred_nodes = stress_fn_elastic(X_nodes, params)
            sigma_pred_nodes = tf.cast(sigma_pred_nodes, dtype)

            sigma_nodes_full = tf.gather(sigma_pred_nodes, rev)
            sigma_cells = tf.reshape(sigma_nodes_full, (tf.shape(dof_idx)[0], 4, -1))
            sigma_cells = tf.reduce_mean(sigma_cells, axis=1)
            sigma_cells = sigma_cells[:, : tf.shape(sigma_phys)[1]]

            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if use_sigma:
                diff = sigma_cells - sigma_phys
                diff_n = diff / sigma_ref
                loss_sigma = tf.reduce_mean(diff_n * diff_n)
                parts["E_sigma"] = loss_sigma * tf.cast(
                    getattr(self.elasticity.cfg, "stress_loss_weight", 1.0), dtype
                )
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_cells * sigma_cells) + 1e-20)

            if use_eq:
                div_sigma = elastic_cache.get("div_sigma")
                if div_sigma is not None:
                    div_sigma = tf.cast(div_sigma, dtype) / sigma_ref
                    res = tf.reduce_sum(div_sigma * div_sigma, axis=1)
                    parts["E_eq"] = tf.reduce_mean(res)
                    stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + 1e-20)

        return parts, stats

    def _compute_parts_residual(
        self,
        u_fn,
        params,
        tape=None,
        stress_fn=None,
        *,
        allow_legacy_contact_fallback: bool = True,
    ):
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_bc": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
            "E_reg": zero,
            "E_bi": zero,
            "E_ed": zero,
            "E_unc": zero,
            "E_data": zero,
            "E_delta_data": zero,
            "E_optical_modal": zero,
            "E_smooth": zero,
            "R_const": zero,
            "R_eq": zero,
            "R_u": zero,
            "R_t": zero,
            "R_tr": zero,
            "R_contact_comp": zero,
            "R_fric_comp": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        w_reg = float(getattr(self.cfg, "w_reg", 0.0))
        stress_weight = float(getattr(getattr(self.elasticity, "cfg", None), "stress_loss_weight", 0.0))
        strict_requested = self._strict_mixed_requested()
        u_fn_elastic = self._resolve_bound_variant(u_fn, "u_fn_pointwise")
        stress_fn_elastic = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        need_sigma = stress_fn_elastic is not None and w_sigma > 1e-15 and stress_weight > 0.0
        need_eq = w_eq > 1e-15
        need_reg = w_reg > 1e-15

        u_nodes = None
        elastic_cache = None
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            use_mixed_terms = (
                stress_fn_elastic is not None
                and (need_sigma or strict_requested)
                and hasattr(self.elasticity, "mixed_residual_terms")
            )
            if use_mixed_terms:
                mixed_terms = self.elasticity.mixed_residual_terms(
                    u_fn_elastic,
                    stress_fn_elastic,
                    params,
                    return_cache=True,
                )
                estates = {}
                elastic_cache = dict(mixed_terms.get("cache") or {})
            else:
                estates, elastic_cache = self.elasticity.residual_cache(
                    u_fn_elastic,
                    params,
                    stress_fn=stress_fn_elastic,
                    need_sigma=need_sigma,
                    need_eq=need_eq,
                )
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            strict_active, strict_parts, strict_stats = self._strict_mixed_contact_terms(
                u_fn,
                params,
                u_nodes=u_nodes,
                stress_fn=stress_fn,
            )
            stats.update(strict_stats)
            if strict_active:
                for key in ("E_cn", "E_ct", "E_bi", "R_t", "R_tr", "R_fric_comp", "R_contact_comp"):
                    if key in strict_parts:
                        parts[key] = tf.cast(strict_parts[key], dtype)
            elif (not strict_requested) or allow_legacy_contact_fallback:
                _, cparts, stats_cn, stats_ct = self.contact.residual(u_fn, params, u_nodes=u_nodes)
                if "E_cn" in cparts:
                    parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
                elif "E_n" in cparts:
                    parts["E_cn"] = tf.cast(cparts["E_n"], dtype)
                if "E_ct" in cparts:
                    parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
                elif "E_t" in cparts:
                    parts["E_ct"] = tf.cast(cparts["E_t"], dtype)
                parts["R_t"] = tf.cast(parts["E_cn"], dtype)
                parts["R_tr"] = tf.cast(parts["E_ct"], dtype)
                stats.update(stats_cn)
                stats.update(stats_ct)
                if "R_fric_comp" in stats_ct:
                    parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
                if "R_contact_comp" in stats_cn:
                    parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
                if "E_bi" in stats_ct:
                    parts["E_bi"] = tf.cast(stats_ct["E_bi"], dtype)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                L_bc_i, si = b.residual(u_fn, params)
                bc_terms.append(tf.cast(L_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)
                parts["R_u"] = tf.cast(parts["E_bc"], dtype)

        if self.tightening is not None:
            E_tight, tstats = self.tightening.residual(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        E_data, E_smooth, data_stats = self._compute_data_supervision_terms(u_fn, params)
        parts["E_data"] = tf.cast(E_data, dtype)
        parts["E_smooth"] = tf.cast(E_smooth, dtype)
        stats.update(data_stats)
        E_optical_modal, optical_modal_stats = self._compute_optical_modal_supervision_terms(u_fn, params)
        parts["E_optical_modal"] = tf.cast(E_optical_modal, dtype)
        stats.update(optical_modal_stats)

        use_stress = stress_fn_elastic is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and stress_weight > 0.0
        use_eq = elastic_cache is not None and w_eq > 1e-15
        use_reg = elastic_cache is not None and w_reg > 1e-15
        need_const = use_stress and (strict_requested or use_sigma)
        need_eq_term = elastic_cache is not None and (strict_requested or use_eq)

        if need_const or need_eq_term or use_reg:
            eps_vec = elastic_cache.get("eps_vec") if isinstance(elastic_cache, dict) else None
            sigma_phys_head = elastic_cache.get("sigma_phys") if isinstance(elastic_cache, dict) else None
            sigma_pred = elastic_cache.get("sigma_pred") if isinstance(elastic_cache, dict) else None
            div_sigma = elastic_cache.get("div_sigma") if isinstance(elastic_cache, dict) else None
            w_sel = elastic_cache.get("w_sel") if isinstance(elastic_cache, dict) else None

            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if need_const and sigma_pred is not None and sigma_phys_head is not None:
                sigma_pred = tf.cast(sigma_pred, dtype)
                sigma_phys_head = tf.cast(sigma_phys_head, dtype)
                # Align residual-path physics stress ordering to stress-head convention:
                # phys: [xx,yy,zz,yz,xz,xy] -> head: [xx,yy,zz,xy,yz,xz].
                n_sigma_comp = sigma_phys_head.shape[-1]
                if n_sigma_comp is None:
                    sigma_cols = tf.shape(sigma_phys_head)[1]

                    def _reorder_sigma():
                        core = tf.stack(
                            [
                                sigma_phys_head[:, 0],
                                sigma_phys_head[:, 1],
                                sigma_phys_head[:, 2],
                                sigma_phys_head[:, 5],
                                sigma_phys_head[:, 3],
                                sigma_phys_head[:, 4],
                            ],
                            axis=1,
                        )
                        return tf.concat([core, sigma_phys_head[:, 6:]], axis=1)

                    sigma_phys_head = tf.cond(
                        sigma_cols >= 6, _reorder_sigma, lambda: sigma_phys_head
                    )
                elif n_sigma_comp >= 6:
                    sigma_phys_core = tf.stack(
                        [
                            sigma_phys_head[:, 0],
                            sigma_phys_head[:, 1],
                            sigma_phys_head[:, 2],
                            sigma_phys_head[:, 5],
                            sigma_phys_head[:, 3],
                            sigma_phys_head[:, 4],
                        ],
                        axis=1,
                    )
                    sigma_phys_head = (
                        sigma_phys_core
                        if n_sigma_comp == 6
                        else tf.concat([sigma_phys_core, sigma_phys_head[:, 6:]], axis=1)
                    )
                diff = sigma_pred - sigma_phys_head
                diff_n = diff / sigma_ref
                res = tf.reduce_sum(diff_n * diff_n, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                    loss_sigma = tf.reduce_sum(res) / denom
                else:
                    loss_sigma = tf.reduce_mean(res)
                parts["R_const"] = loss_sigma
                if use_sigma:
                    parts["E_sigma"] = loss_sigma * tf.cast(stress_weight, dtype)
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_pred * sigma_pred) + tf.cast(1e-20, dtype))

            if need_eq_term and div_sigma is not None:
                div_sigma = tf.cast(div_sigma, dtype) / sigma_ref
                res = tf.reduce_sum(div_sigma * div_sigma, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["R_eq"] = tf.reduce_sum(res) / denom
                parts["E_eq"] = tf.cast(parts["R_eq"], dtype)
                stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + tf.cast(1e-20, dtype))

            if use_reg and eps_vec is not None:
                eps_vec = tf.cast(eps_vec, dtype)
                res = tf.reduce_sum(eps_vec * eps_vec, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_reg"] = tf.reduce_sum(res) / denom
                stats["reg_rms"] = tf.sqrt(tf.reduce_mean(res) + tf.cast(1e-20, dtype))

        return parts, stats

    def _combine_parts(self, parts: Dict[str, tf.Tensor]) -> tf.Tensor:
        return (
            self.w_int * parts.get("E_int", tf.cast(0.0, self.dtype))
            + self.w_cn * parts.get("E_cn", tf.cast(0.0, self.dtype))
            + self.w_ct * parts.get("E_ct", tf.cast(0.0, self.dtype))
            + self.w_bc * parts.get("E_bc", tf.cast(0.0, self.dtype))
            + self.w_tight * parts.get("E_tight", tf.cast(0.0, self.dtype))
            + self.w_sigma * parts.get("E_sigma", tf.cast(0.0, self.dtype))
            + self.w_eq * parts.get("E_eq", tf.cast(0.0, self.dtype))
            + self.w_reg * parts.get("E_reg", tf.cast(0.0, self.dtype))
            + self.w_bi * parts.get("E_bi", tf.cast(0.0, self.dtype))
            + self.w_ed * parts.get("E_ed", tf.cast(0.0, self.dtype))
            + self.w_unc * parts.get("E_unc", tf.cast(0.0, self.dtype))
            + self.w_data * parts.get("E_data", tf.cast(0.0, self.dtype))
            + self.w_delta_data * parts.get("E_delta_data", tf.cast(0.0, self.dtype))
            + self.w_optical_modal * parts.get("E_optical_modal", tf.cast(0.0, self.dtype))
            + self.w_smooth * parts.get("E_smooth", tf.cast(0.0, self.dtype))
        )

    def _combine_parts_with_keys(self, parts: Dict[str, tf.Tensor], active_keys) -> tf.Tensor:
        active = set(active_keys or ())
        combined = tf.cast(0.0, self.dtype)
        for key, weight in (
            ("E_int", self.w_int),
            ("E_cn", self.w_cn),
            ("E_ct", self.w_ct),
            ("E_bc", self.w_bc),
            ("E_tight", self.w_tight),
            ("E_sigma", self.w_sigma),
            ("E_eq", self.w_eq),
            ("E_reg", self.w_reg),
            ("E_bi", self.w_bi),
            ("E_ed", self.w_ed),
            ("E_unc", self.w_unc),
            ("E_data", self.w_data),
            ("E_delta_data", self.w_delta_data),
            ("E_optical_modal", self.w_optical_modal),
            ("E_smooth", self.w_smooth),
            ("R_const", self.w_sigma),
            ("R_eq", self.w_eq),
            ("R_u", self.w_bc),
            ("R_t", self.w_cn),
            ("R_tr", self.w_ct),
        ):
            if key not in active:
                continue
            combined = combined + weight * parts.get(key, tf.cast(0.0, self.dtype))
        return combined

    def _energy_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        dtype = self.dtype
        supervision_target_frame = None
        if isinstance(root_params, dict) and "supervision_target_frame" in root_params:
            supervision_target_frame = root_params["supervision_target_frame"]
        keys = [
            "E_int",
            "E_cn",
            "E_ct",
            "E_bc",
            "E_tight",
            "E_sigma",
            "E_eq",
            "E_reg",
            "E_bi",
            "E_ed",
            "E_unc",
            "E_data",
            "E_delta_data",
            "E_optical_modal",
            "E_smooth",
            "R_const",
            "R_eq",
            "R_u",
            "R_t",
            "R_tr",
            "R_contact_comp",
            "R_fric_comp",
        ]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            stage_tensor_x_obs = stages.get("X_obs")
            stage_tensor_u_obs = stages.get("U_obs")
            stage_tensor_u_obs_delta = stages.get("U_obs_delta")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stacked_x_obs = None
                if stage_tensor_x_obs is not None:
                    stacked_x_obs = tf.convert_to_tensor(stage_tensor_x_obs)
                stacked_u_obs = None
                if stage_tensor_u_obs is not None:
                    stacked_u_obs = tf.convert_to_tensor(stage_tensor_u_obs)
                stacked_u_obs_delta = None
                if stage_tensor_u_obs_delta is not None:
                    stacked_u_obs_delta = tf.convert_to_tensor(stage_tensor_u_obs_delta)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if supervision_target_frame is not None:
                        entry["supervision_target_frame"] = supervision_target_frame
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    if stacked_x_obs is not None:
                        entry["X_obs"] = stacked_x_obs[idx]
                    if stacked_u_obs is not None:
                        entry["U_obs"] = stacked_u_obs[idx]
                    if stacked_u_obs_delta is not None and idx > 0:
                        entry["U_obs_delta"] = stacked_u_obs_delta[idx - 1]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    entry = dict(item)
                    if supervision_target_frame is not None and "supervision_target_frame" not in entry:
                        entry["supervision_target_frame"] = supervision_target_frame
                    stage_seq.append(entry)
                else:
                    p_val, z_val = item
                    entry = {"P": p_val, "P_hat": z_val}
                    if supervision_target_frame is not None:
                        entry["supervision_target_frame"] = supervision_target_frame
                    stage_seq.append(entry)

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_E_int: Optional[tf.Tensor] = None
        prev_stage_params_for_delta: Optional[Dict[str, tf.Tensor]] = None
        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype)
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)

            stage_parts, stage_stats = self._compute_parts(u_fn, stage_params, tape, stress_fn=stress_fn)
            stage_parts = dict(stage_parts)
            if idx > 0 and prev_stage_params_for_delta is not None:
                delta_loss, delta_stats = self._compute_stage_delta_supervision_terms(
                    u_fn,
                    prev_stage_params_for_delta,
                    stage_params,
                )
                stage_parts["E_delta_data"] = tf.cast(delta_loss, dtype)
                stage_stats.update(delta_stats)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            for key in keys:
                cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(tf.convert_to_tensor(stage_params.get("P", [])), dtype)
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)
                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path
                if bool(getattr(self.cfg, "ed_enabled", False)):
                    cur_e_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
                    d_el = cur_e_int - (prev_E_int if prev_E_int is not None else tf.cast(0.0, dtype))
                    d_fric = tf.abs(tf.cast(stage_parts.get("E_ct", tf.cast(0.0, dtype)), dtype))
                    w_ext = tf.cast(getattr(self.cfg, "ed_external_scale", 1.0), dtype) * load_jump
                    ed_pen = compute_incremental_ed_penalty(
                        d_el,
                        d_fric,
                        w_ext,
                        margin=tf.cast(getattr(self.cfg, "ed_margin", 0.0), dtype),
                        use_relu=bool(getattr(self.cfg, "ed_use_relu", True)),
                        squared=bool(getattr(self.cfg, "ed_square", True)),
                    )
                    stage_parts["E_ed"] = tf.cast(ed_pen, dtype)
                    totals["E_ed"] = totals["E_ed"] + tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_E_ed"] = tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_ed_delta_el"] = tf.cast(d_el, dtype)
                    stats_all[f"s{idx+1}_ed_d_fric"] = tf.cast(d_fric, dtype)
                    stats_all[f"s{idx+1}_ed_w_ext"] = tf.cast(w_ext, dtype)
            stage_path_penalty = w_path * stage_path + w_fric_path * stage_fric_path
            if stage_path != 0.0:
                path_penalty_total = path_penalty_total + stage_path
                stats_all[f"s{idx+1}_path_penalty"] = stage_path
                stats_all[f"s{idx+1}_path_penalty_w"] = w_path

            if self._strict_mixed_requested():
                stage_mech = self._combine_parts_with_keys(stage_parts, self.STRICT_MIXED_ACTIVE_KEYS)
            else:
                stage_mech = self._combine_parts(stage_parts)
            stage_pi_step = stage_mech + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step
            stats_all[f"s{idx+1}_Pi_mech"] = stage_mech

            Pi_accum = Pi_accum + stage_pi_step

            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_E_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
            prev_stage_params_for_delta = dict(stage_params)
            if self.contact is not None:
                try:
                    stage_params_detached = {k: tf.stop_gradient(v) if isinstance(v, tf.Tensor) else v for k, v in stage_params.items()}
                    self.contact.update_multipliers(u_fn, stage_params_detached)
                except Exception:
                    pass
            if self.bcs:
                for bc in self.bcs:
                    try:
                        bc.update_multipliers(u_fn, stage_params)
                    except Exception:
                        pass

        if isinstance(root_params, dict):
            if "stage_order" in root_params:
                stats_all["stage_order"] = root_params["stage_order"]
            if "stage_rank" in root_params:
                stats_all["stage_rank"] = root_params["stage_rank"]
            if "stage_count" in root_params:
                stats_all["stage_count"] = root_params["stage_count"]

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total

        Pi = Pi_accum
        return Pi, totals, stats_all

    def _residual_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        dtype = self.dtype
        supervision_target_frame = None
        if isinstance(root_params, dict) and "supervision_target_frame" in root_params:
            supervision_target_frame = root_params["supervision_target_frame"]
        keys = [
            "E_int",
            "E_cn",
            "E_ct",
            "E_bc",
            "E_tight",
            "E_sigma",
            "E_eq",
            "E_reg",
            "E_bi",
            "E_ed",
            "E_unc",
            "E_data",
            "E_delta_data",
            "E_optical_modal",
            "E_smooth",
            "R_const",
            "R_eq",
            "R_u",
            "R_t",
            "R_tr",
            "R_contact_comp",
            "R_fric_comp",
        ]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            stage_tensor_x_obs = stages.get("X_obs")
            stage_tensor_u_obs = stages.get("U_obs")
            stage_tensor_u_obs_delta = stages.get("U_obs_delta")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stacked_x_obs = None
                if stage_tensor_x_obs is not None:
                    stacked_x_obs = tf.convert_to_tensor(stage_tensor_x_obs)
                stacked_u_obs = None
                if stage_tensor_u_obs is not None:
                    stacked_u_obs = tf.convert_to_tensor(stage_tensor_u_obs)
                stacked_u_obs_delta = None
                if stage_tensor_u_obs_delta is not None:
                    stacked_u_obs_delta = tf.convert_to_tensor(stage_tensor_u_obs_delta)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if supervision_target_frame is not None:
                        entry["supervision_target_frame"] = supervision_target_frame
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    if stacked_x_obs is not None:
                        entry["X_obs"] = stacked_x_obs[idx]
                    if stacked_u_obs is not None:
                        entry["U_obs"] = stacked_u_obs[idx]
                    if stacked_u_obs_delta is not None and idx > 0:
                        entry["U_obs_delta"] = stacked_u_obs_delta[idx - 1]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    entry = dict(item)
                    if supervision_target_frame is not None and "supervision_target_frame" not in entry:
                        entry["supervision_target_frame"] = supervision_target_frame
                    stage_seq.append(entry)
                else:
                    p_val, z_val = item
                    entry = {"P": p_val, "P_hat": z_val}
                    if supervision_target_frame is not None:
                        entry["supervision_target_frame"] = supervision_target_frame
                    stage_seq.append(entry)

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_E_int: Optional[tf.Tensor] = None
        prev_stage_params_for_delta: Optional[Dict[str, tf.Tensor]] = None
        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype)
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)

            stage_parts, stage_stats = self._compute_parts_residual(u_fn, stage_params, tape, stress_fn=stress_fn)
            stage_parts = dict(stage_parts)
            if idx > 0 and prev_stage_params_for_delta is not None:
                delta_loss, delta_stats = self._compute_stage_delta_supervision_terms(
                    u_fn,
                    prev_stage_params_for_delta,
                    stage_params,
                )
                stage_parts["E_delta_data"] = tf.cast(delta_loss, dtype)
                stage_stats.update(delta_stats)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            for key in keys:
                cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(tf.convert_to_tensor(stage_params.get("P", [])), dtype)
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)
                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path
                if bool(getattr(self.cfg, "ed_enabled", False)):
                    cur_e_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
                    d_el = cur_e_int - (prev_E_int if prev_E_int is not None else tf.cast(0.0, dtype))
                    d_fric = tf.abs(tf.cast(stage_parts.get("E_ct", tf.cast(0.0, dtype)), dtype))
                    w_ext = tf.cast(getattr(self.cfg, "ed_external_scale", 1.0), dtype) * load_jump
                    ed_pen = compute_incremental_ed_penalty(
                        d_el,
                        d_fric,
                        w_ext,
                        margin=tf.cast(getattr(self.cfg, "ed_margin", 0.0), dtype),
                        use_relu=bool(getattr(self.cfg, "ed_use_relu", True)),
                        squared=bool(getattr(self.cfg, "ed_square", True)),
                    )
                    stage_parts["E_ed"] = tf.cast(ed_pen, dtype)
                    totals["E_ed"] = totals["E_ed"] + tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_E_ed"] = tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_ed_delta_el"] = tf.cast(d_el, dtype)
                    stats_all[f"s{idx+1}_ed_d_fric"] = tf.cast(d_fric, dtype)
                    stats_all[f"s{idx+1}_ed_w_ext"] = tf.cast(w_ext, dtype)
            stage_path_penalty = w_path * stage_path + w_fric_path * stage_fric_path
            if stage_path != 0.0:
                path_penalty_total = path_penalty_total + stage_path
                stats_all[f"s{idx+1}_path_penalty"] = stage_path
                stats_all[f"s{idx+1}_path_penalty_w"] = w_path

            if self._strict_mixed_requested():
                stage_mech = self._combine_parts_with_keys(stage_parts, self.STRICT_MIXED_ACTIVE_KEYS)
            else:
                stage_mech = self._combine_parts(stage_parts)
            stage_pi_step = stage_mech + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step
            stats_all[f"s{idx+1}_Pi_mech"] = stage_mech

            Pi_accum = Pi_accum + stage_pi_step

            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_E_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
            prev_stage_params_for_delta = dict(stage_params)

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total

        Pi = Pi_accum
        return Pi, totals, stats_all

    def update_multipliers(self, u_fn, params=None):
        target_params = params
        staged_updates: List[Dict[str, tf.Tensor]] = []
        if isinstance(params, dict) and params.get("stages"):
            stages = params["stages"]
            if isinstance(stages, dict):
                stage_tensor_P = stages.get("P")
                stage_tensor_feat = stages.get("P_hat")
                stage_tensor_rank = stages.get("stage_rank")
                if stage_tensor_P is not None and stage_tensor_feat is not None:
                    for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                        entry: Dict[str, tf.Tensor] = {"P": p, "P_hat": z}
                        if stage_tensor_rank is not None:
                            entry["stage_rank"] = stage_tensor_rank[idx] if stage_tensor_rank.shape.rank == 2 else stage_tensor_rank
                        staged_updates.append(entry)
                        target_params = entry
            elif isinstance(stages, (list, tuple)) and stages:
                for stage in stages:
                    if isinstance(stage, dict):
                        staged_updates.append(stage)
                        target_params = stage
                    else:
                        p_val, z_val = stage
                        entry = {"P": p_val, "P_hat": z_val}
                        staged_updates.append(entry)
                        target_params = entry

        if self.contact is not None and not self._strict_mixed_last_active:
            if staged_updates:
                for st_params in staged_updates:
                    u_nodes = None
                    if self.elasticity is not None:
                        u_nodes = self.elasticity._eval_u_on_nodes(u_fn, st_params)
                    self.contact.update_multipliers(u_fn, st_params, u_nodes=u_nodes)
            else:
                u_nodes = None
                if self.elasticity is not None:
                    u_nodes = self.elasticity._eval_u_on_nodes(u_fn, target_params)
                self.contact.update_multipliers(u_fn, target_params, u_nodes=u_nodes)
        if self.bcs:
            if staged_updates:
                for st_params in staged_updates:
                    for bc in self.bcs:
                        bc.update_multipliers(u_fn, st_params)
            else:
                for bc in self.bcs:
                    bc.update_multipliers(u_fn, target_params)

    def set_coeffs(self, w_int: Optional[float] = None,
                   w_cn: Optional[float] = None,
                   w_ct: Optional[float] = None):
        if w_int is not None:
            self.w_int.assign(tf.cast(w_int, self.dtype))
        if w_cn is not None:
            self.w_cn.assign(tf.cast(w_cn, self.dtype))
        if w_ct is not None:
            self.w_ct.assign(tf.cast(w_ct, self.dtype))
