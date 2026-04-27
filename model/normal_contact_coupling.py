#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal-contact-first outer-inner coupling assembly."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

import tensorflow as tf

from physics.contact.contact_operator import traction_matching_terms
from physics.contact.contact_inner_kernel_primitives import compose_contact_traction
from physics.contact.contact_inner_solver import flatten_contact_inputs, flatten_contact_state
from physics.contact.contact_implicit_backward import attach_normal_contact_implicit_backward
from physics.traction_utils import normal_tangential_components


def assemble_normal_contact_coupling(
    *,
    inner_result,
    strict_inputs: Mapping[str, tf.Tensor],
    route_mode: str,
    detach_inner_solution: bool,
    policy,
    stress_fn_contact,
    stress_params,
    protocol_traction_scale: float,
    dtype,
    linearization_contract_is_valid: Callable[[Any, str], bool],
    normal_ift_stats_fn: Callable[[Any, str], Dict[str, tf.Tensor]],
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Assemble strict outer-inner coupling terms for the normal-contact-first mainline."""

    zero = tf.cast(0.0, dtype)
    diagnostics = dict(getattr(inner_result, "diagnostics", {}) or {})
    linearization = getattr(inner_result, "linearization", None)
    implicit_backward_enabled = (
        route_mode == "normal_ready"
        and not bool(detach_inner_solution)
        and isinstance(linearization, dict)
        and "block_jacobians" in linearization
        and linearization_contract_is_valid(linearization, route_mode)
    )

    lambda_n = tf.cast(inner_result.state.lambda_n, dtype)
    lambda_t = tf.cast(inner_result.state.lambda_t, dtype)
    traction_vec = tf.cast(inner_result.traction_vec, dtype)
    ds_t = tf.cast(strict_inputs["ds_t"], dtype)
    if implicit_backward_enabled:
        flat_state = flatten_contact_state(
            tf.cast(inner_result.state.lambda_n, tf.float32),
            tf.cast(inner_result.state.lambda_t, tf.float32),
        )
        flat_inputs = flatten_contact_inputs(
            tf.cast(strict_inputs["g_n"], tf.float32),
            tf.cast(strict_inputs["ds_t"], tf.float32),
        )
        flat_state = attach_normal_contact_implicit_backward(flat_state, flat_inputs, linearization)
        lambda_n_size = tf.size(lambda_n)
        lambda_n = tf.reshape(tf.cast(flat_state[:lambda_n_size], dtype), tf.shape(lambda_n))
        lambda_t = tf.reshape(tf.cast(flat_state[lambda_n_size:], dtype), tf.shape(lambda_t))
        traction_vec = tf.cast(
            compose_contact_traction(
                lambda_n,
                lambda_t,
                strict_inputs["normals"],
                strict_inputs["t1"],
                strict_inputs["t2"],
            ),
            dtype,
        )
    elif detach_inner_solution:
        lambda_n = tf.stop_gradient(lambda_n)
        lambda_t = tf.stop_gradient(lambda_t)
        traction_vec = tf.stop_gradient(traction_vec)
        ds_t = tf.stop_gradient(ds_t)

    _, sigma_s = stress_fn_contact(strict_inputs["xs"], stress_params)
    _, sigma_m = stress_fn_contact(strict_inputs["xm"], stress_params)
    sigma_s = tf.cast(sigma_s, dtype)
    sigma_m = tf.cast(sigma_m, dtype)

    basis = tf.stack([strict_inputs["t1"], strict_inputs["t2"]], axis=1)
    inner_for_match = type("StrictInnerMatch", (), {"traction_vec": traction_vec})()
    rs, rm = traction_matching_terms(
        sigma_s,
        sigma_m,
        strict_inputs["normals"],
        strict_inputs["t1"],
        strict_inputs["t2"],
        inner_for_match,
    )
    rs = tf.cast(rs, dtype)
    rm = tf.cast(rm, dtype)

    rs_n, rs_t = normal_tangential_components(rs, strict_inputs["normals"], basis)
    rm_n, rm_t = normal_tangential_components(rm, strict_inputs["normals"], basis)
    rs_n = tf.squeeze(tf.cast(rs_n, dtype), axis=-1)
    rm_n = tf.squeeze(tf.cast(rm_n, dtype), axis=-1)
    rs_t = tf.cast(rs_t, dtype)
    rm_t = tf.cast(rm_t, dtype)

    weights = tf.cast(strict_inputs["weights"], dtype)
    denom = tf.reduce_sum(weights) + tf.cast(1.0e-12, dtype)
    rn_sq = tf.square(rs_n) + tf.square(rm_n)
    rt_sq = tf.reduce_sum(tf.square(rs_t), axis=1) + tf.reduce_sum(tf.square(rm_t), axis=1)
    runtime_traction_scale = tf.cast(policy.traction_scale, dtype)
    phase_traction_scale = tf.cast(protocol_traction_scale, dtype)
    effective_traction_scale = runtime_traction_scale * phase_traction_scale
    e_cn_raw = tf.reduce_sum(weights * rn_sq) / (2.0 * denom)
    e_ct_raw = tf.reduce_sum(weights * rt_sq) / (2.0 * denom)
    e_cn = effective_traction_scale * e_cn_raw
    e_ct = effective_traction_scale * e_ct_raw

    rt_norm = tf.sqrt(tf.reduce_sum(tf.square(rs_t), axis=1) + tf.cast(1.0e-12, dtype))
    rm_t_norm = tf.sqrt(tf.reduce_sum(tf.square(rm_t), axis=1) + tf.cast(1.0e-12, dtype))
    r_contact = effective_traction_scale * tf.reduce_sum(weights * 0.5 * (tf.abs(rs_n) + tf.abs(rm_n)))
    r_fric = effective_traction_scale * tf.reduce_sum(weights * 0.5 * (rt_norm + rm_t_norm))

    mu = tf.cast(strict_inputs["mu"], dtype)
    eps_bi = tf.cast(1.0e-8, dtype)
    st_norm = tf.sqrt(tf.reduce_sum(tf.square(ds_t), axis=1) + eps_bi)
    bi_raw = mu * tf.maximum(lambda_n, tf.cast(0.0, dtype)) * st_norm - tf.reduce_sum(lambda_t * ds_t, axis=1)
    bi_pos = tf.nn.relu(bi_raw)
    e_bi = tf.reduce_sum(weights * bi_pos * bi_pos) / denom

    fn_norm = tf.cast(diagnostics.get("fn_norm", zero), dtype)
    ft_norm = tf.cast(diagnostics.get("ft_norm", zero), dtype)
    cone_violation = tf.cast(diagnostics.get("cone_violation", zero), dtype)
    max_penetration = tf.cast(diagnostics.get("max_penetration", zero), dtype)
    fb_residual_norm = tf.cast(diagnostics.get("fb_residual_norm", zero), dtype)
    normal_step_norm = tf.cast(diagnostics.get("normal_step_norm", zero), dtype)
    tangential_step_norm = tf.cast(diagnostics.get("tangential_step_norm", zero), dtype)
    fallback_used = tf.cast(diagnostics.get("fallback_used", zero), dtype)
    converged = tf.cast(diagnostics.get("converged", tf.cast(1.0, dtype) - fallback_used), dtype)
    signature_gate_applied = tf.cast(diagnostics.get("signature_gate_applied", zero), dtype)
    fallback_reason_code = tf.cast(diagnostics.get("fallback_reason_code", zero), dtype)
    tol_t = tf.cast(diagnostics.get("tol_t", zero), dtype)
    ft_residual_norm = tf.cast(diagnostics.get("ft_residual_norm", ft_norm), dtype)
    effective_alpha_scale = tf.cast(diagnostics.get("effective_alpha_scale", zero), dtype)
    tail_has_effective_step = tf.cast(diagnostics.get("tail_has_effective_step", zero), dtype)
    tangential_step_mode = ""
    fallback_trigger_reason = ""
    signature_gate_name = str(diagnostics.get("signature_gate_name", "") or "")
    trace = diagnostics.get("iteration_trace")
    if isinstance(trace, dict):
        fallback_trigger_reason = str(trace.get("fallback_trigger_reason", "") or "")
        iterations = trace.get("iterations")
        if isinstance(iterations, (list, tuple)) and len(iterations) > 0 and isinstance(iterations[-1], dict):
            last_iter = iterations[-1]
            tangential_step_mode = str(last_iter.get("tangential_step_mode", "") or "")
            if "effective_alpha_scale" in last_iter:
                effective_alpha_scale = tf.cast(last_iter["effective_alpha_scale"], dtype)
            if "tail_has_effective_step" in last_iter:
                tail_has_effective_step = tf.cast(
                    1.0 if bool(last_iter["tail_has_effective_step"]) else 0.0,
                    dtype,
                )
            if "ft_residual_after" in last_iter:
                ft_residual_norm = tf.cast(last_iter["ft_residual_after"], dtype)

    linearization_contract_valid = linearization_contract_is_valid(linearization, route_mode)
    if linearization_contract_valid and isinstance(linearization, dict) and "residual" in linearization:
        ift_linear_residual = tf.sqrt(
            tf.reduce_sum(tf.square(tf.cast(linearization["residual"], dtype))) + tf.cast(1.0e-20, dtype)
        )
    else:
        ift_linear_residual = zero

    normal_ift_stats = normal_ift_stats_fn(linearization, route_mode)
    strict_contract_mode = str(diagnostics.get("contract_mode", "") or "")
    if not strict_contract_mode and isinstance(linearization, dict):
        strict_contract_mode = str(linearization.get("contract_mode", "") or "")
    if not strict_contract_mode:
        strict_contract_mode = "normal_contact_first"

    strict_tangential_role = str(diagnostics.get("tangential_contract_role", "") or "")
    if not strict_tangential_role and isinstance(linearization, dict):
        strict_tangential_role = str(linearization.get("tangential_contract_role", "") or "")
    if strict_tangential_role == "auxiliary_friction_fixed_point":
        strict_tangential_role = "auxiliary_friction_residual"
    elif not strict_tangential_role:
        strict_tangential_role = "auxiliary_friction_residual"

    parts = {
        "R_t": e_cn,
        "R_tr": e_ct,
        "E_cn": e_cn,
        "E_ct": e_ct,
        "E_bi": e_bi,
        "R_contact_comp": r_contact,
        "R_fric_comp": r_fric,
    }
    stats = {
        "mixed_strict_active": tf.cast(1.0, dtype),
        "mixed_strict_skipped": tf.cast(0.0, dtype),
        "fn_norm": fn_norm,
        "ft_norm": ft_norm,
        "cone_violation": cone_violation,
        "max_penetration": max_penetration,
        "fallback_used": fallback_used,
        "converged": converged,
        "skip_batch": tf.cast(0.0, dtype),
        "inner_converged": converged,
        "inner_skip_batch": tf.cast(0.0, dtype),
        "inner_fn_norm": fn_norm,
        "inner_ft_norm": ft_norm,
        "inner_cone_violation": cone_violation,
        "inner_max_penetration": max_penetration,
        "inner_fb_residual_norm": fb_residual_norm,
        "inner_normal_step_norm": normal_step_norm,
        "inner_tangential_step_norm": tangential_step_norm,
        "inner_fallback_used": fallback_used,
        "signature_gate_applied": signature_gate_applied,
        "fallback_reason_code": fallback_reason_code,
        "tol_t": tol_t,
        "ft_residual_norm": ft_residual_norm,
        "effective_alpha_scale": effective_alpha_scale,
        "tail_has_effective_step": tail_has_effective_step,
        "ift_linear_residual": ift_linear_residual,
        "R_contact_comp": r_contact,
        "R_fric_comp": r_fric,
        "traction_match_n_rms": tf.sqrt(e_cn + tf.cast(1.0e-20, dtype)),
        "traction_match_t_rms": tf.sqrt(e_ct + tf.cast(1.0e-20, dtype)),
        "strict_normal_traction_rms": tf.sqrt(e_cn + tf.cast(1.0e-20, dtype)),
        "strict_tangential_traction_rms": tf.sqrt(e_ct + tf.cast(1.0e-20, dtype)),
        "coupling_phase_traction_scale": phase_traction_scale,
        "strict_effective_traction_scale": effective_traction_scale,
    }
    stats["strict_contract_mode"] = tf.constant(strict_contract_mode, dtype=tf.string)
    stats["strict_primary_coupling"] = tf.constant("normal_traction_consistency", dtype=tf.string)
    stats["strict_tangential_coupling_role"] = tf.constant(strict_tangential_role, dtype=tf.string)
    if tangential_step_mode:
        stats["tangential_step_mode"] = tf.constant(tangential_step_mode, dtype=tf.string)
    if fallback_trigger_reason:
        stats["fallback_trigger_reason"] = tf.constant(fallback_trigger_reason, dtype=tf.string)
    if signature_gate_name:
        stats["signature_gate_name"] = tf.constant(signature_gate_name, dtype=tf.string)
    stats.update(normal_ift_stats)
    stats.update({key: tf.cast(value, dtype) for key, value in policy.as_stats(include_text=False).items()})
    return parts, stats
