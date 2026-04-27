#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stateless strict-bilevel inner-contact solver with explicit state/result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf

from .contact_inner_kernel_primitives import (
    check_contact_feasibility,
    compose_contact_traction,
    friction_fixed_point_residual,
    inner_normal_jacobian,
    inner_normal_residual,
    project_to_coulomb_disk,
    smooth_penetration_target,
)

FALLBACK_REASON_NONE = 0
FALLBACK_REASON_ITERATION_BUDGET_EXHAUSTED = 1
FALLBACK_REASON_POLICY_PENETRATION_GATE = 2
FALLBACK_REASON_NORMAL_FB_RESIDUAL_NOT_REDUCED = 3
FALLBACK_REASON_TANGENTIAL_RESIDUAL_NOT_REDUCED = 4
FALLBACK_REASON_INVALID_DIAG = 5
FALLBACK_REASON_NAN_OR_INF = 6

FALLBACK_REASON_LABELS = {
    FALLBACK_REASON_NONE: "",
    FALLBACK_REASON_ITERATION_BUDGET_EXHAUSTED: "iteration_budget_exhausted",
    FALLBACK_REASON_POLICY_PENETRATION_GATE: "policy_penetration_gate",
    FALLBACK_REASON_NORMAL_FB_RESIDUAL_NOT_REDUCED: "normal_fb_residual_not_reduced",
    FALLBACK_REASON_TANGENTIAL_RESIDUAL_NOT_REDUCED: "tangential_residual_not_reduced",
    FALLBACK_REASON_INVALID_DIAG: "invalid_diag",
    FALLBACK_REASON_NAN_OR_INF: "nan_or_inf",
}

LINEARIZATION_DENSE_MAX_ELEMENTS = 1_000_000
NORMAL_CONTACT_FIRST_CONTRACT_MODE = "normal_contact_first"
NORMAL_CONTACT_FIRST_TANGENTIAL_ROLE = "auxiliary_friction_fixed_point"


@dataclass
class ContactInnerState:
    lambda_n: tf.Tensor
    lambda_t: tf.Tensor
    converged: bool = False
    iters: int = 0
    res_norm: float = 0.0
    fallback_used: bool = False


@dataclass
class ContactInnerResult:
    state: ContactInnerState
    traction_vec: tf.Tensor
    traction_tangent: tf.Tensor
    diagnostics: Dict[str, object]
    linearization: Optional[Dict[str, object]] = None


def _to_float_tensor(x) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _smooth_penetration_target(g_n: tf.Tensor, eps_n: tf.Tensor) -> tf.Tensor:
    """Smooth approximation of max(0, -g_n) derived from the FB kernel at lambda_n=0."""

    return smooth_penetration_target(g_n, eps_n)


def _max_abs(x: tf.Tensor) -> tf.Tensor:
    x = _to_float_tensor(x)
    return tf.reduce_max(tf.abs(x))


def _max_row_norm(x: tf.Tensor) -> tf.Tensor:
    x = _to_float_tensor(x)
    if x.shape.rank == 0:
        return tf.abs(x)
    if x.shape.rank == 1:
        return tf.sqrt(tf.reduce_sum(tf.square(x)) + 1.0e-12)
    row_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1) + 1.0e-12)
    return tf.reduce_max(row_norm)


def _normal_residual_norm(g_n: tf.Tensor, lambda_n: tf.Tensor, eps_n) -> tf.Tensor:
    return _max_abs(inner_normal_residual(g_n, lambda_n, eps_n))


def _tangential_residual_norm(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps_n,
) -> tf.Tensor:
    return _max_abs(
        friction_fixed_point_residual(
            lambda_t,
            ds_t,
            lambda_n,
            mu,
            k_t,
            eps=eps_n,
        )
    )


def _python_scalar(value, cast_fn):
    if isinstance(value, tf.Tensor) and tf.executing_eagerly():
        return cast_fn(value.numpy())
    return value


def _trace_scalar(value):
    if value is None:
        return None
    return _python_scalar(tf.cast(value, tf.float32), float)


def _all_finite(*values) -> tf.Tensor:
    checks = []
    for value in values:
        tensor = _to_float_tensor(value)
        checks.append(tf.reduce_all(tf.math.is_finite(tensor)))
    if not checks:
        return tf.constant(True)
    return tf.reduce_all(tf.stack(checks))


def _tangential_damping_schedule(damping: float):
    """Generate a short acceptance/backtracking schedule for tangential updates."""

    base = float(min(max(damping, 0.0), 1.0))
    if base <= 0.0:
        return [0.0]
    schedule = []
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        alpha = base * scale
        if alpha <= 1.0e-12:
            continue
        if schedule and abs(alpha - schedule[-1]) <= 1.0e-12:
            continue
        schedule.append(alpha)
    return schedule or [base]


def _tangential_tail_schedule(damping: float):
    """Generate smaller fallback alphas for late-stage tangential stagnation."""

    base = float(min(max(damping, 0.0), 1.0))
    if base <= 0.0:
        return []
    schedule = []
    for scale in (0.03125, 0.015625, 0.0078125):
        alpha = base * scale
        if alpha <= 1.0e-12:
            continue
        if schedule and abs(alpha - schedule[-1]) <= 1.0e-12:
            continue
        schedule.append(alpha)
    return schedule


def _tangential_qn_tail_schedule(damping: float):
    """Backtracking schedule for quasi-Newton tangential tail steps."""

    return _tangential_damping_schedule(damping) + _tangential_tail_schedule(damping)


def _regularize_tangential_qn_diag(
    diag: tf.Tensor,
    *,
    floor: float = 1.0e-3,
    gamma: float = 0.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Make a diagonal quasi-Newton approximation safely invertible."""

    diag = _to_float_tensor(diag)
    floor = tf.cast(floor, tf.float32)
    gamma = tf.cast(gamma, tf.float32)
    finite_mask = tf.math.is_finite(diag)
    abs_diag = tf.abs(tf.where(finite_mask, diag, tf.zeros_like(diag)))
    invalid_mask = tf.logical_or(tf.logical_not(finite_mask), abs_diag < floor)
    sign = tf.where(diag < 0.0, -tf.ones_like(diag), tf.ones_like(diag))
    floored = sign * tf.maximum(abs_diag, floor)
    safe_diag = tf.where(floored >= 0.0, floored + gamma, floored - gamma)
    invalid_ratio = tf.reduce_mean(tf.cast(invalid_mask, tf.float32))
    return safe_diag, abs_diag, tf.reduce_min(tf.abs(safe_diag)), invalid_ratio


def _clip_tangential_qn_step(
    step: tf.Tensor,
    residual: tf.Tensor,
    *,
    min_radius: float = 1.0e-3,
) -> tf.Tensor:
    """Apply a light trust region to keep tail quasi-Newton proposals admissible."""

    step = _to_float_tensor(step)
    residual = _to_float_tensor(residual)
    min_radius = tf.cast(min_radius, tf.float32)
    step_norm = tf.sqrt(tf.reduce_sum(tf.square(step), axis=-1, keepdims=True) + 1.0e-12)
    residual_norm = tf.sqrt(tf.reduce_sum(tf.square(residual), axis=-1, keepdims=True) + 1.0e-12)
    trust_radius = tf.maximum(residual_norm, min_radius)
    scale = tf.minimum(1.0, trust_radius / step_norm)
    return step * scale


def _approx_tangential_qn_diag(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps_n,
    *,
    fd_eps: float = 1.0e-3,
) -> tf.Tensor:
    """Approximate the diagonal of dF_t / d lambda_t via forward differences."""

    lambda_t = _to_float_tensor(lambda_t)
    residual = friction_fixed_point_residual(
        lambda_t,
        ds_t,
        lambda_n,
        mu,
        k_t,
        eps=eps_n,
    )
    fd_eps = tf.cast(fd_eps, tf.float32)
    diag_terms = []
    for axis in range(2):
        basis = tf.one_hot(axis, 2, dtype=tf.float32)[None, :]
        perturbed = lambda_t + fd_eps * basis
        perturbed_residual = friction_fixed_point_residual(
            perturbed,
            ds_t,
            lambda_n,
            mu,
            k_t,
            eps=eps_n,
        )
        diag_terms.append((perturbed_residual[:, axis] - residual[:, axis]) / fd_eps)
    return tf.stack(diag_terms, axis=1)


def _stabilized_tangential_qn_step(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps_n,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return a diagonal quasi-Newton tangential step and its diagonal approximation."""

    residual = friction_fixed_point_residual(
        lambda_t,
        ds_t,
        lambda_n,
        mu,
        k_t,
        eps=eps_n,
    )
    diag = _approx_tangential_qn_diag(
        lambda_t,
        ds_t,
        lambda_n,
        mu,
        k_t,
        eps_n,
    )
    safe_diag, raw_abs_diag, min_safe_diag, invalid_ratio = _regularize_tangential_qn_diag(
        diag,
    )
    qn_step = -residual / safe_diag
    clipped_qn_step = tf.where(
        invalid_ratio >= tf.cast(0.5, tf.float32),
        _clip_tangential_qn_step(qn_step, residual),
        qn_step,
    )
    return clipped_qn_step, diag, raw_abs_diag, min_safe_diag, invalid_ratio


def _estimate_tangential_bb_alpha(
    lambda_t: tf.Tensor,
    residual: tf.Tensor,
    prev_lambda_t: tf.Tensor,
    prev_residual: tf.Tensor,
    *,
    min_alpha: float = 1.0e-3,
    max_alpha: float = 8.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Estimate a positive BB-style scalar step from the last accepted tangential update."""

    lambda_t = _to_float_tensor(lambda_t)
    residual = _to_float_tensor(residual)
    prev_lambda_t = _to_float_tensor(prev_lambda_t)
    prev_residual = _to_float_tensor(prev_residual)
    s = tf.reshape(lambda_t - prev_lambda_t, (-1,))
    y = tf.reshape(residual - prev_residual, (-1,))
    ss = tf.reduce_sum(tf.square(s))
    sy = tf.reduce_sum(s * y)
    yy = tf.reduce_sum(tf.square(y))
    min_alpha = tf.cast(min_alpha, tf.float32)
    max_alpha = tf.cast(max_alpha, tf.float32)

    bb1_valid = sy > tf.cast(1.0e-12, tf.float32)
    bb2_valid = tf.logical_and(tf.logical_not(bb1_valid), yy > tf.cast(1.0e-12, tf.float32))
    raw_alpha = tf.where(
        bb1_valid,
        ss / sy,
        tf.where(
            bb2_valid,
            sy / yy,
            tf.cast(0.0, tf.float32),
        ),
    )
    valid = tf.logical_and(
        _all_finite(raw_alpha, ss, sy, yy),
        tf.logical_or(bb1_valid, bb2_valid),
    )
    safe_alpha = tf.clip_by_value(raw_alpha, min_alpha, max_alpha)
    return safe_alpha, raw_alpha, valid


def _stabilized_normal_correction(
    g_n: tf.Tensor,
    lambda_n: tf.Tensor,
    eps_n: tf.Tensor,
    *,
    damping: float,
    cap_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return a clipped diagonal correction that keeps the FB update stable."""

    gain = float(max(0.0, damping))
    cap_scale = float(max(1.0, cap_scale))
    normal_residual = inner_normal_residual(g_n, lambda_n, eps_n)
    normal_diag = inner_normal_jacobian(g_n, lambda_n, eps_n)
    safe_diag = tf.where(
        tf.abs(normal_diag) > 1.0e-6,
        normal_diag,
        tf.ones_like(normal_diag),
    )
    raw_correction = -tf.cast(gain, tf.float32) * normal_residual / safe_diag
    to_target = tf.abs(normal_residual)
    base_max_correction = tf.maximum(to_target, tf.abs(g_n))
    max_correction = tf.cast(max(1.0, gain) * cap_scale, tf.float32) * base_max_correction
    clipped_magnitude = tf.minimum(tf.abs(raw_correction), tf.minimum(max_correction, to_target))
    clipped_correction = tf.sign(raw_correction) * clipped_magnitude
    return clipped_correction, normal_residual, normal_diag


def _fallback_trigger_reason(
    *,
    fallback_used,
    normal_fb_residual_norm,
    tangential_residual_norm,
    max_penetration,
    diag_is_finite,
    all_values_finite,
    normal_reduced,
    tangential_reduced,
    tol_fb: float,
    tol_n: float,
    tol_t: float,
):
    if not tf.executing_eagerly():
        return tf.constant("", dtype=tf.string)

    if not bool(_python_scalar(all_values_finite, bool)):
        return "nan_or_inf"
    if not bool(_python_scalar(diag_is_finite, bool)):
        return "invalid_diag"
    if bool(_python_scalar(fallback_used, bool)):
        if (
            float(_trace_scalar(normal_fb_residual_norm)) > float(tol_fb)
            and not bool(_python_scalar(normal_reduced, bool))
        ):
            return "normal_fb_residual_not_reduced"
        if (
            float(_trace_scalar(tangential_residual_norm)) > float(tol_t)
            and not bool(_python_scalar(tangential_reduced, bool))
        ):
            return "tangential_residual_not_reduced"
        if (
            float(_trace_scalar(normal_fb_residual_norm)) > float(tol_fb)
            or float(_trace_scalar(tangential_residual_norm)) > float(tol_t)
        ):
            return "iteration_budget_exhausted"
    if float(_trace_scalar(max_penetration)) > 1.0e-3:
        return "policy_penetration_gate"
    return "converged"


def _fallback_reason_code(
    *,
    fallback_used,
    normal_fb_residual_norm,
    tangential_residual_norm,
    max_penetration,
    diag_is_finite,
    all_values_finite,
    normal_reduced,
    tangential_reduced,
    tol_fb: float,
    tol_n: float,
    tol_t: float,
) -> tf.Tensor:
    del tol_n

    tol_fb_tensor = tf.cast(tol_fb, tf.float32)
    tol_t_tensor = tf.cast(tol_t, tf.float32)
    reason_code = tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)

    reason_code = tf.where(
        tf.logical_not(all_values_finite),
        tf.constant(FALLBACK_REASON_NAN_OR_INF, dtype=tf.int32),
        reason_code,
    )
    reason_code = tf.where(
        tf.logical_and(
            tf.equal(reason_code, tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)),
            tf.logical_not(diag_is_finite),
        ),
        tf.constant(FALLBACK_REASON_INVALID_DIAG, dtype=tf.int32),
        reason_code,
    )
    normal_failed = tf.logical_and(
        fallback_used,
        tf.logical_and(
            normal_fb_residual_norm > tol_fb_tensor,
            tf.logical_not(normal_reduced),
        ),
    )
    tangential_failed = tf.logical_and(
        fallback_used,
        tf.logical_and(
            tangential_residual_norm > tol_t_tensor,
            tf.logical_not(tangential_reduced),
        ),
    )
    budget_exhausted = tf.logical_and(
        fallback_used,
        tf.logical_or(
            normal_fb_residual_norm > tol_fb_tensor,
            tangential_residual_norm > tol_t_tensor,
        ),
    )
    reason_code = tf.where(
        tf.logical_and(
            tf.equal(reason_code, tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)),
            normal_failed,
        ),
        tf.constant(FALLBACK_REASON_NORMAL_FB_RESIDUAL_NOT_REDUCED, dtype=tf.int32),
        reason_code,
    )
    reason_code = tf.where(
        tf.logical_and(
            tf.equal(reason_code, tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)),
            tangential_failed,
        ),
        tf.constant(FALLBACK_REASON_TANGENTIAL_RESIDUAL_NOT_REDUCED, dtype=tf.int32),
        reason_code,
    )
    reason_code = tf.where(
        tf.logical_and(
            tf.equal(reason_code, tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)),
            budget_exhausted,
        ),
        tf.constant(FALLBACK_REASON_ITERATION_BUDGET_EXHAUSTED, dtype=tf.int32),
        reason_code,
    )
    reason_code = tf.where(
        tf.logical_and(
            tf.equal(reason_code, tf.constant(FALLBACK_REASON_NONE, dtype=tf.int32)),
            max_penetration > tf.cast(1.0e-3, tf.float32),
        ),
        tf.constant(FALLBACK_REASON_POLICY_PENETRATION_GATE, dtype=tf.int32),
        reason_code,
    )
    return reason_code


def flatten_contact_state(lambda_n: tf.Tensor, lambda_t: tf.Tensor) -> tf.Tensor:
    """Flatten `[lambda_n, lambda_t]` using a fixed `[N, N*2]` ordering."""

    return tf.concat(
        [
            tf.reshape(tf.cast(lambda_n, tf.float32), (-1,)),
            tf.reshape(tf.cast(lambda_t, tf.float32), (-1,)),
        ],
        axis=0,
    )


def flatten_contact_inputs(g_n: tf.Tensor, ds_t: tf.Tensor) -> tf.Tensor:
    """Flatten `[g_n, ds_t]` using a fixed `[N, N*2]` ordering."""

    return tf.concat(
        [
            tf.reshape(tf.cast(g_n, tf.float32), (-1,)),
            tf.reshape(tf.cast(ds_t, tf.float32), (-1,)),
        ],
        axis=0,
    )


def _flatten_jacobian_block(jacobian: tf.Tensor, output_size: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(jacobian, tf.float32), (output_size, -1))


def _normal_residual_input_jacobian(g_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Derivative of the inner normal residual with respect to the gap input."""

    g_n = _to_float_tensor(g_n)
    eps_n = _to_float_tensor(eps_n)
    denom = tf.sqrt(g_n * g_n + eps_n * eps_n)
    return 0.5 * (1.0 - g_n / (denom + 1.0e-12))


def _pack_sparse_entries(rows: tf.Tensor, cols: tf.Tensor, values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    rows = tf.reshape(tf.cast(rows, tf.int64), (-1,))
    cols = tf.reshape(tf.cast(cols, tf.int64), (-1,))
    values = tf.reshape(tf.cast(values, tf.float32), (-1,))
    keep_mask = tf.logical_or(
        tf.math.not_equal(values, tf.cast(0.0, tf.float32)),
        tf.logical_not(tf.math.is_finite(values)),
    )
    return (
        tf.stack(
            [
                tf.boolean_mask(rows, keep_mask),
                tf.boolean_mask(cols, keep_mask),
            ],
            axis=1,
        ),
        tf.boolean_mask(values, keep_mask),
    )


def _concat_sparse_blocks(
    blocks: Tuple[Tuple[tf.Tensor, tf.Tensor], ...],
    *,
    dense_shape: tf.Tensor,
) -> tf.SparseTensor:
    if not blocks:
        return tf.SparseTensor(
            indices=tf.zeros((0, 2), dtype=tf.int64),
            values=tf.zeros((0,), dtype=tf.float32),
            dense_shape=tf.cast(dense_shape, tf.int64),
        )
    indices = tf.concat([indices for indices, _ in blocks], axis=0)
    values = tf.concat([values for _, values in blocks], axis=0)
    return tf.sparse.reorder(
        tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=tf.cast(dense_shape, tf.int64),
        )
    )


def _tangential_linearization_blocks(
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    *,
    mu,
    k_t,
    eps_n,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return exact per-contact Jacobian blocks for the tangential residual."""

    lambda_t_lin = tf.identity(tf.cast(lambda_t, tf.float32))
    ds_t_lin = tf.identity(tf.cast(ds_t, tf.float32))
    lambda_n_lin = tf.reshape(tf.identity(tf.cast(lambda_n, tf.float32)), (-1, 1))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambda_t_lin)
        tape.watch(ds_t_lin)
        tape.watch(lambda_n_lin)
        tangential_residual = friction_fixed_point_residual(
            lambda_t_lin,
            ds_t_lin,
            tf.reshape(lambda_n_lin, (-1,)),
            mu,
            k_t,
            eps=eps_n,
        )
    jac_lambda_t = tf.cast(
        tape.batch_jacobian(
            tangential_residual,
            lambda_t_lin,
            parallel_iterations=1,
            experimental_use_pfor=False,
        ),
        tf.float32,
    )
    jac_ds_t = tf.cast(
        tape.batch_jacobian(
            tangential_residual,
            ds_t_lin,
            parallel_iterations=1,
            experimental_use_pfor=False,
        ),
        tf.float32,
    )
    jac_lambda_n = tf.cast(
        tape.batch_jacobian(
            tangential_residual,
            lambda_n_lin,
            parallel_iterations=1,
            experimental_use_pfor=False,
        ),
        tf.float32,
    )
    del tape
    return jac_lambda_n, jac_lambda_t, jac_ds_t


def _build_contact_linearization_jacobians(
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    g_n: tf.Tensor,
    ds_t: tf.Tensor,
    *,
    mu,
    k_t,
    eps_n,
) -> Tuple[tf.SparseTensor, tf.SparseTensor, Dict[str, tf.Tensor]]:
    """Build exact sparse Jacobians for the strict mixed linearization payload."""

    lambda_n = tf.cast(lambda_n, tf.float32)
    lambda_t = tf.cast(lambda_t, tf.float32)
    g_n = tf.cast(g_n, tf.float32)
    ds_t = tf.cast(ds_t, tf.float32)
    n = tf.shape(lambda_n, out_type=tf.int64)[0]
    row_base = tf.range(n, dtype=tf.int64)
    tangential_row_base = n + 2 * row_base
    tangential_col_base = n + 2 * row_base
    tangential_rows = tf.stack([tangential_row_base, tangential_row_base + 1], axis=1)
    tangential_cols = tf.stack([tangential_col_base, tangential_col_base + 1], axis=1)

    normal_state_block = _pack_sparse_entries(
        row_base,
        row_base,
        tf.ones_like(lambda_n, dtype=tf.float32),
    )
    normal_input_block = _pack_sparse_entries(
        row_base,
        row_base,
        _normal_residual_input_jacobian(g_n, eps_n),
    )

    tangential_lambda_n, tangential_lambda_t, tangential_ds_t = _tangential_linearization_blocks(
        lambda_n,
        lambda_t,
        ds_t,
        mu=mu,
        k_t=k_t,
        eps_n=eps_n,
    )
    tangential_lambda_n_block = _pack_sparse_entries(
        tangential_rows,
        tf.broadcast_to(row_base[:, None], tf.shape(tangential_rows)),
        tf.squeeze(tangential_lambda_n, axis=-1),
    )
    tangential_lambda_t_block = _pack_sparse_entries(
        tf.broadcast_to(tangential_rows[:, :, None], tf.shape(tangential_lambda_t)),
        tf.broadcast_to(tangential_cols[:, None, :], tf.shape(tangential_lambda_t)),
        tangential_lambda_t,
    )
    tangential_ds_t_block = _pack_sparse_entries(
        tf.broadcast_to(tangential_rows[:, :, None], tf.shape(tangential_ds_t)),
        tf.broadcast_to(tangential_cols[:, None, :], tf.shape(tangential_ds_t)),
        tangential_ds_t,
    )

    block_jacobians = {
        "normal_state_diag": tf.ones_like(lambda_n, dtype=tf.float32),
        "normal_input_diag": _normal_residual_input_jacobian(g_n, eps_n),
        "tangential_lambda_n": tangential_lambda_n,
        "tangential_lambda_t": tangential_lambda_t,
        "tangential_ds_t": tangential_ds_t,
    }

    dense_shape = tf.stack([3 * n, 3 * n], axis=0)
    jac_z = _concat_sparse_blocks(
        (
            normal_state_block,
            tangential_lambda_n_block,
            tangential_lambda_t_block,
        ),
        dense_shape=dense_shape,
    )
    jac_inputs = _concat_sparse_blocks(
        (
            normal_input_block,
            tangential_ds_t_block,
        ),
        dense_shape=dense_shape,
    )
    return jac_z, jac_inputs, block_jacobians


def _known_dim0(tensor: tf.Tensor) -> Optional[int]:
    static_dim = tensor.shape[0]
    if static_dim is not None:
        return int(static_dim)
    if tf.executing_eagerly():
        return int(tf.shape(tensor, out_type=tf.int64)[0].numpy())
    return None


def _materialize_dense_linearization(output_size: Optional[int], width: Optional[int]) -> bool:
    if output_size is None or width is None:
        return False
    return int(output_size) * int(width) <= int(LINEARIZATION_DENSE_MAX_ELEMENTS)


def solve_contact_inner(
    g_n: tf.Tensor,
    ds_t: tf.Tensor,
    normals: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
    *,
    mu,
    eps_n,
    k_t,
    init_state: Optional[ContactInnerState] = None,
    init_state_available=None,
    return_linearization: bool = False,
    return_iteration_trace: bool = False,
    tol_n: float = 1.0e-5,
    tol_t: float = 1.0e-5,
    tol_fb: Optional[float] = None,
    max_inner_iters: int = 8,
    max_tail_qn_iters: int = 0,
    damping: float = 1.0,
    normal_correction_cap_scale: float = 1.0,
) -> ContactInnerResult:
    """Solve a geometry-driven strict-bilevel inner contact state without hidden mutable state."""

    g_n = _to_float_tensor(g_n)
    ds_t = _to_float_tensor(ds_t)
    normals = _to_float_tensor(normals)
    t1 = _to_float_tensor(t1)
    t2 = _to_float_tensor(t2)
    mu = _to_float_tensor(mu)
    eps_n = _to_float_tensor(eps_n)
    k_t = _to_float_tensor(k_t)
    damping = float(max(0.0, damping))
    tol_fb = float(tol_n if tol_fb is None else tol_fb)

    target_lambda_n = tf.maximum(_smooth_penetration_target(g_n, eps_n), 0.0)
    if init_state is not None:
        init_lambda_n_raw = _to_float_tensor(init_state.lambda_n)
        init_lambda_t_raw = _to_float_tensor(init_state.lambda_t)
    else:
        init_lambda_n_raw = tf.zeros_like(target_lambda_n)
        init_lambda_t_raw = tf.zeros_like(ds_t)
    if init_state_available is None:
        init_state_available = init_state is not None
    init_state_available = tf.cast(init_state_available, tf.bool)
    init_lambda_n = tf.where(init_state_available, init_lambda_n_raw, tf.zeros_like(target_lambda_n))
    init_lambda_t = tf.where(init_state_available, init_lambda_t_raw, tf.zeros_like(ds_t))

    lambda_n = init_lambda_n
    lambda_t = init_lambda_t
    normal_step = tf.zeros_like(lambda_n)
    tangential_step = tf.zeros_like(lambda_t)
    tangential_residual = friction_fixed_point_residual(
        lambda_t,
        ds_t,
        lambda_n,
        mu,
        k_t,
        eps=eps_n,
    )
    converged = tf.constant(False)
    iters = tf.constant(0, dtype=tf.int32)
    res_norm = tf.constant(float("inf"), dtype=tf.float32)
    done = tf.constant(False)
    last_normal_step_norm = tf.constant(0.0, dtype=tf.float32)
    last_normal_fb_residual_norm = _normal_residual_norm(g_n, lambda_n, eps_n)
    last_tangential_residual_norm = _max_abs(tangential_residual)
    last_effective_alpha_scale = tf.constant(0.0, dtype=tf.float32)
    last_tail_has_effective_step = tf.constant(0.0, dtype=tf.float32)
    last_normal_reduced = tf.constant(False)
    last_tangential_reduced = tf.constant(False)
    any_normal_reduced = tf.constant(False)
    any_tangential_reduced = tf.constant(False)
    last_feasibility = check_contact_feasibility(
        g_n,
        lambda_n,
        lambda_t,
        mu,
        tol_n=tol_n,
        tol_t=tol_t,
    )
    last_diag_is_finite = tf.constant(True)
    last_all_values_finite = _all_finite(g_n, lambda_n, lambda_t, tangential_residual)
    prev_tangential_lambda = tf.identity(lambda_t)
    prev_tangential_residual = tf.identity(tangential_residual)
    prev_tangential_history_valid = tf.constant(False)
    iteration_trace = [] if return_iteration_trace and tf.executing_eagerly() else None
    base_inner_iters = max(0, int(max_inner_iters))
    extra_tail_qn_iters = max(0, int(max_tail_qn_iters))
    total_inner_iters = base_inner_iters + extra_tail_qn_iters
    tail_budget_continue = tf.constant(False)
    tail_budget_activated = tf.constant(False)
    tail_extra_iters_granted = tf.constant(0, dtype=tf.int32)
    for it in range(total_inner_iters):
        within_base_budget = it < base_inner_iters
        active_iter = (
            tf.logical_not(done)
            if within_base_budget
            else tf.logical_and(tf.logical_not(done), tail_budget_continue)
        )
        lambda_t_before_state = tf.identity(lambda_t)
        normal_fb_residual_before = _normal_residual_norm(g_n, lambda_n, eps_n)
        fn_residual_before = None
        ft_residual_before = None
        lambda_t_before_norm = None
        target_lambda_t_norm = None
        cone_violation_before = None
        slip_norm = None
        if return_iteration_trace:
            fn_residual_before = normal_fb_residual_before
            lambda_t_before_norm = _max_row_norm(lambda_t)
            cone_violation_before = check_contact_feasibility(
                g_n,
                lambda_n,
                lambda_t,
                mu,
                tol_n=tol_n,
                tol_t=tol_t,
            )["cone_violation"]
            slip_norm = _max_row_norm(ds_t)
        normal_correction, normal_fb_residual, normal_diag = _stabilized_normal_correction(
            g_n,
            lambda_n,
            eps_n,
            damping=damping,
            cap_scale=normal_correction_cap_scale,
        )
        next_lambda_n = tf.maximum(lambda_n + normal_correction, 0.0)
        tangential_residual_before = friction_fixed_point_residual(
            lambda_t,
            ds_t,
            next_lambda_n,
            mu,
            k_t,
            eps=eps_n,
        )
        tangential_residual_before_norm = _max_abs(tangential_residual_before)
        if return_iteration_trace:
            ft_residual_before = tangential_residual_before_norm
        target_lambda_t = project_to_coulomb_disk(
            lambda_t - tangential_residual_before,
            mu * next_lambda_n,
            eps=eps_n,
        )
        accepted_target_lambda_t = tf.identity(target_lambda_t)
        if return_iteration_trace:
            target_lambda_t_norm = _max_row_norm(target_lambda_t)
        next_lambda_t = tf.identity(lambda_t)
        tangential_residual = tangential_residual_before
        tangential_step = tf.zeros_like(lambda_t)
        tangential_reduced = tf.constant(False)
        accepted_k_t_scale = tf.constant(1.0, dtype=tf.float32)
        accepted_alpha_scale = tf.constant(0.0, dtype=tf.float32)
        tail_has_effective_step = tf.constant(False)
        qn_diag_min_raw = None
        qn_diag_min_safe = None
        qn_reg_gamma = None
        qn_invalid_ratio = None
        qn_diag_min = None
        qn_diag_max = None
        qn_step_norm = None
        qn_available = tf.constant(False)
        base_schedule = _tangential_damping_schedule(damping)
        tail_schedule = _tangential_tail_schedule(damping)
        tangential_backtrack_steps = tf.cast(len(base_schedule), tf.int32)
        tangential_step_mode = "residual_driven"
        for alpha_index, tangential_alpha in enumerate(base_schedule):
            candidate_lambda_t = project_to_coulomb_disk(
                lambda_t - tf.cast(tangential_alpha, tf.float32) * tangential_residual_before,
                mu * next_lambda_n,
                eps=eps_n,
            )
            candidate_residual = friction_fixed_point_residual(
                candidate_lambda_t,
                ds_t,
                next_lambda_n,
                mu,
                k_t,
                eps=eps_n,
            )
            candidate_residual_norm = _max_abs(candidate_residual)
            take_candidate = tf.logical_and(
                tf.logical_not(tangential_reduced),
                candidate_residual_norm < (tangential_residual_before_norm - tf.cast(1.0e-12, tf.float32)),
            )
            next_lambda_t = tf.where(take_candidate, candidate_lambda_t, next_lambda_t)
            tangential_residual = tf.where(take_candidate, candidate_residual, tangential_residual)
            tangential_step = tf.where(take_candidate, candidate_lambda_t - lambda_t, tangential_step)
            accepted_target_lambda_t = tf.where(
                take_candidate,
                candidate_lambda_t,
                accepted_target_lambda_t,
            )
            accepted_alpha_scale = tf.where(
                take_candidate,
                tf.cast(tangential_alpha, tf.float32),
                accepted_alpha_scale,
            )
            tangential_backtrack_steps = tf.where(
                take_candidate,
                tf.cast(alpha_index, tf.int32),
                tangential_backtrack_steps,
            )
            tangential_reduced = tf.logical_or(tangential_reduced, take_candidate)
        if tail_schedule:
            tail_needed = tf.logical_not(tangential_reduced)
            bb_alpha, _, bb_available = _estimate_tangential_bb_alpha(
                lambda_t,
                tangential_residual_before,
                prev_tangential_lambda,
                prev_tangential_residual,
            )
            bb_available = tf.logical_and(
                tail_needed,
                tf.logical_and(prev_tangential_history_valid, bb_available),
            )
            bb_candidate_lambda_t = project_to_coulomb_disk(
                lambda_t - bb_alpha * tangential_residual_before,
                mu * next_lambda_n,
                eps=eps_n,
            )
            bb_candidate_residual = friction_fixed_point_residual(
                bb_candidate_lambda_t,
                ds_t,
                next_lambda_n,
                mu,
                k_t,
                eps=eps_n,
            )
            bb_candidate_residual_norm = _max_abs(bb_candidate_residual)
            bb_take_candidate = tf.logical_and(
                bb_available,
                tf.logical_and(
                    tf.logical_not(tangential_reduced),
                    bb_candidate_residual_norm < (tangential_residual_before_norm - tf.cast(1.0e-12, tf.float32)),
                ),
            )
            next_lambda_t = tf.where(bb_take_candidate, bb_candidate_lambda_t, next_lambda_t)
            tangential_residual = tf.where(bb_take_candidate, bb_candidate_residual, tangential_residual)
            tangential_step = tf.where(bb_take_candidate, bb_candidate_lambda_t - lambda_t, tangential_step)
            accepted_target_lambda_t = tf.where(
                bb_take_candidate,
                bb_candidate_lambda_t,
                accepted_target_lambda_t,
            )
            accepted_alpha_scale = tf.where(
                bb_take_candidate,
                bb_alpha,
                accepted_alpha_scale,
            )
            tangential_backtrack_steps = tf.where(
                bb_take_candidate,
                tf.cast(len(base_schedule), tf.int32),
                tangential_backtrack_steps,
            )
            tail_has_effective_step = tf.logical_or(tail_has_effective_step, bb_take_candidate)
            tangential_reduced = tf.logical_or(tangential_reduced, bb_take_candidate)
            if return_iteration_trace and tf.executing_eagerly() and bool(_python_scalar(bb_take_candidate, bool)):
                tangential_step_mode = "residual_driven_tail_bb"
            qn_step, qn_diag, qn_diag_abs, qn_diag_safe_min, qn_invalid_ratio = _stabilized_tangential_qn_step(
                lambda_t,
                ds_t,
                next_lambda_n,
                mu,
                k_t,
                eps_n,
            )
            qn_available = tf.logical_and(
                _all_finite(qn_step, qn_diag),
                tf.reduce_any(tf.abs(qn_step) > tf.cast(1.0e-12, tf.float32)),
            )
            qn_diag_min_raw = tf.reduce_min(qn_diag_abs)
            qn_diag_min_safe = qn_diag_safe_min
            qn_reg_gamma = tf.cast(0.0, tf.float32)
            qn_diag_min = qn_diag_min_raw
            qn_diag_max = tf.reduce_max(qn_diag_abs)
            qn_step_norm = _max_row_norm(qn_step)
            if return_iteration_trace and tf.executing_eagerly() and bool(
                _python_scalar(tf.logical_and(tail_needed, tf.logical_not(tangential_reduced)), bool)
            ):
                tangential_step_mode = (
                    "residual_driven_tail_qn"
                    if bool(_python_scalar(qn_available, bool))
                    else "residual_driven_tail"
                )
            qn_tail_schedule = _tangential_qn_tail_schedule(damping)
            tangential_backtrack_steps = tf.where(
                tf.logical_and(tf.logical_and(tail_needed, tf.logical_not(tangential_reduced)), qn_available),
                tf.cast(len(base_schedule) + len(qn_tail_schedule), tf.int32),
                tangential_backtrack_steps,
            )
            for tail_offset, tangential_alpha in enumerate(qn_tail_schedule, start=len(base_schedule)):
                candidate_lambda_t = project_to_coulomb_disk(
                    lambda_t + tf.cast(tangential_alpha, tf.float32) * qn_step,
                    mu * next_lambda_n,
                    eps=eps_n,
                )
                candidate_residual = friction_fixed_point_residual(
                    candidate_lambda_t,
                    ds_t,
                    next_lambda_n,
                    mu,
                    k_t,
                    eps=eps_n,
                )
                candidate_residual_norm = _max_abs(candidate_residual)
                take_candidate = tf.logical_and(
                    tf.logical_and(tail_needed, qn_available),
                    tf.logical_and(
                        tf.logical_not(tangential_reduced),
                        candidate_residual_norm < (tangential_residual_before_norm - tf.cast(1.0e-12, tf.float32)),
                    ),
                )
                next_lambda_t = tf.where(take_candidate, candidate_lambda_t, next_lambda_t)
                tangential_residual = tf.where(take_candidate, candidate_residual, tangential_residual)
                tangential_step = tf.where(take_candidate, candidate_lambda_t - lambda_t, tangential_step)
                accepted_target_lambda_t = tf.where(
                    take_candidate,
                    candidate_lambda_t,
                    accepted_target_lambda_t,
                )
                accepted_alpha_scale = tf.where(
                    take_candidate,
                    tf.cast(tangential_alpha, tf.float32),
                    accepted_alpha_scale,
                )
                tangential_backtrack_steps = tf.where(
                    take_candidate,
                    tf.cast(tail_offset, tf.int32),
                    tangential_backtrack_steps,
                )
                tail_has_effective_step = tf.logical_or(tail_has_effective_step, take_candidate)
                tangential_reduced = tf.logical_or(tangential_reduced, take_candidate)

            tangential_backtrack_steps = tf.where(
                tf.logical_and(
                    tf.logical_and(tail_needed, tf.logical_not(tangential_reduced)),
                    tf.logical_not(qn_available),
                ),
                tf.cast(len(base_schedule) + len(tail_schedule), tf.int32),
                tangential_backtrack_steps,
            )
            for tail_offset, tangential_alpha in enumerate(tail_schedule, start=len(base_schedule)):
                candidate_lambda_t = project_to_coulomb_disk(
                    lambda_t - tf.cast(tangential_alpha, tf.float32) * tangential_residual_before,
                    mu * next_lambda_n,
                    eps=eps_n,
                )
                candidate_residual = friction_fixed_point_residual(
                    candidate_lambda_t,
                    ds_t,
                    next_lambda_n,
                    mu,
                    k_t,
                    eps=eps_n,
                )
                candidate_residual_norm = _max_abs(candidate_residual)
                take_candidate = tf.logical_and(
                    tf.logical_and(tail_needed, tf.logical_not(qn_available)),
                    tf.logical_and(
                        tf.logical_not(tangential_reduced),
                        candidate_residual_norm < (tangential_residual_before_norm - tf.cast(1.0e-12, tf.float32)),
                    ),
                )
                next_lambda_t = tf.where(take_candidate, candidate_lambda_t, next_lambda_t)
                tangential_residual = tf.where(take_candidate, candidate_residual, tangential_residual)
                tangential_step = tf.where(take_candidate, candidate_lambda_t - lambda_t, tangential_step)
                accepted_target_lambda_t = tf.where(
                    take_candidate,
                    candidate_lambda_t,
                    accepted_target_lambda_t,
                )
                accepted_alpha_scale = tf.where(
                    take_candidate,
                    tf.cast(tangential_alpha, tf.float32),
                    accepted_alpha_scale,
                )
                tangential_backtrack_steps = tf.where(
                    take_candidate,
                    tf.cast(tail_offset, tf.int32),
                    tangential_backtrack_steps,
                )
                tail_has_effective_step = tf.logical_or(tail_has_effective_step, take_candidate)
                tangential_reduced = tf.logical_or(tangential_reduced, take_candidate)
        if return_iteration_trace:
            target_lambda_t_norm = tf.where(
                tangential_reduced,
                _max_row_norm(accepted_target_lambda_t),
                target_lambda_t_norm,
            )

        normal_step = next_lambda_n - lambda_n
        feasibility = check_contact_feasibility(
            g_n,
            next_lambda_n,
            next_lambda_t,
            mu,
            tol_n=tol_n,
            tol_t=tol_t,
        )
        normal_step_norm = _max_abs(normal_step)
        normal_fb_residual_norm = _normal_residual_norm(g_n, next_lambda_n, eps_n)
        tangential_residual_norm = _max_abs(tangential_residual)
        tangential_step_norm = _max_abs(tangential_step)
        last_normal_step_norm = tf.where(active_iter, normal_step_norm, last_normal_step_norm)
        last_normal_fb_residual_norm = tf.where(
            active_iter,
            normal_fb_residual_norm,
            last_normal_fb_residual_norm,
        )
        last_tangential_residual_norm = tf.where(
            active_iter,
            tangential_residual_norm,
            last_tangential_residual_norm,
        )
        last_effective_alpha_scale = tf.where(
            active_iter,
            accepted_alpha_scale,
            last_effective_alpha_scale,
        )
        last_tail_has_effective_step = tf.where(
            active_iter,
            tf.cast(tail_has_effective_step, tf.float32),
            last_tail_has_effective_step,
        )
        last_normal_reduced = tf.logical_and(
            active_iter,
            normal_fb_residual_norm < (normal_fb_residual_before - tf.cast(1.0e-12, tf.float32)),
        )
        last_tangential_reduced = tf.logical_and(active_iter, tangential_reduced)
        any_normal_reduced = tf.logical_or(any_normal_reduced, last_normal_reduced)
        any_tangential_reduced = tf.logical_or(any_tangential_reduced, last_tangential_reduced)
        last_feasibility = {
            "lambda_neg_violation": tf.where(
                active_iter,
                feasibility["lambda_neg_violation"],
                last_feasibility["lambda_neg_violation"],
            ),
            "cone_violation": tf.where(
                active_iter,
                feasibility["cone_violation"],
                last_feasibility["cone_violation"],
            ),
            "max_penetration": tf.where(
                active_iter,
                feasibility["max_penetration"],
                last_feasibility["max_penetration"],
            ),
            "feasible": tf.where(
                active_iter,
                feasibility["feasible"],
                last_feasibility["feasible"],
            ),
        }
        last_diag_is_finite = tf.where(active_iter, _all_finite(normal_diag), last_diag_is_finite)
        last_all_values_finite = tf.where(
            active_iter,
            _all_finite(
                next_lambda_n,
                next_lambda_t,
                normal_fb_residual,
                tangential_residual,
                normal_diag,
            ),
            last_all_values_finite,
        )
        res_norm_candidate = tf.maximum(
            normal_fb_residual_norm,
            tf.maximum(
                normal_step_norm,
                tf.maximum(tangential_step_norm, tangential_residual_norm),
            ),
        )
        res_norm = tf.where(active_iter, res_norm_candidate, res_norm)
        this_converged = tf.logical_and(
            normal_step_norm <= tf.cast(tol_n, tf.float32),
            tf.logical_and(
                normal_fb_residual_norm <= tf.cast(tol_fb, tf.float32),
                tf.logical_and(
                    tangential_residual_norm <= tf.cast(tol_t, tf.float32),
                    tf.cast(feasibility["feasible"], tf.bool),
                ),
            ),
        )
        update = tf.cast(active_iter, tf.float32)
        lambda_n = update * next_lambda_n + (1.0 - update) * lambda_n
        lambda_t = update * next_lambda_t + (1.0 - update) * lambda_t
        iters = tf.where(active_iter, tf.cast(it + 1, tf.int32), iters)
        converged = tf.logical_or(converged, tf.logical_and(active_iter, this_converged))
        done = tf.logical_or(done, tf.logical_and(active_iter, this_converged))
        if iteration_trace is not None and bool(_python_scalar(active_iter, bool)):
            iteration_trace.append(
                {
                    "iter": int(it + 1),
                    "fn_residual_before": _trace_scalar(fn_residual_before),
                    "fn_residual_after": _trace_scalar(_normal_residual_norm(g_n, next_lambda_n, eps_n)),
                    "ft_residual_before": _trace_scalar(ft_residual_before),
                    "ft_residual_after": _trace_scalar(tangential_residual_norm),
                    "delta_lambda_n_norm": _trace_scalar(normal_step_norm),
                    "delta_lambda_t_norm": _trace_scalar(tangential_step_norm),
                    "lambda_t_before_norm": _trace_scalar(lambda_t_before_norm),
                    "lambda_t_after_norm": _trace_scalar(_max_row_norm(next_lambda_t)),
                    "cone_violation_before": _trace_scalar(cone_violation_before),
                    "cone_violation_after": _trace_scalar(feasibility["cone_violation"]),
                    "slip_norm": _trace_scalar(slip_norm),
                    "target_lambda_t_norm": _trace_scalar(target_lambda_t_norm),
                    "ft_reduction_ratio": _trace_scalar(
                        tangential_residual_norm
                        / tf.maximum(ft_residual_before, tf.cast(1.0e-12, tf.float32))
                    ),
                    "tangential_backtrack_steps": int(_python_scalar(tangential_backtrack_steps, int)),
                    "effective_k_t_scale": _trace_scalar(accepted_k_t_scale),
                    "tangential_step_mode": tangential_step_mode,
                    "effective_alpha_scale": _trace_scalar(accepted_alpha_scale),
                    "qn_diag_min_raw": _trace_scalar(qn_diag_min_raw),
                    "qn_diag_min_safe": _trace_scalar(qn_diag_min_safe),
                    "qn_reg_gamma": _trace_scalar(qn_reg_gamma),
                    "qn_invalid_ratio": _trace_scalar(qn_invalid_ratio),
                    "qn_diag_min": _trace_scalar(qn_diag_min),
                    "qn_diag_max": _trace_scalar(qn_diag_max),
                    "qn_step_norm": _trace_scalar(qn_step_norm),
                    "tail_has_effective_step": bool(_python_scalar(tail_has_effective_step, bool)),
                }
            )
        store_tangential_history = tf.logical_and(active_iter, tangential_reduced)
        prev_tangential_lambda = tf.where(
            store_tangential_history,
            lambda_t_before_state,
            prev_tangential_lambda,
        )
        prev_tangential_residual = tf.where(
            store_tangential_history,
            tangential_residual_before,
            prev_tangential_residual,
        )
        prev_tangential_history_valid = tf.logical_or(
            prev_tangential_history_valid,
            store_tangential_history,
        )
        tail_effective_step = tf.logical_and(
            active_iter,
            tail_has_effective_step,
        )
        if within_base_budget:
            if (it + 1) == base_inner_iters:
                tail_budget_continue = tf.logical_and(
                    tf.cast(extra_tail_qn_iters > 0, tf.bool),
                    tail_effective_step,
                )
            else:
                tail_budget_continue = tf.constant(False)
        else:
            tail_budget_activated = tf.logical_or(tail_budget_activated, active_iter)
            tail_extra_iters_granted = tf.where(
                active_iter,
                tail_extra_iters_granted + 1,
                tail_extra_iters_granted,
            )
            tail_budget_continue = tf.logical_and(active_iter, last_tangential_reduced)

    fallback_used = tf.logical_not(converged)
    reused_init = tf.constant(False)
    if init_state is not None:
        init_feasibility = check_contact_feasibility(
            g_n,
            init_lambda_n,
            init_lambda_t,
            mu,
            tol_n=tol_n,
            tol_t=tol_t,
        )
        reused_init = tf.logical_and(
            fallback_used,
            tf.logical_and(init_state_available, tf.cast(init_feasibility["feasible"], tf.bool)),
        )

    projected_lambda_t = project_to_coulomb_disk(k_t * ds_t, mu * target_lambda_n, eps=eps_n)
    fallback_lambda_n = tf.cond(reused_init, lambda: init_lambda_n, lambda: target_lambda_n)
    fallback_lambda_t = tf.cond(reused_init, lambda: init_lambda_t, lambda: projected_lambda_t)
    fallback_res_norm = tf.maximum(
        _normal_residual_norm(g_n, fallback_lambda_n, eps_n),
        _max_abs(
            friction_fixed_point_residual(
                fallback_lambda_t,
                ds_t,
                fallback_lambda_n,
                mu,
                k_t,
                eps=eps_n,
            )
        ),
    )
    keep_iterate = tf.logical_and(
        fallback_used,
        tf.logical_or(any_normal_reduced, any_tangential_reduced),
    )
    lambda_n = tf.cond(
        tf.logical_and(fallback_used, tf.logical_not(keep_iterate)),
        lambda: fallback_lambda_n,
        lambda: lambda_n,
    )
    lambda_t = tf.cond(
        tf.logical_and(fallback_used, tf.logical_not(keep_iterate)),
        lambda: fallback_lambda_t,
        lambda: lambda_t,
    )
    res_norm = tf.where(
        tf.logical_and(
            tf.logical_and(fallback_used, tf.logical_not(reused_init)),
            tf.logical_not(keep_iterate),
        ),
        fallback_res_norm,
        res_norm,
    )

    state = ContactInnerState(
        lambda_n=lambda_n,
        lambda_t=lambda_t,
        converged=_python_scalar(converged, bool),
        iters=_python_scalar(iters, int),
        res_norm=_python_scalar(tf.cast(res_norm, tf.float32), float),
        fallback_used=_python_scalar(fallback_used, bool),
    )

    traction_vec = compose_contact_traction(state.lambda_n, state.lambda_t, normals, t1, t2)
    feasibility = check_contact_feasibility(
        g_n,
        state.lambda_n,
        state.lambda_t,
        mu,
        tol_n=tol_n,
        tol_t=tol_t,
    )
    diagnostics = {
        "contract_mode": NORMAL_CONTACT_FIRST_CONTRACT_MODE,
        "tangential_contract_role": NORMAL_CONTACT_FIRST_TANGENTIAL_ROLE,
        "normal_contact_first_ready": tf.cast(1.0, tf.float32),
        "fn_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_n))),
        "normal_multiplier_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_n))),
        "ft_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_t))),
        "ft_residual_norm": tf.cast(last_tangential_residual_norm, tf.float32),
        "cone_violation": tf.cast(feasibility["cone_violation"], tf.float32),
        "max_penetration": tf.cast(feasibility["max_penetration"], tf.float32),
        "normal_penetration_max": tf.cast(feasibility["max_penetration"], tf.float32),
        "converged": tf.cast(converged, tf.float32),
        "skip_batch": tf.cast(0.0, tf.float32),
        "tol_t": tf.cast(tol_t, tf.float32),
        "fb_residual_norm": tf.sqrt(
            tf.reduce_mean(tf.square(inner_normal_residual(g_n, state.lambda_n, eps_n)))
            + 1.0e-20
        ),
        "normal_residual_norm": tf.sqrt(
            tf.reduce_mean(tf.square(inner_normal_residual(g_n, state.lambda_n, eps_n)))
            + 1.0e-20
        ),
        "normal_step_norm": _max_abs(normal_step),
        "tangential_step_norm": _max_abs(tangential_step),
        "fallback_used": tf.cast(fallback_used, tf.float32),
        "iters": tf.cast(iters, tf.float32),
        "tail_budget_activated": tf.cast(tail_budget_activated, tf.float32),
        "tail_extra_iters_granted": tf.cast(tail_extra_iters_granted, tf.float32),
        "effective_alpha_scale": tf.cast(last_effective_alpha_scale, tf.float32),
        "tail_has_effective_step": tf.cast(last_tail_has_effective_step, tf.float32),
        "fallback_reason_code": _fallback_reason_code(
            fallback_used=fallback_used,
            normal_fb_residual_norm=last_normal_fb_residual_norm,
            tangential_residual_norm=last_tangential_residual_norm,
            max_penetration=feasibility["max_penetration"],
            diag_is_finite=last_diag_is_finite,
            all_values_finite=last_all_values_finite,
            normal_reduced=any_normal_reduced,
            tangential_reduced=any_tangential_reduced,
            tol_fb=tol_fb,
            tol_n=tol_n,
            tol_t=tol_t,
        ),
    }
    if return_iteration_trace:
        diagnostics["iteration_trace"] = {
            "iterations": [] if iteration_trace is None else iteration_trace,
            "fallback_trigger_reason": _fallback_trigger_reason(
                fallback_used=fallback_used,
                normal_fb_residual_norm=last_normal_fb_residual_norm,
                tangential_residual_norm=last_tangential_residual_norm,
                max_penetration=last_feasibility["max_penetration"],
                diag_is_finite=last_diag_is_finite,
                all_values_finite=last_all_values_finite,
                normal_reduced=any_normal_reduced,
                tangential_reduced=any_tangential_reduced,
                tol_fb=tol_fb,
                tol_n=tol_n,
                tol_t=tol_t,
            ),
        }
    linearization = None
    if return_linearization:
        lambda_n_lin = tf.identity(tf.cast(state.lambda_n, tf.float32))
        lambda_t_lin = tf.identity(tf.cast(state.lambda_t, tf.float32))
        g_n_lin = tf.identity(tf.cast(g_n, tf.float32))
        ds_t_lin = tf.identity(tf.cast(ds_t, tf.float32))
        normal_residual = inner_normal_residual(g_n_lin, lambda_n_lin, eps_n)
        tangential_residual = friction_fixed_point_residual(
            lambda_t_lin,
            ds_t_lin,
            lambda_n_lin,
            mu,
            k_t,
            eps=eps_n,
        )
        flat_residual = tf.concat(
            [
                tf.reshape(normal_residual, (-1,)),
                tf.reshape(tangential_residual, (-1,)),
            ],
            axis=0,
        )
        jac_z_sparse, jac_inputs_sparse, block_jacobians = _build_contact_linearization_jacobians(
            lambda_n_lin,
            lambda_t_lin,
            g_n_lin,
            ds_t_lin,
            mu=mu,
            k_t=k_t,
            eps_n=eps_n,
        )
        flat_state = flatten_contact_state(lambda_n_lin, lambda_t_lin)
        flat_inputs = flatten_contact_inputs(g_n_lin, ds_t_lin)
        output_size = _known_dim0(flat_residual)
        state_size = _known_dim0(flat_state)
        input_size = _known_dim0(flat_inputs)
        use_dense_jacobians = _materialize_dense_linearization(output_size, state_size) and _materialize_dense_linearization(
            output_size,
            input_size,
        )
        lambda_n_shape = list(lambda_n_lin.shape.as_list() or [])
        lambda_t_shape = list(lambda_t_lin.shape.as_list() or [])
        g_n_shape = list(g_n_lin.shape.as_list() or [])
        ds_t_shape = list(ds_t_lin.shape.as_list() or [])
        linearization = {
            "schema_version": "strict_mixed_v2",
            "route_mode": "normal_ready",
            "contract_mode": NORMAL_CONTACT_FIRST_CONTRACT_MODE,
            "primary_state": "lambda_n",
            "tangential_contract_role": NORMAL_CONTACT_FIRST_TANGENTIAL_ROLE,
            "is_exact": False,
            "tangential_mode": "smooth_not_enabled",
            "jac_z": tf.sparse.to_dense(jac_z_sparse) if use_dense_jacobians else jac_z_sparse,
            "jac_inputs": tf.sparse.to_dense(jac_inputs_sparse) if use_dense_jacobians else jac_inputs_sparse,
            "state_layout": {
                "order": ["lambda_n", "lambda_t"],
                "lambda_n_shape": lambda_n_shape,
                "lambda_t_shape": lambda_t_shape,
            },
            "input_layout": {
                "order": ["g_n", "ds_t"],
                "g_n_shape": g_n_shape,
                "ds_t_shape": ds_t_shape,
            },
            "flat_z": flat_state,
            "flat_inputs": flat_inputs,
            "z_splits": {"lambda_n": 1, "lambda_t": 2},
            "input_splits": {"g_n": 1, "ds_t": 2},
            "residual": flat_residual,
            "residual_at_solution": flat_residual,
            "block_jacobians": block_jacobians,
            "normal_step": tf.reshape(tf.cast(normal_step, tf.float32), (-1,)),
            "tangential_step": tf.reshape(tf.cast(tangential_step, tf.float32), (-1,)),
        }
    return ContactInnerResult(
        state=state,
        traction_vec=traction_vec,
        traction_tangent=state.lambda_t,
        diagnostics=diagnostics,
        linearization=linearization,
    )
