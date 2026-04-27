#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local error-bound diagnostics for the normal-contact-first implicit scheme."""

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from physics.contact.contact_implicit_backward import _validate_normal_contact_linearization


def _validate_residual_perturbation(block_jacobians: Dict[str, tf.Tensor], residual_perturbation):
    if not isinstance(residual_perturbation, dict):
        raise ValueError("local error-bound analysis requires residual_perturbation as a dict")
    if "normal" not in residual_perturbation or "tangential" not in residual_perturbation:
        raise ValueError("local error-bound analysis requires normal and tangential perturbation entries")

    delta_r_n = tf.convert_to_tensor(residual_perturbation["normal"], dtype=tf.float32)
    delta_r_t = tf.convert_to_tensor(residual_perturbation["tangential"], dtype=tf.float32)

    expected_normal_shape = tuple(block_jacobians["normal_state_diag"].shape.as_list())
    expected_tangential_shape = tuple(block_jacobians["tangential_lambda_n"].shape.as_list()[:1] + [2])

    if delta_r_n.shape.rank != 1:
        raise ValueError("local error-bound analysis expects residual_perturbation['normal'] shape [n]")
    if delta_r_t.shape.rank != 2 or delta_r_t.shape[-1] != 2:
        raise ValueError("local error-bound analysis expects residual_perturbation['tangential'] shape [n,2]")
    if expected_normal_shape and tuple(delta_r_n.shape.as_list()) != expected_normal_shape:
        raise ValueError("local error-bound analysis received mismatched normal perturbation shape")
    if expected_tangential_shape and tuple(delta_r_t.shape.as_list()) != expected_tangential_shape:
        raise ValueError("local error-bound analysis received mismatched tangential perturbation shape")
    return delta_r_n, delta_r_t


def _solve_tangential_block_rhs(tangential_lambda_t: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve A_tt x = rhs per contact with an explicit 2x2 inverse."""

    a11 = tangential_lambda_t[:, 0, 0]
    a12 = tangential_lambda_t[:, 0, 1]
    a21 = tangential_lambda_t[:, 1, 0]
    a22 = tangential_lambda_t[:, 1, 1]
    det = a11 * a22 - a12 * a21
    b1 = rhs[:, 0]
    b2 = rhs[:, 1]
    return tf.stack(
        [
            (a22 * b1 - a12 * b2) / det,
            (-a21 * b1 + a11 * b2) / det,
        ],
        axis=1,
    )


def _vector_norm(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(tf.reshape(x, (-1,)), tf.float32)
    return tf.sqrt(tf.reduce_sum(tf.square(x)))


def analyze_local_error_bounds(linearization, residual_perturbation):
    block_jacobians = _validate_normal_contact_linearization(linearization)
    delta_r_n, delta_r_t = _validate_residual_perturbation(block_jacobians, residual_perturbation)

    normal_state_diag = tf.reshape(block_jacobians["normal_state_diag"], (-1,))
    tangential_lambda_n = tf.cast(block_jacobians["tangential_lambda_n"], tf.float32)
    tangential_lambda_t = tf.cast(block_jacobians["tangential_lambda_t"], tf.float32)
    tangential_ds_t = tf.cast(block_jacobians["tangential_ds_t"], tf.float32)

    delta_lambda_n = -(delta_r_n / normal_state_diag)
    propagated_rhs_t = delta_r_t + tf.squeeze(tangential_lambda_n, axis=-1) * delta_lambda_n[:, None]
    delta_lambda_t = -_solve_tangential_block_rhs(tangential_lambda_t, propagated_rhs_t)

    ds11 = tangential_ds_t[:, 0, 0]
    ds12 = tangential_ds_t[:, 0, 1]
    ds21 = tangential_ds_t[:, 1, 0]
    ds22 = tangential_ds_t[:, 1, 1]
    dl1 = delta_lambda_t[:, 0]
    dl2 = delta_lambda_t[:, 1]
    gradient_error_proxy = tf.stack(
        [
            ds11 * dl1 + ds21 * dl2,
            ds12 * dl1 + ds22 * dl2,
        ],
        axis=1,
    )

    tangential_det = (
        tangential_lambda_t[:, 0, 0] * tangential_lambda_t[:, 1, 1]
        - tangential_lambda_t[:, 0, 1] * tangential_lambda_t[:, 1, 0]
    )
    normal_condition_margin = tf.reduce_min(tf.abs(normal_state_diag))
    tangential_condition_margin = tf.reduce_min(tf.abs(tangential_det))

    assumption_flags = {
        "local_contract_mode_valid": tf.constant(
            str(linearization.get("contract_mode", "") or "") == "normal_contact_first",
            dtype=tf.bool,
        ),
        "normal_block_nonsingular": tf.cast(normal_condition_margin > 0.0, tf.bool),
        "tangential_block_nonsingular": tf.cast(tangential_condition_margin > 0.0, tf.bool),
        "auxiliary_tangential_role_valid": tf.constant(
            str(linearization.get("tangential_contract_role", "") or "") == "auxiliary_friction_fixed_point",
            dtype=tf.bool,
        ),
    }
    regularity_ok = tf.reduce_all(tf.stack(list(assumption_flags.values())))

    return {
        "normal_state_error_bound": _vector_norm(delta_lambda_n),
        "tangential_state_error_bound": _vector_norm(delta_lambda_t),
        "total_state_error_bound": _vector_norm(
            tf.concat([delta_lambda_n, tf.reshape(delta_lambda_t, (-1,))], axis=0)
        ),
        "implicit_gradient_error_bound": _vector_norm(gradient_error_proxy),
        "normal_condition_margin": normal_condition_margin,
        "tangential_condition_margin": tangential_condition_margin,
        "regularity_ok": regularity_ok,
        "assumption_flags": assumption_flags,
    }
