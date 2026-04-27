#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stateless low-level kernels for the strict-bilevel inner contact solve."""

from __future__ import annotations

from typing import Dict

import tensorflow as tf


def _to_float_tensor(x) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def fb_normal_residual(g_n: tf.Tensor, lambda_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Smooth Fischer-Burmeister residual for normal contact complementarity."""

    g_n = _to_float_tensor(g_n)
    lambda_n = _to_float_tensor(lambda_n)
    eps_n = _to_float_tensor(eps_n)
    return tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n) - g_n - lambda_n


def fb_normal_jacobian(g_n: tf.Tensor, lambda_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Derivative of the FB residual with respect to the normal multiplier."""

    g_n = _to_float_tensor(g_n)
    lambda_n = _to_float_tensor(lambda_n)
    eps_n = _to_float_tensor(eps_n)
    denom = tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n)
    return lambda_n / (denom + 1.0e-12) - 1.0


def smooth_penetration_target(g_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Smooth nonnegative penetration target derived from the FB kernel at ``lambda_n=0``."""

    g_n = _to_float_tensor(g_n)
    eps_n = _to_float_tensor(eps_n)
    return 0.5 * (tf.sqrt(g_n * g_n + eps_n * eps_n) - g_n)


def inner_normal_residual(g_n: tf.Tensor, lambda_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Residual used by the frozen-geometry inner normal solve."""

    g_n = _to_float_tensor(g_n)
    lambda_n = _to_float_tensor(lambda_n)
    return lambda_n - smooth_penetration_target(g_n, eps_n)


def inner_normal_jacobian(g_n: tf.Tensor, lambda_n: tf.Tensor, eps_n) -> tf.Tensor:
    """Derivative of :func:`inner_normal_residual` with respect to ``lambda_n``."""

    del g_n, eps_n
    lambda_n = _to_float_tensor(lambda_n)
    return tf.ones_like(lambda_n)


def project_to_coulomb_disk(tau_trial: tf.Tensor, radius: tf.Tensor, eps=1.0e-6) -> tf.Tensor:
    """Project tangential traction onto the Coulomb disk ||tau|| <= radius."""

    tau_trial = _to_float_tensor(tau_trial)
    radius = tf.maximum(_to_float_tensor(radius), 0.0)
    eps = _to_float_tensor(eps)
    tau_norm = tf.sqrt(tf.reduce_sum(tf.square(tau_trial), axis=1) + eps * eps)
    scale = tf.minimum(tf.ones_like(tau_norm), radius / (tau_norm + 1.0e-12))
    return tau_trial * scale[:, None]


def tangential_update_map(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps=1.0e-6,
) -> tf.Tensor:
    """Tangential projection update map used by the current inner solver."""

    lambda_t = _to_float_tensor(lambda_t)
    ds_t = _to_float_tensor(ds_t)
    lambda_n = _to_float_tensor(lambda_n)
    mu = _to_float_tensor(mu)
    k_t = _to_float_tensor(k_t)
    tau_trial = lambda_t + k_t * ds_t
    return project_to_coulomb_disk(tau_trial, mu * tf.maximum(lambda_n, 0.0), eps=eps)


def tangential_fixed_point_gap(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps=1.0e-6,
) -> tf.Tensor:
    """Fixed-point gap for the current tangential update map."""

    lambda_t = _to_float_tensor(lambda_t)
    target = tangential_update_map(lambda_t, ds_t, lambda_n, mu, k_t, eps=eps)
    return lambda_t - target


def friction_fixed_point_residual(
    lambda_t: tf.Tensor,
    ds_t: tf.Tensor,
    lambda_n: tf.Tensor,
    mu,
    k_t,
    eps=1.0e-6,
) -> tf.Tensor:
    """Production tangential residual aligned with the current fixed-point map."""

    return tangential_fixed_point_gap(lambda_t, ds_t, lambda_n, mu, k_t, eps=eps)


def compose_contact_traction(
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    normals: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
) -> tf.Tensor:
    """Compose 3D traction from normal and tangential contact components."""

    lambda_n = _to_float_tensor(lambda_n)
    lambda_t = _to_float_tensor(lambda_t)
    normals = _to_float_tensor(normals)
    t1 = _to_float_tensor(t1)
    t2 = _to_float_tensor(t2)
    return (
        lambda_n[:, None] * normals
        + lambda_t[:, 0:1] * t1
        + lambda_t[:, 1:2] * t2
    )


def check_contact_feasibility(
    g_n: tf.Tensor,
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    mu,
    tol_n: float,
    tol_t: float,
) -> Dict[str, tf.Tensor]:
    """Evaluate simple feasibility metrics for a candidate contact state."""

    g_n = _to_float_tensor(g_n)
    lambda_n = _to_float_tensor(lambda_n)
    lambda_t = _to_float_tensor(lambda_t)
    mu = _to_float_tensor(mu)

    tang_norm = tf.sqrt(tf.reduce_sum(tf.square(lambda_t), axis=1) + 1.0e-12)
    cone_violation = tf.reduce_max(tf.nn.relu(tang_norm - mu * tf.maximum(lambda_n, 0.0)))
    lambda_neg_violation = tf.reduce_max(tf.nn.relu(-lambda_n))
    max_penetration = tf.reduce_max(tf.nn.relu(-g_n))
    feasible = tf.logical_and(
        lambda_neg_violation <= tf.cast(tol_n, tf.float32),
        cone_violation <= tf.cast(tol_t, tf.float32),
    )
    return {
        "feasible": feasible,
        "cone_violation": tf.cast(cone_violation, tf.float32),
        "lambda_neg_violation": tf.cast(lambda_neg_violation, tf.float32),
        "max_penetration": tf.cast(max_penetration, tf.float32),
    }
