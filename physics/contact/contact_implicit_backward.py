#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implicit backward helpers for the normal-contact-first inner layer."""

from __future__ import annotations

from typing import Dict

import tensorflow as tf


def _static_shape_tuple(tensor: tf.Tensor):
    shape = tensor.shape
    if shape.rank is None:
        return None
    return tuple(shape.as_list())


def _validate_normal_contact_linearization(linearization) -> Dict[str, tf.Tensor]:
    if not isinstance(linearization, dict):
        raise ValueError("normal-contact implicit backward requires a linearization dict")
    if str(linearization.get("contract_mode", "") or "") != "normal_contact_first":
        raise ValueError("normal-contact implicit backward requires contract_mode=normal_contact_first")
    if str(linearization.get("primary_state", "") or "") != "lambda_n":
        raise ValueError("normal-contact implicit backward requires primary_state=lambda_n")
    if str(linearization.get("tangential_contract_role", "") or "") != "auxiliary_friction_fixed_point":
        raise ValueError("normal-contact implicit backward requires auxiliary_friction_fixed_point tangential role")
    state_layout = linearization.get("state_layout")
    input_layout = linearization.get("input_layout")
    if not isinstance(state_layout, dict) or list(state_layout.get("order") or []) != ["lambda_n", "lambda_t"]:
        raise ValueError("normal-contact implicit backward requires [lambda_n, lambda_t] state layout")
    if not isinstance(input_layout, dict) or list(input_layout.get("order") or []) != ["g_n", "ds_t"]:
        raise ValueError("normal-contact implicit backward requires [g_n, ds_t] input layout")
    block_jacobians = linearization.get("block_jacobians")
    if not isinstance(block_jacobians, dict):
        raise ValueError("normal-contact implicit backward requires block_jacobians")
    required = (
        "normal_state_diag",
        "normal_input_diag",
        "tangential_lambda_n",
        "tangential_lambda_t",
        "tangential_ds_t",
    )
    missing = [key for key in required if key not in block_jacobians]
    if missing:
        raise ValueError(f"normal-contact implicit backward missing block_jacobians: {missing}")
    normal_state_diag = tf.convert_to_tensor(block_jacobians["normal_state_diag"], dtype=tf.float32)
    normal_input_diag = tf.convert_to_tensor(block_jacobians["normal_input_diag"], dtype=tf.float32)
    tangential_lambda_n = tf.convert_to_tensor(block_jacobians["tangential_lambda_n"], dtype=tf.float32)
    tangential_lambda_t = tf.convert_to_tensor(block_jacobians["tangential_lambda_t"], dtype=tf.float32)
    tangential_ds_t = tf.convert_to_tensor(block_jacobians["tangential_ds_t"], dtype=tf.float32)

    normal_shape = _static_shape_tuple(normal_state_diag)
    input_shape = _static_shape_tuple(normal_input_diag)
    lambda_n_shape = _static_shape_tuple(tangential_lambda_n)
    lambda_t_shape = _static_shape_tuple(tangential_lambda_t)
    ds_t_shape = _static_shape_tuple(tangential_ds_t)

    if normal_shape is not None and len(normal_shape) != 1:
        raise ValueError("normal-contact implicit backward expects normal_state_diag shape [n]")
    if input_shape is not None and input_shape != normal_shape:
        raise ValueError("normal-contact implicit backward expects normal_input_diag shape [n]")
    if lambda_n_shape is not None and (len(lambda_n_shape) != 3 or lambda_n_shape[1:] != (2, 1)):
        raise ValueError("normal-contact implicit backward expects tangential_lambda_n shape [n,2,1]")
    if lambda_t_shape is not None and (len(lambda_t_shape) != 3 or lambda_t_shape[1:] != (2, 2)):
        raise ValueError("normal-contact implicit backward expects tangential_lambda_t shape [n,2,2]")
    if ds_t_shape is not None and (len(ds_t_shape) != 3 or ds_t_shape[1:] != (2, 2)):
        raise ValueError("normal-contact implicit backward expects tangential_ds_t shape [n,2,2]")

    block_jacobians = {
        "normal_state_diag": normal_state_diag,
        "normal_input_diag": normal_input_diag,
        "tangential_lambda_n": tangential_lambda_n,
        "tangential_lambda_t": tangential_lambda_t,
        "tangential_ds_t": tangential_ds_t,
    }
    return block_jacobians


def solve_normal_contact_structured_adjoint(
    upstream_state_grad: tf.Tensor,
    block_jacobians: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """Solve the structured adjoint system and return dL/d[g_n, ds_t]."""

    grad_z = tf.cast(tf.reshape(upstream_state_grad, (-1,)), tf.float32)
    normal_state_diag = tf.cast(tf.reshape(block_jacobians["normal_state_diag"], (-1,)), tf.float32)
    normal_input_diag = tf.cast(tf.reshape(block_jacobians["normal_input_diag"], (-1,)), tf.float32)
    tangential_lambda_n = tf.cast(block_jacobians["tangential_lambda_n"], tf.float32)
    tangential_lambda_t = tf.cast(block_jacobians["tangential_lambda_t"], tf.float32)
    tangential_ds_t = tf.cast(block_jacobians["tangential_ds_t"], tf.float32)

    n = tf.shape(normal_state_diag)[0]
    grad_lambda_n = grad_z[:n]
    grad_lambda_t = tf.reshape(grad_z[n:], (n, 2))

    # Solve A_tt^T v_t = dL/dlambda_t per contact with an explicit 2x2 inverse.
    a11 = tangential_lambda_t[:, 0, 0]
    a12 = tangential_lambda_t[:, 0, 1]
    a21 = tangential_lambda_t[:, 1, 0]
    a22 = tangential_lambda_t[:, 1, 1]
    det = a11 * a22 - a12 * a21
    g1 = grad_lambda_t[:, 0]
    g2 = grad_lambda_t[:, 1]
    v_t = tf.stack(
        [
            (a22 * g1 - a21 * g2) / det,
            (-a12 * g1 + a11 * g2) / det,
        ],
        axis=1,
    )

    coupling = tf.reduce_sum(tf.squeeze(tangential_lambda_n, axis=-1) * v_t, axis=1)
    v_n = (grad_lambda_n - coupling) / normal_state_diag

    grad_g_n = -normal_input_diag * v_n
    grad_ds_t = -tf.linalg.matvec(
        tf.linalg.matrix_transpose(tangential_ds_t),
        v_t,
    )
    return tf.concat([grad_g_n, tf.reshape(grad_ds_t, (-1,))], axis=0)


def attach_normal_contact_implicit_backward(
    flat_state: tf.Tensor,
    flat_inputs: tf.Tensor,
    linearization,
) -> tf.Tensor:
    """Return the solved flat state with implicit gradients on flattened inputs."""

    flat_state = tf.cast(tf.reshape(flat_state, (-1,)), tf.float32)
    flat_inputs = tf.cast(tf.reshape(flat_inputs, (-1,)), tf.float32)
    block_jacobians = _validate_normal_contact_linearization(linearization)

    @tf.custom_gradient
    def _wrapped(flat_inputs_tensor: tf.Tensor):
        del flat_inputs_tensor
        forward_state = tf.identity(flat_state)

        def grad(upstream_state_grad: tf.Tensor):
            grad_inputs = solve_normal_contact_structured_adjoint(upstream_state_grad, block_jacobians)
            return tf.cast(grad_inputs, flat_inputs.dtype)

        return forward_state, grad

    return _wrapped(flat_inputs)
