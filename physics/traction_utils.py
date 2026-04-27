#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Traction and component helpers based on canonical Voigt stress vectors."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from model.voigt_utils import voigt6_to_tensor


def traction_from_sigma_voigt(sigma_vec: tf.Tensor, normals: tf.Tensor) -> tf.Tensor:
    """Compute traction t=sigma*n from canonical Voigt stress vectors."""
    sigma_tensor = voigt6_to_tensor(sigma_vec)
    normals = tf.convert_to_tensor(normals, dtype=tf.float32)
    return tf.einsum("...ij,...j->...i", sigma_tensor, normals)


def normal_tangential_components(
    traction: tf.Tensor,
    normals: tf.Tensor,
    tangential_basis: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Split traction into normal scalar and 2D tangential components."""
    traction = tf.convert_to_tensor(traction, dtype=tf.float32)
    normals = tf.convert_to_tensor(normals, dtype=tf.float32)
    tangential_basis = tf.convert_to_tensor(tangential_basis, dtype=tf.float32)

    tn = tf.reduce_sum(traction * normals, axis=-1, keepdims=True)
    tt = tf.einsum("...ij,...j->...i", tangential_basis, traction)
    return tn, tt
