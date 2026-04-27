#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Voigt-order tensor conversion helpers."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

VOIGT6_ORDER: Tuple[str, str, str, str, str, str] = ("xx", "yy", "zz", "xy", "yz", "xz")


def voigt6_to_tensor(sigma_vec: tf.Tensor) -> tf.Tensor:
    """Convert [...,6] Voigt stress vectors to [...,3,3] symmetric tensors."""
    sigma_vec = tf.convert_to_tensor(sigma_vec, dtype=tf.float32)
    xx, yy, zz, xy, yz, xz = tf.unstack(sigma_vec, axis=-1)
    return tf.stack(
        [
            tf.stack([xx, xy, xz], axis=-1),
            tf.stack([xy, yy, yz], axis=-1),
            tf.stack([xz, yz, zz], axis=-1),
        ],
        axis=-2,
    )


def tensor_to_voigt6(sigma_tensor: tf.Tensor) -> tf.Tensor:
    """Convert [...,3,3] symmetric tensors to [...,6] canonical Voigt vectors."""
    sigma_tensor = tf.convert_to_tensor(sigma_tensor, dtype=tf.float32)
    return tf.stack(
        [
            sigma_tensor[..., 0, 0],
            sigma_tensor[..., 1, 1],
            sigma_tensor[..., 2, 2],
            sigma_tensor[..., 0, 1],
            sigma_tensor[..., 1, 2],
            sigma_tensor[..., 0, 2],
        ],
        axis=-1,
    )
