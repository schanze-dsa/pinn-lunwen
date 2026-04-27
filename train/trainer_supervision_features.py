# -*- coding: utf-8 -*-
"""Feature helpers for pure-supervision supervision routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class RingFeatureConfig:
    center_x: float
    center_y: float
    r_in: float
    r_out: float
    fourier_order: int = 6
    eps: float = 1.0e-8

    def validated(self) -> "RingFeatureConfig":
        if int(self.fourier_order) <= 0:
            raise ValueError(f"fourier_order must be positive, got {self.fourier_order!r}")
        width = float(self.r_out) - float(self.r_in)
        if not np.isfinite(width) or width <= float(self.eps):
            raise ValueError(
                f"r_out must be greater than r_in by more than eps; got r_in={self.r_in!r}, r_out={self.r_out!r}"
            )
        if not np.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {self.eps!r}")
        return self


def _as_xyz(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected array with shape (N,2) or (N,3), got {arr.shape!r}")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    return arr.astype(np.float32, copy=False)


def _as_xyz_tf(values: tf.Tensor) -> tf.Tensor:
    arr = tf.convert_to_tensor(values, dtype=tf.float32)
    tf.debugging.assert_rank(arr, 2, message="Expected rank-2 tensor with shape (N,2) or (N,3).")
    last_dim = tf.shape(arr)[1]
    tf.debugging.assert_greater_equal(last_dim, 2, message="Expected at least two coordinate columns.")
    tf.debugging.assert_less_equal(last_dim, 3, message="Expected at most three coordinate columns.")
    # Avoid tf.cond here: when the input already has static width 3, TensorFlow
    # still traces the width-2 branch and may infer a spurious width-4 forward
    # shape, which later breaks compiled gradients in the cylindrical->xyz path.
    pad_cols = tf.maximum(0, 3 - last_dim)
    arr = tf.pad(arr, paddings=[[0, 0], [0, pad_cols]])
    arr = arr[:, :3]
    arr.set_shape([None, 3])
    return tf.cast(arr, tf.float32)


def compute_ring_coordinate_components(
    xyz: np.ndarray,
    cfg: RingFeatureConfig,
) -> Dict[str, np.ndarray]:
    cfg = cfg.validated()
    xyz_arr = _as_xyz(xyz)
    dx = xyz_arr[:, 0] - float(cfg.center_x)
    dy = xyz_arr[:, 1] - float(cfg.center_y)
    r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    r_safe = np.maximum(r, np.float32(cfg.eps))
    theta = np.arctan2(dy, dx).astype(np.float32)
    rho = ((r - float(cfg.r_in)) / (float(cfg.r_out) - float(cfg.r_in))).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    return {
        "z": xyz_arr[:, 2].astype(np.float32),
        "dx": dx.astype(np.float32),
        "dy": dy.astype(np.float32),
        "r": r.astype(np.float32),
        "r_safe": r_safe.astype(np.float32),
        "rho": rho.astype(np.float32),
        "theta": theta.astype(np.float32),
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
    }


def compute_ring_coordinate_components_tf(
    xyz: tf.Tensor,
    cfg: RingFeatureConfig,
) -> Dict[str, tf.Tensor]:
    cfg = cfg.validated()
    xyz_arr = _as_xyz_tf(xyz)
    dx = xyz_arr[:, 0] - tf.cast(cfg.center_x, tf.float32)
    dy = xyz_arr[:, 1] - tf.cast(cfg.center_y, tf.float32)
    r = tf.sqrt(dx * dx + dy * dy)
    r_safe = tf.maximum(r, tf.cast(cfg.eps, tf.float32))
    theta = tf.atan2(dy, dx)
    rho = (r - tf.cast(cfg.r_in, tf.float32)) / tf.cast(float(cfg.r_out) - float(cfg.r_in), tf.float32)
    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    return {
        "z": tf.cast(xyz_arr[:, 2], tf.float32),
        "dx": tf.cast(dx, tf.float32),
        "dy": tf.cast(dy, tf.float32),
        "r": tf.cast(r, tf.float32),
        "r_safe": tf.cast(r_safe, tf.float32),
        "rho": tf.cast(rho, tf.float32),
        "theta": tf.cast(theta, tf.float32),
        "sin_theta": tf.cast(sin_theta, tf.float32),
        "cos_theta": tf.cast(cos_theta, tf.float32),
    }


def build_ring_aware_input_features(
    xyz: np.ndarray,
    cfg: RingFeatureConfig,
) -> np.ndarray:
    comps = compute_ring_coordinate_components(xyz, cfg)
    features = [comps["z"][:, None], comps["rho"][:, None]]
    theta = comps["theta"]
    for order in range(1, int(cfg.fourier_order) + 1):
        features.append(np.sin(order * theta).astype(np.float32)[:, None])
        features.append(np.cos(order * theta).astype(np.float32)[:, None])
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def build_ring_aware_input_features_tf(
    xyz: tf.Tensor,
    cfg: RingFeatureConfig,
) -> tf.Tensor:
    comps = compute_ring_coordinate_components_tf(xyz, cfg)
    features = [comps["z"][:, None], comps["rho"][:, None]]
    theta = comps["theta"]
    for order in range(1, int(cfg.fourier_order) + 1):
        order_f = tf.cast(order, tf.float32)
        features.append(tf.sin(order_f * theta)[:, None])
        features.append(tf.cos(order_f * theta)[:, None])
    return tf.cast(tf.concat(features, axis=1), tf.float32)


def convert_xyz_displacements_to_cylindrical(
    xyz: np.ndarray,
    u_xyz: np.ndarray,
    cfg: RingFeatureConfig,
) -> np.ndarray:
    comps = compute_ring_coordinate_components(xyz, cfg)
    disp = _as_xyz(u_xyz)
    cos_theta = comps["dx"] / comps["r_safe"]
    sin_theta = comps["dy"] / comps["r_safe"]
    u_r = disp[:, 0] * cos_theta + disp[:, 1] * sin_theta
    u_theta = -disp[:, 0] * sin_theta + disp[:, 1] * cos_theta
    u_z = disp[:, 2]
    return np.stack([u_r, u_theta, u_z], axis=1).astype(np.float32, copy=False)


def convert_xyz_displacements_to_cylindrical_tf(
    xyz: tf.Tensor,
    u_xyz: tf.Tensor,
    cfg: RingFeatureConfig,
) -> tf.Tensor:
    comps = compute_ring_coordinate_components_tf(xyz, cfg)
    disp = _as_xyz_tf(u_xyz)
    cos_theta = comps["dx"] / comps["r_safe"]
    sin_theta = comps["dy"] / comps["r_safe"]
    u_r = disp[:, 0] * cos_theta + disp[:, 1] * sin_theta
    u_theta = -disp[:, 0] * sin_theta + disp[:, 1] * cos_theta
    u_z = disp[:, 2]
    return tf.cast(tf.stack([u_r, u_theta, u_z], axis=1), tf.float32)


def convert_cylindrical_displacements_to_xyz(
    xyz: np.ndarray,
    u_cyl: np.ndarray,
    cfg: RingFeatureConfig,
) -> np.ndarray:
    comps = compute_ring_coordinate_components(xyz, cfg)
    disp = _as_xyz(u_cyl)
    cos_theta = comps["dx"] / comps["r_safe"]
    sin_theta = comps["dy"] / comps["r_safe"]
    u_x = disp[:, 0] * cos_theta - disp[:, 1] * sin_theta
    u_y = disp[:, 0] * sin_theta + disp[:, 1] * cos_theta
    u_z = disp[:, 2]
    return np.stack([u_x, u_y, u_z], axis=1).astype(np.float32, copy=False)


def convert_cylindrical_displacements_to_xyz_tf(
    xyz: tf.Tensor,
    u_cyl: tf.Tensor,
    cfg: RingFeatureConfig,
) -> tf.Tensor:
    comps = compute_ring_coordinate_components_tf(xyz, cfg)
    disp = _as_xyz_tf(u_cyl)
    cos_theta = comps["dx"] / comps["r_safe"]
    sin_theta = comps["dy"] / comps["r_safe"]
    u_x = disp[:, 0] * cos_theta - disp[:, 1] * sin_theta
    u_y = disp[:, 0] * sin_theta + disp[:, 1] * cos_theta
    u_z = disp[:, 2]
    return tf.cast(tf.stack([u_x, u_y, u_z], axis=1), tf.float32)
