#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tightening_model.py
-------------------
Penalty model for nut tightening via prescribed rotation about each nut axis.

Each nut is represented by sampled surface points X. For a given tightening
angle theta, the target displacement is:
    u_tgt = R(axis, theta) * (X - center) + center - X

Energy (penalty form):
    E_tight = 0.5 * alpha * sum_i w_i * || u(X_i) - u_tgt_i ||^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from mesh.surface_utils import triangulate_part_boundary, compute_tri_geometry, sample_points_on_surface
from mesh.interp_utils import interp_bary_tf


@dataclass
class NutSpec:
    name: str
    part: str
    axis: Optional[Tuple[float, float, float]] = None
    center: Optional[Tuple[float, float, float]] = None


@dataclass
class TighteningConfig:
    alpha: float = 1.0e3
    angle_unit: str = "deg"  # "deg" or "rad"
    clockwise: bool = True
    forward_chunk: int = 2048


@dataclass
class NutSampleData:
    name: str
    X: np.ndarray
    w: np.ndarray
    tri_node_idx: Optional[np.ndarray]
    bary: Optional[np.ndarray]
    axis: np.ndarray
    center: np.ndarray


def _sorted_node_ids(asm: Any) -> np.ndarray:
    return np.asarray(sorted(int(nid) for nid in asm.nodes.keys()), dtype=np.int64)


def _map_node_ids_to_idx(sorted_node_ids: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    nid = np.asarray(node_ids, dtype=np.int64)
    idx = np.searchsorted(sorted_node_ids, nid)
    if idx.size == 0:
        return idx.astype(np.int32)
    bad = (
        (idx < 0)
        | (idx >= sorted_node_ids.shape[0])
        | (sorted_node_ids[idx] != nid)
    )
    if np.any(bad):
        missing = np.unique(nid[bad])[:10]
        raise KeyError(f"Some node IDs are missing in asm.nodes (example: {missing}).")
    return idx.astype(np.int32)


def _compute_area_weights(tri_idx: np.ndarray, tri_areas: np.ndarray) -> np.ndarray:
    tri_idx = np.asarray(tri_idx, dtype=np.int64).reshape(-1)
    tri_areas = np.asarray(tri_areas, dtype=np.float64).reshape(-1)
    if tri_idx.size == 0 or tri_areas.size == 0:
        return np.zeros((0,), dtype=np.float64)
    counts = np.bincount(tri_idx, minlength=tri_areas.shape[0]).astype(np.float64)
    w = tri_areas[tri_idx] / (counts[tri_idx] + 1e-16)
    total_area = float(tri_areas.sum())
    w_sum = float(w.sum())
    if w_sum > 0:
        w *= total_area / w_sum
    return w


def _normalize_axis(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size != 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _auto_axis_from_nodes(nodes_xyz: Dict[int, Tuple[float, float, float]]) -> np.ndarray:
    coords = np.asarray(list(nodes_xyz.values()), dtype=np.float64)
    if coords.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = coords.mean(axis=0)
    Xc = coords - c
    cov = (Xc.T @ Xc) / max(coords.shape[0], 1)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmin(w))]
    axis = _normalize_axis(axis)
    # Stabilize sign with +Z to keep "clockwise" consistent.
    if np.dot(axis, np.array([0.0, 0.0, 1.0])) < 0.0:
        axis = -axis
    return axis


class NutTighteningPenalty:
    def __init__(self, cfg: Optional[TighteningConfig] = None):
        self.cfg = cfg or TighteningConfig()
        self._nuts: List[NutSampleData] = []

    def build_from_specs(
        self,
        asm,
        specs: List[NutSpec],
        n_points_each: int = 800,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        nuts: List[NutSampleData] = []
        sorted_node_ids = _sorted_node_ids(asm)
        for sp in specs:
            if sp.part not in asm.parts:
                raise KeyError(f"Nut part not found: {sp.part}")
            part = asm.parts[sp.part]
            ts = triangulate_part_boundary(part, sp.part, log_summary=False)
            X, tri_idx, bary, _ = sample_points_on_surface(part, ts, n_points_each, rng=rng)
            tri_areas, _, _ = compute_tri_geometry(part, ts)
            w = _compute_area_weights(tri_idx, tri_areas)

            tri_node_ids = ts.tri_node_ids[tri_idx.astype(np.int64)]
            tri_node_idx = _map_node_ids_to_idx(sorted_node_ids, tri_node_ids)

            axis_vec = None
            if sp.axis is not None:
                axis_vec = _normalize_axis(np.asarray(sp.axis, dtype=np.float64))
            else:
                axis_vec = _auto_axis_from_nodes(part.nodes_xyz)

            if sp.center is not None:
                center = np.asarray(sp.center, dtype=np.float64)
            else:
                center = np.mean(X, axis=0)

            nuts.append(
                NutSampleData(
                    name=sp.name,
                    X=X.astype(np.float32),
                    w=w.astype(np.float32),
                    tri_node_idx=tri_node_idx.astype(np.int32),
                    bary=bary.astype(np.float32),
                    axis=axis_vec.astype(np.float32),
                    center=center.astype(np.float32),
                )
            )
        self._nuts = nuts

    def _u_fn_chunked(self, u_fn, params, X, batch: int = None) -> tf.Tensor:
        if batch is None:
            batch = int(getattr(self.cfg, "forward_chunk", 2048))
        batch = max(1, int(batch))
        X = tf.convert_to_tensor(X)
        if X.dtype != tf.float32:
            X = tf.cast(X, tf.float32)
        n = int(X.shape[0])
        outs = []
        for s in range(0, n, batch):
            e = min(n, s + batch)
            outs.append(tf.cast(u_fn(X[s:e], params), tf.float32))
        return tf.concat(outs, axis=0)

    def _angle_to_rad(self, theta: tf.Tensor) -> tf.Tensor:
        unit = str(getattr(self.cfg, "angle_unit", "deg") or "deg").lower()
        if unit.startswith("deg"):
            return theta * (tf.constant(np.pi / 180.0, dtype=theta.dtype))
        return theta

    def _rotation_displacement(
        self, X: tf.Tensor, axis: tf.Tensor, center: tf.Tensor, theta: tf.Tensor
    ) -> tf.Tensor:
        # Rodrigues' rotation formula around axis passing through center.
        a = tf.reshape(axis, (1, 3))
        a = a / (tf.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        c = tf.reshape(center, (1, 3))
        r = X - c
        a_b = tf.broadcast_to(a, tf.shape(r))
        proj = tf.reduce_sum(r * a_b, axis=1, keepdims=True) * a_b
        radial = r - proj
        ct = tf.cos(theta)
        st = tf.sin(theta)
        cross = tf.linalg.cross(a_b, radial)
        radial_rot = (
            radial * ct
            + cross * st
            + a_b * (tf.reduce_sum(a_b * radial, axis=1, keepdims=True)) * (1.0 - ct)
        )
        X_rot = c + proj + radial_rot
        return X_rot - X

    def energy(self, u_fn, params: Dict[str, tf.Tensor], *, u_nodes: Optional[tf.Tensor] = None):
        if not self._nuts:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, {"tightening": {"nut_angles": tf.zeros((0,), tf.float32)}}

        theta = params.get("theta", None)
        if theta is None:
            theta = params.get("P", None)
        if theta is None:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, {"tightening": {"nut_angles": tf.zeros((0,), tf.float32)}}

        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        nb = len(self._nuts)
        nb_tf = tf.constant(nb, dtype=tf.int32)
        t_len = tf.shape(theta)[0]

        def _pad():
            pad = nb_tf - t_len
            zeros = tf.zeros((pad,), dtype=tf.float32)
            return tf.concat([theta, zeros], axis=0)

        def _truncate():
            return theta[:nb]

        theta = tf.cond(t_len < nb_tf, _pad, _truncate)
        theta = theta[:nb]

        theta = self._angle_to_rad(theta)
        if bool(getattr(self.cfg, "clockwise", True)):
            theta = -theta

        alpha = tf.cast(getattr(self.cfg, "alpha", 1.0e3), tf.float32)
        total = tf.cast(0.0, tf.float32)
        rms_list = []

        for i, nut in enumerate(self._nuts):
            X = tf.convert_to_tensor(nut.X, tf.float32)
            w = tf.convert_to_tensor(nut.w, tf.float32)
            axis = tf.convert_to_tensor(nut.axis, tf.float32)
            center = tf.convert_to_tensor(nut.center, tf.float32)
            if u_nodes is not None and nut.tri_node_idx is not None and nut.bary is not None:
                u_pred = interp_bary_tf(
                    tf.cast(u_nodes, tf.float32),
                    tf.convert_to_tensor(nut.tri_node_idx, dtype=tf.int32),
                    tf.convert_to_tensor(nut.bary, dtype=tf.float32),
                )
            else:
                u_pred = self._u_fn_chunked(u_fn, params, X, batch=int(getattr(self.cfg, "forward_chunk", 2048)))
            u_tgt = self._rotation_displacement(X, axis, center, theta[i])
            r = u_pred - u_tgt
            r2 = tf.reduce_sum(r * r, axis=1)
            total = total + 0.5 * alpha * tf.reduce_sum(w * r2)
            rms = tf.sqrt(tf.reduce_mean(r2) + 1e-20)
            rms_list.append(rms)

        stats = {
            "tightening": {
                "nut_angles": theta,
                "rms": tf.stack(rms_list, axis=0) if rms_list else tf.zeros((0,), tf.float32),
            }
        }
        return total, stats

    def residual(self, u_fn, params: Dict[str, tf.Tensor], *, u_nodes: Optional[tf.Tensor] = None):
        if not self._nuts:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, {"tightening": {"nut_angles": tf.zeros((0,), tf.float32)}}
        E, stats = self.energy(u_fn, params, u_nodes=u_nodes)
        # Convert to mean residual (normalize by total weight).
        denom = tf.constant(0.0, tf.float32)
        for nut in self._nuts:
            denom = denom + tf.reduce_sum(tf.convert_to_tensor(nut.w, tf.float32))
        denom = tf.maximum(denom, tf.constant(1e-12, tf.float32))
        return E / denom, stats
