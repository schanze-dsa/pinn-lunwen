#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_pairs.py
----------------
Build contact-pair sampling maps:
  - Sample points on the slave surface (area-proportional)
  - Project them to the closest points on the master surface
  - Return normals (from master tri), tangent basis t1/t2, and area weights

This module relies on:
  - src/io/inp_parser.AssemblyModel
  - src/mesh/surface_utils for surface triangulation, sampling, and projection

Output data structures are light-weight numpy arrays, ready for TF/JAX/PyTorch
wrappers in the contact operator (ALM with friction).

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from inp_io.inp_parser import AssemblyModel, PartMesh
from mesh.surface_utils import (
    resolve_surface_to_tris,
    compute_tri_geometry,
    sample_points_on_surface,
    project_points_onto_surface,
    build_contact_surfaces,
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ContactPairSpec:
    """User-level specification of a contact pair (surface keys in asm.surfaces)."""
    slave_key: str
    master_key: str
    name: str = ""  # optional alias for logging


@dataclass
class ContactPairData:
    """
    Concrete sampled data for one contact pair (size = n_pts).
    Arrays are np.float64 unless noted. Shapes:
      xs, xm, n, t1, t2 : (n, 3)
      w_area            : (n,)
      xs_node_idx       : (n, 3) int32  global node indices for slave triangle vertices
      xs_bary           : (n, 3) float32 barycentric weights for xs on slave triangles
      xm_node_idx       : (n, 3) int32  global node indices for master triangle vertices
      xm_bary           : (n, 3) float32 barycentric weights for xm on master triangles
      slave_tri_idx     : (n,) int64
      master_tri_idx    : (n,) int64
      dist              : (n,)  Euclidean distance from xs to xm
      pair_id           : (n,)  int64, an integer label for this pair (for grouping)
    """
    slave_part: str
    master_part: str
    xs: np.ndarray
    xm: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    w_area: np.ndarray
    xs_node_idx: np.ndarray
    xs_bary: np.ndarray
    xm_node_idx: np.ndarray
    xm_bary: np.ndarray
    slave_tri_idx: np.ndarray
    master_tri_idx: np.ndarray
    dist: np.ndarray
    pair_id: np.ndarray
    name: str = ""


@dataclass
class ContactMap:
    """
    A collection of ContactPairData for multiple pairs.
    Provides helpers to get concatenated arrays for batched computation.
    """
    pairs: List[ContactPairData]

    def concatenate(self) -> Dict[str, np.ndarray]:
        """Concatenate arrays from all pairs along the sample dimension."""
        if not self.pairs:
            return {}
        xs = np.concatenate([p.xs for p in self.pairs], axis=0)
        xm = np.concatenate([p.xm for p in self.pairs], axis=0)
        n  = np.concatenate([p.n  for p in self.pairs], axis=0)
        t1 = np.concatenate([p.t1 for p in self.pairs], axis=0)
        t2 = np.concatenate([p.t2 for p in self.pairs], axis=0)
        w  = np.concatenate([p.w_area for p in self.pairs], axis=0)
        xs_node_idx = np.concatenate([p.xs_node_idx for p in self.pairs], axis=0)
        xs_bary = np.concatenate([p.xs_bary for p in self.pairs], axis=0)
        xm_node_idx = np.concatenate([p.xm_node_idx for p in self.pairs], axis=0)
        xm_bary = np.concatenate([p.xm_bary for p in self.pairs], axis=0)
        sid = np.concatenate([p.slave_tri_idx for p in self.pairs], axis=0)
        mid = np.concatenate([p.master_tri_idx for p in self.pairs], axis=0)
        dist = np.concatenate([p.dist for p in self.pairs], axis=0)
        pid = np.concatenate([p.pair_id for p in self.pairs], axis=0)
        return dict(
            xs=xs,
            xm=xm,
            n=n,
            t1=t1,
            t2=t2,
            w_area=w,
            xs_node_idx=xs_node_idx,
            xs_bary=xs_bary,
            xm_node_idx=xm_node_idx,
            xm_bary=xm_bary,
            slave_tri_idx=sid,
            master_tri_idx=mid,
            dist=dist,
            pair_id=pid,
        )

    def __len__(self) -> int:
        return int(sum(p.xs.shape[0] for p in self.pairs))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _orthonormal_tangent_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given normals (N,3), compute two orthonormal tangent vectors t1,t2 for each.
    Strategy: pick an arbitrary reference axis not parallel to n; do cross products.

    Returns:
        t1, t2 : (N,3) each
    """
    N = normals.shape[0]
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-16)

    # pick reference axis per normal (avoid near-parallel with [1,0,0])
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    mask = np.abs(n[:, 0]) > 0.9  # if too parallel to x-axis, use y-axis instead
    ref[mask] = np.array([0.0, 1.0, 0.0])

    # t1 = normalize(ref x n)
    t1 = np.cross(ref, n)
    t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-16)
    # t2 = n x t1
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-16)
    return t1, t2


def _fetch_xyz(part_or_asm, node_ids: np.ndarray) -> np.ndarray:
    """
    Fetch (N,3) coords for node ids from either PartMesh.nodes_xyz or AssemblyModel.nodes.
    Local helper (mirrors mesh.surface_utils._fetch_xyz).
    """
    if hasattr(part_or_asm, "nodes_xyz"):
        mapping = part_or_asm.nodes_xyz  # PartMesh
    else:
        mapping = part_or_asm.nodes      # AssemblyModel
    out = np.empty((node_ids.shape[0], 3), dtype=np.float64)
    for i, nid in enumerate(node_ids):
        out[i] = mapping[int(nid)]
    return out


def _triangle_gauss_rule(n_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return barycentric points (G,3) and weights (G,) for triangle quadrature.
    Weights sum to 1.0 (area will be multiplied later).
    Supported: 1, 3, 7 points.
    """
    if n_pts == 1:
        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
        w = np.array([1.0], dtype=np.float64)
        return bary, w
    if n_pts == 3:
        a = 2.0 / 3.0
        b = 1.0 / 6.0
        bary = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
        ], dtype=np.float64)
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return bary, w
    if n_pts == 7:
        a = 0.101286507323456
        b = 0.797426985353087
        w1 = 0.125939180544827
        c = 0.470142064105115
        d = 0.059715871789770
        w2 = 0.132394152788506
        w3 = 0.225000000000000
        bary = np.array([
            [a, a, b],
            [a, b, a],
            [b, a, a],
            [c, c, d],
            [c, d, c],
            [d, c, c],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ], dtype=np.float64)
        w = np.array([w1, w1, w1, w2, w2, w2, w3], dtype=np.float64)
        return bary, w
    raise ValueError(f"Unsupported triangle gauss rule n_pts={n_pts}. Use 1, 3, or 7.")


def _mortar_points_on_surface(part_or_asm, ts, n_gauss: int):
    """
    Deterministic Gauss-point sampling on each triangle (mortar-style).

    Returns:
        X   : (N,3) gauss points
        tri_idx : (N,) triangle indices
        bary : (N,3) barycentric weights on slave triangles
        n   : (N,3) triangle normals (unit)
        w_area : (N,) area weights (sum = total surface area)
    """
    bary_gp, w_gp = _triangle_gauss_rule(n_gauss)
    areas, normals, _ = compute_tri_geometry(part_or_asm, ts)

    tri_node_ids = ts.tri_node_ids
    T = tri_node_ids.shape[0]
    G = bary_gp.shape[0]

    tri_xyz = np.empty((T, 3, 3), dtype=np.float64)
    for i, tri in enumerate(tri_node_ids):
        tri_xyz[i] = _fetch_xyz(part_or_asm, np.asarray(tri, dtype=np.int64))

    v0 = tri_xyz[:, 0, :]
    v1 = tri_xyz[:, 1, :]
    v2 = tri_xyz[:, 2, :]

    b0 = bary_gp[:, 0][None, :, None]
    b1 = bary_gp[:, 1][None, :, None]
    b2 = bary_gp[:, 2][None, :, None]
    X = b0 * v0[:, None, :] + b1 * v1[:, None, :] + b2 * v2[:, None, :]
    X = X.reshape(T * G, 3)

    tri_idx = np.repeat(np.arange(T, dtype=np.int64), G)
    bary = np.repeat(bary_gp[None, :, :], T, axis=0).reshape(T * G, 3)
    n = np.repeat(normals, G, axis=0)
    w_area = (areas[:, None] * w_gp[None, :]).reshape(T * G)
    return X, tri_idx, bary, n, w_area


def _compute_area_weights(tri_idx: np.ndarray, tri_areas: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Compute per-sample area weights ensuring unbiased Monte Carlo integration on the surface.

    We sampled triangles proportional to area, so a simple constant weight
    w = total_area / n_samples is unbiased. For slightly lower variance,
    we can do per-triangle counting: w_k = area(tri) / count(tri).
    """
    total_area = float(tri_areas.sum())
    if n_samples == 0 or total_area <= 0:
        return np.zeros((0,), dtype=np.float64)

    # Count how many times each triangle was sampled
    counts = np.bincount(tri_idx, minlength=tri_areas.shape[0]).astype(np.float64)
    w = tri_areas[tri_idx] / (counts[tri_idx] + 1e-16)  # area per sample inside that tri
    # Normalize to match total_area (optional, keeps sums comparable across pairs)
    scale = total_area / (w.sum() + 1e-16) * n_samples / n_samples
    return w * scale


# ---------------------------------------------------------------------
# Node-id -> global node index mapping (must match DFEM node ordering)
# ---------------------------------------------------------------------

def _sorted_node_ids(asm: AssemblyModel) -> np.ndarray:
    return np.asarray(sorted(int(nid) for nid in asm.nodes.keys()), dtype=np.int64)


def _map_node_ids_to_idx(sorted_node_ids: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """
    Map Abaqus node IDs to 0-based indices under the DFEM ordering (sorted node IDs).
    """
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


# ---------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------

def build_contact_pair_data(
    asm: AssemblyModel,
    slave_key: str,
    master_key: str,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
    prefilter_k: int = 8,
    chunk: int = 4096,
    pair_id: int = 0,
    name: str = "",
) -> ContactPairData:
    """
    为一个接触对构建采样/投影数据。
    - 在从属面（slave surface）上采样 n_points
    - 投影到主面（master surface）（最近点）
    - 计算法向量（从主面）、切向基底和面积权重

    返回：
        ContactPairData（为摩擦接触能量准备）
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_node_ids = _sorted_node_ids(asm)

    # 解析两个接触面的部分和三角形表面
    part_s, ts_s, part_m, ts_m = build_contact_surfaces(asm, slave_key, master_key)

    # 在从属面上采样
    xs, tri_idx_s, bary_s, n_s_dummy = sample_points_on_surface(part_s, ts_s, n_points, rng=rng)

    # 投影到主面
    xm, n_m, tri_idx_m, dist, bary_m = project_points_onto_surface(
        part_m, ts_m, xs, prefilter_k=prefilter_k, chunk=chunk
    )

    # Interpolation metadata (tri vertices + barycentric weights)
    xs_tri_node_ids = ts_s.tri_node_ids[tri_idx_s.astype(np.int64)]
    xm_tri_node_ids = ts_m.tri_node_ids[tri_idx_m.astype(np.int64)]
    xs_node_idx = _map_node_ids_to_idx(sorted_node_ids, xs_tri_node_ids)
    xm_node_idx = _map_node_ids_to_idx(sorted_node_ids, xm_tri_node_ids)

    # 从主面法向量计算切向基底
    t1, t2 = _orthonormal_tangent_basis(n_m)

    # 使用从属面的三角形面积计算权重（因为在标准接触公式中，积分在从属面上进行）
    tri_areas_s, _, _ = compute_tri_geometry(part_s, ts_s)
    w = _compute_area_weights(tri_idx_s, tri_areas_s, n_points)

    return ContactPairData(
        slave_part=part_s.name,
        master_part=part_m.name,
        xs=xs, xm=xm, n=n_m, t1=t1, t2=t2, w_area=w,
        xs_node_idx=xs_node_idx,
        xs_bary=bary_s.astype(np.float32),
        xm_node_idx=xm_node_idx,
        xm_bary=bary_m.astype(np.float32),
        slave_tri_idx=tri_idx_s.astype(np.int64),
        master_tri_idx=tri_idx_m.astype(np.int64),
        dist=dist.astype(np.float64),
        pair_id=(np.full((n_points,), pair_id, dtype=np.int64)),
        name=name or f"{slave_key}__{master_key}",
    )


def build_contact_pair_data_mortar(
    asm: AssemblyModel,
    slave_key: str,
    master_key: str,
    n_gauss: int = 3,
    rng: Optional[np.random.Generator] = None,
    prefilter_k: int = 8,
    chunk: int = 4096,
    pair_id: int = 0,
    name: str = "",
    max_points: int = 0,
) -> ContactPairData:
    """
    Mortar-style deterministic contact data:
    - Use triangle Gauss points on slave surface (no random sampling)
    - Project to master surface (closest point)
    - Use per-Gauss-point Lagrange multipliers in ALM (discrete LM field)
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_node_ids = _sorted_node_ids(asm)

    part_s, ts_s, part_m, ts_m = build_contact_surfaces(asm, slave_key, master_key)

    xs, tri_idx_s, bary_s, n_s_dummy, w_area = _mortar_points_on_surface(part_s, ts_s, n_gauss)

    # Optional cap (approximate integral with importance sampling)
    if max_points and xs.shape[0] > max_points:
        probs = w_area / (w_area.sum() + 1e-16)
        idx = rng.choice(xs.shape[0], size=max_points, replace=True, p=probs)
        xs = xs[idx]
        tri_idx_s = tri_idx_s[idx]
        bary_s = bary_s[idx]
        n_s_dummy = n_s_dummy[idx]
        # MC estimator: constant weight = total_area / m
        w_area = np.full((max_points,), float(w_area.sum()) / float(max_points), dtype=np.float64)

    # Project to master surface
    xm, n_m, tri_idx_m, dist, bary_m = project_points_onto_surface(
        part_m, ts_m, xs, prefilter_k=prefilter_k, chunk=chunk
    )

    xs_tri_node_ids = ts_s.tri_node_ids[tri_idx_s.astype(np.int64)]
    xm_tri_node_ids = ts_m.tri_node_ids[tri_idx_m.astype(np.int64)]
    xs_node_idx = _map_node_ids_to_idx(sorted_node_ids, xs_tri_node_ids)
    xm_node_idx = _map_node_ids_to_idx(sorted_node_ids, xm_tri_node_ids)

    t1, t2 = _orthonormal_tangent_basis(n_m)

    n_points = int(xs.shape[0])
    return ContactPairData(
        slave_part=part_s.name,
        master_part=part_m.name,
        xs=xs, xm=xm, n=n_m, t1=t1, t2=t2, w_area=w_area,
        xs_node_idx=xs_node_idx,
        xs_bary=bary_s.astype(np.float32),
        xm_node_idx=xm_node_idx,
        xm_bary=bary_m.astype(np.float32),
        slave_tri_idx=tri_idx_s.astype(np.int64),
        master_tri_idx=tri_idx_m.astype(np.int64),
        dist=dist.astype(np.float64),
        pair_id=(np.full((n_points,), pair_id, dtype=np.int64)),
        name=name or f"{slave_key}__{master_key}",
    )

def build_contact_map(
    asm: AssemblyModel,
    specs: List[ContactPairSpec],
    n_points_per_pair: int,
    seed: Optional[int] = None,
    prefilter_k: int = 8,
    chunk: int = 4096,
    two_pass: bool = False,
    mode: str = "sample",  # "sample" | "mortar"
    mortar_gauss: int = 3,
    mortar_max_points: int = 0,
) -> ContactMap:
    """
    Build ContactMap for a list of pairs. Each pair gets n_points_per_pair samples.

    If two_pass=True, split samples across both directions (slave->master and master->slave),
    so the total per original pair remains n_points_per_pair.

    If mode="mortar", uses deterministic Gauss points on the slave surface (n_points_per_pair ignored),
    and per-GP Lagrange multipliers in the ALM operator.
    """
    rng = np.random.default_rng(seed)
    pairs: List[ContactPairData] = []
    kept = 0
    skipped = 0
    mode_norm = str(mode or "sample").strip().lower()
    for pid, spec in enumerate(specs):
        base_name = spec.name or f"pair_{pid}"
        any_ok = False

        if mode_norm == "mortar":
            try:
                data = build_contact_pair_data_mortar(
                    asm=asm,
                    slave_key=spec.slave_key,
                    master_key=spec.master_key,
                    n_gauss=mortar_gauss,
                    rng=rng,
                    prefilter_k=prefilter_k,
                    chunk=chunk,
                    pair_id=kept,
                    name=base_name,
                    max_points=mortar_max_points,
                )
                if data.xs.shape[0] > 0:
                    pairs.append(data)
                    any_ok = True
            except Exception as exc:
                print(
                    f"[contact] Skip pair {spec.slave_key}/{spec.master_key} (mortar): {exc}"
                )
        elif two_pass:
            n_fwd = int(n_points_per_pair // 2)
            n_rev = int(n_points_per_pair - n_fwd)

            if n_fwd > 0:
                try:
                    data = build_contact_pair_data(
                        asm=asm,
                        slave_key=spec.slave_key,
                        master_key=spec.master_key,
                        n_points=n_fwd,
                        rng=rng,
                        prefilter_k=prefilter_k,
                        chunk=chunk,
                        pair_id=kept,
                        name=base_name,
                    )
                    if data.xs.shape[0] > 0:
                        pairs.append(data)
                        any_ok = True
                except Exception as exc:
                    print(
                        f"[contact] Skip pair {spec.slave_key}/{spec.master_key} (fwd): {exc}"
                    )

            if n_rev > 0:
                try:
                    data = build_contact_pair_data(
                        asm=asm,
                        slave_key=spec.master_key,
                        master_key=spec.slave_key,
                        n_points=n_rev,
                        rng=rng,
                        prefilter_k=prefilter_k,
                        chunk=chunk,
                        pair_id=kept,
                        name=f"{base_name}__rev",
                    )
                    if data.xs.shape[0] > 0:
                        pairs.append(data)
                        any_ok = True
                except Exception as exc:
                    print(
                        f"[contact] Skip pair {spec.master_key}/{spec.slave_key} (rev): {exc}"
                    )
        else:
            try:
                data = build_contact_pair_data(
                    asm=asm,
                    slave_key=spec.slave_key,
                    master_key=spec.master_key,
                    n_points=n_points_per_pair,
                    rng=rng,
                    prefilter_k=prefilter_k,
                    chunk=chunk,
                    pair_id=kept,
                    name=base_name,
                )
                if data.xs.shape[0] > 0:
                    pairs.append(data)
                    any_ok = True
            except Exception as exc:
                print(
                    f"[contact] Skip pair {spec.slave_key}/{spec.master_key}: {exc}"
                )

        if any_ok:
            kept += 1
        else:
            skipped += 1
    if skipped > 0:
        print(f"[contact] Skipped {skipped} invalid pair(s); using {kept} pairs.")
    return ContactMap(pairs=pairs)


def resample_contact_map(
    asm: AssemblyModel,
    specs: List[ContactPairSpec],
    n_points_per_pair: int,
    base_seed: Optional[int],
    step_index: int,
    prefilter_k: int = 8,
    chunk: int = 4096,
    two_pass: bool = False,
    mode: str = "sample",
    mortar_gauss: int = 3,
    mortar_max_points: int = 0,
) -> ContactMap:
    """
    Convenience function for training loops:
    - Each step call can resample contact points with a varying seed to avoid overfitting to fixed locations.
    """
    seed = None if base_seed is None else (base_seed + step_index * 9973)  # decorrelate seeds
    return build_contact_map(
        asm=asm,
        specs=specs,
        n_points_per_pair=n_points_per_pair,
        seed=seed,
        prefilter_k=prefilter_k,
        chunk=chunk,
        two_pass=two_pass,
        mode=mode,
        mortar_gauss=mortar_gauss,
        mortar_max_points=mortar_max_points,
    )


# ---------------------------------------------------------------------
# Optional utilities
# ---------------------------------------------------------------------

def guess_surface_key(asm: AssemblyModel, bare_name: str) -> Optional[str]:
    """
    Try to guess a full surface key in asm.surfaces from a bare name (case-insensitive substring match).
    Returns the first unique match or None if ambiguous/not found.
    """
    target_cs = bare_name.strip()
    target = target_cs.lower()

    # - 若用户给的是裸的几何名，优先尝试自动补成 ABAQUS 的 key 形式 ASM::"<name>"
    #   这样不会把包含相同片段的其他面误判成冲突（例如 “bolt3 mirror up”）。
    if "::" not in target_cs:
        asm_key_cs = f'ASM::"{target_cs}"'
        if asm_key_cs in asm.surfaces:
            return asm_key_cs

    # 0) 完全大小写敏感匹配（优先用用户提供的精确写法）
    cs_exact = [k for k, s in asm.surfaces.items()
                if s.name.strip() == target_cs or k.strip() == target_cs]
    if len(cs_exact) == 1:
        return cs_exact[0]

    # 1) 完全大小写不敏感匹配（若仍唯一则返回，否则视为歧义）
    exact = [k for k, s in asm.surfaces.items()
             if s.name.strip().lower() == target or k.strip().lower() == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # 仍然由上层提示冲突

    # 2) 片段匹配（若唯一则返回）
    matches = [k for k, s in asm.surfaces.items()
               if target in k.lower() or target in s.name.strip().lower()]
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous or not found; let the caller handle


# ---------------------------------------------------------------------
# Minimal self-test (optional; no command line required)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Quick check when running this file alone; requires env INP_PATH or default path present.
    import os
    from inp_io.inp_parser import load_inp

    inp = os.environ.get("INP_PATH", "data/shuangfan.inp")
    if not os.path.exists(inp):
        print("[contact_pairs] INP not found. Set INP_PATH env or place file at data/shuangfan.inp.")
        exit(0)

    asm = load_inp(inp)

    # Heuristically pick two part-scope surfaces for a smoke test
    surface_keys = [k for k, v in asm.surfaces.items() if v.scope == "part"]
    if len(surface_keys) < 2:
        print("[contact_pairs] Not enough part-scope surfaces to test.")
        exit(0)

    # Use the first two as (slave, master)
    specs = [ContactPairSpec(slave_key=surface_keys[0], master_key=surface_keys[1], name="test_pair")]

    cmap = build_contact_map(asm, specs, n_points_per_pair=2000, seed=123)
    cat = cmap.concatenate()
