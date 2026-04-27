#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
volume_quadrature.py
--------------------
Generate volume integration points for elasticity assembly:
  - X_vol : (N,3)  element centroids (or other rules in future)
  - w_vol : (N,)   element volumes (weights)
  - mat_id: (N,)   integer material ids for each point

Supported elements: C3D4 (tet), C3D8 (hex).  Hex volume is computed by a
robust tetrahedral decomposition.

Typical usage:
    from inp_io.inp_parser import load_inp
    from physics.material_lib import MaterialLibrary
    asm = load_inp("data/shuangfan.inp")

    # map part -> material tag
    part2mat = {"MIRROR": "mirror_al", "BOLT1": "steel", ...}
    matlib = MaterialLibrary({
        "mirror_al": (70000.0, 0.33),
        "steel": (210000.0, 0.30),
    })

    X_vol, w_vol, mat_id = build_volume_points(asm, part2mat, matlib)

Notes:
- This module does *not* read material assignments from the INP automatically;
  pass a dict `part2mat` 明确指派每个 Part 的材料标签（否则使用 default_tag）。
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

from inp_io.inp_parser import AssemblyModel, PartMesh, ElementBlock


# -----------------------------
# Public API
# -----------------------------

def build_volume_points(asm: AssemblyModel,
                        part2mat: Dict[str, str],
                        matlib,
                        default_tag: str = "steel"
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_vol, w_vol, mat_id) by visiting all supported elements in assembly parts.

    Args:
        asm: parsed AssemblyModel
        part2mat: mapping from part_name -> material tag (must exist in matlib)
        matlib: MaterialLibrary instance (for encoding tag -> id)
        default_tag: used when part not present in part2mat

    Returns:
        X_vol: (N,3) float64
        w_vol: (N,)  float64
        mat_id: (N,) int64
    """
    Xs: List[np.ndarray] = []
    Ws: List[np.ndarray] = []
    Mids: List[np.ndarray] = []

    for pname, pm in asm.parts.items():
        tag = part2mat.get(pname, default_tag)
        mid = matlib.id_of(tag)  # may raise if unknown -> better fail-fast

        x_part, w_part = _volume_points_for_part(pm)   # centroids & volumes
        if x_part.size == 0:
            continue
        Xs.append(x_part)
        Ws.append(w_part)
        Mids.append(np.full((x_part.shape[0],), mid, dtype=np.int64))

    if not Xs:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64))

    X_vol = np.vstack(Xs).astype(np.float64)
    w_vol = np.concatenate(Ws).astype(np.float64)
    mat_id = np.concatenate(Mids).astype(np.int64)
    return X_vol, w_vol, mat_id


# -----------------------------
# Per-part processing
# -----------------------------

def _volume_points_for_part(pm: PartMesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single part, gather centroids and volumes of all supported elements.
    Returns:
        X: (Ne,3) centroids
        W: (Ne,)  volumes
    """
    Xs: List[np.ndarray] = []
    Ws: List[np.ndarray] = []

    for blk in pm.element_blocks:
        et = (blk.elem_type or "").upper()
        if et == "C3D4":
            x, w = _centroid_weight_c3d4_block(pm, blk)
        elif et in {"C3D8", "SOLID185"}:
            x, w = _centroid_weight_c3d8_block(pm, blk)
        else:
            # unsupported element type -> skip
            continue
        if x.size:
            Xs.append(x)
            Ws.append(w)

    if not Xs:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    return np.vstack(Xs).astype(np.float64), np.concatenate(Ws).astype(np.float64)


# -----------------------------
# Element block handlers
# -----------------------------
# --- tet (C3D4) ---
def _centroid_weight_c3d4_block(pm: PartMesh, blk: ElementBlock) -> Tuple[np.ndarray, np.ndarray]:
    nn = len(blk.elem_ids)
    X = np.zeros((nn, 3), dtype=np.float64)
    W = np.zeros((nn,), dtype=np.float64)

    for k, conn in enumerate(blk.connectivity):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        n1, n2, n3, n4 = [np.asarray(pm.nodes_xyz[int(n)], dtype=np.float64) for n in conn[:4]]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        X[k] = (n1 + n2 + n3 + n4) / 4.0
        b1 = n2 - n1; b2 = n3 - n1; b3 = n4 - n1
        vol = abs(np.linalg.det(np.stack([b1, b2, b3], axis=1))) / 6.0
        W[k] = vol
    return X, W

# --- hex (C3D8) ---
def _centroid_weight_c3d8_block(pm: PartMesh, blk: ElementBlock) -> Tuple[np.ndarray, np.ndarray]:
    nn = len(blk.elem_ids)
    X = np.zeros((nn, 3), dtype=np.float64)
    W = np.zeros((nn,), dtype=np.float64)

    def tet_vol(a, b, c, d) -> float:
        return abs(np.linalg.det(np.stack([b - a, c - a, d - a], axis=1))) / 6.0

    for k, conn in enumerate(blk.connectivity):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        P = [np.asarray(pm.nodes_xyz[int(n)], dtype=np.float64) for n in conn[:8]]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        X[k] = np.mean(P, axis=0)
        tets = [(0,1,3,4),(1,2,3,6),(1,6,3,4),(1,5,6,4),(3,6,7,4)]
        vol = 0.0
        for (i,j,l,m) in tets:
            vol += tet_vol(P[i], P[j], P[l], P[m])
        W[k] = vol
    return X, W
