# -*- coding: utf-8 -*-
"""
attach_ties_bcs.py
------------------
Attach boundary conditions parsed from the assembly model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig

_DOF_LABELS = {
    "UX": 0, "UY": 1, "UZ": 2,
    "X": 0, "Y": 1, "Z": 2,
    "1": 0, "2": 1, "3": 2,
}


def _parse_bc_line(line: str) -> Tuple[int | None, List[int], float]:
    """Parse an ANSYS D command line into (node_id, dof_indices, value)."""
    parts = [p.strip() for p in line.split(",")]
    if not parts or parts[0].upper() != "D":
        return None, [], 0.0
    try:
        node = int(float(parts[1]))
    except Exception:
        node = None
    dof_raw = parts[2].strip().upper() if len(parts) > 2 else ""
    value = 0.0
    if len(parts) > 3:
        try:
            value = float(parts[3])
        except Exception:
            value = 0.0
    if dof_raw == "ALL":
        return node, [0, 1, 2], value
    if dof_raw in _DOF_LABELS:
        return node, [_DOF_LABELS[dof_raw]], value
    return node, [], value


def attach_bcs_from_asm(total, asm, cfg) -> List[BoundaryPenalty]:
    """
    Build BoundaryPenalty from asm.boundaries and attach to TotalEnergy.
    """
    boundaries = list(getattr(asm, "boundaries", []) or [])
    if not boundaries:
        return []

    bc_alpha = float(getattr(cfg, "bc_alpha", 1.0e4))
    bc_mu = float(getattr(cfg, "bc_mu", 1.0e3))
    bc_mode = str(getattr(cfg, "bc_mode", "penalty")).lower()
    bc_cfg = BoundaryConfig(alpha=bc_alpha, mode=bc_mode, mu=bc_mu)

    node_map: Dict[int, Dict[str, Any]] = {}
    for entry in boundaries:
        raw = getattr(entry, "raw", "") or ""
        nid, dof_idx, value = _parse_bc_line(raw)
        if nid is None or not dof_idx:
            continue
        rec = node_map.setdefault(nid, {"mask": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]})
        for di in dof_idx:
            rec["mask"][di] = 1.0
            rec["target"][di] = float(value)

    if not node_map:
        return []

    X_list: List[Tuple[float, float, float]] = []
    mask_list: List[List[float]] = []
    target_list: List[List[float]] = []
    for nid, rec in node_map.items():
        if nid not in asm.nodes:
            continue
        X_list.append(asm.nodes[nid])
        mask_list.append(rec["mask"])
        target_list.append(rec["target"])

    if not X_list:
        return []

    X_bc = np.asarray(X_list, dtype=np.float32)
    mask = np.asarray(mask_list, dtype=np.float32)
    u_target = np.asarray(target_list, dtype=np.float32)

    bc = BoundaryPenalty(cfg=bc_cfg)
    bc.build_from_numpy(X_bc, mask, u_target, w_bc=None)
    total.attach(bcs=[bc])
    return [bc]
