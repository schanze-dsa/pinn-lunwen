#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
material_lib.py
---------------
Lightweight material library for linear isotropic elasticity in 3D.

Conventions:
- Units follow your global choice (mm–N–MPa recommended). Keep consistent.
- Voigt notation uses engineering shear strains:
      e_voigt = [ exx, eyy, ezz, 2*eyz, 2*exz, 2*exy ]^T
  With this convention, the isotropic stiffness matrix C (6x6) has μ on the
  shear diagonals (C44=C55=C66=μ), and the upper-left 3x3 block uses λ, μ.

Typical usage:
    # 1) Build from dict tag -> (E, nu)
    lib = MaterialLibrary({"steel": (210000.0, 0.30), "aluminum": (70000.0, 0.33)})

    # 2) Map a list of per-point tags to integer material ids
    mat_ids = lib.encode_tags(["steel", "steel", "aluminum", ...])  # -> np.ndarray[int]

    # 3) In TF code, gather per-point C:
    C_table = lib.C_table_tf(dtype=tf.float32)        # (M,6,6)
    C_pts = tf.gather(C_table, mat_id_tensor)         # (N,6,6)

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------
# Core helpers
# -----------------------------

def lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """Compute Lamé parameters (λ, μ) from E and ν."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    return lam, mu


def isotropic_C_6x6(E: float, nu: float) -> np.ndarray:
    """
    Build 6x6 isotropic stiffness matrix (Voigt, engineering shear).
    """
    lam, mu = lame_from_E_nu(E, nu)
    C = np.zeros((6, 6), dtype=np.float64)
    # upper-left 3x3
    C[0, 0] = lam + 2.0 * mu
    C[1, 1] = lam + 2.0 * mu
    C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam
    # shear
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


# -----------------------------
# Material library
# -----------------------------

@dataclass(frozen=True)
class MaterialSpec:
    tag: str
    E: float
    nu: float


class MaterialLibrary:
    """
    Store and serve multiple isotropic materials.

    - Keeps deterministic ordering of tags.
    - Provides integer ids for tags (0..M-1).
    - Exposes NumPy and TensorFlow tables for C(6x6).

    Note:
    - This library does not enforce positivity checks on E, nu; do it in config validation if needed.
    """

    def __init__(self, materials: Dict[str, Any]):
        """
        Args:
            materials: dict {tag: (E, nu)} or {tag: {"E": ..., "nu": ...}}
        """
        def _extract_E_nu(spec: Any) -> Tuple[float, float]:
            if isinstance(spec, (tuple, list)) and len(spec) >= 2:
                return float(spec[0]), float(spec[1])
            if isinstance(spec, dict):
                return float(spec["E"]), float(spec["nu"])
            raise TypeError(
                "Material spec must be (E, nu) or a dict with keys 'E' and 'nu', "
                f"got {type(spec)}"
            )

        # Stable order
        tags = sorted(list(materials.keys()))
        self._specs: List[MaterialSpec] = []
        self._tag2id: Dict[str, int] = {}
        C_list: List[np.ndarray] = []

        for i, tag in enumerate(tags):
            E, nu = _extract_E_nu(materials[tag])
            self._specs.append(MaterialSpec(tag, float(E), float(nu)))
            self._tag2id[tag] = i
            C_list.append(isotropic_C_6x6(float(E), float(nu)))

        self._C_table_np = np.stack(C_list, axis=0)  # (M,6,6), float64 by default
        self._tf_cache: Dict[str, tf.Tensor] = {}    # dtype->tensor

    # ----- queries -----

    @property
    def tags(self) -> List[str]:
        return [s.tag for s in self._specs]

    def num_materials(self) -> int:
        return len(self._specs)

    def encode_tags(self, tag_list: List[str]) -> np.ndarray:
        """
        Map a list of material tags to integer ids. Unknown tags raise KeyError.
        """
        ids = []
        for t in tag_list:
            if t not in self._tag2id:
                raise KeyError(f"[MaterialLibrary] Unknown material tag '{t}'. Known: {self.tags}")
            ids.append(self._tag2id[t])
        return np.asarray(ids, dtype=np.int64)

    def id_of(self, tag: str) -> int:
        return self._tag2id[tag]

    def C_table_np(self) -> np.ndarray:
        """Return NumPy C table, shape (M,6,6), dtype float64."""
        return self._C_table_np

    def C_table_tf(self, dtype=tf.float32) -> tf.Tensor:
        """Return (and cache) TF tensor of C table, shape (M,6,6), requested dtype."""
        key = str(dtype.name if hasattr(dtype, "name") else dtype)
        if key not in self._tf_cache:
            self._tf_cache[key] = tf.convert_to_tensor(self._C_table_np, dtype=dtype)
        return self._tf_cache[key]

    # ----- pretty print / debug -----

    def summary(self) -> str:
        lines = ["MaterialLibrary:"]
        for i, s in enumerate(self._specs):
            lines.append(f"  [{i}] {s.tag:<10s}  E={s.E:.6g}  nu={s.nu:.4f}")
        return "\n".join(lines)


# -----------------------------
# Small self-test
# -----------------------------
if __name__ == "__main__":
    lib = MaterialLibrary({"steel": (210000.0, 0.30), "aluminum": (70000.0, 0.33)})
    print(lib.summary())
    Ctf = lib.C_table_tf(dtype=tf.float32)
    print("C_table_tf shape:", Ctf.shape)
    ids = lib.encode_tags(["steel", "aluminum", "steel"])
    print("ids:", ids)
