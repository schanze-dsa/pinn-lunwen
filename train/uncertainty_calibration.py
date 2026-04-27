#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residual-driven uncertainty calibration helpers."""

from __future__ import annotations

import numpy as np


def calibrate_sigma_by_residual(
    sigma: np.ndarray,
    residual_proxy: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Calibrate predictive sigma with a monotonic residual-driven scale.

    sigma_cal = sigma * (1 + alpha * r_norm**beta)
    where r_norm is residual normalized to [0,1].
    """

    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    r = np.asarray(residual_proxy, dtype=np.float64).reshape(-1)
    if s.shape != r.shape:
        raise ValueError(f"sigma and residual_proxy shape mismatch: {s.shape} vs {r.shape}")
    if s.size == 0:
        return s.astype(np.float64)

    a = max(0.0, float(alpha))
    b = max(0.0, float(beta))
    s = np.maximum(s, eps)

    r0 = np.nanmin(r)
    r1 = np.nanmax(r)
    span = max(r1 - r0, eps)
    rn = np.clip((r - r0) / span, 0.0, 1.0)
    scale = 1.0 + a * np.power(rn, b)
    return s * scale

