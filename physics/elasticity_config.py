# -*- coding: utf-8 -*-
"""
elasticity_config.py
--------------------
Config container for residual elasticity (PINN) computations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ElasticityConfig:
    # Coordinate scaling (if mesh is not normalized)
    coord_scale: float = 1.0
    # Chunk size for batching gradient computations (0 -> no chunking)
    chunk_size: int = 0
    # Placeholder for compatibility
    use_pfor: bool = False
    # Enable numeric checks (NaN/Inf)
    check_nan: bool = False
    # Number of volume points per step (RAR target); optional
    n_points_per_step: int = 0
    # Stress supervision weight (0 disables stress loss)
    stress_loss_weight: float = 1.0
    # Use forward-mode JVP to compute strain (faster than 3x reverse-mode grads)
    use_forward_mode: bool = True
    # Cache per-sample metrics (psi/idx) for volume-RAR diagnostics.
    # Disable this to avoid extra device->host synchronization.
    cache_sample_metrics: bool = True
