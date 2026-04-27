# -*- coding: utf-8 -*-
"""Initialization mixin extracted from Trainer.__init__."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


class TrainerInitMixin:
    def _init_runtime_state(self, cfg) -> None:
        self._preload_dim: int = 3
        self._preload_lhs_rng = np.random.default_rng(cfg.seed + 11)
        self._preload_lhs_points: np.ndarray = np.zeros((0, self._preload_dim), dtype=np.float32)
        self._preload_lhs_index: int = 0

        self.device_summary = "Unknown"
        self._step_stage_times: List[Tuple[str, float]] = []
        self._pi_baseline: Optional[float] = None
        self._pi_ema: Optional[float] = None
        self._prev_pi: Optional[float] = None
        self._best_pi_ema: Optional[float] = None
        self._nonfinite_streak: int = 0
        self._diverge_streak: int = 0
        self._contact_route_ema: Optional[float] = None
        self._contact_route_ref: Optional[float] = None
        self._preload_sequence: List[np.ndarray] = []
        self._preload_sequence_orders: List[Optional[np.ndarray]] = []
        self._preload_sequence_index: int = 0
        self._preload_sequence_hold: int = 0
        self._preload_current_target: Optional[np.ndarray] = None
        self._preload_current_order: Optional[np.ndarray] = None
        self._last_preload_order: Optional[np.ndarray] = None
        self._last_preload_case: Optional[Dict[str, np.ndarray]] = None
        self._train_vars: List[tf.Variable] = []
        self._total_ref = None
        self._base_weights: Dict[str, float] = {}
        self._loss_keys: List[str] = []
        self._static_weight_vector: Optional[tf.Tensor] = None
        self._active_weight_overrides: Dict[str, float] = {}
        self._apply_gradients_kwargs: Dict[str, Any] = {}
        self._current_contact_cat: Optional[Dict[str, np.ndarray]] = None
        self._contact_hardening_targets: Optional[Dict[str, float]] = None
        self._contact_hardening_frozen: bool = False
        self._strict_bilevel_stats: Dict[str, int] = {
            "total": 0,
            "converged": 0,
            "fallback": 0,
            "skipped": 0,
        }
        self._strict_bilevel_freeze_requested: bool = False
        self._strict_bilevel_backoff_requested: bool = False
        self._strict_bilevel_force_detach: bool = False
        self._strict_bilevel_traction_scale: float = 1.0
        self._continuation_freeze_events: int = 0
        self._supervision_dataset = None
        self._tqdm_enabled: bool = self._resolve_tqdm_enabled()
        self._viz_reference_cache_path: Optional[str] = None
        self._viz_reference_cache: Optional[Dict[str, Any]] = None
        self._asm_node_ids: Optional[set[int]] = None
        self._latest_val_summary: Optional[Dict[str, float]] = None
        self._latest_val_step: int = 0
        self._val_plateau_best: Optional[float] = None
        self._val_plateau_bad_count: int = 0
        self._best_ckpt_path: Optional[str] = None
        self._final_ckpt_path: Optional[str] = None

    def _init_preload_sequence(self, cfg) -> None:
        if cfg.preload_specs:
            self._set_preload_dim(len(cfg.preload_specs))

        if cfg.preload_sequence:
            sanitized: List[np.ndarray] = []
            sanitized_orders: List[Optional[np.ndarray]] = []
            for idx, entry in enumerate(cfg.preload_sequence):
                order_entry = None
                values_entry: Any = entry
                if isinstance(entry, dict):
                    order_entry = entry.get("order")
                    for key in ("values", "loads", "P", "p", "preload", "forces"):
                        if key in entry:
                            values_entry = entry[key]
                            break
                try:
                    arr = np.array(values_entry, dtype=np.float32).reshape(-1)
                except Exception:
                    print(f"[tightening] ignore preload_sequence[{idx}]: invalid numeric array: {entry}")
                    sanitized_orders.append(None)
                    continue
                if arr.size == 0:
                    print(f"[tightening] ignore preload_sequence[{idx}]: empty values")
                    sanitized_orders.append(None)
                    continue
                nb = int(getattr(self, "_preload_dim", 0) or len(cfg.preload_specs) or 1)
                if arr.size == 1:
                    arr = np.repeat(arr, nb)
                if arr.size != nb:
                    print(
                        f"[tightening] ignore preload_sequence[{idx}]: expected {nb} values, got {arr.size}"
                    )
                    sanitized_orders.append(None)
                    continue

                order_arr: Optional[np.ndarray] = None
                if order_entry is not None:
                    try:
                        order_raw = np.array(order_entry, dtype=np.int32).reshape(-1)
                    except Exception:
                        print(f"[tightening] ignore preload_sequence[{idx}] order: cannot parse {order_entry}")
                        order_raw = None
                    if order_raw is not None:
                        nb = arr.size
                        if order_raw.size != nb:
                            print(f"[tightening] ignore preload_sequence[{idx}] order: expected {nb} entries")
                        else:
                            if np.all(order_raw >= 1) and np.max(order_raw) <= nb and np.min(order_raw) >= 1:
                                order_raw = order_raw - 1
                            unique = sorted(set(order_raw.tolist()))
                            if unique != list(range(nb)):
                                print(
                                    f"[tightening] ignore preload_sequence[{idx}] order: must be a permutation of 0..{nb-1} (or 1..{nb})"
                                )
                            else:
                                order_arr = order_raw.astype(np.int32)

                sanitized.append(arr.astype(np.float32))
                sanitized_orders.append(order_arr.copy() if order_arr is not None else None)

            if sanitized:
                if cfg.preload_sequence_shuffle:
                    perm = np.random.permutation(len(sanitized))
                    sanitized = [sanitized[i] for i in perm]
                    sanitized_orders = [sanitized_orders[i] for i in perm]
                self._preload_sequence = sanitized
                self._preload_sequence_orders = sanitized_orders
                self._preload_current_target = self._preload_sequence[0].copy()
                if self._preload_sequence_orders:
                    self._preload_current_order = (
                        None
                        if self._preload_sequence_orders[0] is None
                        else self._preload_sequence_orders[0].copy()
                    )
                hold = max(1, cfg.preload_sequence_repeat)
                print(
                    f"[tightening] using preload sequence: {len(self._preload_sequence)} groups, each held for {hold} step(s)"
                )
                if cfg.preload_sequence_jitter > 0:
                    print(
                        f"[tightening] preload sequence applies +/-{cfg.preload_sequence_jitter}N uniform jitter"
                    )
                self._set_preload_dim(self._preload_sequence[0].size)
            else:
                print("[tightening] preload_sequence has no valid items; fallback to random sampling")
        if cfg.model_cfg.preload_scale:
            print(
                f"[tightening] normalize preload: shift={cfg.model_cfg.preload_shift:.2f}, "
                f"scale={cfg.model_cfg.preload_scale:.2f}"
            )

    def _init_device_and_precision(self, cfg) -> None:
        gpus = tf.config.list_physical_devices("GPU")
        gpu_labels = []
        for idx, g in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
            label = getattr(g, "name", None)
            if label:
                label = label.split("/")[-1]
                parts = label.split(":")
                if len(parts) >= 2:
                    label = ":".join(parts[-2:])
            else:
                label = f"GPU:{idx}"
            gpu_labels.append(label)

        if gpu_labels:
            self.device_summary = f"GPU ({', '.join(gpu_labels)})"
            print(f"[trainer] training on GPU: {', '.join(gpu_labels)}")
        else:
            self.device_summary = "CPU"
            print("[trainer] no GPU detected, training on CPU.")

        if cfg.mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy(cfg.mixed_precision)
                print(f"[pinn_model] Mixed precision policy set to: {cfg.mixed_precision}")
            except Exception as e:
                print("[pinn_model] Failed to set mixed precision:", e)

        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def _init_runtime_components(self) -> None:
        self.asm = None
        self.matlib = None
        self.model = None
        self.optimizer = None

        self.elasticity = None
        self.contact = None
        self.tightening = None
        self.bcs_ops = []
        self._cp_specs = []

        self.ckpt = None
        self.ckpt_manager = None
        self.best_metric = float("inf")
        self._best_ckpt_path = None
        self._final_ckpt_path = None
        self._resumed_ckpt_path = None

        self.X_vol = None
        self.w_vol = None
        self.mat_id = None
        self.enum_names = []
        self.id2props_map = {}
        self.loss_state = None
