# -*- coding: utf-8 -*-
"""Preload sampling/staging mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf


class TrainerPreloadMixin:
    def _set_preload_dim(self, nb: int):
        nb_int = int(nb)
        if nb_int <= 0:
            nb_int = 3
        if nb_int != getattr(self, "_preload_dim", None):
            self._preload_dim = nb_int
            self._preload_lhs_points = np.zeros((0, nb_int), dtype=np.float32)
            self._preload_lhs_index = 0

    def _generate_lhs_points(self, n_samples: int, n_dim: int, lo: float, hi: float) -> np.ndarray:
        """Simple Latin Hypercube sampler returning shape (n_samples, n_dim)."""

        if n_samples <= 0:
            return np.zeros((0, n_dim), dtype=np.float32)
        unit = np.zeros((n_samples, n_dim), dtype=np.float32)
        for j in range(n_dim):
            seg = (np.arange(n_samples, dtype=np.float32) + self._preload_lhs_rng.random(n_samples)) / float(n_samples)
            self._preload_lhs_rng.shuffle(seg)
            unit[:, j] = seg
        scale = hi - lo
        return (lo + unit * scale).astype(np.float32)

    def _next_lhs_preload(self, n_dim: int, lo: float, hi: float) -> np.ndarray:
        batch = max(1, int(self.cfg.preload_lhs_size))
        if self._preload_lhs_points.shape[1] != n_dim or len(self._preload_lhs_points) == 0:
            self._preload_lhs_points = self._generate_lhs_points(batch, n_dim, lo, hi)
            self._preload_lhs_index = 0
        if self._preload_lhs_index >= len(self._preload_lhs_points):
            self._preload_lhs_points = self._generate_lhs_points(batch, n_dim, lo, hi)
            self._preload_lhs_index = 0
        out = self._preload_lhs_points[self._preload_lhs_index].copy()
        self._preload_lhs_index += 1
        return out

    def _sample_P(self) -> np.ndarray:
        if self._preload_sequence:
            if self._preload_current_target is None:
                idx = self._preload_sequence_index
                self._preload_current_target = self._preload_sequence[idx].copy()
                base_order = (
                    self._preload_sequence_orders[idx]
                    if idx < len(self._preload_sequence_orders)
                    else None
                )
                self._preload_current_order = (
                    None if base_order is None else base_order.copy()
                )
            target = self._preload_current_target.copy()
            current_order = (
                None if self._preload_current_order is None else self._preload_current_order.copy()
            )
            jitter = float(self.cfg.preload_sequence_jitter or 0.0)
            if jitter > 0.0:
                noise = np.random.uniform(-jitter, jitter, size=target.shape)
                target = target + noise.astype(np.float32)
            lo, hi = self.cfg.preload_min, self.cfg.preload_max
            target = np.clip(target, lo, hi)

            self._preload_sequence_hold += 1
            if self._preload_sequence_hold >= max(1, self.cfg.preload_sequence_repeat):
                self._preload_sequence_hold = 0
                self._preload_sequence_index = (self._preload_sequence_index + 1) % len(
                    self._preload_sequence
                )
                if self._preload_sequence_index == 0 and self.cfg.preload_sequence_shuffle:
                    perm = np.random.permutation(len(self._preload_sequence))
                    self._preload_sequence = [self._preload_sequence[i] for i in perm]
                    self._preload_sequence_orders = [
                        self._preload_sequence_orders[i] if i < len(self._preload_sequence_orders) else None
                        for i in perm
                    ]
                idx = self._preload_sequence_index
                self._preload_current_target = self._preload_sequence[idx].copy()
                base_order = (
                    self._preload_sequence_orders[idx]
                    if idx < len(self._preload_sequence_orders)
                    else None
                )
                self._preload_current_order = (
                    None if base_order is None else base_order.copy()
                )

            self._last_preload_order = None if current_order is None else current_order.copy()
            return target.astype(np.float32)

        lo, hi = self.cfg.preload_min, self.cfg.preload_max
        nb = int(self._preload_dim)
        repeat = max(1, int(getattr(self.cfg, "preload_sequence_repeat", 1) or 1))
        if repeat > 1:
            # Hold the same preload vector (and optionally the same tightening order)
            # for a few consecutive optimization steps, so each sampled case has a
            # chance to converge instead of being "seen once and forgotten".
            if self._preload_current_target is None:
                sampling = (self.cfg.preload_sampling or "uniform").lower()
                if sampling == "lhs":
                    out = self._next_lhs_preload(nb, lo, hi)
                else:
                    out = np.random.uniform(lo, hi, size=(nb,)).astype(np.float32)
                self._preload_current_target = out.astype(np.float32).copy()

                if self.cfg.preload_use_stages:
                    if self.cfg.preload_randomize_order:
                        self._preload_current_order = np.random.permutation(nb).astype(np.int32)
                    else:
                        self._preload_current_order = np.arange(nb, dtype=np.int32)
                else:
                    self._preload_current_order = None
                self._preload_sequence_hold = 0

            target = self._preload_current_target.copy()
            current_order = (
                None if self._preload_current_order is None else self._preload_current_order.copy()
            )

            self._preload_sequence_hold += 1
            if self._preload_sequence_hold >= repeat:
                self._preload_sequence_hold = 0
                self._preload_current_target = None
                self._preload_current_order = None

            self._last_preload_order = None if current_order is None else current_order.copy()
            return target.astype(np.float32)

        sampling = (self.cfg.preload_sampling or "uniform").lower()
        if sampling == "lhs":
            out = self._next_lhs_preload(nb, lo, hi)
        else:
            out = np.random.uniform(lo, hi, size=(nb,)).astype(np.float32)
        self._last_preload_order = None
        return out.astype(np.float32)

    def _normalize_order(self, order: Optional[Any], nb: int) -> Optional[np.ndarray]:
        if order is None:
            return None
        arr = np.array(order, dtype=np.int32).reshape(-1)
        if arr.size != nb:
            raise ValueError(f"order length must be {nb}, got {arr.size}.")
        if np.all(arr >= 1) and np.max(arr) <= nb and np.min(arr) >= 1:
            arr = arr - 1
        unique = sorted(set(arr.tolist()))
        if unique != list(range(nb)):
            raise ValueError(
                f"order must be a permutation of 0~{nb-1} (or 1~{nb}), got {list(arr)}."
            )
        return arr.astype(np.int32)

    def _build_stage_case(self, P: np.ndarray, order: np.ndarray) -> Dict[str, np.ndarray]:
        nb = int(P.shape[0])
        order = np.asarray(order, dtype=np.int32).reshape(-1)
        if order.size != nb:
            raise ValueError(f"order length must be {nb}, got {order.size}.")
        stage_loads = []
        stage_masks = []
        stage_last = []
        cumulative = np.zeros_like(P)
        mask = np.zeros_like(P)
        rank = np.zeros((nb,), dtype=np.float32)
        for pos, idx in enumerate(order):
            idx_int = int(idx)
            cumulative[idx_int] = P[idx_int]
            mask[idx_int] = 1.0
            stage_loads.append(cumulative.copy())
            stage_masks.append(mask.copy())
            onehot = np.zeros_like(P)
            onehot[idx_int] = 1.0
            stage_last.append(onehot)
            rank[idx_int] = float(pos)

        mode = str(getattr(self.cfg.total_cfg, "preload_stage_mode", "") or "")
        mode = mode.strip().lower().replace("-", "_")
        append_release_stage = bool(getattr(self.cfg, "preload_append_release_stage", True))
        if mode == "force_then_lock" and append_release_stage:
            # Append a final "release" stage so the last stage represents the post-tightening
            # equilibrium (all bolts locked, no active force control). This is where tightening
            # order effects should manifest most clearly.
            stage_loads.append(cumulative.copy())
            stage_masks.append(mask.copy())
            stage_last.append(np.zeros_like(P))
        if nb > 1:
            rank = rank / float(nb - 1)
        else:
            rank = np.zeros_like(rank)
        rank_matrix = np.tile(rank.reshape(1, -1), (len(stage_loads), 1))
        return {
            "stages": np.stack(stage_loads).astype(np.float32),
            "stage_masks": np.stack(stage_masks).astype(np.float32),
            "stage_last": np.stack(stage_last).astype(np.float32),
            "stage_rank": rank.astype(np.float32),
            "stage_rank_matrix": rank_matrix.astype(np.float32),
        }

    def _sample_preload_case(self) -> Dict[str, np.ndarray]:
        dataset = getattr(self, "_supervision_dataset", None)
        if dataset is not None:
            case = dataset.next_case("train")
            if self.cfg.preload_use_stages and "stages" not in case:
                case.update(self._build_stage_case(case["P"], case["order"]))
            return case

        P = self._sample_P()
        case: Dict[str, np.ndarray] = {"P": P}
        if not self.cfg.preload_use_stages:
            return case

        base_order = None if self._last_preload_order is None else self._last_preload_order.copy()
        if base_order is None:
            if self.cfg.preload_randomize_order:
                order = np.random.permutation(P.shape[0]).astype(np.int32)
            else:
                order = np.arange(P.shape[0], dtype=np.int32)
        else:
            order = base_order.astype(np.int32)

        case["order"] = order
        case.update(self._build_stage_case(P, order))
        return case

    def _make_preload_params(self, case: Dict[str, np.ndarray]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "P": tf.convert_to_tensor(case["P"], dtype=tf.float32)
        }
        if "X_obs" in case and "U_obs" in case and not self.cfg.preload_use_stages:
            params["X_obs"] = tf.convert_to_tensor(case["X_obs"], dtype=tf.float32)
            params["U_obs"] = tf.convert_to_tensor(case["U_obs"], dtype=tf.float32)
            if "obs_morphology_weight" in case:
                params["data_weight"] = tf.convert_to_tensor(case["obs_morphology_weight"], dtype=tf.float32)
        if not self.cfg.preload_use_stages or "stages" not in case:
            return params

        stages = case.get("stages")
        masks = case.get("stage_masks")
        lasts = case.get("stage_last")
        order_np = case.get("order")
        rank_np = case.get("stage_rank")
        rank_matrix_np = case.get("stage_rank_matrix")
        if order_np is None:
            order_np = np.arange(case["P"].shape[0], dtype=np.int32)
        order_tf = tf.convert_to_tensor(order_np, dtype=tf.int32)
        rank_tf = None
        if rank_np is not None:
            rank_tf = tf.convert_to_tensor(rank_np, dtype=tf.float32)
        rank_matrix_tf = None
        if rank_matrix_np is not None:
            rank_matrix_tf = tf.convert_to_tensor(rank_matrix_np, dtype=tf.float32)
        elif rank_tf is not None:
            rank_matrix_tf = tf.repeat(
                tf.expand_dims(rank_tf, axis=0), repeats=int(len(stages)), axis=0
            )
        mask_tensor = (
            tf.convert_to_tensor(masks, dtype=tf.float32) if masks is not None else None
        )
        last_tensor = (
            tf.convert_to_tensor(lasts, dtype=tf.float32) if lasts is not None else None
        )

        stage_params_P: List[tf.Tensor] = []
        stage_params_feat: List[tf.Tensor] = []
        stage_count = int(len(stages))
        shift = tf.cast(self.cfg.model_cfg.preload_shift, tf.float32)
        scale = tf.cast(self.cfg.model_cfg.preload_scale, tf.float32)
        n_bolts = int(case["P"].shape[0])
        feat_dim = n_bolts
        if masks is not None:
            feat_dim += n_bolts
        if lasts is not None:
            feat_dim += n_bolts
        if rank_tf is not None:
            feat_dim += n_bolts
        tighten_time = None
        if rank_tf is not None:
            tighten_time = rank_tf
        else:
            try:
                pos = tf.range(n_bolts, dtype=tf.float32)
                rank_raw = tf.scatter_nd(tf.reshape(order_tf, (-1, 1)), pos, [n_bolts])
                if n_bolts > 1:
                    rank_raw = rank_raw / tf.cast(n_bolts - 1, tf.float32)
                else:
                    rank_raw = tf.zeros_like(rank_raw)
                tighten_time = rank_raw
            except Exception:
                tighten_time = None
        if tighten_time is not None:
            feat_dim += 1 + n_bolts

        for idx in range(stage_count):
            p_stage = tf.convert_to_tensor(stages[idx], dtype=tf.float32)
            norm = (p_stage - shift) / scale
            feat_parts = [norm]
            if mask_tensor is not None:
                feat_parts.append(mask_tensor[idx])
            if last_tensor is not None:
                feat_parts.append(last_tensor[idx])
            if rank_tf is not None:
                feat_parts.append(rank_tf)
            if tighten_time is not None:
                if stage_count > 1:
                    t_stage = tf.cast(idx, tf.float32) / tf.cast(stage_count - 1, tf.float32)
                else:
                    t_stage = tf.cast(0.0, tf.float32)
                t_stage_vec = tf.reshape(t_stage, (1,))
                delta_t = tf.maximum(tf.cast(0.0, tf.float32), t_stage - tighten_time)
                feat_parts.append(t_stage_vec)
                feat_parts.append(delta_t)
            features = tf.concat(feat_parts, axis=0)
            features.set_shape((feat_dim,))
            stage_params_P.append(p_stage)
            stage_params_feat.append(features)
        stage_tensor_P = tf.stack(stage_params_P, axis=0)
        stage_tensor_P.set_shape((stage_count, n_bolts))
        stage_tensor_feat = tf.stack(stage_params_feat, axis=0)
        stage_tensor_feat.set_shape((stage_count, feat_dim))
        stage_dict: Dict[str, tf.Tensor] = {
            "P": stage_tensor_P,
            "P_hat": stage_tensor_feat,
        }
        if mask_tensor is not None:
            mask_tensor.set_shape((stage_count, n_bolts))
            stage_dict["stage_mask"] = mask_tensor
        if last_tensor is not None:
            last_tensor.set_shape((stage_count, n_bolts))
            stage_dict["stage_last"] = last_tensor
        if rank_matrix_tf is not None:
            stage_dict["stage_rank"] = rank_matrix_tf
        if "X_obs" in case and "U_obs" in case:
            stage_dict["X_obs"] = tf.convert_to_tensor(case["X_obs"], dtype=tf.float32)
            stage_dict["U_obs"] = tf.convert_to_tensor(case["U_obs"], dtype=tf.float32)
            if "U_obs_delta" in case:
                stage_dict["U_obs_delta"] = tf.convert_to_tensor(case["U_obs_delta"], dtype=tf.float32)
            if "obs_morphology_weight" in case:
                stage_dict["data_weight"] = tf.convert_to_tensor(case["obs_morphology_weight"], dtype=tf.float32)
        params["stages"] = stage_dict
        params["stage_order"] = order_tf
        if rank_tf is not None:
            params["stage_rank"] = rank_tf
        params["stage_count"] = tf.constant(stage_count, dtype=tf.int32)
        return params

    @staticmethod
    def _static_last_dim(arr: Any) -> Optional[int]:
        try:
            dim = getattr(arr, "shape", None)
            if dim is None:
                return None
            last = dim[-1]
            return None if last is None else int(last)
        except Exception:
            return None

    def _infer_preload_feat_dim(self, params: Dict[str, Any]) -> Optional[int]:
        """Static infer of P_hat dimension with staged-preload preference."""

        if not isinstance(params, dict):
            return None

        stages = params.get("stages")
        if isinstance(stages, dict):
            feat = stages.get("P_hat")
            dim = self._static_last_dim(feat)
            if dim:
                return dim

        if "P_hat" in params:
            dim = self._static_last_dim(params.get("P_hat"))
            if dim:
                return dim

        return self._static_last_dim(params.get("P"))

    def _extract_final_stage_params(
        self, params: Dict[str, Any], keep_context: bool = False
    ) -> Dict[str, Any]:
        """Return the last staged parameter set, optionally carrying context."""

        if not (
            self.cfg.preload_use_stages
            and isinstance(params, dict)
            and "stages" in params
        ):
            return params

        stages = params["stages"]
        final: Optional[Dict[str, tf.Tensor]] = None
        if isinstance(stages, dict) and stages:
            last_P = stages.get("P")
            last_feat = stages.get("P_hat")
            if last_P is not None and last_feat is not None:
                final = {"P": last_P[-1], "P_hat": last_feat[-1]}
                rank_tensor = stages.get("stage_rank")
                if rank_tensor is not None:
                    if getattr(rank_tensor, "shape", None) is not None and rank_tensor.shape.rank == 2:
                        final["stage_rank"] = rank_tensor[-1]
                    else:
                        final["stage_rank"] = rank_tensor
                obs_x = stages.get("X_obs")
                if obs_x is not None and getattr(obs_x, "shape", None) is not None and obs_x.shape.rank == 3:
                    final["X_obs"] = obs_x[-1]
                obs_u = stages.get("U_obs")
                if obs_u is not None and getattr(obs_u, "shape", None) is not None and obs_u.shape.rank == 3:
                    final["U_obs"] = obs_u[-1]
                data_weight = stages.get("data_weight")
                if data_weight is not None and getattr(data_weight, "shape", None) is not None and data_weight.shape.rank == 3:
                    final["data_weight"] = data_weight[-1]
        elif isinstance(stages, (list, tuple)) and stages:
            last_stage = stages[-1]
            if isinstance(last_stage, dict):
                final = dict(last_stage)
            else:
                p_val, z_val = last_stage
                final = {"P": p_val, "P_hat": z_val}

        if final is None:
            return params

        if keep_context:
            for key in (
                "stage_order",
                "stage_rank",
                "stage_count",
                "stage_mask",
                "stage_last",
            ):
                if key in params and key not in final:
                    final[key] = params[key]
        return final

    def _extract_stage_params(
        self, params: Dict[str, Any], stage_index: int, keep_context: bool = False
    ) -> Dict[str, Any]:
        """Return the indexed staged parameter set (0-based), optionally carrying context."""

        if not (
            self.cfg.preload_use_stages
            and isinstance(params, dict)
            and "stages" in params
        ):
            return params

        stages = params["stages"]
        out: Optional[Dict[str, Any]] = None
        if isinstance(stages, dict) and stages:
            stage_P = stages.get("P")
            stage_feat = stages.get("P_hat")
            if stage_P is not None and stage_feat is not None:
                stage_count = 0
                try:
                    stage_count = int(stage_P.shape[0])
                except Exception:
                    stage_count = 0
                if stage_count <= 0:
                    stage_count = int(tf.shape(stage_P)[0].numpy())
                idx = stage_index % stage_count
                out = {"P": stage_P[idx], "P_hat": stage_feat[idx]}

                rank_tensor = stages.get("stage_rank")
                if rank_tensor is not None:
                    if getattr(rank_tensor, "shape", None) is not None and rank_tensor.shape.rank == 2:
                        out["stage_rank"] = rank_tensor[idx]
                    else:
                        out["stage_rank"] = rank_tensor

                mask_tensor = stages.get("stage_mask")
                if mask_tensor is not None and getattr(mask_tensor, "shape", None) is not None and mask_tensor.shape.rank == 2:
                    out["stage_mask"] = mask_tensor[idx]

                last_tensor = stages.get("stage_last")
                if last_tensor is not None and getattr(last_tensor, "shape", None) is not None and last_tensor.shape.rank == 2:
                    out["stage_last"] = last_tensor[idx]
                obs_x = stages.get("X_obs")
                if obs_x is not None and getattr(obs_x, "shape", None) is not None and obs_x.shape.rank == 3:
                    out["X_obs"] = obs_x[idx]
                obs_u = stages.get("U_obs")
                if obs_u is not None and getattr(obs_u, "shape", None) is not None and obs_u.shape.rank == 3:
                    out["U_obs"] = obs_u[idx]
                obs_delta = stages.get("U_obs_delta")
                if (
                    obs_delta is not None
                    and getattr(obs_delta, "shape", None) is not None
                    and obs_delta.shape.rank == 3
                    and idx > 0
                ):
                    out["U_obs_delta"] = obs_delta[idx - 1]
                data_weight = stages.get("data_weight")
                if data_weight is not None and getattr(data_weight, "shape", None) is not None and data_weight.shape.rank == 3:
                    out["data_weight"] = data_weight[idx]
        elif isinstance(stages, (list, tuple)) and stages:
            idx = stage_index % len(stages)
            stage_item = stages[idx]
            if isinstance(stage_item, dict):
                out = dict(stage_item)
            else:
                p_val, z_val = stage_item
                out = {"P": p_val, "P_hat": z_val}

        if out is None:
            return params

        if keep_context:
            for key in (
                "stage_order",
                "stage_rank",
                "stage_count",
            ):
                if key in params and key not in out:
                    out[key] = params[key]
        return out

    def _get_stage_count(self, params: Dict[str, Any]) -> int:
        """Infer stage count from params; falls back to 1."""
        if not (self.cfg.preload_use_stages and isinstance(params, dict) and "stages" in params):
            return 1
        stages = params["stages"]
        if isinstance(stages, dict) and stages:
            stage_P = stages.get("P")
            if stage_P is not None:
                try:
                    return int(stage_P.shape[0])
                except Exception:
                    try:
                        return int(tf.shape(stage_P)[0].numpy())
                    except Exception:
                        return 1
        if isinstance(stages, (list, tuple)):
            return max(1, len(stages))
        return 1

    def _active_stage_count(self, step: Optional[int], stage_count: int) -> int:
        """Determine how many stages to include based on a schedule."""
        if step is None or stage_count <= 1:
            return stage_count
        schedule = getattr(self.cfg, "stage_schedule_steps", None) or []
        if not schedule or len(schedule) < stage_count:
            return stage_count
        cum = 0
        for idx, span in enumerate(schedule[:stage_count]):
            try:
                span_i = int(span)
            except Exception:
                span_i = 0
            if span_i <= 0:
                continue
            cum += span_i
            if step <= cum:
                return idx + 1
        return stage_count

    def _make_warmup_case(self) -> Dict[str, np.ndarray]:
        mid = 0.5 * (float(self.cfg.preload_min) + float(self.cfg.preload_max))
        base = np.full((3,), mid, dtype=np.float32)
        case: Dict[str, np.ndarray] = {"P": base}
        if self.cfg.preload_use_stages:
            order = np.arange(base.shape[0], dtype=np.int32)
            case["order"] = order
            case.update(self._build_stage_case(base, order))
        return case
