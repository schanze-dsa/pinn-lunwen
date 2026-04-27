# -*- coding: utf-8 -*-
"""Training-loop mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import copy
import os
import time
from typing import Dict, Mapping, Optional

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from train.attach_ties_bcs import attach_bcs_from_asm
from train.loss_weights import LossWeightState


class TrainerRunMixin:
    def _set_condition_residual_runtime_scale(self, step: int) -> Optional[float]:
        encoder = getattr(getattr(self, "model", None), "encoder", None)
        if encoder is None or not hasattr(encoder, "set_runtime_residual_scale"):
            return None
        warmup_steps = max(0, int(getattr(getattr(self, "cfg", None), "ase_residual_warmup_steps", 0) or 0))
        if warmup_steps <= 0:
            return float(encoder.set_runtime_residual_scale(1.0))
        scale = min(1.0, float(max(0, int(step))) / float(max(1, warmup_steps)))
        return float(encoder.set_runtime_residual_scale(scale))

    def _strict_mixed_runtime_enabled(self) -> bool:
        flags = getattr(self, "_mixed_phase_flags", {}) or {}
        phase_name = str(flags.get("phase_name", "phase0") or "phase0").strip().lower()
        return phase_name not in {"", "phase0"}

    def _restore_resume_checkpoint_if_needed(self) -> None:
        ckpt = getattr(self, "ckpt", None)
        ckpt_path = str(getattr(getattr(self, "cfg", None), "resume_ckpt_path", "") or "").strip()
        if ckpt is None or not ckpt_path:
            return
        reset_optimizer = bool(getattr(getattr(self, "cfg", None), "resume_reset_optimizer", False))
        if reset_optimizer:
            model_ckpt = tf.train.Checkpoint(
                encoder=self.model.encoder,
                field=self.model.field,
            )
            status = model_ckpt.restore(ckpt_path)
            print("[trainer] Resume optimizer reset: restored model/field/encoder variables only.")
        else:
            status = ckpt.restore(ckpt_path)
        try:
            status.expect_partial()
        except Exception:
            pass
        self._resumed_ckpt_path = ckpt_path
        cfg_lr = getattr(getattr(self, "cfg", None), "lr", None)
        if cfg_lr is not None and hasattr(self, "_set_optimizer_learning_rate"):
            old_lr = float("nan")
            if hasattr(self, "_get_optimizer_learning_rate"):
                old_lr = self._get_optimizer_learning_rate()
            applied_lr = self._set_optimizer_learning_rate(float(cfg_lr))
            if np.isfinite(applied_lr):
                if np.isfinite(old_lr):
                    print(f"[trainer] Resume learning rate override: {old_lr:.4e} -> {applied_lr:.4e}")
                else:
                    print(f"[trainer] Resume learning rate override: {applied_lr:.4e}")

    def _resolve_resume_start_step(self) -> int:
        cfg = getattr(self, "cfg", None)
        max_steps = max(0, int(getattr(cfg, "max_steps", 0) or 0))
        resume_start_step = max(0, int(getattr(cfg, "resume_start_step", 0) or 0))
        if resume_start_step <= 0:
            return 0
        ckpt_path = str(getattr(cfg, "resume_ckpt_path", "") or "").strip()
        if not ckpt_path:
            raise ValueError("resume_start_step requires resume_ckpt_path; refusing to skip training steps without a checkpoint.")
        return min(resume_start_step, max_steps)

    def _resolve_training_step_range(self) -> range:
        max_steps = max(0, int(getattr(getattr(self, "cfg", None), "max_steps", 0) or 0))
        resume_start_step = self._resolve_resume_start_step()
        return range(resume_start_step + 1, max_steps + 1)

    def _save_final_checkpoint(self, step: int) -> Optional[str]:
        ckpt_path = self._save_checkpoint_best_effort(step)
        if not ckpt_path:
            return None
        self._final_ckpt_path = str(ckpt_path)
        return str(ckpt_path)

    def _snapshot_scalar(self, value):
        if value is None:
            return None
        try:
            if hasattr(value, "numpy"):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                if value.shape != ():
                    return None
                value = value.item()
            scalar = float(value)
        except Exception:
            return None
        if not np.isfinite(scalar):
            return None
        return scalar

    def _snapshot_text(self, value):
        if value is None:
            return None
        try:
            if hasattr(value, "numpy"):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                if value.shape != ():
                    return None
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    def _benchmark_metric_mean(self, key: str):
        sums = getattr(self, "_benchmark_metric_sums", None)
        counts = getattr(self, "_benchmark_metric_counts", None)
        if not isinstance(sums, Mapping) or not isinstance(counts, Mapping):
            return None
        total = self._snapshot_scalar(sums.get(key))
        count = self._snapshot_scalar(counts.get(key))
        if total is None or count is None or count <= 0.0:
            return None
        return total / count

    def _record_benchmark_step_metrics(self, Pi, parts, stats) -> None:
        sums = dict(getattr(self, "_benchmark_metric_sums", {}) or {})
        counts = dict(getattr(self, "_benchmark_metric_counts", {}) or {})
        parts = parts if isinstance(parts, Mapping) else {}
        stats = stats if isinstance(stats, Mapping) else {}

        tracked_metrics = {
            "Pi": self._snapshot_scalar(Pi),
            "E_data": self._snapshot_scalar(parts.get("E_data")),
            "ft_residual_norm": self._snapshot_scalar(stats.get("ft_residual_norm")),
            "inner_convergence_rate": self._snapshot_scalar(stats.get("inner_convergence_rate")),
            "inner_fallback_rate": self._snapshot_scalar(stats.get("inner_fallback_rate")),
        }
        for key, value in tracked_metrics.items():
            if value is None:
                continue
            sums[key] = float(sums.get(key, 0.0)) + float(value)
            counts[key] = float(counts.get(key, 0.0)) + 1.0

        self._benchmark_metric_sums = sums
        self._benchmark_metric_counts = counts
        self._last_train_parts_snapshot = {
            "E_data": tracked_metrics["E_data"],
        }
        self._last_train_stats_snapshot = {
            "strict_route_mode": self._snapshot_text(stats.get("strict_route_mode")),
            "normal_ift_ready": self._snapshot_scalar(stats.get("normal_ift_ready")),
            "normal_ift_consumed": self._snapshot_scalar(stats.get("normal_ift_consumed")),
            "strict_effective_traction_scale": self._snapshot_scalar(
                stats.get("strict_effective_traction_scale")
            ),
            "coupling_phase_traction_scale": self._snapshot_scalar(
                stats.get("coupling_phase_traction_scale")
            ),
            "coupling_refinement_steps": self._snapshot_scalar(
                stats.get("coupling_refinement_steps")
            ),
            "coupling_tail_qn_budget": self._snapshot_scalar(
                stats.get("coupling_tail_qn_budget")
            ),
            "ft_residual_norm": tracked_metrics["ft_residual_norm"],
            "inner_convergence_rate": tracked_metrics["inner_convergence_rate"],
            "inner_fallback_rate": tracked_metrics["inner_fallback_rate"],
        }

    def get_compact_final_metrics_snapshot(self) -> Dict[str, object]:
        """Return a compact, JSON-friendly snapshot of the final trainer state."""

        latest_val_summary = getattr(self, "_latest_val_summary", None)
        if not isinstance(latest_val_summary, Mapping):
            latest_val_summary = {}
        latest_train_stats = getattr(self, "_last_train_stats_snapshot", None)
        if not isinstance(latest_train_stats, Mapping):
            latest_train_stats = {}

        best_metric = self._snapshot_scalar(getattr(self, "best_metric", None))
        best_ckpt_path = str(getattr(self, "_best_ckpt_path", "") or "").strip() or None
        final_ckpt_path = str(getattr(self, "_final_ckpt_path", "") or "").strip() or None

        snapshot = {
            "final_step": self._snapshot_scalar(getattr(self, "_last_completed_step", getattr(self, "_last_train_step", None))),
            "pi": self._snapshot_scalar(getattr(self, "_last_train_pi", None)),
            "grad_norm": self._snapshot_scalar(getattr(self, "_last_train_grad_norm", None)),
            "route_score": self._snapshot_scalar(getattr(self, "_last_route_score", None)),
            "best_metric": best_metric,
            "best_ckpt_path": best_ckpt_path,
            "final_ckpt_path": final_ckpt_path,
            "route_mode": self._snapshot_text(latest_train_stats.get("strict_route_mode")),
            "normal_ift_ready": self._snapshot_scalar(latest_train_stats.get("normal_ift_ready")),
            "normal_ift_consumed": self._snapshot_scalar(latest_train_stats.get("normal_ift_consumed")),
            "strict_effective_traction_scale": self._snapshot_scalar(
                latest_train_stats.get("strict_effective_traction_scale")
            ),
            "coupling_phase_traction_scale": self._snapshot_scalar(
                latest_train_stats.get("coupling_phase_traction_scale")
            ),
            "coupling_refinement_steps": self._snapshot_scalar(
                latest_train_stats.get("coupling_refinement_steps")
            ),
            "coupling_tail_qn_budget": self._snapshot_scalar(
                latest_train_stats.get("coupling_tail_qn_budget")
            ),
            "mean_Pi": self._benchmark_metric_mean("Pi"),
            "mean_E_data": self._benchmark_metric_mean("E_data"),
            "mean_ft_residual_norm": self._benchmark_metric_mean("ft_residual_norm"),
            "mean_inner_convergence_rate": self._benchmark_metric_mean("inner_convergence_rate"),
            "mean_inner_fallback_rate": self._benchmark_metric_mean("inner_fallback_rate"),
            "val_step": self._snapshot_scalar(getattr(self, "_latest_val_step", None)),
            "val_drrms_mean": self._snapshot_scalar(latest_val_summary.get("val_drrms_mean")),
            "val_ratio_median": self._snapshot_scalar(latest_val_summary.get("val_ratio_median")),
        }
        return snapshot

    def _configure_volume_sampling_for_step(self) -> str:
        """Apply per-step elasticity subsampling when configured."""

        if self.elasticity is None or not hasattr(self.elasticity, "set_sample_indices"):
            return ""

        target_raw = getattr(getattr(self.cfg, "elas_cfg", None), "n_points_per_step", 0)
        try:
            target = int(target_raw) if target_raw is not None else 0
        except Exception:
            target = 0

        n_cells = int(getattr(self.elasticity, "n_cells", 0) or 0)
        if target <= 0 or n_cells <= 0 or target >= n_cells:
            self.elasticity.set_sample_indices(None)
            return ""

        indices = np.random.choice(n_cells, size=target, replace=False).astype(np.int64)
        self.elasticity.set_sample_indices(indices)
        return f"vol={target}/{n_cells}"

    def _validation_eval_every(self) -> int:
        raw = int(getattr(self.cfg, "validation_eval_every", 0) or 0)
        if raw > 0:
            return raw
        return max(1, int(getattr(self.cfg, "log_every", 1) or 1))

    def _should_run_validation_eval(self, step: int) -> bool:
        dataset = getattr(self, "_supervision_dataset", None)
        sup_cfg = getattr(self.cfg, "supervision", None)
        if dataset is None or sup_cfg is None:
            return False
        if not bool(getattr(sup_cfg, "enabled", False)):
            return False
        every = self._validation_eval_every()
        return every > 0 and step >= every and (step % every) == 0

    def _resolve_best_metric_value(
        self,
        pi_val: float,
        parts: Mapping[str, object],
        val_summary: Optional[Mapping[str, object]] = None,
    ) -> Optional[float]:
        metric_name = str(getattr(self.cfg, "save_best_on", "Pi") or "Pi").strip().lower()
        if metric_name == "pi":
            return float(pi_val)
        if metric_name == "val_drrms":
            if not isinstance(val_summary, Mapping):
                return None
            value = val_summary.get("val_drrms_mean")
            return None if value is None else float(value)
        if metric_name == "val_ratio":
            if not isinstance(val_summary, Mapping):
                return None
            value = val_summary.get("val_ratio_median")
            return None if value is None else float(value)

        value = parts.get("E_int")
        if value is None:
            return None
        try:
            if hasattr(value, "numpy"):
                return float(value.numpy())
            return float(value)
        except Exception:
            return None

    def _maybe_save_best_checkpoint(
        self,
        step: int,
        pi_val: float,
        parts: Mapping[str, object],
        val_summary: Optional[Mapping[str, object]] = None,
    ) -> str:
        metric_val = self._resolve_best_metric_value(pi_val, parts, val_summary)
        if metric_val is None or not np.isfinite(metric_val):
            return ""
        if metric_val >= self.best_metric:
            return ""

        ckpt_path = self._save_checkpoint_best_effort(step)
        if not ckpt_path:
            return "checkpoint 淇濆瓨澶辫触(宸茶烦杩?)"
        self.best_metric = metric_val
        self._best_ckpt_path = str(ckpt_path)
        return f"宸蹭繚瀛?{os.path.basename(ckpt_path)}"

    def _maybe_apply_val_plateau_lr_decay(
        self,
        step: int,
        val_summary: Optional[Mapping[str, object]],
    ) -> Optional[str]:
        if not bool(getattr(self.cfg, "val_plateau_lr_decay_enabled", False)):
            return None
        if step < max(0, int(getattr(self.cfg, "val_plateau_lr_decay_warmup", 0) or 0)):
            return None
        if not isinstance(val_summary, Mapping):
            return None

        metric_key = str(
            getattr(self.cfg, "val_plateau_lr_decay_metric", "val_drrms") or "val_drrms"
        ).strip().lower()
        metric_name = "val_ratio_median" if metric_key == "val_ratio" else "val_drrms_mean"
        metric_val_raw = val_summary.get(metric_name)
        if metric_val_raw is None:
            return None
        metric_val = float(metric_val_raw)
        if not np.isfinite(metric_val):
            return None

        if self._val_plateau_best is None or metric_val < float(self._val_plateau_best):
            self._val_plateau_best = metric_val
            self._val_plateau_bad_count = 0
            return None

        patience = max(1, int(getattr(self.cfg, "val_plateau_lr_decay_patience", 0) or 1))
        self._val_plateau_bad_count += 1
        if self._val_plateau_bad_count < patience:
            return None

        factor = float(getattr(self.cfg, "val_plateau_lr_decay_factor", 0.5) or 0.5)
        min_lr = float(getattr(self.cfg, "val_plateau_lr_decay_min_lr", 1.0e-6) or 1.0e-6)
        old_lr = self._get_optimizer_learning_rate()
        if not np.isfinite(old_lr):
            self._val_plateau_bad_count = 0
            return None
        new_lr = max(min_lr, old_lr * factor)
        applied_lr = self._set_optimizer_learning_rate(new_lr)
        self._val_plateau_bad_count = 0
        return f"lr_decay={old_lr:.4e}->{applied_lr:.4e}"

    # ----------------- 训练 -----------------
    def run(self):
        self.build()
        self._restore_resume_checkpoint_if_needed()
        print(f"[trainer] 当前训练设备：{self.device_summary}")
        total = self._assemble_total()
        self.bcs_ops = attach_bcs_from_asm(
            total=total,
            asm=self.asm,
            cfg=self.cfg,
        )
        if self.bcs_ops:
            print(f"[bc] 已挂载 {len(self.bcs_ops)} 组边界约束")
        else:
            print("[bc] 未发现边界约束，跳过挂载")
        self._total_ref = total

        # ---- 初始化自适应损失权重状态 ----
        # 以 TotalConfig 里的 w_int / w_cn / ... 作为基准权重
        base_weights = {
            "E_int": self.cfg.total_cfg.w_int,
            "E_cn": self.cfg.total_cfg.w_cn,
            "E_ct": self.cfg.total_cfg.w_ct,
            "E_bc": self.cfg.total_cfg.w_bc,
            "E_tight": self.cfg.total_cfg.w_tight,
            "E_sigma": self.cfg.total_cfg.w_sigma,
            "E_eq": getattr(self.cfg.total_cfg, "w_eq", 0.0),
            "E_reg": getattr(self.cfg.total_cfg, "w_reg", 0.0),
            "E_bi": getattr(self.cfg.total_cfg, "w_bi", 0.0),
            "E_ed": getattr(self.cfg.total_cfg, "w_ed", 0.0),
            "E_unc": getattr(self.cfg, "uncertainty_loss_weight", 0.0),
            "E_data": getattr(self.cfg.total_cfg, "w_data", 0.0),
            "E_delta_data": getattr(self.cfg.total_cfg, "w_delta_data", 0.0),
            "E_optical_modal": getattr(self.cfg.total_cfg, "w_optical_modal", 0.0),
            "E_smooth": getattr(self.cfg.total_cfg, "w_smooth", 0.0),
            "path_penalty_total": getattr(self.cfg.total_cfg, "path_penalty_weight", 0.0),
            "fric_path_penalty_total": getattr(self.cfg.total_cfg, "fric_path_penalty_weight", 0.0),
            "R_fric_comp": 0.0,
            "R_contact_comp": 0.0,
        }
        self._base_weights = base_weights
        self._loss_keys = list(base_weights.keys())

        adaptive_enabled = bool(getattr(self.cfg, "loss_adaptive_enabled", False))
        sign_overrides = {}
        if adaptive_enabled:
            scheme = getattr(self.cfg.total_cfg, "adaptive_scheme", "contact_only")
            focus_terms = getattr(self.cfg, "loss_focus_terms", tuple())
            self.loss_state = LossWeightState.from_config(
                base_weights=base_weights,
                adaptive_scheme=scheme,
                ema_decay=getattr(self.cfg, "loss_ema_decay", 0.95),
                min_factor=getattr(self.cfg, "loss_min_factor", 0.25),
                max_factor=getattr(self.cfg, "loss_max_factor", 4.0),
                min_weight=getattr(self.cfg, "loss_min_weight", None),
                max_weight=getattr(self.cfg, "loss_max_weight", None),
                gamma=getattr(self.cfg, "loss_gamma", 2.0),
                focus_terms=focus_terms,
                update_every=getattr(self.cfg, "loss_update_every", 1),
                sign_overrides=sign_overrides,
            )
        else:
            self.loss_state = None
        self._refresh_static_weight_vector()
        self._benchmark_metric_sums = {}
        self._benchmark_metric_counts = {}
        self._last_train_parts_snapshot = {}
        self._last_train_stats_snapshot = {}
        resume_start_step = self._resolve_resume_start_step()
        residual_warmup_steps = max(0, int(getattr(self.cfg, "ase_residual_warmup_steps", 0) or 0))
        if residual_warmup_steps > 0:
            self._set_condition_residual_runtime_scale(resume_start_step)
            print(f"[trainer] ASE residual warmup steps = {residual_warmup_steps}")
        if resume_start_step > 0:
            print(
                f"[trainer] Resume step control: restored '{getattr(self, '_resumed_ckpt_path', None)}', "
                f"continuing from step {resume_start_step + 1} / {self.cfg.max_steps}."
            )
        step_range = self._resolve_training_step_range()
        train_desc = "训练"
        train_pb_kwargs = dict(
            total=max(0, int(self.cfg.max_steps) - resume_start_step),
            desc=train_desc,
            leave=True,
            disable=not (self._tqdm_enabled and self.cfg.train_bar_enabled),
        )
        step_detail_enabled = self._step_detail_enabled()
        last_step = resume_start_step
        stop_reason = None
        if self.cfg.train_bar_color:
            train_pb_kwargs["colour"] = self.cfg.train_bar_color
        with tqdm(**train_pb_kwargs) as p_train:
            for step in step_range:
                stop_this_step = False
                # 子进度条：本 step 的 4 个动作
                step_pb_kwargs = dict(
                    total=4,
                    leave=False,
                    disable=not step_detail_enabled,
                )
                if self.cfg.step_bar_color:
                    step_pb_kwargs["colour"] = self.cfg.step_bar_color
                with tqdm(**step_pb_kwargs) as p_step:
                    # 1) 接触重采样
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 接触重采样")
                    t0 = time.perf_counter()
                    contact_note = "跳过"
                    if self.contact is None:
                        contact_note = "跳过 (无接触体)"
                    else:
                        contact_note = "跳过 (路线锁定: 沿用构建采样)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("resample", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{contact_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                    # 2) 前向 + 反传（随机采样三螺栓预紧力）
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 前向/反传")
                    t0 = time.perf_counter()
                    preload_case = self._sample_preload_case()
                    # 动态提升接触惩罚/ALM 参数（软→硬）
                    self._maybe_update_contact_hardening(step)
                    self._set_condition_residual_runtime_scale(step)
                    vol_note = self._configure_volume_sampling_for_step()
                    self._push_contact_route_hint()
                    Pi, parts, stats, grad_norm = self._train_step(total, preload_case, step=step)
                    P_np = preload_case["P"]
                    order_np = preload_case.get("order")
                    self._last_preload_case = copy.deepcopy(preload_case)
                    self._last_train_step = step
                    self._last_train_pi = Pi
                    self._last_train_grad_norm = grad_norm
                    if self._should_update_contact_route(step):
                        route_score = self._update_contact_route_metric(parts)
                    else:
                        route_score = self._contact_route_score()
                    self._last_route_score = route_score
                    self._record_benchmark_step_metrics(Pi, parts, stats)

                    should_collect_scalars = self._should_collect_step_scalars(step)
                    pi_val = float("nan")
                    grad_val = float("nan")
                    rel_pi = None
                    rel_delta = None
                    if should_collect_scalars:
                        pi_val = float(Pi.numpy())
                        if self._pi_baseline is None:
                            self._pi_baseline = pi_val if pi_val != 0.0 else 1.0
                        if self._pi_ema is None:
                            self._pi_ema = pi_val
                        else:
                            ema_alpha = 0.1
                            self._pi_ema = (1 - ema_alpha) * self._pi_ema + ema_alpha * pi_val
                        rel_pi = pi_val / (self._pi_baseline or pi_val or 1.0)
                        if self._prev_pi is not None and self._prev_pi != 0.0:
                            rel_delta = (self._prev_pi - pi_val) / abs(self._prev_pi)
                        self._prev_pi = pi_val
                        grad_val = float(grad_norm.numpy()) if hasattr(grad_norm, "numpy") else float(grad_norm)
                    elif self._prev_pi is not None:
                        pi_val = float(self._prev_pi)
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("train", elapsed))
                    device = self._short_device_name(getattr(Pi, "device", None))
                    if step_detail_enabled:
                        rel_pct = rel_pi * 100.0 if rel_pi is not None else None
                        rel_txt = (
                            f"Πrel={rel_pct:.2f}%" if rel_pct is not None else "Πrel=--"
                        )
                        d_txt = (
                            f"ΔΠ={rel_delta * 100:+.1f}%"
                            if rel_delta is not None
                            else "ΔΠ=--"
                        )
                        ema_txt = f"Πema={self._pi_ema:.2e}" if self._pi_ema is not None else "Πema=--"
                        order_txt = ""
                        if order_np is not None:
                            order_txt = " order=" + "-".join(str(int(x) + 1) for x in order_np)
                        energy_summary = self._format_energy_summary_if_needed(parts)
                        energy_txt = f" | {energy_summary}" if energy_summary else ""
                        if vol_note:
                            energy_txt += f" | {vol_note}"
                        train_note = (
                            f"P=[{int(P_np[0])},{int(P_np[1])},{int(P_np[2])}]"
                            f"{order_txt}{energy_txt} | Π={pi_val:.2e} {rel_txt} {d_txt} "
                            f"grad={grad_val:.2e} {ema_txt} route={route_score:.2f}"
                        )
                        if step == 1:
                            train_note += " | 首轮包含图追踪/缓存构建"
                        self._set_pbar_postfix(
                            p_step,
                            f"{train_note} | {self._format_seconds(elapsed)} | dev={device}"
                        )
                    p_step.update(1)

                    stop_reason = None
                    if self._should_check_early_exit(step):
                        stop_reason = self._check_early_exit(step, pi_val, grad_val)
                    if stop_reason:
                        stop_this_step = True
                        print(
                            f"[trainer] Early exit at step {step}: {stop_reason}",
                            flush=True,
                        )
                        if step_detail_enabled:
                            self._set_pbar_postfix(
                                p_step,
                                f"触发 early-exit | {self._format_seconds(elapsed)}"
                            )

                    # 3) ALM 更新
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: ALM 更新")
                    t0 = time.perf_counter()
                    alm_note = "跳过"
                    if stop_this_step:
                        alm_note = "跳过 (early-exit)"
                    else:
                        alm_note = "跳过 (路线锁定: 阶段内更新)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("alm", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{alm_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                    # 4) 日志/检查点
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 日志/检查点")
                    t0 = time.perf_counter()
                    log_note = "跳过"
                    if stop_this_step:
                        log_note = "跳过 (early-exit)"
                    elif self.cfg.log_every <= 0:
                        log_note = "跳过 (已禁用)"
                    else:
                        should_log = step == 1 or step % self.cfg.log_every == 0
                        if should_log:
                            val_summary = self._latest_val_summary
                            if self._should_run_validation_eval(step):
                                val_summary = self._compute_validation_supervision_summary()
                                self._latest_val_summary = val_summary
                                self._latest_val_step = step
                            lr_note = self._maybe_apply_val_plateau_lr_decay(step, val_summary)
                            postfix, log_note = self._format_train_log_postfix(
                                P_np,
                                Pi,
                                parts,
                                stats,
                                grad_val,
                                rel_pi,
                                rel_delta,
                                order_np,
                                val_summary=val_summary,
                            )
                            if postfix:
                                p_train.set_postfix_str(postfix)
                                # 额外打印到终端（确保不被进度条覆盖）
                                print(f"\n[Step {step}] {postfix}", flush=True)
                            best_note = self._maybe_save_best_checkpoint(
                                step=step,
                                pi_val=pi_val,
                                parts=parts,
                                val_summary=val_summary,
                            )
                            if best_note:
                                if getattr(self, "ckpt_manager", None) is not None:
                                    self._best_ckpt_path = getattr(self.ckpt_manager, "latest_checkpoint", None)
                                log_note += f" | {best_note}"
                            if lr_note:
                                log_note += f" | {lr_note}"

                    if (
                        not stop_this_step
                        and self.cfg.log_every > 0
                        and not (step == 1 or step % self.cfg.log_every == 0)
                    ):
                        remaining = self.cfg.log_every - (step % self.cfg.log_every)
                        log_note = f"跳过 (距下次还有 {remaining} 步)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("log", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{log_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                p_train.update(1)
                last_step = step

                if step % max(1, self.cfg.log_every) == 0:
                    total_spent = sum(t for _, t in self._step_stage_times)
                    if total_spent > 0:
                        label_map = {
                            "resample": "采样",
                            "train": "前向/反传",
                            "alm": "ALM",
                            "log": "日志",
                        }
                        stage_totals: Dict[str, float] = {}
                        for name, t in self._step_stage_times:
                            stage_totals[name] = stage_totals.get(name, 0.0) + float(t)
                        n_steps = max(1, int(round(len(self._step_stage_times) / 4.0)))
                        avg_step = total_spent / n_steps
                        ordered = ["resample", "train", "alm", "log"]
                        parts_txt = ", ".join(
                            f"{label_map.get(name, name)}:{stage_totals.get(name, 0.0) / total_spent * 100:.0f}%"
                            for name in ordered
                        )
                        summary_note = (
                            f"step{step}平均耗时 {self._format_seconds(avg_step)} ({parts_txt})"
                        )
                        if step == 1:
                            summary_note += " | 首轮额外包括图追踪/初次缓存"
                        self._set_pbar_postfix(p_train, summary_note)
                    self._step_stage_times.clear()

                if stop_this_step:
                    break

        # 训练结束：再存一次
        if self.ckpt_manager is not None:
            final_step = last_step if last_step > 0 else self.cfg.max_steps
            final_ckpt = self._save_final_checkpoint(final_step)
            if final_ckpt:
                print(f"[trainer] 训练结束已保存 checkpoint -> {final_ckpt}")
            else:
                print("[trainer] WARNING: 训练结束 checkpoint 保存失败(已跳过)")

        self._last_completed_step = last_step if last_step > 0 else None
        self._final_train_metrics = self.get_compact_final_metrics_snapshot()
        self._visualize_after_training(n_samples=self.cfg.viz_samples_after_train)
