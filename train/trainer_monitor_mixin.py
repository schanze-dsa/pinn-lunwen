# -*- coding: utf-8 -*-
"""Monitoring/logging utility mixin extracted from Trainer."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf


class TrainerMonitorMixin:
    def _optimizer_with_learning_rate(self):
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return None
        for candidate in (opt, getattr(opt, "inner_optimizer", None), getattr(opt, "optimizer", None)):
            if candidate is not None and hasattr(candidate, "learning_rate"):
                return candidate
        return None

    def _get_optimizer_learning_rate(self) -> float:
        opt = self._optimizer_with_learning_rate()
        if opt is None:
            return float("nan")
        lr = getattr(opt, "learning_rate", None)
        if lr is None:
            return float("nan")
        try:
            if hasattr(lr, "numpy"):
                return float(lr.numpy())
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            try:
                return float(lr)
            except Exception:
                return float("nan")

    def _set_optimizer_learning_rate(self, value: float) -> float:
        opt = self._optimizer_with_learning_rate()
        if opt is None:
            return float("nan")
        lr = getattr(opt, "learning_rate", None)
        if lr is None:
            return float("nan")
        lr_value = float(value)
        try:
            if hasattr(lr, "assign"):
                lr.assign(lr_value)
            else:
                setattr(opt, "learning_rate", lr_value)
        except Exception:
            setattr(opt, "learning_rate", lr_value)
        return self._get_optimizer_learning_rate()

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.0f}µs"
        if seconds < 1:
            return f"{seconds * 1e3:.1f}ms"
        return f"{seconds:.2f}s"

    @staticmethod
    def _short_device_name(device: Optional[str]) -> str:
        if not device:
            return "?"
        if "/device:" in device:
            return device.split("/device:")[-1]
        if device.startswith("/"):
            return device.split(":")[-1]
        return device

    def _step_detail_enabled(self) -> bool:
        return bool(self._tqdm_enabled and getattr(self.cfg, "step_bar_enabled", False))

    def _route_update_every(self) -> int:
        return max(1, int(getattr(self.cfg, "contact_route_update_every", 1) or 1))

    def _should_update_contact_route(self, step: int) -> bool:
        if step <= 1:
            return True
        return (step % self._route_update_every()) == 0

    def _early_exit_check_every(self) -> int:
        return max(1, int(getattr(self.cfg, "early_exit_check_every", 1) or 1))

    def _should_check_early_exit(self, step: int) -> bool:
        if not bool(getattr(self.cfg, "early_exit_enabled", False)):
            return False
        if step <= 1:
            return True
        return (step % self._early_exit_check_every()) == 0

    def _should_collect_step_scalars(self, step: int) -> bool:
        # Any detailed step bar needs full scalar diagnostics every step.
        if self._step_detail_enabled():
            return True
        log_every = int(getattr(self.cfg, "log_every", 0) or 0)
        if log_every > 0 and (step == 1 or (step % log_every) == 0):
            return True
        if self._should_check_early_exit(step):
            return True
        return False

    @staticmethod
    def extract_bilevel_diagnostics(stats: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Pick strict-bilevel diagnostics from stats for monitoring/logging."""

        out: Dict[str, Any] = {}
        if not isinstance(stats, Mapping):
            return out
        for key in (
            "inner_fn_norm",
            "inner_ft_norm",
            "inner_cone_violation",
            "inner_max_penetration",
            "inner_fb_residual_norm",
            "inner_normal_step_norm",
            "inner_tangential_step_norm",
            "inner_fallback_used",
            "inner_converged",
            "inner_skip_batch",
            "inner_convergence_rate",
            "inner_fallback_rate",
            "inner_skip_rate",
            "continuation_frozen",
            "continuation_freeze_events",
            "ift_linear_residual",
            "normal_ift_ready",
            "normal_ift_consumed",
            "normal_ift_condition_metric",
            "normal_ift_inputs_present",
            "normal_ift_core_valid_ratio",
            "ft_residual_norm",
            "signature_gate_applied",
            "fallback_reason_code",
            "tangential_step_mode",
            "effective_alpha_scale",
            "tail_has_effective_step",
            "fallback_trigger_reason",
            "risk_guard_bucket_raw",
            "risk_guard_bucket",
            "strict_step_class",
            "risk_guard_applied",
            "risk_guard_scale",
            "strict_step_update_scale",
            "total_step_update_scale",
            "protect_prefix_active",
            "protect_prefix_reason",
            "first_c_event_seen_before_step",
            "first_c_event_seen_after_step",
            "delayed_guard_ready",
            "grad_u_norm",
            "grad_sigma_norm",
            "strict_phase_hold",
            "strict_continuation_backoff",
            "continuation_backoff_applied",
            "strict_force_detach",
            "strict_traction_scale",
            "phase_hold_reason",
            "inner_solver_not_stable_count",
        ):
            if key not in stats:
                continue
            value = stats.get(key)
            try:
                if isinstance(value, tf.Tensor):
                    if value.dtype == tf.string:
                        out[key] = value.numpy().decode("utf-8")
                    else:
                        out[key] = float(tf.cast(value, tf.float32).numpy())
                else:
                    out[key] = str(value) if key in {
                        "phase_hold_reason",
                        "fallback_trigger_reason",
                        "tangential_step_mode",
                        "risk_guard_bucket_raw",
                        "risk_guard_bucket",
                        "strict_step_class",
                        "protect_prefix_reason",
                    } else float(value)
            except Exception:
                continue
        return out

    def _contact_route_score(self) -> float:
        if self._contact_route_ema is None:
            return 0.0
        if self._contact_route_ref is None or self._contact_route_ref <= 0.0:
            return float(self._contact_route_ema)
        return float(self._contact_route_ema) / float(self._contact_route_ref)

    def _update_contact_route_metric(self, parts: Mapping[str, Any]) -> float:
        raw = None
        for key in ("R_contact_comp", "E_cn"):
            if key not in parts:
                continue
            val = parts.get(key)
            try:
                if isinstance(val, tf.Tensor):
                    if val.shape.rank == 0:
                        raw = float(tf.cast(val, tf.float32).numpy())
                        break
                else:
                    raw = float(val)
                    break
            except Exception:
                continue

        if raw is None or not np.isfinite(raw):
            return self._contact_route_score()

        raw = abs(float(raw))
        if self._contact_route_ema is None:
            self._contact_route_ema = raw
        else:
            ema_decay = 0.9
            self._contact_route_ema = (
                ema_decay * self._contact_route_ema + (1.0 - ema_decay) * raw
            )

        if self._contact_route_ref is None:
            self._contact_route_ref = max(self._contact_route_ema, 1.0e-6)
        else:
            ref_decay = 0.99
            self._contact_route_ref = (
                ref_decay * self._contact_route_ref
                + (1.0 - ref_decay) * max(self._contact_route_ema, 1.0e-6)
            )

        return self._contact_route_score()

    def _push_contact_route_hint(self) -> None:
        if self.model is None or not hasattr(self.model, "field"):
            return
        field = getattr(self.model, "field", None)
        if field is None:
            return
        route_src = str(
            getattr(getattr(field, "cfg", None), "adaptive_depth_route_source", "")
            or ""
        ).strip().lower()
        if route_src != "contact_residual":
            return
        if hasattr(field, "set_contact_residual_hint"):
            field.set_contact_residual_hint(self._contact_route_score())

    def _check_early_exit(self, step: int, pi_val: float, grad_val: float) -> Optional[str]:
        if not bool(getattr(self.cfg, "early_exit_enabled", False)):
            return None

        nonfinite_patience = max(
            1, int(getattr(self.cfg, "early_exit_nonfinite_patience", 1) or 1)
        )
        div_patience = max(
            1, int(getattr(self.cfg, "early_exit_divergence_patience", 1) or 1)
        )
        warmup_steps = max(0, int(getattr(self.cfg, "early_exit_warmup_steps", 0) or 0))
        grad_thr = float(
            getattr(self.cfg, "early_exit_grad_norm_threshold", 0.0) or 0.0
        )
        ema_rel_inc = max(
            0.0, float(getattr(self.cfg, "early_exit_pi_ema_rel_increase", 0.0) or 0.0)
        )

        finite_pi = bool(np.isfinite(pi_val))
        finite_grad = bool(np.isfinite(grad_val))
        if not (finite_pi and finite_grad):
            self._nonfinite_streak += 1
        else:
            self._nonfinite_streak = 0

        if self._nonfinite_streak >= nonfinite_patience:
            return (
                f"non-finite detected for {self._nonfinite_streak} consecutive steps "
                f"(patience={nonfinite_patience})"
            )

        pi_ema = float(self._pi_ema) if self._pi_ema is not None else None
        if pi_ema is not None and np.isfinite(pi_ema):
            if self._best_pi_ema is None or pi_ema < self._best_pi_ema:
                self._best_pi_ema = pi_ema

        if step <= warmup_steps:
            self._diverge_streak = 0
            return None

        grad_high = grad_thr > 0.0 and finite_grad and float(grad_val) >= grad_thr
        ema_worse = False
        if (
            pi_ema is not None
            and np.isfinite(pi_ema)
            and self._best_pi_ema is not None
            and np.isfinite(self._best_pi_ema)
        ):
            baseline = max(abs(float(self._best_pi_ema)), 1.0e-12)
            ema_worse = (pi_ema - float(self._best_pi_ema)) / baseline >= ema_rel_inc

        if grad_high and ema_worse:
            self._diverge_streak += 1
        else:
            self._diverge_streak = 0

        if self._diverge_streak >= div_patience:
            return (
                f"divergence detected: grad_norm={grad_val:.3e} >= {grad_thr:.3e} and "
                f"Pi_ema has worsened for {self._diverge_streak} consecutive steps "
                f"(patience={div_patience})"
            )

        return None

    def _format_energy_summary_if_needed(self, parts: Mapping[str, tf.Tensor]) -> str:
        if not self._step_detail_enabled():
            return ""
        return self._format_energy_summary(parts)

    def _resolve_tqdm_enabled(self) -> bool:
        if getattr(self.cfg, "tqdm_disable", False):
            return False
        if getattr(self.cfg, "tqdm_disable_if_not_tty", True):
            try:
                if not sys.stderr.isatty():
                    return False
            except Exception:
                return False
        return True

    def _graph_cache_path(self, n_nodes: int) -> str:
        base = self.cfg.graph_cache_dir or os.path.join(self.cfg.out_dir or "outputs", "graph_cache")
        os.makedirs(base, exist_ok=True)
        if self.cfg.graph_cache_name:
            return os.path.join(base, self.cfg.graph_cache_name)
        mesh_tag = os.path.splitext(os.path.basename(self.cfg.inp_path))[0]
        k = int(getattr(self.cfg.model_cfg.field, "graph_k", 0) or 0)
        return os.path.join(base, f"knn_{mesh_tag}_n{n_nodes}_k{k}.npz")

    def _loss_weight_lookup(self) -> Dict[str, float]:
        """Assemble the latest per-term loss weights for logging."""

        weights = {
            "E_int": getattr(self.cfg.total_cfg, "w_int", 1.0),
            "E_cn": getattr(self.cfg.total_cfg, "w_cn", 1.0),
            "E_ct": getattr(self.cfg.total_cfg, "w_ct", 1.0),
            "E_bc": getattr(self.cfg.total_cfg, "w_bc", 1.0),
            "E_tight": getattr(self.cfg.total_cfg, "w_tight", 1.0),
            "E_sigma": getattr(self.cfg.total_cfg, "w_sigma", 1.0),
            "E_eq": getattr(self.cfg.total_cfg, "w_eq", 0.0),
            "E_reg": getattr(self.cfg.total_cfg, "w_reg", 0.0),
            "E_bi": getattr(self.cfg.total_cfg, "w_bi", 0.0),
            "E_ed": getattr(self.cfg.total_cfg, "w_ed", 0.0),
            "E_data": getattr(self.cfg.total_cfg, "w_data", 0.0),
            "E_smooth": getattr(self.cfg.total_cfg, "w_smooth", 0.0),
            "E_unc": getattr(self.cfg, "uncertainty_loss_weight", 0.0),
        }
        if self.loss_state is not None:
            for key, value in self.loss_state.current.items():
                try:
                    weights[key] = float(value)
                except Exception:
                    weights[key] = value
        overrides = getattr(self, "_active_weight_overrides", {}) or {}
        for key, value in overrides.items():
            try:
                weights[key] = float(value)
            except Exception:
                weights[key] = value
        return weights

    @staticmethod
    def _extract_part_scalar(parts: Mapping[str, tf.Tensor], *keys: str) -> Optional[float]:
        for key in keys:
            if key not in parts:
                continue
            value = parts[key]
            try:
                if isinstance(value, tf.Tensor):
                    return float(value.numpy())
                if isinstance(value, np.ndarray):
                    return float(value)
                return float(value)
            except Exception:
                continue
        return None

    def _format_energy_summary(self, parts: Mapping[str, tf.Tensor]) -> str:
        display = [
            ("E_cn", "Ecn"),
            ("E_ct", "Ect"),
            ("E_bi", "Ebi"),
            ("E_bc", "Ebc"),
            ("E_tight", "Etight"),
            ("E_sigma", "Esig"),
            ("E_eq", "Eeq"),
            ("E_reg", "Ereg"),
            ("E_ed", "Eed"),
            ("E_data", "Edata"),
            ("E_smooth", "Esm"),
            ("E_unc", "Eunc"),
        ]
        aliases = {
            "E_cn": ("E_cn", "E_n"),
            "E_ct": ("E_ct", "E_t"),
        }
        weights = self._loss_weight_lookup()
        entries: List[str] = []
        for key, label in display:
            weight = weights.get(key, 0.0)
            # Skip if weight is effectively zero
            if abs(weight) < 1e-15:
                continue
            val = self._extract_part_scalar(parts, *aliases.get(key, (key,)))
            if val is None:
                continue
            entries.append(f"{label}={val:.6e}(w={weight:.6g})")
        return " ".join(entries)

    def _format_train_log_postfix(
        self,
        P_np: np.ndarray,
        Pi: tf.Tensor,
        parts: Mapping[str, tf.Tensor],
        stats: Optional[Mapping[str, Any]],
        grad_val: float,
        rel_pi: float,
        rel_delta: Optional[float],
        order: Optional[np.ndarray] = None,
        val_summary: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[str], str]:
        """Compose the detailed training log postfix for the outer progress bar.

        Returns a tuple of ``(postfix, note)`` where ``postfix`` is the formatted
        text (or ``None`` when formatting fails) and ``note`` summarises whether
        logging succeeded.
        """

        try:
            angles = [float(x) for x in P_np.tolist()]
            pin = float(Pi.numpy())
            energy_disp = self._format_energy_summary(parts)

            tight_txt = ""
            if isinstance(stats, Mapping):
                tstats = stats.get("tightening")
                if isinstance(tstats, Mapping):
                    rms = tstats.get("rms")
                    if hasattr(rms, "numpy"):
                        rms = rms.numpy()
                    try:
                        vals = [float(x) for x in list(rms)[:3]]
                        if vals:
                            tight_txt = " rms=[" + ",".join(f"{v:.4e}" for v in vals) + "]"
                    except Exception:
                        pass

            pen_ratio = None
            stick_ratio = None
            slip_ratio = None
            mean_gap = None

            def _get_stat_float(*keys: str) -> Optional[float]:
                """
                Extract a scalar from stats. Supports staged keys like 's3_cn_mean_gap'
                by taking the highest stage index found.
                """
                if not isinstance(stats, Mapping):
                    return None

                # 1) direct lookup
                for key in keys:
                    val = stats.get(key)
                    if val is None:
                        continue
                    try:
                        if hasattr(val, "numpy"):
                            return float(val.numpy())
                        return float(val)
                    except Exception:
                        continue

                # 2) staged lookup: pick latest stage sN_key
                best_val = None
                best_stage = -1
                stage_re = re.compile(r"s(\\d+)_")
                for name, val in stats.items():
                    for key in keys:
                        if not name.endswith(key):
                            continue
                        m = stage_re.match(name)
                        stage_idx = int(m.group(1)) if m else 0
                        if stage_idx < best_stage:
                            continue
                        try:
                            if hasattr(val, "numpy"):
                                v = float(val.numpy())
                            else:
                                v = float(val)
                            best_stage = stage_idx
                            best_val = v
                        except Exception:
                            continue
                return best_val

            def _get_stat_text(*keys: str) -> Optional[str]:
                if not isinstance(stats, Mapping):
                    return None
                for key in keys:
                    val = stats.get(key)
                    if val is None:
                        continue
                    try:
                        if isinstance(val, tf.Tensor):
                            if val.dtype == tf.string:
                                return val.numpy().decode("utf-8")
                            if val.shape.rank == 0:
                                return str(val.numpy())
                        return str(val)
                    except Exception:
                        continue
                return None

            pen_ratio = _get_stat_float("n_pen_ratio", "cn_pen_ratio", "pen_ratio")
            stick_ratio = _get_stat_float("t_stick_ratio", "stick_ratio")
            slip_ratio = _get_stat_float("t_slip_ratio", "slip_ratio")
            min_gap = _get_stat_float("n_min_gap", "cn_min_gap", "min_gap")
            mean_gap = _get_stat_float("n_mean_gap", "cn_mean_gap", "mean_gap")

            grad_disp = f"grad={grad_val:.4e}"
            rel_pct = rel_pi * 100.0 if rel_pi is not None else None
            rel_disp = (
                f"Πrel={rel_pct:.3f}%" if rel_pct is not None else "Πrel=--"
            )
            delta_disp = (
                f"ΔΠ={rel_delta * 100:+.2f}%" if rel_delta is not None else "ΔΠ=--"
            )
            pen_disp = (
                f"pen={pen_ratio * 100:.2f}%" if pen_ratio is not None else "pen=--"
            )
            stick_disp = (
                f"stick={stick_ratio * 100:.2f}%" if stick_ratio is not None else "stick=--"
            )
            slip_disp = (
                f"slip={slip_ratio * 100:.2f}%" if slip_ratio is not None else "slip=--"
            )
            gap_p01 = None
            if self.contact is not None:
                try:
                    metrics = self.contact.last_sample_metrics()
                    gap_arr = metrics.get("gap") if isinstance(metrics, dict) else None
                    if gap_arr is not None:
                        g = np.asarray(gap_arr, dtype=np.float64).reshape(-1)
                        g = g[np.isfinite(g)]
                        if g.size > 0:
                            gap_p01 = float(np.quantile(g, 0.01))
                except Exception:
                    pass

            gap_terms: List[str] = []
            if min_gap is not None:
                gap_terms.append(f"gmin={min_gap:.4e}")
            if gap_p01 is not None:
                gap_terms.append(f"g01={gap_p01:.4e}")
            if mean_gap is not None:
                gap_terms.append(f"gmean={mean_gap:.4e}")
            gap_disp = " ".join(gap_terms) if gap_terms else "gmean=--"

            weights = self._loss_weight_lookup()
            eq_rms = _get_stat_float("eq_rms")
            reg_rms = _get_stat_float("reg_rms")
            eq_terms: List[str] = []
            if weights.get("E_eq", 0.0) > 1e-15:
                eq_terms.append(f"eqrms={eq_rms:.4e}" if eq_rms is not None else "eqrms=--")
            if weights.get("E_reg", 0.0) > 1e-15:
                eq_terms.append(f"regrms={reg_rms:.4e}" if reg_rms is not None else "regrms=--")
            eq_disp = " ".join(eq_terms)

            data_rms = _get_stat_float("data_rms")
            data_mae = _get_stat_float("data_mae")
            data_ref_rms = _get_stat_float("data_ref_rms")
            data_rel_rms = _get_stat_float("data_rel_rms")
            data_rel_mae = _get_stat_float("data_rel_mae")
            data_smooth_rel_rms = _get_stat_float("data_smooth_rel_rms")
            data_smooth_rms = _get_stat_float("data_smooth_rms")
            data_eff_w = _get_stat_float("data_eff_w")
            data_floor_active = _get_stat_float("data_floor_active")
            data_terms: List[str] = []
            if data_rms is not None:
                data_terms.append(f"drms={data_rms:.4e}")
            if data_mae is not None:
                data_terms.append(f"dmae={data_mae:.4e}")
            if data_ref_rms is not None:
                data_terms.append(f"dref={data_ref_rms:.4e}")
            if data_rel_rms is not None:
                data_terms.append(f"drrms={data_rel_rms:.4e}")
            if data_rel_mae is not None:
                data_terms.append(f"drmae={data_rel_mae:.4e}")
            if data_smooth_rel_rms is not None:
                data_terms.append(f"smrms={data_smooth_rel_rms:.4e}")
            elif data_smooth_rms is not None:
                data_terms.append(f"smrms={data_smooth_rms:.4e}")
            if data_eff_w is not None:
                data_terms.append(f"dsmw={data_eff_w:.4e}")
            if data_floor_active is not None:
                data_terms.append(f"dsmf={int(data_floor_active > 0.5)}")
            if isinstance(val_summary, Mapping):
                val_drrms = val_summary.get("val_drrms_mean")
                val_ratio = val_summary.get("val_ratio_median")
                if val_drrms is not None:
                    data_terms.append(f"vdr={float(val_drrms):.4e}")
                if val_ratio is not None:
                    data_terms.append(f"vrat={float(val_ratio):.4e}")
            lr_val = self._get_optimizer_learning_rate()
            if np.isfinite(lr_val):
                data_terms.append(f"vlr={lr_val:.4e}")
            data_disp = " ".join(data_terms)

            strict_terms: List[str] = []
            strict_route = _get_stat_text("strict_route_mode")
            contact_backend = None
            try:
                contact_backend = self._resolve_contact_backend()
            except Exception:
                contact_backend = _get_stat_text("contact_backend")
            normal_ift_ready = _get_stat_float("normal_ift_ready")
            normal_ift_consumed = _get_stat_float("normal_ift_consumed")
            normal_ift_inputs_present = _get_stat_float("normal_ift_inputs_present")
            normal_ift_core_valid_ratio = _get_stat_float("normal_ift_core_valid_ratio")
            ft_residual_norm = _get_stat_float("ft_residual_norm", "inner_ft_norm")
            signature_gate_applied = _get_stat_float("signature_gate_applied")
            fallback_reason_code = _get_stat_float("fallback_reason_code")
            effective_alpha_scale = _get_stat_float("effective_alpha_scale")
            tail_has_effective_step = _get_stat_float("tail_has_effective_step")
            tangential_step_mode = _get_stat_text("tangential_step_mode")
            fallback_trigger_reason = _get_stat_text("fallback_trigger_reason")
            strict_step_class = _get_stat_text("strict_step_class")
            risk_guard_bucket_raw = _get_stat_text("risk_guard_bucket_raw")
            risk_guard_bucket = _get_stat_text("risk_guard_bucket")
            protect_prefix_active = _get_stat_float("protect_prefix_active")
            protect_prefix_reason = _get_stat_text("protect_prefix_reason")
            first_c_event_seen_before_step = _get_stat_float("first_c_event_seen_before_step")
            first_c_event_seen_after_step = _get_stat_float("first_c_event_seen_after_step")
            delayed_guard_ready = _get_stat_float("delayed_guard_ready")
            strict_step_update_scale = _get_stat_float("strict_step_update_scale", "step_update_scale")
            total_step_update_scale = _get_stat_float("total_step_update_scale")
            inner_convergence_rate = _get_stat_float("inner_convergence_rate")
            inner_fallback_rate = _get_stat_float("inner_fallback_rate")
            inner_skip_rate = _get_stat_float("inner_skip_rate")
            continuation_frozen = _get_stat_float("continuation_frozen")
            continuation_freeze_events = _get_stat_float("continuation_freeze_events")
            if strict_route:
                strict_terms.append(f"smode={strict_route}")
            if contact_backend:
                strict_terms.append(f"cback={contact_backend}")
            if inner_convergence_rate is not None:
                strict_terms.append(f"iconv={inner_convergence_rate:.4e}")
            if inner_fallback_rate is not None:
                strict_terms.append(f"ifb={inner_fallback_rate:.4e}")
            if inner_skip_rate is not None:
                strict_terms.append(f"iskip={inner_skip_rate:.4e}")
            if continuation_frozen is not None:
                strict_terms.append(f"cfrz={int(continuation_frozen > 0.5)}")
            if continuation_freeze_events is not None:
                strict_terms.append(f"cfrze={int(round(continuation_freeze_events))}")
            if normal_ift_ready is not None:
                strict_terms.append(f"normal_ift_ready={int(normal_ift_ready > 0.5)}")
            if normal_ift_consumed is not None:
                strict_terms.append(f"normal_ift_consumed={int(normal_ift_consumed > 0.5)}")
            if normal_ift_inputs_present is not None:
                strict_terms.append(f"normal_ift_inputs_present={int(normal_ift_inputs_present > 0.5)}")
            if normal_ift_core_valid_ratio is not None:
                strict_terms.append(f"normal_ift_core_valid_ratio={normal_ift_core_valid_ratio:.4e}")
            strict_terms.append(
                f"max_tail_qn_iters={max(0, int(getattr(self.cfg, 'max_tail_qn_iters', 0) or 0))}"
            )
            if signature_gate_applied is not None:
                strict_terms.append(f"gate_applied={int(signature_gate_applied > 0.5)}")
            if fallback_reason_code is not None:
                strict_terms.append(f"fallback_reason_code={int(round(fallback_reason_code))}")
            if tangential_step_mode:
                strict_terms.append(f"tangential_step_mode={tangential_step_mode}")
            if strict_step_class:
                strict_terms.append(f"strict_step_class={strict_step_class}")
            if risk_guard_bucket_raw:
                strict_terms.append(f"risk_guard_bucket_raw={risk_guard_bucket_raw}")
            if risk_guard_bucket:
                strict_terms.append(f"risk_guard_bucket={risk_guard_bucket}")
            if protect_prefix_active is not None:
                strict_terms.append(f"protect_prefix_active={int(protect_prefix_active > 0.5)}")
            if protect_prefix_reason:
                strict_terms.append(f"protect_prefix_reason={protect_prefix_reason}")
            if first_c_event_seen_before_step is not None:
                strict_terms.append(
                    f"first_c_event_seen_before_step={int(first_c_event_seen_before_step > 0.5)}"
                )
            if first_c_event_seen_after_step is not None:
                strict_terms.append(
                    f"first_c_event_seen_after_step={int(first_c_event_seen_after_step > 0.5)}"
                )
            if delayed_guard_ready is not None:
                strict_terms.append(f"delayed_guard_ready={int(delayed_guard_ready > 0.5)}")
            if total_step_update_scale is not None:
                strict_terms.append(f"total_step_update_scale={total_step_update_scale:.4e}")
            if effective_alpha_scale is not None:
                strict_terms.append(f"effective_alpha_scale={effective_alpha_scale:.4e}")
            if fallback_trigger_reason:
                strict_terms.append(f"fallback_trigger_reason={fallback_trigger_reason}")
            if ft_residual_norm is not None:
                strict_terms.append(f"ft_residual_norm={ft_residual_norm:.4e}")
            if tail_has_effective_step is not None:
                strict_terms.append(f"tail_has_effective_step={int(tail_has_effective_step > 0.5)}")
            if strict_step_update_scale is not None:
                strict_terms.append(f"strict_step_update_scale={strict_step_update_scale:.4e}")
            strict_disp = " ".join(strict_terms)

            # Von Mises 应力及屈服比（若提供 yield_strength）
            vm_phys_max = _get_stat_float("stress_vm_phys_max")
            vm_pred_max = _get_stat_float("stress_vm_pred_max")
            vm_ref = vm_phys_max if vm_phys_max is not None else vm_pred_max
            if vm_phys_max is not None and vm_pred_max is not None:
                vm_disp = f"σvm={vm_phys_max:.4e}(pred={vm_pred_max:.4e})"
            elif vm_phys_max is not None:
                vm_disp = f"σvm={vm_phys_max:.4e}"
            elif vm_pred_max is not None:
                vm_disp = f"σvm_pred={vm_pred_max:.4e}"
            else:
                vm_disp = ""
            vm_ratio_disp = ""
            if vm_ref is not None and getattr(self.cfg, "yield_strength", None):
                y = float(self.cfg.yield_strength)
                if y > 0:
                    vm_ratio_disp = f"σvm/σy={vm_ref / y:.3f}"

            order_txt = ""
            if order is not None:
                try:
                    order_list = [int(x) for x in list(order)]
                    human_order = "-".join(str(idx + 1) for idx in order_list)
                    ordered_values: Optional[List[int]] = None
                    if P_np is not None and len(order_list) == len(P_np):
                        ordered_values = []
                        for idx in order_list:
                            if 0 <= idx < len(P_np):
                                ordered_values.append(int(P_np[idx]))
                            else:
                                ordered_values = None
                                break
                    if ordered_values:
                        order_txt = (
                            f" order={human_order}(P序=["
                            + ",".join(str(val) for val in ordered_values)
                            + "])"
                        )
                    else:
                        order_txt = f" order={human_order}"
                except Exception:
                    order_txt = " order=?"
            parts_disp = energy_disp or ""
            unit = str(getattr(self.cfg.tightening_cfg, "angle_unit", "deg") or "deg")
            angle_txt = ",".join(f"{a:.4f}" for a in angles)
            postfix = (
                f"theta=[{angle_txt}]{unit}{order_txt} Π={pin:.6e} | {parts_disp}{tight_txt} "
                f"| {grad_disp} {pen_disp} {stick_disp} {slip_disp} {gap_disp} {eq_disp} {data_disp} {strict_disp} {vm_disp} {vm_ratio_disp}"
            )
            return postfix, "已记录"
        except Exception:
            return None, "记录异常"

