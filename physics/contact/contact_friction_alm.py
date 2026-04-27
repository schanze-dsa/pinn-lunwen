#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_friction_alm.py
-----------------------
Augmented Lagrangian (ALM) tangential / friction contact operator.

核心思路：
- 切向相对滑移：
      s  = ((xs + u(xs)) - (xm + u(xm)))
      s_t = [ t1·s, t2·s ]      (切向基 {t1, t2} 下的 2D 分量)
- 试探切向应力：
      tau_trial = lambda_t + k_t * s_t     (R^2, 每个接触点两个分量)
- 摩擦圆锥（库仑摩擦）：
      p_eff = max(0, lambda_n + mu_n * phi(g))   (来自法向 ALM 的有效压缩)
      tau_c = mu_f * p_eff                       (圆锥半径)
- 光滑投影到摩擦圆锥：
      ||·||_eps = sqrt( ·^2 + eps^2 )
      scale = min(1, tau_c / ||tau_trial||_eps)
      tau   = scale * tau_trial
- 残差：
      r_t = tau_trial - tau

两种摩擦“能量路径”（由 config.use_smooth_friction 控制）：

1) 经典 ALM 伪能量（默认）：
      Et = sum w_eff * (1/(2*mu_t)) * ||r_t||^2

2) 可选的 C^1 平滑摩擦能量（推荐，在训练阶段更稳定）：
      r   = ||s_t||                        (切向滑移量)
      psi = s0^2 ( sqrt(1 + (r/s0)^2) - 1 )
      Et  = sum w_eff * mu_f * p_eff * psi

    其中 s0 = smooth_s0 是一条光滑尺度，r<<s0 时近似 0.5*r^2，
    r>>s0 时近似 s0*|r|，兼具小滑移二次、大发滑近似线性的特性。

外层 ALM 更新（保持原有形式）：
      lambda_t <- lambda_t + eta * (k_t * s_t - tau)
    其中 eta = step_scale 为可调步长因子。

Inputs (per batch):
  xs, xm : (N,3) slave/master coordinates
  t1, t2 : (N,3) orthonormal tangential basis (from master tri normals)
  w_area : (N,) area weights
  normal_op : NormalContactALM, to retrieve p_eff

本模块与 DFEM 内核解耦，只依赖 u_fn(X, params) 接口。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from mesh.interp_utils import interp_bary_tf
from .contact_inner_kernel_primitives import project_to_coulomb_disk

# Optional import for type hints (no runtime cyclic dep)
try:
    from .contact_normal_alm import NormalContactALM, softplus_neg, _ensure_2d, _to_tf
except Exception:  # 允许单文件调试时的降级实现
    def _to_tf(x, dtype=tf.float32):
        if isinstance(x, tf.Tensor):
            return tf.cast(x, dtype)
        return tf.convert_to_tensor(x, dtype=dtype)

    def _ensure_2d(u: tf.Tensor) -> tf.Tensor:
        if u.shape.rank == 1:
            return tf.reshape(u, (-1, 3))
        return u

    def softplus_neg(x: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        return tf.nn.softplus(-x * beta) / (beta + 1e-12)

    class NormalContactALM:  # minimal stub for type hints
        beta: tf.Variable
        mu_n: tf.Variable
        lmbda: tf.Variable

        def _gap(self, u_fn, params=None) -> tf.Tensor:  # type: ignore
            raise NotImplementedError


def delta_slip_from_pair(
    us_now: tf.Tensor,
    um_now: tf.Tensor,
    us_prev: tf.Tensor,
    um_prev: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
) -> tf.Tensor:
    """Stateless incremental tangential slip kernel."""

    us_now = tf.cast(us_now, tf.float32)
    um_now = tf.cast(um_now, tf.float32)
    us_prev = tf.cast(us_prev, tf.float32)
    um_prev = tf.cast(um_prev, tf.float32)
    t1 = tf.cast(t1, tf.float32)
    t2 = tf.cast(t2, tf.float32)

    du = (us_now - um_now) - (us_prev - um_prev)
    return tf.stack(
        [tf.reduce_sum(du * t1, axis=-1), tf.reduce_sum(du * t2, axis=-1)],
        axis=-1,
    )


# -----------------------------
# Config
# -----------------------------

@dataclass
class FrictionALMConfig:
    # 物理/数值参数
    enabled: bool = True      # master switch for friction (False -> skip friction entirely)
    mu_f: float = 0.15          # Coulomb friction coefficient
    k_t: float = 5.0e2          # tangential penalty "stiffness" (for trial traction)
    mu_t: float = 1.0e3         # ALM coefficient for tangential residual pseudo-energy
    eps: float = 1.0e-6         # smoothing for ||·||_eps
    use_delta_st: bool = False  # use incremental tangential slip (delta) for friction update

    # 是否使用“有效法向压力” p_eff（建议 True）
    use_effective_normal: bool = True

    # 是否启用平滑摩擦能量路径（推荐训练用）
    use_smooth_friction: bool = False
    smooth_s0: float = 1.0e-3   # 光滑尺度 s0，量纲与 |s_t| 相同
    smooth_blend: float = 1.0   # 0=严格(ALM), 1=平滑；(0,1) 线性混合
    use_bipotential_residual: bool = False
    bipotential_weight: float = 0.0
    bipotential_eps: float = 1.0e-8

    dtype: str = "float32"


class FrictionContactALM:
    """
    Tangential / friction ALM operator.

    Typical usage per batch:
        fn = FrictionContactALM(cfg)
        fn.link_normal(normal_op)                               # to fetch p_eff
        fn.build_from_numpy(xs, xm, t1, t2, w_area, extra_w)    # reset lambda_t
        Et, stats = fn.energy(u_fn, params, extra_weights=w_t)  # backprop on Et
        fn.update_multipliers(u_fn, params)                     # outer-loop update

    Notes:
    - lambda_t is a per-sample 2D vector (N,2).
    - If you re-sample contact points every step, call build_* each time (it resets lambda_t).
      If you want to persist lambda_t across steps, manage indices yourself and avoid rebuild.
    """

    def __init__(self, cfg: Optional[FrictionALMConfig] = None):
        self.cfg = cfg or FrictionALMConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # batch tensors
        self.xs: Optional[tf.Tensor] = None  # (N,3)
        self.xm: Optional[tf.Tensor] = None  # (N,3)
        self.t1: Optional[tf.Tensor] = None  # (N,3)
        self.t2: Optional[tf.Tensor] = None  # (N,3)
        self.w:  Optional[tf.Tensor] = None  # (N,)

        # Optional interpolation metadata (Route-2)
        self.xs_node_idx: Optional[tf.Tensor] = None  # (N,3) int32
        self.xs_bary: Optional[tf.Tensor] = None      # (N,3) float32/float64
        self.xm_node_idx: Optional[tf.Tensor] = None  # (N,3) int32
        self.xm_bary: Optional[tf.Tensor] = None      # (N,3) float32/float64

        # multipliers (tangential, 2D)
        self.lmbda_t: Optional[tf.Variable] = None  # (N,2)

        # schedules / scalars
        self.mu_t = tf.Variable(self.cfg.mu_t, dtype=self.dtype, trainable=False, name="mu_t")
        self.k_t  = tf.Variable(self.cfg.k_t,  dtype=self.dtype, trainable=False, name="k_t")
        self.mu_f = tf.Variable(self.cfg.mu_f, dtype=self.dtype, trainable=False, name="mu_f")
        self.eps  = tf.Variable(self.cfg.eps,  dtype=self.dtype, trainable=False, name="eps")
        self.s0   = tf.Variable(self.cfg.smooth_s0, dtype=self.dtype, trainable=False, name="s0")

        # link to normal operator (for p_eff)
        self.normal_op: Optional[NormalContactALM] = None

        # cache
        self._last_st: Optional[tf.Tensor] = None
        self._ref_st: Optional[tf.Tensor] = None
        self._last_tau_trial: Optional[tf.Tensor] = None
        self._last_tau: Optional[tf.Tensor] = None
        self._last_r_norm: Optional[tf.Tensor] = None

        self._built_N: int = 0

    # ---------- linking & building ----------

    def link_normal(self, normal_op: NormalContactALM):
        """Link a NormalContactALM instance so we can compute p_eff for the friction cone."""
        self.normal_op = normal_op

    def build_from_numpy(
        self,
        xs: np.ndarray,
        xm: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        w_area: np.ndarray,
        extra_weights: Optional[np.ndarray] = None,
        *,
        xs_node_idx: Optional[np.ndarray] = None,
        xs_bary: Optional[np.ndarray] = None,
        xm_node_idx: Optional[np.ndarray] = None,
        xm_bary: Optional[np.ndarray] = None,
    ):
        """
        Initialize per-batch tensors from NumPy arrays.
        - xs, xm : (N,3)
        - t1, t2 : (N,3) orthonormal tangential basis
        - w_area : (N,) area weights (will be multiplied by extra_weights if provided)
        """
        assert xs.shape == xm.shape and xs.shape[1] == 3
        assert t1.shape == xs.shape and t2.shape == xs.shape and w_area.shape[0] == xs.shape[0]

        Xs = _to_tf(xs, self.dtype)
        Xm = _to_tf(xm, self.dtype)
        T1 = _to_tf(t1, self.dtype)
        T2 = _to_tf(t2, self.dtype)
        W  = _to_tf(w_area, self.dtype)

        if extra_weights is not None:
            W = W * _to_tf(extra_weights, self.dtype)

        # Normalize basis defensively (should already be orthonormal)
        T1 = T1 / (tf.norm(T1, axis=1, keepdims=True) + tf.cast(1e-12, self.dtype))
        # T2 由上游保证与 T1 正交即可，这里不再强制修正

        self.xs, self.xm, self.t1, self.t2, self.w = Xs, Xm, T1, T2, W
        self._built_N = int(Xs.shape[0])

        if xs_node_idx is not None and xs_bary is not None and xm_node_idx is not None and xm_bary is not None:
            self.xs_node_idx = tf.convert_to_tensor(xs_node_idx, dtype=tf.int32)
            self.xs_bary = tf.convert_to_tensor(xs_bary, dtype=self.dtype)
            self.xm_node_idx = tf.convert_to_tensor(xm_node_idx, dtype=tf.int32)
            self.xm_bary = tf.convert_to_tensor(xm_bary, dtype=self.dtype)
        else:
            self.xs_node_idx = None
            self.xs_bary = None
            self.xm_node_idx = None
            self.xm_bary = None

        # reset tangential multipliers
        self.lmbda_t = tf.Variable(
            tf.zeros((self._built_N, 2), dtype=self.dtype),
            trainable=False,
            name="lambda_t",
        )

        # clear caches
        self._last_st = None
        self._ref_st = None
        self._last_tau_trial = None
        self._last_tau = None
        self._last_r_norm = None

    # ---------- stage state helpers ----------

    def snapshot_state(self) -> Dict[str, tf.Tensor]:
        """Take a lightweight snapshot of stateful friction variables for staged loading."""
        if self.lmbda_t is None:
            return {"lmbda_t": None}

        return {
            "lmbda_t": tf.identity(self.lmbda_t),
            "_last_st": None if self._last_st is None else tf.identity(self._last_st),
            "_ref_st": None if self._ref_st is None else tf.identity(self._ref_st),
        }

    def restore_state(self, state: Dict[str, tf.Tensor]):
        """Restore tangential multipliers/cache from :meth:`snapshot_state`."""
        if not state:
            return

        lmbda_t = state.get("lmbda_t", None)
        if lmbda_t is not None and self.lmbda_t is not None:
            self.lmbda_t.assign(tf.cast(lmbda_t, self.dtype))

        st = state.get("_last_st", None)
        if st is None:
            self._last_st = None
        else:
            self._last_st = tf.cast(st, self.dtype)
        ref = state.get("_ref_st", None)
        if ref is None:
            self._ref_st = None
        else:
            self._ref_st = tf.cast(ref, self.dtype)

    def last_slip(self) -> Optional[tf.Tensor]:
        """Return cached tangential slip (N,2) from the last energy call, if any."""
        return self._last_st

    def capture_reference(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> Optional[tf.Tensor]:
        """Capture current absolute tangential slip as the incremental reference."""
        st = self._absolute_slip_t(u_fn, params, u_nodes=u_nodes)
        self._ref_st = tf.identity(st) if st is not None else None
        return st

    def commit_reference(self):
        """Commit the latest slip as the new incremental reference."""
        if self._last_st is None:
            return
        self._ref_st = tf.identity(self._last_st)

    def reset_reference(self):
        """Clear incremental reference so the next step uses absolute slip."""
        self._ref_st = None

    def build_from_cat(
        self,
        cat: Dict[str, np.ndarray],
        extra_weights: Optional[np.ndarray] = None,
    ):
        """
        Build from ContactMap.concatenate() dict:
            xs, xm, t1, t2, w_area
        """
        self.build_from_numpy(
            cat["xs"], cat["xm"], cat["t1"], cat["t2"], cat["w_area"],
            extra_weights=extra_weights,
            xs_node_idx=cat.get("xs_node_idx"),
            xs_bary=cat.get("xs_bary"),
            xm_node_idx=cat.get("xm_node_idx"),
            xm_bary=cat.get("xm_bary"),
        )

    # ---------- internals ----------

    def _absolute_slip_t(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
        update_cache: bool = True,
    ) -> tf.Tensor:
        """Compute absolute tangential slip components in the [t1,t2] basis."""
        xs = tf.cast(self.xs, self.dtype)
        xm = tf.cast(self.xm, self.dtype)
        t1 = tf.cast(self.t1, self.dtype)
        t2 = tf.cast(self.t2, self.dtype)

        use_interp = (
            u_nodes is not None
            and self.xs_node_idx is not None
            and self.xs_bary is not None
            and self.xm_node_idx is not None
            and self.xm_bary is not None
        )
        if use_interp:
            u_nodes = tf.cast(u_nodes, self.dtype)
            u_s = tf.cast(interp_bary_tf(u_nodes, self.xs_node_idx, self.xs_bary), self.dtype)
            u_m = tf.cast(interp_bary_tf(u_nodes, self.xm_node_idx, self.xm_bary), self.dtype)
        else:
            u_s = tf.cast(_ensure_2d(u_fn(xs, params)), self.dtype)
            u_m = tf.cast(_ensure_2d(u_fn(xm, params)), self.dtype)

        s = (xs + u_s) - (xm + u_m)                  # (N,3)
        s1 = tf.reduce_sum(t1 * s, axis=1)           # (N,)
        s2 = tf.reduce_sum(t2 * s, axis=1)           # (N,)
        st = tf.stack([s1, s2], axis=1)              # (N,2)

        if update_cache:
            self._last_st = st
        return st

    def _relative_slip_t(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
        update_cache: bool = True,
    ) -> tf.Tensor:
        """
        Return tangential slip for friction update. If use_delta_st is enabled,
        return incremental slip relative to the stored reference.
        """
        st = self._absolute_slip_t(u_fn, params, u_nodes=u_nodes, update_cache=update_cache)
        if self.cfg.use_delta_st:
            if self._ref_st is None:
                return st
            return st - self._ref_st
        return st

    def _effective_normal_pressure(self, u_fn, params=None, *, u_nodes: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute p_eff, the effective normal compression used in the friction cone:
            p_eff = max(0, lambda_n + mu_n * phi(g)), phi(g)=softplus(-g; beta)
        Requires that link_normal() has been called.
        """
        if self.normal_op is None:
            raise RuntimeError("FrictionContactALM needs link_normal(normal_op) before use.")

        if hasattr(self.normal_op, "effective_normal_pressure"):
            return tf.cast(
                self.normal_op.effective_normal_pressure(u_fn, params, u_nodes=u_nodes),
                self.dtype,
            )

        g = self.normal_op._gap(u_fn, params, u_nodes=u_nodes)                      # (N,)
        phi = softplus_neg(g, self.normal_op.beta)                 # (N,)
        p_eff = self.normal_op.lmbda + self.normal_op.mu_n * phi   # (N,)
        p_eff = tf.maximum(p_eff, tf.cast(0.0, self.dtype))        # no tension
        return p_eff

    # ---------- energy & update ----------

    def energy(
        self,
        u_fn,
        params=None,
        extra_weights: Optional[tf.Tensor] = None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute friction energy and stats.

        若 cfg.use_smooth_friction=False（默认）：
            - 使用 ALM 残差伪能量：
                  r_t = tau_trial - tau
                  Et  = sum w_eff * (1/(2*mu_t)) * ||r_t||^2

        若 cfg.use_smooth_friction=True：
            - 使用 C^1 光滑摩擦能量：
                  r   = ||s_t||
                  psi = s0^2 ( sqrt(1 + (r/s0)^2) - 1 )
                  Et  = sum w_eff * mu_f * p_eff * psi

        其中 w_eff = w * extra_weights（若 extra_weights 为 None，则 w_eff = w）。
        """
        # 切向滑移
        st = self._relative_slip_t(u_fn, params, u_nodes=u_nodes, update_cache=True)  # (N,2)

        # 有效权重：保留 self.w 作为几何/面积基权重，额外权重只在本次调用中生效
        w_eff = self.w
        if extra_weights is not None:
            w_eff = w_eff * tf.cast(extra_weights, self.dtype)

        # 有效法向压力 p_eff（用于摩擦圆锥半径 & 平滑能量）
        if self.cfg.use_effective_normal:
            p_eff = self._effective_normal_pressure(u_fn, params, u_nodes=u_nodes)
        else:
            if self.normal_op is None:
                raise RuntimeError("use_effective_normal=False still needs link_normal(normal_op).")
            p_eff = tf.maximum(self.normal_op.lmbda, tf.cast(0.0, self.dtype))

        blend = float(getattr(self.cfg, "smooth_blend", 1.0))
        blend = max(0.0, min(1.0, blend))
        use_smooth = bool(self.cfg.use_smooth_friction) and blend > 0.0
        use_alm = (not self.cfg.use_smooth_friction) or blend < 1.0
        blend_t = tf.cast(blend, self.dtype)

        # 预分配
        Et_smooth = None
        Et_alm = None
        r = None
        r_norm = None
        tau_trial = None
        tau = None
        stats_s: Dict[str, tf.Tensor] = {}
        stats_a: Dict[str, tf.Tensor] = {}

        # 模式 1：平滑摩擦能量路径
        if use_smooth:
            # r = ||s_t||, 光滑伪势 psi
            r = tf.sqrt(tf.reduce_sum(st * st, axis=1) + tf.cast(1e-12, self.dtype))  # (N,)
            s0 = self.s0
            r_over_s0 = r / (s0 + tf.cast(1e-12, self.dtype))
            psi = (s0 * s0) * (tf.sqrt(1.0 + r_over_s0 * r_over_s0) - 1.0)           # (N,)

            Et_smooth = tf.reduce_sum(w_eff * self.mu_f * p_eff * psi)

            stats_s = {
                "mode": tf.constant("smooth", dtype=tf.string),
                "slip_mean": tf.reduce_mean(r),
                "psi_mean": tf.reduce_mean(psi),
                "p_eff_mean": tf.reduce_mean(p_eff),
            }
            stats_s["R_fric_comp"] = tf.reduce_sum(w_eff * r)

        # 模式 2：经典 ALM 残差伪能量
        if use_alm:
            tau_trial = self.lmbda_t + self.k_t * st                # (N,2)
            tau_c = self.mu_f * p_eff                               # (N,)

            eps = tf.cast(self.eps, self.dtype)
            norm_trial = tf.sqrt(
                tf.reduce_sum(tau_trial * tau_trial, axis=1) + eps * eps
            )                                                        # (N,)
            tau = tf.cast(project_to_coulomb_disk(tau_trial, tau_c, eps=eps), self.dtype)

            r_t = tau_trial - tau                                    # (N,2)
            r_norm = tf.sqrt(tf.reduce_sum(r_t * r_t, axis=1) + 1e-12)

            Et_alm = tf.reduce_sum(
                w_eff * (0.5 / self.mu_t) * tf.reduce_sum(r_t * r_t, axis=1)
            )

            stick = tf.cast(scale >= 0.999, self.dtype)              # scale ~1 => stick
            slip = tf.cast(scale < 0.999, self.dtype)

            stats_a = {
                "mode": tf.constant("alm_residual", dtype=tf.string),
                "stick_ratio": tf.reduce_mean(stick),
                "slip_ratio": tf.reduce_mean(slip),
                "tau_trial_mean": tf.reduce_mean(norm_trial),
                "tau_mean": tf.reduce_mean(tf.sqrt(tf.reduce_sum(tau * tau, axis=1))),
                "R_fric_comp": tf.reduce_sum(w_eff * r_norm),
            }

        def _bipotential_residual_terms(tau_use: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            eps_bi = tf.cast(getattr(self.cfg, "bipotential_eps", 1.0e-8), self.dtype)
            st_norm = tf.sqrt(tf.reduce_sum(st * st, axis=1) + eps_bi)
            bi_raw = self.mu_f * p_eff * st_norm - tf.reduce_sum(tau_use * st, axis=1)
            bi_pos = tf.nn.relu(bi_raw)
            E_bi = tf.reduce_sum(w_eff * bi_pos * bi_pos)
            R_bi = tf.reduce_sum(w_eff * bi_pos)
            return E_bi, R_bi

        # 单一路径：平滑
        if use_smooth and not use_alm:
            if bool(getattr(self.cfg, "use_bipotential_residual", False)) and float(getattr(self.cfg, "bipotential_weight", 0.0)) > 0.0:
                tau_trial_s = self.lmbda_t + self.k_t * st
                E_bi, R_bi = _bipotential_residual_terms(tau_trial_s)
                w_bi = tf.cast(getattr(self.cfg, "bipotential_weight", 0.0), self.dtype)
                Et_smooth = Et_smooth + w_bi * E_bi
                stats_s["E_bi"] = E_bi
                stats_s["R_bi_comp"] = R_bi
            self._last_tau_trial = None
            self._last_tau = None
            self._last_r_norm = r
            return Et_smooth, stats_s

        # 单一路径：ALM
        if use_alm and not use_smooth:
            if bool(getattr(self.cfg, "use_bipotential_residual", False)) and float(getattr(self.cfg, "bipotential_weight", 0.0)) > 0.0:
                E_bi, R_bi = _bipotential_residual_terms(tau)
                w_bi = tf.cast(getattr(self.cfg, "bipotential_weight", 0.0), self.dtype)
                Et_alm = Et_alm + w_bi * E_bi
                stats_a["E_bi"] = E_bi
                stats_a["R_bi_comp"] = R_bi
            self._last_tau_trial = tau_trial
            self._last_tau = tau
            self._last_r_norm = r_norm
            return Et_alm, stats_a

        # 混合路径
        Et = blend_t * tf.cast(Et_smooth, self.dtype) + (1.0 - blend_t) * tf.cast(Et_alm, self.dtype)
        stats = {
            "mode": tf.constant("blend", dtype=tf.string),
            "smooth_blend": tf.cast(blend_t, self.dtype),
            "slip_mean": stats_s.get("slip_mean", tf.reduce_mean(r)),
            "psi_mean": stats_s.get("psi_mean", tf.reduce_mean(r)),
            "tau_trial_mean": stats_a.get("tau_trial_mean", tf.reduce_mean(r_norm)),
            "tau_mean": stats_a.get("tau_mean", tf.reduce_mean(r_norm)),
            "p_eff_mean": stats_s.get("p_eff_mean", tf.reduce_mean(p_eff)),
        }
        stats["R_fric_comp"] = (
            blend_t * tf.reduce_sum(w_eff * r)
            + (1.0 - blend_t) * tf.reduce_sum(w_eff * r_norm)
        )
        if bool(getattr(self.cfg, "use_bipotential_residual", False)) and float(getattr(self.cfg, "bipotential_weight", 0.0)) > 0.0:
            E_bi, R_bi = _bipotential_residual_terms(tau)
            w_bi = tf.cast(getattr(self.cfg, "bipotential_weight", 0.0), self.dtype)
            Et = Et + w_bi * E_bi
            stats["E_bi"] = E_bi
            stats["R_bi_comp"] = R_bi

        self._last_tau_trial = tau_trial
        self._last_tau = tau
        self._last_r_norm = blend_t * r + (1.0 - blend_t) * r_norm
        return Et, stats

    # ---------- residual (no energy) ----------

    def residual(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Residual-only friction term using projection residual:
            r_t = tau_trial - proj_{||tau||<=mu*p_eff}(tau_trial)
        """
        if self.xs is None or self.xm is None or self.t1 is None or self.t2 is None or self.w is None:
            raise RuntimeError("[FrictionContactALM] build_from_numpy/build_from_cat must be called first.")
        if self.lmbda_t is None:
            self.lmbda_t = tf.Variable(
                tf.zeros((tf.shape(self.xs)[0], 2), dtype=self.dtype),
                trainable=False,
                name="lambda_t",
            )

        st = self._relative_slip_t(u_fn, params, u_nodes=u_nodes, update_cache=True)  # (N,2)

        if self.cfg.use_effective_normal:
            p_eff = self._effective_normal_pressure(u_fn, params, u_nodes=u_nodes)
        else:
            if self.normal_op is None:
                raise RuntimeError("use_effective_normal=False still needs link_normal(normal_op).")
            p_eff = tf.maximum(self.normal_op.lmbda, tf.cast(0.0, self.dtype))

        tau_trial = self.lmbda_t + self.k_t * st
        tau_c = self.mu_f * p_eff

        eps = tf.cast(self.eps, self.dtype)
        norm_trial = tf.sqrt(tf.reduce_sum(tau_trial * tau_trial, axis=1) + eps * eps)
        tau = tf.cast(project_to_coulomb_disk(tau_trial, tau_c, eps=eps), self.dtype)
        r_t = tau_trial - tau

        r_norm = tf.sqrt(tf.reduce_sum(r_t * r_t, axis=1) + 1e-12)
        w_eff = tf.cast(self.w, self.dtype)
        denom = tf.reduce_sum(w_eff) + tf.cast(1e-12, self.dtype)
        L_ct = tf.reduce_sum(w_eff * r_norm * r_norm) / denom

        if bool(getattr(self.cfg, "use_bipotential_residual", False)) and float(getattr(self.cfg, "bipotential_weight", 0.0)) > 0.0:
            eps_bi = tf.cast(getattr(self.cfg, "bipotential_eps", 1.0e-8), self.dtype)
            st_norm = tf.sqrt(tf.reduce_sum(st * st, axis=1) + eps_bi)
            bi_raw = self.mu_f * p_eff * st_norm - tf.reduce_sum(tau * st, axis=1)
            bi_pos = tf.nn.relu(bi_raw)
            E_bi = tf.reduce_sum(w_eff * bi_pos * bi_pos) / denom
            w_bi = tf.cast(getattr(self.cfg, "bipotential_weight", 0.0), self.dtype)
            L_ct = L_ct + w_bi * E_bi
        else:
            E_bi = tf.cast(0.0, self.dtype)
            bi_pos = tf.zeros_like(r_norm)

        stick = tf.cast(scale >= 0.999, self.dtype)
        slip = tf.cast(scale < 0.999, self.dtype)
        stats = {
            "mode": tf.constant("residual", dtype=tf.string),
            "stick_ratio": tf.reduce_mean(stick),
            "slip_ratio": tf.reduce_mean(slip),
            "tau_trial_mean": tf.reduce_mean(norm_trial),
            "tau_mean": tf.reduce_mean(tf.sqrt(tf.reduce_sum(tau * tau, axis=1))),
            "R_fric_comp": tf.reduce_sum(w_eff * r_norm),
            "E_bi": E_bi,
            "R_bi_comp": tf.reduce_sum(w_eff * bi_pos),
        }

        self._last_tau_trial = tau_trial
        self._last_tau = tau
        self._last_r_norm = r_norm
        return L_ct, stats

    def update_multipliers(
        self,
        u_fn,
        params=None,
        step_scale: float = 1.0,
        u_nodes: Optional[tf.Tensor] = None,
    ):
        """
        Outer update for tangential multipliers (增强型 ALM)：

            lambda_t <- lambda_t + eta * (k_t * s_t - tau)

        其中 eta = step_scale 为可调步长因子。保持接口向后兼容。
        """
        # 切向滑移
        st = self._relative_slip_t(u_fn, params, u_nodes=u_nodes, update_cache=False)  # (N,2)

        # 试探应力
        tau_trial = self.lmbda_t + self.k_t * st

        # 摩擦圆锥半径（与 energy 中保持一致）
        if self.cfg.use_effective_normal:
            p_eff = self._effective_normal_pressure(u_fn, params, u_nodes=u_nodes)
        else:
            p_eff = tf.maximum(self.normal_op.lmbda, tf.cast(0.0, self.dtype))
        tau_c = self.mu_f * p_eff

        eps = tf.cast(self.eps, self.dtype)
        norm_trial = tf.sqrt(
            tf.reduce_sum(tau_trial * tau_trial, axis=1) + eps * eps
        )
        tau = tf.cast(project_to_coulomb_disk(tau_trial, tau_c, eps=eps), self.dtype)

        eta = tf.cast(step_scale, self.dtype)
        new_lambda_t = self.lmbda_t + eta * (self.k_t * st - tau)
        self.lmbda_t.assign(new_lambda_t)

    # ---------- schedules / setters ----------

    def set_mu_t(self, mu_t: float):
        self.mu_t.assign(tf.cast(mu_t, self.dtype))

    def set_k_t(self, k_t: float):
        self.k_t.assign(tf.cast(k_t, self.dtype))

    def set_mu_f(self, mu_f: float):
        """Set Coulomb friction coefficient."""
        self.mu_f.assign(tf.cast(mu_f, self.dtype))

    def set_smooth_friction(self, flag: bool):
        """Turn on/off smooth friction energy path."""
        self.cfg.use_smooth_friction = bool(flag)

    def set_s0(self, s0: float):
        """Update smoothness scale s0."""
        self.s0.assign(tf.cast(s0, self.dtype))

    def set_smooth_blend(self, blend: float):
        """Set smooth/strict blend weight in [0, 1]."""
        try:
            b = float(blend)
        except Exception:
            b = 0.0
        b = max(0.0, min(1.0, b))
        self.cfg.smooth_blend = b

    def multiply_weights(self, extra_w: np.ndarray):
        """Permanently multiply current per-sample weights by extra_w (Weighted PINN hooks)."""
        ew = _to_tf(extra_w, self.dtype)
        self.w.assign(self.w * ew)

    def reset_for_new_batch(self):
        """Clear batch tensors/state to allow rebuild with new samples."""
        self.xs = self.xm = self.t1 = self.t2 = self.w = None
        self.xs_node_idx = self.xs_bary = None
        self.xm_node_idx = self.xm_bary = None
        self.lmbda_t = None
        self._built_N = 0
        self._last_st = None
        self._ref_st = None
        self._last_tau_trial = None
        self._last_tau = None
        self._last_r_norm = None

    def reset_multipliers(self, reset_reference: bool = True):
        """Reset ALM multipliers for the current batch (keep geometry)."""
        if self.lmbda_t is not None:
            self.lmbda_t.assign(tf.zeros_like(self.lmbda_t))
        self._last_tau_trial = None
        self._last_tau = None
        self._last_r_norm = None
        if reset_reference:
            self._last_st = None
            self._ref_st = None
