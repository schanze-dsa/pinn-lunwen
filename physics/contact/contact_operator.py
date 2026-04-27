#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_operator.py
-------------------
Unified contact operator that wraps:
  - NormalContactALM   (normal, frictionless, ALM)
  - FrictionContactALM (tangential, Coulomb / smooth friction, ALM)

典型用法（每个训练 batch）::

    op = ContactOperator(cfg)
    op.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)

    # 在损失里：
    E_c, parts_c, stats_cn, stats_ct = op.energy(u_fn, params)
    # parts_c: {"E_n": En, "E_t": Et}
    # stats_cn: 法向残差 / 间隙统计
    # stats_ct: 摩擦 stick/slip 比例、τ 等

    # 在外层 ALM 更新时（比如每 K 步一次）：
    op.update_multipliers(u_fn, params)

Weighted PINN:
    - extra_weights: np.ndarray, shape (N,)
    - 会在 build_from_cat 时与面积权重相乘，用于法向和摩擦两部分能量
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .contact_normal_alm import NormalContactALM, NormalALMConfig
from .contact_friction_alm import FrictionContactALM, FrictionALMConfig
from .contact_inner_solver import ContactInnerState, ContactInnerResult, solve_contact_inner
from .contact_inner_kernel_primitives import compose_contact_traction
from physics.traction_utils import traction_from_sigma_voigt


# -----------------------------
# Config for the adapter/dispatcher
# -----------------------------

@dataclass
class ContactOperatorConfig:
    # 子模块超参数
    normal: NormalALMConfig = field(
        default_factory=lambda: NormalALMConfig(beta=50.0, mu_n=1.0e3, dtype="float32")
    )
    friction: FrictionALMConfig = field(
        default_factory=lambda: FrictionALMConfig(
            mu_f=0.15, k_t=5.0e2, mu_t=1.0e3, dtype="float32"
        )
    )

    # ALM 外层更新节奏：若 <=0，则每一步都更新；否则每 update_every_steps 步更新一次
    update_every_steps: int = 150

    # 摩擦相关选项（可选，用于和 FrictionContactALM 协同）
    use_smooth_friction: bool = False      # True 时偏向使用 C^1 平滑摩擦伪势
    fric_weight_mode: str = "residual"     # 后续在 Trainer 里可根据该字段选择加权策略

    # 精度
    dtype: str = "float32"


@dataclass
class StrictMixedContactInputs:
    """Typed strict-mixed adapter contract for one assembled contact batch."""

    g_n: tf.Tensor
    ds_t: tf.Tensor
    normals: tf.Tensor
    t1: tf.Tensor
    t2: tf.Tensor
    weights: tf.Tensor
    xs: tf.Tensor
    xm: tf.Tensor
    mu: tf.Tensor
    eps_n: tf.Tensor
    k_t: tf.Tensor
    init_state: Optional[ContactInnerState] = None
    init_state_available: Optional[object] = None
    contact_ids: Optional[tf.Tensor] = None
    batch_meta: Optional[Dict[str, tf.Tensor]] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _clone_inner_state(state: Optional[ContactInnerState]) -> Optional[ContactInnerState]:
    if state is None:
        return None
    return ContactInnerState(
        lambda_n=tf.identity(tf.cast(state.lambda_n, tf.float32)),
        lambda_t=tf.identity(tf.cast(state.lambda_t, tf.float32)),
        converged=getattr(state, "converged", False),
        iters=getattr(state, "iters", 0),
        res_norm=getattr(state, "res_norm", 0.0),
        fallback_used=getattr(state, "fallback_used", False),
    )


def _as_python_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return float(value)
    except Exception:
        return float(default)


def _iteration_acceptance_label(iteration: Dict[str, object]) -> str:
    before = _as_python_float(iteration.get("ft_residual_before"), default=float("nan"))
    after = _as_python_float(iteration.get("ft_residual_after"), default=float("nan"))
    if not np.isfinite(before) or not np.isfinite(after):
        return "unknown"
    return "accepted" if after < (before - 1.0e-12) else "rejected"


def _matches_inner_budget_signature(result: ContactInnerResult, gate_name: str) -> bool:
    gate_name = str(gate_name or "").strip().lower()
    if not gate_name:
        return False
    if gate_name != "b":
        raise ValueError(f"Unsupported inner-budget signature gate '{gate_name}'.")

    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    trace = diagnostics.get("iteration_trace")
    if not isinstance(trace, dict):
        return False
    if str(trace.get("fallback_trigger_reason", "") or "") != "iteration_budget_exhausted":
        return False
    iterations = trace.get("iterations")
    if not isinstance(iterations, (list, tuple)) or len(iterations) == 0 or not isinstance(iterations[-1], dict):
        return False

    tail_rows = list(iterations[-3:])
    accepted_pattern = [_iteration_acceptance_label(row) for row in tail_rows]
    if "accepted" not in accepted_pattern:
        return False
    if accepted_pattern == ["rejected", "rejected", "rejected"]:
        return False

    last_iter = tail_rows[-1]
    if _as_python_float(last_iter.get("effective_alpha_scale")) <= 1.0e-12:
        return False

    ft_after = _as_python_float(last_iter.get("ft_residual_after"), default=_as_python_float(diagnostics.get("ft_residual_norm")))
    tol_t = _as_python_float(diagnostics.get("tol_t"), default=1.0e-5)
    if ft_after <= tol_t:
        return False

    if not any(
        _as_python_float(row.get("ft_residual_after")) < (_as_python_float(row.get("ft_residual_before")) - 1.0e-12)
        for row in tail_rows
    ):
        return False

    return True


def traction_matching_terms(sigma_s, sigma_m, normals, t1, t2, inner_result):
    """Residual matching terms based on solved inner-contact traction."""

    del t1, t2
    traction_s = traction_from_sigma_voigt(sigma_s, normals)
    traction_m = traction_from_sigma_voigt(sigma_m, normals)
    tc = tf.cast(inner_result.traction_vec, tf.float32)
    return traction_s + tc, traction_m - tc


class ContactOperator:
    """
    Combine normal-ALM and friction-ALM into a single, convenient interface.

    关键接口：
        - build_from_cat(cat, extra_weights=None, auto_orient=True)
        - energy(u_fn, params=None)
        - update_multipliers(u_fn, params=None)
        - multiply_weights(extra_w)  # runtime 再叠乘一层权重（比如 IRLS）
    """
    # This is an adapter/dispatcher around one batch contact frame, not a
    # second contact formulation. The strict-mixed route consumes the typed
    # contract from `strict_mixed_inputs()` and reuses the same geometry/state.
    BACKEND_LEGACY_ALM = "legacy_alm"
    BACKEND_INNER_SOLVER = "inner_solver"
    DEFAULT_BACKEND = BACKEND_LEGACY_ALM
    VALID_BACKENDS = frozenset({BACKEND_LEGACY_ALM, BACKEND_INNER_SOLVER})

    @classmethod
    def resolve_backend(cls, backend: Optional[str] = None) -> str:
        resolved = str(backend or cls.DEFAULT_BACKEND).strip().lower() or cls.DEFAULT_BACKEND
        if resolved not in cls.VALID_BACKENDS:
            valid = ", ".join(sorted(cls.VALID_BACKENDS))
            raise ValueError(f"Unsupported contact backend '{resolved}'. Expected one of: {valid}.")
        return resolved

    @classmethod
    def uses_inner_solver_backend(cls, backend: Optional[str] = None) -> bool:
        return cls.resolve_backend(backend) == cls.BACKEND_INNER_SOLVER

    def __init__(self, cfg: Optional[ContactOperatorConfig] = None):
        self.cfg = cfg or ContactOperatorConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-operators
        self.normal = NormalContactALM(self.cfg.normal)
        self.friction = FrictionContactALM(self.cfg.friction)
        self.friction.link_normal(self.normal)

        # 如果 FrictionContactALM 已经实现平滑摩擦开关，这里做一次同步（有则用，无则跳过）
        if hasattr(self.friction, "set_smooth_friction"):
            try:
                self.friction.set_smooth_friction(self.cfg.use_smooth_friction)  # type: ignore[attr-defined]
            except Exception:
                pass
        elif hasattr(self.friction, "cfg") and hasattr(self.friction.cfg, "use_smooth_friction"):
            try:
                self.friction.cfg.use_smooth_friction = self.cfg.use_smooth_friction  # type: ignore[assignment]
            except Exception:
                pass

        # bookkeeping
        self._built: bool = False
        self._N: int = 0
        self._step: int = 0
        self._meta: Dict[str, np.ndarray] = {}
        self._last_inner_state: Optional[ContactInnerState] = None
        self._graph_warm_start_lambda_n = tf.Variable(
            tf.zeros((0,), dtype=tf.float32),
            trainable=False,
            shape=tf.TensorShape([None]),
            validate_shape=False,
        )
        self._graph_warm_start_lambda_t = tf.Variable(
            tf.zeros((0, 2), dtype=tf.float32),
            trainable=False,
            shape=tf.TensorShape([None, 2]),
            validate_shape=False,
        )
        self._graph_warm_start_valid = tf.Variable(False, trainable=False, dtype=tf.bool)

    def _friction_active(self) -> bool:
        cfg = getattr(self.cfg, "friction", None)
        if cfg is None:
            return False
        if not bool(getattr(cfg, "enabled", True)):
            return False
        try:
            mu_f = float(getattr(cfg, "mu_f", 0.0) or 0.0)
            k_t = float(getattr(cfg, "k_t", 0.0) or 0.0)
        except Exception:
            return True
        if mu_f <= 0.0 or k_t <= 0.0:
            return False
        return True

    def _reset_graph_warm_start(self, contact_count: int = 0):
        contact_count = max(0, int(contact_count or 0))
        self._graph_warm_start_lambda_n.assign(tf.zeros((contact_count,), dtype=tf.float32))
        self._graph_warm_start_lambda_t.assign(tf.zeros((contact_count, 2), dtype=tf.float32))
        self._graph_warm_start_valid.assign(False)

    def _graph_warm_start_state(self) -> ContactInnerState:
        return ContactInnerState(
            lambda_n=tf.identity(self._graph_warm_start_lambda_n),
            lambda_t=tf.identity(self._graph_warm_start_lambda_t),
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

    def _store_graph_warm_start(self, state: ContactInnerState):
        self._graph_warm_start_lambda_n.assign(tf.cast(state.lambda_n, tf.float32))
        self._graph_warm_start_lambda_t.assign(tf.cast(state.lambda_t, tf.float32))
        self._graph_warm_start_valid.assign(True)

    # ---------- build per batch ----------

    def build_from_cat(
        self,
        cat: Dict[str, np.ndarray],
        extra_weights: Optional[np.ndarray] = None,
        auto_orient: bool = True,
    ):
        """
        Build both normal and friction operators from concatenated contact arrays.

        Parameters
        ----------
        cat : dict
            必须包含键: "xs", "xm", "n", "t1", "t2", "w_area"
        extra_weights : np.ndarray, shape (N,), optional
            额外的加权（如 weighted PINN 或 IRLS 权重），会与 w_area 相乘。
        auto_orient : bool
            若为 True，normal ALM 会在 build 阶段根据零位移间隙自动翻转法向。
        """
        required = ["xs", "xm", "n", "t1", "t2", "w_area"]
        for k in required:
            if k not in cat:
                raise KeyError(f"[ContactOperator] cat missing key '{k}'")

        # normal
        interp_keys = ["xs_node_idx", "xs_bary", "xm_node_idx", "xm_bary"]
        normal_cat = {"xs": cat["xs"], "xm": cat["xm"], "n": cat["n"], "w_area": cat["w_area"]}
        for k in interp_keys:
            if k in cat:
                normal_cat[k] = cat[k]
        self.normal.build_from_cat(
            normal_cat,
            extra_weights=extra_weights,
            auto_orient=auto_orient,
        )

        # friction (linked to normal)
        fric_cat = {"xs": cat["xs"], "xm": cat["xm"], "t1": cat["t1"], "t2": cat["t2"], "w_area": cat["w_area"]}
        for k in interp_keys:
            if k in cat:
                fric_cat[k] = cat[k]
        if self._friction_active():
            self.friction.build_from_cat(
                fric_cat,
                extra_weights=extra_weights,
            )
        else:
            self.friction.reset_for_new_batch()

        self._N = int(cat["xs"].shape[0])
        self._built = True
        self._step = 0
        keep_keys = ["pair_id", "slave_tri_idx", "master_tri_idx", "w_area", "xs", "xm", "n", "t1", "t2"]
        self._meta = {k: v for k, v in cat.items() if k in keep_keys}
        self._last_inner_state = None
        self._reset_graph_warm_start(self._N)

    def reset_for_new_batch(self):
        """Clear internal state so you can rebuild with a new set of contact samples."""
        self.normal.reset_for_new_batch()
        self.friction.reset_for_new_batch()
        self._built = False
        self._N = 0
        self._step = 0
        self._meta = {}
        self._last_inner_state = None
        self._reset_graph_warm_start(0)

    def reset_multipliers(self, reset_reference: bool = True):
        """Reset ALM multipliers without changing the current contact samples."""
        if not self._built:
            return
        if hasattr(self.normal, "reset_multipliers"):
            self.normal.reset_multipliers()
        if hasattr(self.friction, "reset_multipliers"):
            self.friction.reset_multipliers(reset_reference=reset_reference)
        self._step = 0
        self._last_inner_state = None
        self._reset_graph_warm_start(self._N)

    # ---------- energy & update ----------

    def energy(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Compute total contact energy and return:

            E_contact_total, part_dict, stats_cn, stats_ct

        Returns
        -------
        E_contact_total : tf.Tensor (scalar)
            总接触能量 En + Et
        part_dict : dict
            {"E_n": En, "E_t": Et}
        stats_cn : dict
            法向 ALM 的统计量（由 NormalContactALM.energy 返回）
        stats_ct : dict
            摩擦 ALM 的统计量（由 FrictionContactALM.energy 返回）

        注意：此函数是可微的，通常直接参与 PINN 损失；ALM 乘子更新在
              `update_multipliers` 中完成（不可微）。
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before energy().")

        En, stats_cn = self.normal.energy(u_fn, params, u_nodes=u_nodes)
        if self._friction_active():
            Et, stats_ct = self.friction.energy(u_fn, params, u_nodes=u_nodes)
        else:
            Et = tf.cast(0.0, self.dtype)
            stats_ct = {}

        E = En + Et
        parts = {"E_n": En, "E_t": Et}

        return E, parts, stats_cn, stats_ct

    def residual(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Residual-only contact term (normal + friction), without energy semantics.
        Returns:
            L_contact_total, part_dict, stats_cn, stats_ct
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before residual().")

        L_n, stats_cn = self.normal.residual(u_fn, params, u_nodes=u_nodes)
        if self._friction_active():
            L_t, stats_ct = self.friction.residual(u_fn, params, u_nodes=u_nodes)
        else:
            L_t = tf.cast(0.0, self.dtype)
            stats_ct = {}

        L = L_n + L_t
        parts = {"E_n": L_n, "E_t": L_t}
        return L, parts, stats_cn, stats_ct

    def update_multipliers(self, u_fn, params=None, *, u_nodes: Optional[tf.Tensor] = None):
        """
        Outer-loop ALM update for both normal and friction.

        - 若 cfg.update_every_steps <= 0：每一次调用都更新一次乘子；
        - 否则：只有在 self._step % update_every_steps == 0 时才真正更新。
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before update_multipliers().")

        do_update = False
        if self.cfg.update_every_steps is None or self.cfg.update_every_steps <= 0:
            do_update = True
        else:
            if (self._step % self.cfg.update_every_steps) == 0:
                do_update = True

        if do_update:
            self.normal.update_multipliers(u_fn, params, u_nodes=u_nodes)
            if self._friction_active():
                self.friction.update_multipliers(u_fn, params, u_nodes=u_nodes)

        self._step += 1

    # ---------- residual export (for adaptive sampling) ----------

    def last_sample_metrics(self) -> Dict[str, np.ndarray]:
        """
        Return per-sample residual-like metrics from the latest energy call.

        Useful for residual-adaptive resampling (RAR):
            - "gap": raw normal gap g (negative => penetration)
            - "fric_res": ||r_t|| or |s_t| depending on friction mode
        If contact has not been evaluated since the last build/reset, returns empty dict.
        """

        metrics: Dict[str, np.ndarray] = {}
        if getattr(self.normal, "_last_gap", None) is not None:
            try:
                metrics["gap"] = np.asarray(self.normal._last_gap.numpy()).reshape(-1)
            except Exception:
                pass
        if self._friction_active() and getattr(self.friction, "_last_r_norm", None) is not None:
            try:
                metrics["fric_res"] = np.asarray(self.friction._last_r_norm.numpy()).reshape(-1)
            except Exception:
                pass
        return metrics

    def last_meta(self) -> Dict[str, np.ndarray]:
        """Return shallow-copied metadata for the current batch of contact samples."""

        return dict(self._meta)

    # ---------- staged-loading helpers ----------

    def snapshot_stage_state(self) -> Dict[str, np.ndarray]:
        """Snapshot frictional state so staged preload can carry order-dependent stick/slip."""
        if self._friction_active() and hasattr(self.friction, "snapshot_state"):
            return self.friction.snapshot_state()
        return {}

    def restore_stage_state(self, state: Dict[str, np.ndarray]):
        """Restore frictional state from :meth:`snapshot_stage_state`."""
        if self._friction_active() and hasattr(self.friction, "restore_state"):
            self.friction.restore_state(state)

    def last_friction_slip(self):
        """Expose cached tangential slip for staged path-penalty construction."""
        if self._friction_active() and hasattr(self.friction, "last_slip"):
            return self.friction.last_slip()
        return None

    def current_contact_frame(self) -> Dict[str, tf.Tensor]:
        """Return the current batch geometry/basis tensors for strict mixed assembly."""

        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before requesting contact frame.")

        meta = self._meta or {}
        normals = self.normal.n if getattr(self.normal, "n", None) is not None else meta.get("n")
        xs = self.normal.xs if getattr(self.normal, "xs", None) is not None else meta.get("xs")
        xm = self.normal.xm if getattr(self.normal, "xm", None) is not None else meta.get("xm")
        weights = self.normal.w if getattr(self.normal, "w", None) is not None else meta.get("w_area")
        t1 = getattr(self.friction, "t1", None)
        if t1 is None:
            t1 = meta.get("t1")
        t2 = getattr(self.friction, "t2", None)
        if t2 is None:
            t2 = meta.get("t2")

        if normals is None or xs is None or xm is None or weights is None or t1 is None or t2 is None:
            raise RuntimeError("[ContactOperator] current batch is missing strict mixed contact tensors.")

        return {
            "xs": tf.cast(xs, tf.float32),
            "xm": tf.cast(xm, tf.float32),
            "normals": tf.cast(normals, tf.float32),
            "t1": tf.cast(t1, tf.float32),
            "t2": tf.cast(t2, tf.float32),
            "weights": tf.cast(weights, tf.float32),
        }

    def strict_mixed_inputs(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
    ) -> StrictMixedContactInputs:
        """Assemble the typed strict-mixed adapter contract for the current batch."""

        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before strict_mixed_inputs().")

        frame = self.current_contact_frame()
        g_n = tf.cast(self.normal._gap(u_fn, params, u_nodes=u_nodes), tf.float32)
        if self._friction_active():
            ds_t = tf.cast(
                self.friction._relative_slip_t(u_fn, params, u_nodes=u_nodes, update_cache=True),
                tf.float32,
            )
            mu = tf.cast(self.friction.mu_f, tf.float32)
            k_t = tf.cast(self.friction.k_t, tf.float32)
        else:
            ds_t = tf.zeros((tf.shape(frame["normals"])[0], 2), dtype=tf.float32)
            mu = tf.cast(0.0, tf.float32)
            k_t = tf.cast(0.0, tf.float32)

        if tf.executing_eagerly():
            init_state = _clone_inner_state(self._last_inner_state)
            init_state_available = init_state is not None
        else:
            init_state = self._graph_warm_start_state()
            init_state_available = tf.identity(self._graph_warm_start_valid)
        meta = self._meta or {}
        contact_ids = meta.get("pair_id")
        if contact_ids is not None:
            contact_ids = tf.cast(contact_ids, tf.int32)
        batch_meta = {
            "weights": frame["weights"],
            "xs": frame["xs"],
            "xm": frame["xm"],
        }

        return StrictMixedContactInputs(
            g_n=g_n,
            ds_t=ds_t,
            normals=frame["normals"],
            t1=frame["t1"],
            t2=frame["t2"],
            weights=frame["weights"],
            xs=frame["xs"],
            xm=frame["xm"],
            mu=mu,
            eps_n=tf.cast(getattr(self.normal.cfg, "fb_eps", 1.0e-8), tf.float32),
            k_t=k_t,
            init_state=init_state,
            init_state_available=init_state_available,
            contact_ids=contact_ids,
            batch_meta=batch_meta,
        )

    def solve_strict_inner(
        self,
        u_fn,
        params=None,
        *,
        u_nodes: Optional[tf.Tensor] = None,
        strict_inputs: Optional[StrictMixedContactInputs] = None,
        return_linearization: bool = False,
        return_iteration_trace: bool = False,
        tol_n: float = 1.0e-5,
        tol_t: float = 1.0e-5,
        tol_fb: Optional[float] = None,
        max_inner_iters: int = 8,
        max_tail_qn_iters: int = 0,
        max_inner_iters_signature_gate: str = "",
        signature_gated_max_inner_iters: int = 0,
        damping: float = 1.0,
        normal_correction_cap_scale: float = 1.0,
    ) -> ContactInnerResult:
        """Solve strict mixed inner state from the current contact batch and cache warm start."""

        if strict_inputs is None:
            strict_inputs = self.strict_mixed_inputs(u_fn, params, u_nodes=u_nodes)
        elif not isinstance(strict_inputs, StrictMixedContactInputs):
            strict_inputs = StrictMixedContactInputs(**strict_inputs)
        init_state = strict_inputs.init_state
        init_state_available = strict_inputs.init_state_available
        if init_state is None:
            if tf.executing_eagerly():
                init_state = self._last_inner_state
                init_state_available = init_state is not None
            else:
                init_state = self._graph_warm_start_state()
                init_state_available = tf.identity(self._graph_warm_start_valid)
        elif init_state_available is None:
            init_state_available = init_state is not None
        result = solve_contact_inner(
            strict_inputs.g_n,
            strict_inputs.ds_t,
            strict_inputs.normals,
            strict_inputs.t1,
            strict_inputs.t2,
            mu=strict_inputs.mu,
            eps_n=strict_inputs.eps_n,
            k_t=strict_inputs.k_t,
            init_state=init_state,
            init_state_available=init_state_available,
            return_linearization=return_linearization,
            return_iteration_trace=return_iteration_trace,
            tol_n=tol_n,
            tol_t=tol_t,
            tol_fb=tol_fb,
            max_inner_iters=max_inner_iters,
            max_tail_qn_iters=max_tail_qn_iters,
            damping=damping,
            normal_correction_cap_scale=normal_correction_cap_scale,
        )
        gate_name = str(max_inner_iters_signature_gate or "").strip().lower()
        gate_applied = False
        gated_inner_iters = max(0, int(signature_gated_max_inner_iters or 0))
        if gated_inner_iters > int(max_inner_iters) and _matches_inner_budget_signature(result, gate_name):
            result = solve_contact_inner(
                strict_inputs.g_n,
                strict_inputs.ds_t,
                strict_inputs.normals,
                strict_inputs.t1,
                strict_inputs.t2,
                mu=strict_inputs.mu,
                eps_n=strict_inputs.eps_n,
                k_t=strict_inputs.k_t,
                init_state=init_state,
                init_state_available=init_state_available,
                return_linearization=return_linearization,
                return_iteration_trace=return_iteration_trace,
                tol_n=tol_n,
                tol_t=tol_t,
                tol_fb=tol_fb,
                max_inner_iters=gated_inner_iters,
                max_tail_qn_iters=max_tail_qn_iters,
                damping=damping,
                normal_correction_cap_scale=normal_correction_cap_scale,
            )
            gate_applied = True
        result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        result.diagnostics["signature_gate_applied"] = tf.cast(1.0 if gate_applied else 0.0, tf.float32)
        result.diagnostics["signature_gate_name"] = gate_name if gate_applied else ""
        self._store_graph_warm_start(result.state)
        if tf.executing_eagerly():
            self._last_inner_state = _clone_inner_state(result.state)
        else:
            # Symbolic tensors cannot be kept in Python cache across tf.function traces.
            self._last_inner_state = None
        return result

    def solve_inner_state(
        self,
        lambda_n: tf.Tensor,
        lambda_t: tf.Tensor,
        normals: tf.Tensor,
        t1: tf.Tensor,
        t2: tf.Tensor,
        *,
        force_fail: bool = False,
    ) -> ContactInnerResult:
        """Legacy compatibility wrapper for callers that still provide precomputed lambdas."""

        lambda_n = tf.cast(lambda_n, tf.float32)
        lambda_t = tf.cast(lambda_t, tf.float32)
        normals = tf.cast(normals, tf.float32)
        t1 = tf.cast(t1, tf.float32)
        t2 = tf.cast(t2, tf.float32)

        if force_fail and self._last_inner_state is not None:
            state = ContactInnerState(
                lambda_n=tf.cast(self._last_inner_state.lambda_n, tf.float32),
                lambda_t=tf.cast(self._last_inner_state.lambda_t, tf.float32),
                converged=False,
                iters=0,
                res_norm=getattr(self._last_inner_state, "res_norm", 0.0),
                fallback_used=True,
            )
        else:
            state = ContactInnerState(
                lambda_n=lambda_n,
                lambda_t=lambda_t,
                converged=not force_fail,
                iters=1,
                res_norm=0.0,
                fallback_used=False,
            )

        traction_vec = compose_contact_traction(state.lambda_n, state.lambda_t, normals, t1, t2)
        result = ContactInnerResult(
            state=state,
            traction_vec=traction_vec,
            traction_tangent=state.lambda_t,
            diagnostics={
                "fn_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_n))),
                "ft_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_t))),
                "fallback_used": tf.cast(1.0 if state.fallback_used else 0.0, tf.float32),
            },
        )
        self._store_graph_warm_start(result.state)
        if result.state.converged and not result.state.fallback_used and tf.executing_eagerly():
            self._last_inner_state = _clone_inner_state(result.state)
        return result

    # ---------- schedules / setters ----------

    def set_beta(self, beta: float):
        """Set softplus steepness for normal contact."""
        self.normal.set_beta(beta)

    def set_mu_n(self, mu_n: float):
        """Set ALM coefficient for normal part."""
        self.normal.set_mu_n(mu_n)

    def set_mu_t(self, mu_t: float):
        """Set ALM coefficient for tangential residual energy."""
        self.friction.set_mu_t(mu_t)

    def set_k_t(self, k_t: float):
        """Set tangential penalty stiffness for trial traction."""
        self.friction.set_k_t(k_t)

    def set_mu_f(self, mu_f: float):
        """Set Coulomb friction coefficient μ_f."""
        # 小心之前版本里的拼写错误，这里直接对 variable 赋值
        self.friction.mu_f.assign(tf.cast(mu_f, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """
        Multiply extra weights to both normal and friction energies (Weighted PINN hook).

        注意：这是在 build 之后、训练过程中「再叠一层」权重的接口，
        典型用法是 IRLS 或残差自适应加权。
        """
        self.normal.multiply_weights(extra_w)
        self.friction.multiply_weights(extra_w)

    # ---------- convenience ----------

    @property
    def N(self) -> int:
        return self._N

    @property
    def built(self) -> bool:
        return self._built


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    # 仅做 API 连通性检查，不跑真实接触（缺少 u_fn）
    import numpy as np

    N = 10
    cat = {
        "xs": np.random.randn(N, 3),
        "xm": np.random.randn(N, 3),
        "n": np.tile(np.array([0.0, 0.0, 1.0]), (N, 1)),
        "t1": np.tile(np.array([1.0, 0.0, 0.0]), (N, 1)),
        "t2": np.tile(np.array([0.0, 1.0, 0.0]), (N, 1)),
        "w_area": np.ones((N,), dtype=np.float64),
    }

    op = ContactOperator()
    op.build_from_cat(cat, extra_weights=None, auto_orient=True)

    # dummy u_fn: zero displacement
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.zeros_like(X)

    E_c, parts_c, stats_cn, stats_ct = op.energy(u_fn)
    print("E_contact =", float(E_c.numpy()))
    print("parts:", {k: float(v.numpy()) for k, v in parts_c.items()})
    print("stats_cn keys:", list(stats_cn.keys()))
    print("stats_ct keys:", list(stats_ct.keys()))
