# -*- coding: utf-8 -*-
"""
elasticity_residual.py
----------------------
Residual-based linear elasticity using PINN-style autograd.

Outputs:
  - energy(): strain energy integral over volume points
  - residual_cache(): caches strain/stress and (optionally) div(sigma)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from physics.material_lib import lame_from_E_nu
from physics.elasticity_config import ElasticityConfig


class ElasticityResidual:
    def __init__(
        self,
        asm,
        X_vol: np.ndarray,
        w_vol: np.ndarray,
        mat_id: np.ndarray,
        matlib,
        materials: Dict[str, Any],
        cfg: Optional[ElasticityConfig] = None,
    ):
        self.asm = asm
        self.cfg = cfg or ElasticityConfig()

        self.X_vol = np.asarray(X_vol, dtype=np.float32)
        self.w_vol = np.asarray(w_vol, dtype=np.float32)
        self.mat_id = np.asarray(mat_id, dtype=np.int64)

        self.n_cells = int(self.X_vol.shape[0])
        self._sample_indices: Optional[np.ndarray] = None

        # Node cache (sorted ids to match contact mapping)
        self.sorted_node_ids = np.asarray(sorted(int(n) for n in asm.nodes.keys()), dtype=np.int64)
        self.X_nodes = np.asarray([asm.nodes[int(n)] for n in self.sorted_node_ids], dtype=np.float32)
        self.X_nodes_tf = tf.convert_to_tensor(self.X_nodes, dtype=tf.float32)

        # Material parameters per cell
        enum_names = list(dict.fromkeys(materials.keys()))
        # map material id -> (lam, mu)
        lam_list = []
        mu_list = []
        for tag in matlib.tags:
            spec = materials[tag]
            if isinstance(spec, dict):
                E = float(spec["E"])
                nu = float(spec["nu"])
            else:
                E = float(spec[0])
                nu = float(spec[1])
            lam, mu = lame_from_E_nu(E, nu)
            lam_list.append(lam)
            mu_list.append(mu)

        lam_table = np.asarray(lam_list, dtype=np.float32)
        mu_table = np.asarray(mu_list, dtype=np.float32)

        self.lam = lam_table[self.mat_id]
        self.mu = mu_table[self.mat_id]

        self.X_vol_tf = tf.convert_to_tensor(self.X_vol, dtype=tf.float32)
        self.w_vol_tf = tf.convert_to_tensor(self.w_vol, dtype=tf.float32)
        self.mat_id_tf = tf.convert_to_tensor(self.mat_id, dtype=tf.int64)
        self.lam_tf = tf.convert_to_tensor(self.lam, dtype=tf.float32)
        self.mu_tf = tf.convert_to_tensor(self.mu, dtype=tf.float32)

        self._cache_sample_metrics = bool(
            getattr(self.cfg, "cache_sample_metrics", True)
        )
        self._last_sample_metrics: Optional[Dict[str, Any]] = None

    # ----- public helpers -----

    def set_sample_indices(self, indices: Optional[np.ndarray]):
        if indices is None:
            self._sample_indices = None
            return
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        self._sample_indices = idx

    def set_sample_metrics_cache_enabled(self, enabled: bool):
        self._cache_sample_metrics = bool(enabled)
        if not self._cache_sample_metrics:
            self._last_sample_metrics = None

    def last_sample_metrics(self) -> Optional[Dict[str, Any]]:
        return self._last_sample_metrics

    def _cache_metrics(self, psi: tf.Tensor, idx: Optional[tf.Tensor]) -> None:
        if not self._cache_sample_metrics:
            self._last_sample_metrics = None
            return
        psi_t = tf.cast(psi, tf.float32)
        if idx is None:
            idx_t = tf.range(tf.shape(psi_t)[0], dtype=tf.int64)
        else:
            idx_t = tf.cast(idx, tf.int64)
        self._last_sample_metrics = {"psi": psi_t, "idx": idx_t}

    # ----- core computation -----

    def _select_points(self):
        if self._sample_indices is None:
            return self.X_vol_tf, self.w_vol_tf, self.lam_tf, self.mu_tf, None
        idx = tf.convert_to_tensor(self._sample_indices, dtype=tf.int64)
        X = tf.gather(self.X_vol_tf, idx)
        w = tf.gather(self.w_vol_tf, idx)
        lam = tf.gather(self.lam_tf, idx)
        mu = tf.gather(self.mu_tf, idx)
        return X, w, lam, mu, idx

    def _eval_u_on_nodes(self, u_fn, params):
        return tf.cast(u_fn(self.X_nodes_tf, params), tf.float32)

    def _compute_strain(self, u_fn, params, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if getattr(self.cfg, "use_forward_mode", False):
            return self._compute_strain_forward_mode(u_fn, params, X)
        return self._compute_strain_reverse_mode(u_fn, params, X)

    def _compute_strain_reverse_mode(self, u_fn, params, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            u = tf.cast(u_fn(X, params), tf.float32)  # (N,3)
            u_sum_x = tf.reduce_sum(u[:, 0])
            u_sum_y = tf.reduce_sum(u[:, 1])
            u_sum_z = tf.reduce_sum(u[:, 2])
        # NOTE: u[:,i] is vector-valued; GradientTape requires a scalar target.
        # Use sum-reduction to obtain a dense gradient (PINN-style assumption).
        # For point-wise networks this matches per-sample grads; for graph-coupled
        # models this is an approximation but avoids None gradients.
        du_dx = tape.gradient(
            u_sum_x,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )  # (N,3)
        dv_dx = tape.gradient(
            u_sum_y,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dw_dx = tape.gradient(
            u_sum_z,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del tape

        # Strain components (engineering shear)
        eps_xx = du_dx[:, 0]
        eps_yy = dv_dx[:, 1]
        eps_zz = dw_dx[:, 2]
        gamma_yz = dv_dx[:, 2] + dw_dx[:, 1]
        gamma_xz = du_dx[:, 2] + dw_dx[:, 0]
        gamma_xy = du_dx[:, 1] + dv_dx[:, 0]

        eps_vec = tf.stack([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy], axis=1)
        return u, eps_vec, (du_dx, dv_dx, dw_dx)

    def _compute_strain_forward_mode(self, u_fn, params, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward-mode JVP: compute columns of du/dX with three forward passes.
        This avoids three reverse-mode gradient calls and is typically faster for large N.
        """
        X = tf.cast(X, tf.float32)
        n = tf.shape(X)[0]
        ones = tf.ones((n, 1), dtype=X.dtype)
        zeros = tf.zeros((n, 1), dtype=X.dtype)

        # Tangent basis vectors for x/y/z
        t0 = tf.concat([ones, zeros, zeros], axis=1)
        t1 = tf.concat([zeros, ones, zeros], axis=1)
        t2 = tf.concat([zeros, zeros, ones], axis=1)

        with tf.autodiff.ForwardAccumulator(primals=X, tangents=t0) as acc0:
            u0 = tf.cast(u_fn(X, params), tf.float32)
        jvp0 = tf.cast(acc0.jvp(u0), tf.float32)  # d u / d x

        with tf.autodiff.ForwardAccumulator(primals=X, tangents=t1) as acc1:
            u1 = tf.cast(u_fn(X, params), tf.float32)
        jvp1 = tf.cast(acc1.jvp(u1), tf.float32)  # d u / d y

        with tf.autodiff.ForwardAccumulator(primals=X, tangents=t2) as acc2:
            u2 = tf.cast(u_fn(X, params), tf.float32)
        jvp2 = tf.cast(acc2.jvp(u2), tf.float32)  # d u / d z

        # Assemble rows: du_dx, dv_dx, dw_dx (each N x 3)
        du_dx = tf.stack([jvp0[:, 0], jvp1[:, 0], jvp2[:, 0]], axis=1)
        dv_dx = tf.stack([jvp0[:, 1], jvp1[:, 1], jvp2[:, 1]], axis=1)
        dw_dx = tf.stack([jvp0[:, 2], jvp1[:, 2], jvp2[:, 2]], axis=1)

        # Strain components (engineering shear)
        eps_xx = du_dx[:, 0]
        eps_yy = dv_dx[:, 1]
        eps_zz = dw_dx[:, 2]
        gamma_yz = dv_dx[:, 2] + dw_dx[:, 1]
        gamma_xz = du_dx[:, 2] + dw_dx[:, 0]
        gamma_xy = du_dx[:, 1] + dv_dx[:, 0]

        eps_vec = tf.stack([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy], axis=1)
        return u0, eps_vec, (du_dx, dv_dx, dw_dx)

        # Strain components (engineering shear)
        eps_xx = du_dx[:, 0]
        eps_yy = dv_dx[:, 1]
        eps_zz = dw_dx[:, 2]
        gamma_yz = dv_dx[:, 2] + dw_dx[:, 1]
        gamma_xz = du_dx[:, 2] + dw_dx[:, 0]
        gamma_xy = du_dx[:, 1] + dv_dx[:, 0]

        eps_vec = tf.stack([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy], axis=1)
        return u, eps_vec, (du_dx, dv_dx, dw_dx)

    def _sigma_from_eps(self, eps_vec, lam, mu):
        tr = eps_vec[:, 0] + eps_vec[:, 1] + eps_vec[:, 2]
        sigma_xx = lam * tr + 2.0 * mu * eps_vec[:, 0]
        sigma_yy = lam * tr + 2.0 * mu * eps_vec[:, 1]
        sigma_zz = lam * tr + 2.0 * mu * eps_vec[:, 2]
        sigma_yz = mu * eps_vec[:, 3]
        sigma_xz = mu * eps_vec[:, 4]
        sigma_xy = mu * eps_vec[:, 5]
        sigma = tf.stack([sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy], axis=1)
        return sigma

    @staticmethod
    def _eval_sigma_output(sigma_fn, X: tf.Tensor, params) -> tf.Tensor:
        out = sigma_fn(X, params)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            out = out[1]
        return tf.cast(out, tf.float32)

    def constitutive_residual(self, u_fn, sigma_fn, params):
        """Mixed constitutive residual: sigma_pred - C:epsilon(u)."""

        X, _, lam, mu, _ = self._select_points()
        if self.cfg.coord_scale and self.cfg.coord_scale != 1.0:
            X = X * tf.cast(self.cfg.coord_scale, X.dtype)

        _, eps_vec, _ = self._compute_strain(u_fn, params, X)
        sigma_pred = self._eval_sigma_output(sigma_fn, X, params)
        sigma_phys = self._sigma_from_eps(eps_vec, lam, mu)
        return sigma_pred - sigma_phys

    def equilibrium_residual(self, sigma_fn, params):
        """Mixed equilibrium residual: div(sigma)."""

        X, _, _, _, _ = self._select_points()
        if self.cfg.coord_scale and self.cfg.coord_scale != 1.0:
            X = X * tf.cast(self.cfg.coord_scale, X.dtype)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            sigma = self._eval_sigma_output(sigma_fn, X, params)
            sigma_sum_xx = tf.reduce_sum(sigma[:, 0])
            sigma_sum_yy = tf.reduce_sum(sigma[:, 1])
            sigma_sum_zz = tf.reduce_sum(sigma[:, 2])
            sigma_sum_yz = tf.reduce_sum(sigma[:, 3])
            sigma_sum_xz = tf.reduce_sum(sigma[:, 4])
            sigma_sum_xy = tf.reduce_sum(sigma[:, 5])

        dsig_xx = tape.gradient(
            sigma_sum_xx,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dsig_yy = tape.gradient(
            sigma_sum_yy,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dsig_zz = tape.gradient(
            sigma_sum_zz,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dsig_yz = tape.gradient(
            sigma_sum_yz,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dsig_xz = tape.gradient(
            sigma_sum_xz,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        dsig_xy = tape.gradient(
            sigma_sum_xy,
            X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del tape

        div_x = dsig_xx[:, 0] + dsig_xy[:, 1] + dsig_xz[:, 2]
        div_y = dsig_xy[:, 0] + dsig_yy[:, 1] + dsig_yz[:, 2]
        div_z = dsig_xz[:, 0] + dsig_yz[:, 1] + dsig_zz[:, 2]
        return tf.stack([div_x, div_y, div_z], axis=1)

    def mixed_residual_terms(self, u_fn, sigma_fn, params, *, return_cache: bool = False):
        """Canonical mixed residual contract for strict mixed training."""

        X, w, lam, mu, _ = self._select_points()
        if self.cfg.coord_scale and self.cfg.coord_scale != 1.0:
            X = X * tf.cast(self.cfg.coord_scale, X.dtype)

        _, eps_vec, _ = self._compute_strain(u_fn, params, X)
        sigma_phys = self._sigma_from_eps(eps_vec, lam, mu)
        sigma_pred = self._eval_sigma_output(sigma_fn, X, params)
        r_const = sigma_pred - sigma_phys
        r_eq = self.equilibrium_residual(sigma_fn, params)

        terms = {
            "R_const": r_const,
            "R_eq": r_eq,
        }
        if not return_cache:
            return terms

        terms["cache"] = {
            "eps_vec": eps_vec,
            "sigma_pred": sigma_pred,
            "sigma_phys": sigma_phys,
            "div_sigma": r_eq,
            "w_sel": w,
        }
        return terms

    def energy(self, u_fn, params, tape=None, return_cache: bool = False, u_nodes=None):
        X, w, lam, mu, idx = self._select_points()
        if self.cfg.coord_scale and self.cfg.coord_scale != 1.0:
            X = X * tf.cast(self.cfg.coord_scale, X.dtype)

        u, eps_vec, _ = self._compute_strain(u_fn, params, X)
        tr = eps_vec[:, 0] + eps_vec[:, 1] + eps_vec[:, 2]
        eps_sqr = (
            eps_vec[:, 0] ** 2
            + eps_vec[:, 1] ** 2
            + eps_vec[:, 2] ** 2
            + 0.5 * (eps_vec[:, 3] ** 2 + eps_vec[:, 4] ** 2 + eps_vec[:, 5] ** 2)
        )
        psi = 0.5 * lam * tr * tr + mu * eps_sqr
        E_int = tf.reduce_sum(w * psi)

        stats = {
            "psi_mean": tf.reduce_mean(psi),
            "psi_max": tf.reduce_max(psi),
            "vol_sum": tf.reduce_sum(w),
        }

        # Cache for RAR (tensor form; materialize on host only when needed).
        self._cache_metrics(psi, idx)

        if not return_cache:
            return E_int, stats
        elastic_cache = {
            "eps_vec": eps_vec,
            "lam": lam,
            "mu": mu,
        }
        return E_int, stats, elastic_cache

    def residual_cache(
        self,
        u_fn,
        params,
        stress_fn=None,
        need_sigma: bool = False,
        need_eq: bool = False,
    ):
        X, w, lam, mu, idx = self._select_points()
        if self.cfg.coord_scale and self.cfg.coord_scale != 1.0:
            X = X * tf.cast(self.cfg.coord_scale, X.dtype)

        # Always compute first derivatives (strain)
        u, eps_vec, _ = self._compute_strain(u_fn, params, X)
        sigma_phys = self._sigma_from_eps(eps_vec, lam, mu)
        tr = eps_vec[:, 0] + eps_vec[:, 1] + eps_vec[:, 2]
        eps_sqr = (
            eps_vec[:, 0] ** 2
            + eps_vec[:, 1] ** 2
            + eps_vec[:, 2] ** 2
            + 0.5 * (eps_vec[:, 3] ** 2 + eps_vec[:, 4] ** 2 + eps_vec[:, 5] ** 2)
        )
        psi = 0.5 * lam * tr * tr + mu * eps_sqr

        sigma_pred = None
        if stress_fn is not None and need_sigma:
            out = stress_fn(X, params)
            # stress head may return (u, sigma); accept both forms
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                out = out[1]
            sigma_pred = tf.cast(out, tf.float32)

        div_sigma = None
        if need_eq:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(X)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(X)
                    u2 = tf.cast(u_fn(X, params), tf.float32)
                    u2_sum_x = tf.reduce_sum(u2[:, 0])
                    u2_sum_y = tf.reduce_sum(u2[:, 1])
                    u2_sum_z = tf.reduce_sum(u2[:, 2])
                du_dx = tape1.gradient(
                    u2_sum_x,
                    X,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                dv_dx = tape1.gradient(
                    u2_sum_y,
                    X,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                dw_dx = tape1.gradient(
                    u2_sum_z,
                    X,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                del tape1
                eps_xx = du_dx[:, 0]
                eps_yy = dv_dx[:, 1]
                eps_zz = dw_dx[:, 2]
                gamma_yz = dv_dx[:, 2] + dw_dx[:, 1]
                gamma_xz = du_dx[:, 2] + dw_dx[:, 0]
                gamma_xy = du_dx[:, 1] + dv_dx[:, 0]
                eps_vec2 = tf.stack([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy], axis=1)
                sigma = self._sigma_from_eps(eps_vec2, lam, mu)
                sigma_sum_xx = tf.reduce_sum(sigma[:, 0])
                sigma_sum_yy = tf.reduce_sum(sigma[:, 1])
                sigma_sum_zz = tf.reduce_sum(sigma[:, 2])
                sigma_sum_yz = tf.reduce_sum(sigma[:, 3])
                sigma_sum_xz = tf.reduce_sum(sigma[:, 4])
                sigma_sum_xy = tf.reduce_sum(sigma[:, 5])

            dsig_xx = tape2.gradient(
                sigma_sum_xx,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            dsig_yy = tape2.gradient(
                sigma_sum_yy,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            dsig_zz = tape2.gradient(
                sigma_sum_zz,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            dsig_yz = tape2.gradient(
                sigma_sum_yz,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            dsig_xz = tape2.gradient(
                sigma_sum_xz,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            dsig_xy = tape2.gradient(
                sigma_sum_xy,
                X,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            del tape2

            div_x = dsig_xx[:, 0] + dsig_xy[:, 1] + dsig_xz[:, 2]
            div_y = dsig_xy[:, 0] + dsig_yy[:, 1] + dsig_yz[:, 2]
            div_z = dsig_xz[:, 0] + dsig_yz[:, 1] + dsig_zz[:, 2]
            div_sigma = tf.stack([div_x, div_y, div_z], axis=1)

        stats = {
            "psi_mean": tf.reduce_mean(psi),
            "psi_max": tf.reduce_max(psi),
            "vol_sum": tf.reduce_sum(w),
        }

        # Cache for RAR (tensor form; materialize on host only when needed).
        self._cache_metrics(psi, idx)

        elastic_cache = {
            "eps_vec": eps_vec,
            "sigma_phys": sigma_phys,
            "sigma_pred": sigma_pred,
            "div_sigma": div_sigma,
            "w_sel": w,
        }
        return stats, elastic_cache
