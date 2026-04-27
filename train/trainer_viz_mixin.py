# -*- coding: utf-8 -*-
"""Visualization mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from viz.mirror_viz import plot_mirror_deflection_by_name
from viz.mirror_viz import (
    _fit_plane_basis,
    _interpolate_displacement_on_refined,
    _project_to_plane,
    _refine_surface_samples,
    _remove_rigid_body_motion,
    _smooth_scalar_on_tri_mesh,
    _smooth_vector_on_tri_mesh,
    _unique_nodes_from_tris,
)
from mesh.surface_utils import compute_tri_geometry, resolve_surface_to_tris, triangulate_part_boundary


class TrainerVizMixin:
    @staticmethod
    def _preload_case_key(case: Dict[str, Any]) -> Optional[Tuple[Tuple[float, ...], Tuple[int, ...]]]:
        if not isinstance(case, dict):
            return None
        if "P" not in case:
            return None
        p = np.asarray(case.get("P"), dtype=np.float64).reshape(-1)
        if p.size == 0:
            return None
        order = np.asarray(case.get("order", []), dtype=np.int32).reshape(-1)
        p_key = tuple(float(np.round(x, 6)) for x in p.tolist())
        order_key = tuple(int(x) for x in order.tolist())
        return p_key, order_key

    def _match_supervision_case_for_preload(self, preload_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dataset = getattr(self, "_supervision_dataset", None)
        if dataset is None:
            return None
        target_key = self._preload_case_key(preload_case)
        if target_key is None:
            return None

        cases_by_split = getattr(dataset, "cases_by_split", {}) or {}
        for split_name in sorted(cases_by_split.keys()):
            for case in list(cases_by_split.get(split_name, []) or []):
                if self._preload_case_key(case) == target_key:
                    return copy.deepcopy(case)
        return None

    def _build_surface_only_viz_asm(self, node_ids: np.ndarray):
        asm = getattr(self, "asm", None)
        if asm is None:
            return None

        node_ids_arr = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids_arr.size == 0:
            return None

        asm_nodes = getattr(asm, "nodes", None)
        if not isinstance(asm_nodes, dict):
            return None

        subset_nodes = {
            int(nid): tuple(asm_nodes[int(nid)])
            for nid in node_ids_arr.tolist()
            if int(nid) in asm_nodes
        }
        if len(subset_nodes) != int(node_ids_arr.size):
            return None

        subset_parts: Dict[str, Any] = {}
        for part_name, part in (getattr(asm, "parts", {}) or {}).items():
            nodes_xyz = getattr(part, "nodes_xyz", None)
            if not isinstance(nodes_xyz, dict):
                continue
            part_subset = {
                int(nid): tuple(nodes_xyz[int(nid)])
                for nid in node_ids_arr.tolist()
                if int(nid) in nodes_xyz
            }
            if not part_subset:
                continue
            part_copy = copy.copy(part)
            part_copy.nodes_xyz = part_subset
            if hasattr(part_copy, "node_ids"):
                part_copy.node_ids = list(part_subset.keys())
            subset_parts[str(part_name)] = part_copy

        asm_copy = copy.copy(asm)
        asm_copy.nodes = subset_nodes
        asm_copy.parts = subset_parts
        asm_copy.surfaces = getattr(asm, "surfaces", {})
        return asm_copy

    @staticmethod
    def _make_supervision_truth_lookup_u_fn(xyz: np.ndarray, u_vec: np.ndarray):
        xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        u_arr = np.asarray(u_vec, dtype=np.float32).reshape(-1, 3)
        if xyz_arr.shape[0] != u_arr.shape[0]:
            raise ValueError("xyz and u_vec must contain the same number of rows")

        rounded = np.round(xyz_arr, decimals=8)
        lookup = {
            tuple(float(v) for v in rounded[idx].tolist()): u_arr[idx]
            for idx in range(rounded.shape[0])
        }

        def _u_fn(X, params=None):
            x_np = np.asarray(X.numpy() if hasattr(X, "numpy") else X, dtype=np.float64).reshape(-1, 3)
            out = np.zeros((x_np.shape[0], 3), dtype=np.float32)
            rounded_query = np.round(x_np, decimals=8)
            for idx in range(rounded_query.shape[0]):
                key = tuple(float(v) for v in rounded_query[idx].tolist())
                vec = lookup.get(key)
                if vec is None:
                    delta = np.max(np.abs(xyz_arr - x_np[idx]), axis=1)
                    near = int(np.argmin(delta))
                    if float(delta[near]) > 1.0e-6:
                        raise KeyError(f"same-pipeline FEM lookup missing coordinate {x_np[idx].tolist()}")
                    vec = u_arr[near]
                out[idx] = vec
            return tf.convert_to_tensor(out, dtype=tf.float32)

        return _u_fn

    @staticmethod
    def _summarize_supervision_eval_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {}

        rmse = np.asarray([float(row.get("rmse_vec_mm", np.nan)) for row in rows], dtype=np.float64)
        pred = np.asarray([float(row.get("pred_rms_vec_mm", np.nan)) for row in rows], dtype=np.float64)
        true = np.asarray([float(row.get("true_rms_vec_mm", np.nan)) for row in rows], dtype=np.float64)
        eps = 1.0e-12
        ratio = pred / np.maximum(true, eps)
        drrms = rmse / np.maximum(true, eps)

        finite_rmse = rmse[np.isfinite(rmse)]
        finite_ratio = ratio[np.isfinite(ratio)]
        finite_drrms = drrms[np.isfinite(drrms)]
        if finite_rmse.size == 0 or finite_ratio.size == 0 or finite_drrms.size == 0:
            return {}

        case_ids = {str(row.get("case_id", "")).strip() for row in rows if str(row.get("case_id", "")).strip()}
        return {
            "val_rows": float(len(rows)),
            "val_cases": float(len(case_ids)),
            "val_rmse_vec_mm_mean": float(np.mean(finite_rmse)),
            "val_ratio_median": float(np.median(finite_ratio)),
            "val_drrms_mean": float(np.mean(finite_drrms)),
        }

    def _validation_supervision_split_name(self) -> Optional[str]:
        dataset = getattr(self, "_supervision_dataset", None)
        sup_cfg = getattr(self.cfg, "supervision", None)
        if dataset is None or sup_cfg is None:
            return None

        available = getattr(dataset, "cases_by_split", {}) or {}
        eval_splits = [str(x).strip() for x in (getattr(sup_cfg, "eval_splits", ()) or ()) if str(x).strip()]
        preferred: List[str] = []
        if "val" in eval_splits:
            preferred.append("val")
        preferred.extend(name for name in eval_splits if name != "val")
        for name in preferred:
            cases = list(available.get(name, []) or [])
            if cases:
                return name
        return None

    def _compute_validation_supervision_summary(self) -> Optional[Dict[str, float]]:
        dataset = getattr(self, "_supervision_dataset", None)
        supervision_cfg = getattr(self.cfg, "supervision", None)
        if dataset is None or self.model is None or supervision_cfg is None:
            return None
        if not bool(getattr(supervision_cfg, "enabled", False)):
            return None

        split = self._validation_supervision_split_name()
        if not split:
            return None
        cases = list(getattr(dataset, "cases_by_split", {}).get(split, []) or [])
        if not cases:
            return None

        u_eval_fn = self.model.u_fn
        if bool(getattr(self.cfg, "viz_force_pointwise", False)) and hasattr(self.model, "u_fn_pointwise"):
            u_eval_fn = self.model.u_fn_pointwise

        try:
            rows = self._build_supervision_eval_rows(split, cases, u_eval_fn)
        except Exception as exc:
            print(f"[trainer] WARNING: validation supervision eval failed for split={split}: {exc}")
            return None
        summary = self._summarize_supervision_eval_rows(rows)
        if not summary:
            return None
        summary["val_split"] = str(split)
        return summary

    def _build_stage_comparison_mesh(
        self, common_ids: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Reuse the real mirror-surface topology for stage-comparison plots."""

        asm = getattr(self, "asm", None)
        if asm is None or not getattr(asm, "surfaces", None):
            return None

        hint = str(getattr(self.cfg, "mirror_surface_name", "") or "").strip().lower()
        if not hint:
            return None

        surface_key = None
        for key, surf in asm.surfaces.items():
            surf_name = str(getattr(surf, "name", "") or "").strip().lower()
            if hint in str(key).lower() or hint == surf_name:
                surface_key = key
                break
        if surface_key is None:
            return None

        try:
            ts = resolve_surface_to_tris(asm, surface_key, log_summary=False)
        except Exception:
            return None

        source_key = str(getattr(self.cfg, "viz_surface_source", "surface") or "surface").strip().lower()
        part = getattr(asm, "parts", {}).get(ts.part_name)
        if part is None and getattr(asm, "nodes", None):
            class _AsmPart:
                pass

            part = _AsmPart()
            part.nodes_xyz = getattr(asm, "nodes", {})
            part.node_ids = sorted(part.nodes_xyz.keys()) if part.nodes_xyz else []

        if source_key in {"part", "part_top", "part_boundary"} and ts.part_name in getattr(asm, "parts", {}):
            try:
                rebuilt = triangulate_part_boundary(part, ts.part_name, log_summary=False)
            except Exception:
                rebuilt = None
            rebuilt_tri_ids = np.asarray(getattr(rebuilt, "tri_node_ids", []), dtype=np.int64).reshape(-1, 3)
            if rebuilt is not None and rebuilt_tri_ids.shape[0] > 0:
                if source_key == "part_top":
                    try:
                        areas, normals, _ = compute_tri_geometry(part, rebuilt)
                        if areas.size:
                            weighted = (areas[:, None] * normals).sum(axis=0)
                            norm = float(np.linalg.norm(weighted))
                            if norm > 0.0:
                                n_dom = weighted / norm
                                keep = (normals @ n_dom) > 0.2
                                if not np.any(keep):
                                    keep = (normals @ n_dom) >= 0.0
                                if np.any(keep):
                                    rebuilt.tri_node_ids = rebuilt_tri_ids[np.asarray(keep)]
                    except Exception:
                        pass
                ts = rebuilt
                part = getattr(asm, "parts", {}).get(ts.part_name, part)

        if part is None or not getattr(part, "nodes_xyz", None):
            return None

        common = np.asarray(common_ids, dtype=np.int64).reshape(-1)
        if common.size < 3:
            return None
        common_lookup = {int(nid): idx for idx, nid in enumerate(common.tolist())}

        tri_ids = np.asarray(getattr(ts, "tri_node_ids", []), dtype=np.int64).reshape(-1, 3)
        tri_idx: List[List[int]] = []
        for tri_nodes in tri_ids.tolist():
            mapped: List[int] = []
            for nid in tri_nodes:
                pos = common_lookup.get(int(nid))
                if pos is None:
                    mapped = []
                    break
                mapped.append(int(pos))
            if len(mapped) == 3:
                tri_idx.append(mapped)
        if not tri_idx:
            return None

        nodes_xyz = getattr(part, "nodes_xyz", {}) or {}
        asm_nodes = getattr(asm, "nodes", {}) or {}
        xyz: List[np.ndarray] = []
        for nid in common.tolist():
            coord = nodes_xyz.get(int(nid), asm_nodes.get(int(nid)))
            if coord is None:
                return None
            xyz.append(np.asarray(coord, dtype=np.float64))
        x3d = np.asarray(xyz, dtype=np.float64)
        c, e1, e2, _ = _fit_plane_basis(x3d)
        uv = _project_to_plane(x3d, c, e1, e2)
        return uv, np.asarray(tri_idx, dtype=np.int64)

    def _build_stage_comparison_render_state(
        self,
        common_ids: np.ndarray,
        aligned_sample: Dict[str, np.ndarray],
    ):
        """Build the triangulation and scalar field used by stage-comparison plots."""

        from matplotlib.tri import Triangulation

        common = np.asarray(common_ids, dtype=np.int64).reshape(-1)
        if common.size < 3:
            return None

        mesh = self._build_stage_comparison_mesh(common)
        xyz = np.stack(
            [
                np.asarray(aligned_sample["x"], dtype=np.float64),
                np.asarray(aligned_sample["y"], dtype=np.float64),
                np.asarray(aligned_sample["z"], dtype=np.float64),
            ],
            axis=1,
        )

        if mesh is not None:
            uv, tri_idx = mesh
            tri_idx = np.asarray(tri_idx, dtype=np.int64).reshape(-1, 3)
        else:
            u = np.asarray(aligned_sample["u_plane"], dtype=np.float64).reshape(-1)
            v = np.asarray(aligned_sample["v_plane"], dtype=np.float64).reshape(-1)
            uv = np.stack([u, v], axis=1)
            coarse_tri = Triangulation(u, v)
            if coarse_tri.triangles is not None and len(coarse_tri.triangles) > 0:
                cu, cv = float(np.mean(u)), float(np.mean(v))
                r = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
                if np.any(np.isfinite(r)):
                    rmin = float(np.nanmin(r))
                    rmax = float(np.nanmax(r))
                    if rmax > 0.0:
                        r_inner = rmin * 1.02
                        r_outer = rmax * 0.98
                        tris = np.asarray(coarse_tri.triangles, dtype=np.int64)
                        uc = u[tris].mean(axis=1)
                        vc = v[tris].mean(axis=1)
                        rc = np.sqrt((uc - cu) ** 2 + (vc - cv) ** 2)
                        coarse_tri.set_mask((rc < r_inner) | (rc > r_outer))
            tri_idx = np.asarray(coarse_tri.get_masked_triangles(), dtype=np.int64).reshape(-1, 3)

        if tri_idx.shape[0] == 0:
            return None

        u_vec = np.stack(
            [
                np.asarray(aligned_sample["ux"], dtype=np.float64),
                np.asarray(aligned_sample["uy"], dtype=np.float64),
                np.asarray(aligned_sample["uz"], dtype=np.float64),
            ],
            axis=1,
        )

        applied_subdiv = max(0, int(getattr(self.cfg, "viz_refine_subdivisions", 0) or 0))
        max_pts = getattr(self.cfg, "viz_refine_max_points", None)
        if applied_subdiv > 0 and max_pts is not None:
            max_pts = int(max_pts)
            if max_pts > 0:
                per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
                estimate = int(per_tri * tri_idx.shape[0])
                while estimate > max_pts and applied_subdiv > 0:
                    applied_subdiv -= 1
                    per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
                    estimate = int(per_tri * tri_idx.shape[0])

        if applied_subdiv > 0:
            _, uv_plot, tri_plot, bary_w, bary_parent = _refine_surface_samples(
                xyz,
                uv,
                tri_idx,
                applied_subdiv,
                return_barycentric=True,
            )
            # Stage-comparison plots are reconstructed from exported nodal values,
            # so refinement must interpolate from the FE-node field rather than
            # re-query the network at refined samples.
            u_plot = _interpolate_displacement_on_refined(u_vec, tri_idx, bary_parent, bary_w)
        else:
            uv_plot = uv
            tri_plot = tri_idx
            u_plot = u_vec

        tri = Triangulation(uv_plot[:, 0], uv_plot[:, 1], tri_plot)
        d_plot = np.linalg.norm(u_plot, axis=1)

        smoothing_steps = max(0, int(getattr(self.cfg, "viz_smooth_scalar_iters", 0) or 0))
        if smoothing_steps > 0:
            triangles_for_smoothing = tri.get_masked_triangles()
            if triangles_for_smoothing.size > 0 and np.any(np.isfinite(d_plot)):
                d_plot = _smooth_scalar_on_tri_mesh(
                    d_plot,
                    triangles_for_smoothing,
                    iterations=smoothing_steps,
                    lam=float(getattr(self.cfg, "viz_smooth_scalar_lambda", 0.6) or 0.6),
                )

        return tri, d_plot

    def _build_supervision_compare_render_state(
        self,
        node_ids: np.ndarray,
        xyz: np.ndarray,
        u_vec: np.ndarray,
    ):
        """Build a supervision compare render state on the true mirror FE topology."""

        from matplotlib.tri import Triangulation

        node_ids_arr = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        u_base = np.asarray(u_vec, dtype=np.float64).reshape(-1, 3)
        if node_ids_arr.size < 3 or xyz_arr.shape[0] != node_ids_arr.size or u_base.shape[0] != node_ids_arr.size:
            return None

        mesh = self._build_stage_comparison_mesh(node_ids_arr)
        if mesh is not None:
            uv, tri_idx = mesh
            tri_idx = np.asarray(tri_idx, dtype=np.int64).reshape(-1, 3)
        else:
            c, e1, e2, _ = _fit_plane_basis(xyz_arr)
            uv = _project_to_plane(xyz_arr, c, e1, e2)
            coarse_tri = Triangulation(uv[:, 0], uv[:, 1])
            tri_idx = np.asarray(coarse_tri.get_masked_triangles(), dtype=np.int64).reshape(-1, 3)
        if tri_idx.shape[0] == 0:
            return None

        if bool(getattr(self.cfg, "viz_remove_rigid", False)):
            try:
                u_base, _ = _remove_rigid_body_motion(xyz_arr, u_base)
            except Exception:
                pass

        vector_steps = max(0, int(getattr(self.cfg, "viz_smooth_vector_iters", 0) or 0))
        if vector_steps > 0:
            u_base = _smooth_vector_on_tri_mesh(
                u_base,
                tri_idx,
                iterations=vector_steps,
                lam=float(getattr(self.cfg, "viz_smooth_vector_lambda", 0.35) or 0.35),
                preserve_mean=True,
            )

        applied_subdiv = max(0, int(getattr(self.cfg, "viz_refine_subdivisions", 0) or 0))
        max_pts = getattr(self.cfg, "viz_refine_max_points", None)
        if applied_subdiv > 0 and max_pts is not None:
            max_pts = int(max_pts)
            if max_pts > 0:
                per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
                estimate = int(per_tri * tri_idx.shape[0])
                while estimate > max_pts and applied_subdiv > 0:
                    applied_subdiv -= 1
                    per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
                    estimate = int(per_tri * tri_idx.shape[0])

        if applied_subdiv > 0:
            _, uv_plot, tri_plot, bary_w, bary_parent = _refine_surface_samples(
                xyz_arr,
                uv,
                tri_idx,
                applied_subdiv,
                return_barycentric=True,
            )
            u_plot = _interpolate_displacement_on_refined(u_base, tri_idx, bary_parent, bary_w)
        else:
            uv_plot = uv
            tri_plot = tri_idx
            u_plot = u_base

        tri = Triangulation(uv_plot[:, 0], uv_plot[:, 1], tri_plot)
        d_plot = np.linalg.norm(u_plot, axis=1)

        smoothing_steps = max(0, int(getattr(self.cfg, "viz_smooth_scalar_iters", 0) or 0))
        if smoothing_steps > 0:
            triangles_for_smoothing = tri.get_masked_triangles()
            if triangles_for_smoothing.size > 0 and np.any(np.isfinite(d_plot)):
                d_plot = _smooth_scalar_on_tri_mesh(
                    d_plot,
                    triangles_for_smoothing,
                    iterations=smoothing_steps,
                    lam=float(getattr(self.cfg, "viz_smooth_scalar_lambda", 0.6) or 0.6),
                )

        return tri, u_plot, d_plot

    def _call_viz_with_context(
        self,
        asm_obj,
        u_eval_fn,
        P: np.ndarray,
        params: Dict[str, tf.Tensor],
        out_path: str,
        title: str,
    ):
        bare = self.cfg.mirror_surface_name
        data_path = None
        if self.cfg.viz_write_data and out_path:
            data_path = os.path.splitext(out_path)[0] + ".txt"

        mesh_path = None
        if self.cfg.viz_write_surface_mesh and out_path:
            mesh_path = "auto"

        full_plot_enabled = bool(self.cfg.viz_plot_full_structure)
        full_struct_out = "auto" if (full_plot_enabled and out_path) else None
        full_struct_data = (
            "auto" if (full_plot_enabled and self.cfg.viz_write_full_structure_data and out_path) else None
        )

        diag_out: Dict[str, Any] = {} if self.cfg.viz_diagnose_blanks else None
        return plot_mirror_deflection_by_name(
            asm_obj,
            bare,
            u_eval_fn,
            params,
            P_values=tuple(float(x) for x in np.asarray(P, dtype=np.float64).reshape(-1)),
            out_path=out_path,
            render_surface=self.cfg.viz_surface_enabled,
            surface_source=self.cfg.viz_surface_source,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            data_out_path=data_path,
            surface_mesh_out_path=mesh_path,
            plot_full_structure=full_plot_enabled,
            full_structure_out_path=full_struct_out,
            full_structure_data_out_path=full_struct_data,
            full_structure_part=self.cfg.viz_full_structure_part,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            use_shape_function_interp=self.cfg.viz_use_shape_function_interp,
            smooth_vector_iters=self.cfg.viz_smooth_vector_iters,
            smooth_vector_lambda=self.cfg.viz_smooth_vector_lambda,
            smooth_scalar_iters=self.cfg.viz_smooth_scalar_iters,
            smooth_scalar_lambda=self.cfg.viz_smooth_scalar_lambda,
            retriangulate_2d=self.cfg.viz_retriangulate_2d,
            eval_batch_size=self.cfg.viz_eval_batch_size,
            eval_scope=self.cfg.viz_eval_scope,
            diagnose_blanks=self.cfg.viz_diagnose_blanks,
            auto_fill_blanks=self.cfg.viz_auto_fill_blanks,
            remove_rigid=self.cfg.viz_remove_rigid,
            diag_out=diag_out,
        )

    def _call_viz(self, P: np.ndarray, params: Dict[str, tf.Tensor], out_path: str, title: str):
        u_eval_fn = self.model.u_fn
        if bool(getattr(self.cfg, "viz_force_pointwise", False)) and hasattr(self.model, "u_fn_pointwise"):
            u_eval_fn = self.model.u_fn_pointwise

        return self._call_viz_with_context(self.asm, u_eval_fn, P, params, out_path, title)

    def _write_same_pipeline_supervision_debug_exports(
        self,
        *,
        case_index: int,
        preload_case: Dict[str, Any],
        params_full: Dict[str, Any],
        suffix: str,
        title_prefix: str,
    ) -> List[str]:
        if not bool(getattr(self.cfg, "viz_same_pipeline_supervision_debug", False)):
            return []
        if getattr(self, "asm", None) is None or getattr(self, "model", None) is None:
            return []

        matched = self._match_supervision_case_for_preload(preload_case)
        if matched is None:
            print("[viz] same-pipeline debug skipped: no matching supervision case found.")
            return []

        node_ids = np.asarray(matched.get("node_ids", []), dtype=np.int64).reshape(-1)
        X_all = np.asarray(matched.get("X_obs", []), dtype=np.float32)
        U_all = np.asarray(matched.get("U_obs", []), dtype=np.float32)
        if node_ids.size == 0 or X_all.ndim != 3 or U_all.ndim != 3 or X_all.shape != U_all.shape:
            print("[viz] same-pipeline debug skipped: matched supervision case is missing staged field data.")
            return []

        asm_debug = self._build_surface_only_viz_asm(node_ids)
        if asm_debug is None:
            print("[viz] same-pipeline debug skipped: unable to build surface-only visualization assembly.")
            return []

        u_eval_fn = self.model.u_fn
        if bool(getattr(self.cfg, "viz_force_pointwise", False)) and hasattr(self.model, "u_fn_pointwise"):
            u_eval_fn = self.model.u_fn_pointwise

        exported: List[str] = []
        base_stem = os.path.join(self.cfg.out_dir, f"deflection_{int(case_index):02d}{suffix}")

        def _emit(
            out_png: str,
            plot_u_fn,
            plot_params: Dict[str, Any],
            plot_P: np.ndarray,
            plot_title: str,
        ) -> None:
            self._call_viz_with_context(asm_debug, plot_u_fn, plot_P, plot_params, out_png, plot_title)
            exported.append(out_png)

        final_stage_idx = int(U_all.shape[0]) - 1
        final_params = self._extract_stage_params(params_full, final_stage_idx, keep_context=True)
        final_truth_fn = self._make_supervision_truth_lookup_u_fn(X_all[final_stage_idx], U_all[final_stage_idx])
        stage_values = np.asarray(preload_case.get("stages", []), dtype=np.float32)
        final_P = stage_values[final_stage_idx] if stage_values.ndim == 2 else np.asarray(preload_case.get("P"), dtype=np.float32)

        print(
            f"[viz] same-pipeline debug matched supervision case={matched.get('case_id')} "
            f"(source={matched.get('source', '')}, split={matched.get('split', '')})"
        )
        _emit(
            base_stem + "_samepipe_pinn.png",
            u_eval_fn,
            final_params,
            final_P,
            f"{title_prefix} [samepipe PINN]",
        )
        _emit(
            base_stem + "_samepipe_fem.png",
            final_truth_fn,
            {},
            final_P,
            f"{title_prefix} [samepipe FEM]",
        )

        if self.cfg.viz_plot_stages and stage_values.ndim == 2 and stage_values.shape[0] == U_all.shape[0]:
            stage_indices = self._resolve_stage_plot_indices(preload_case, int(stage_values.shape[0]))
            for s in stage_indices:
                params_s = self._extract_stage_params(params_full, int(s), keep_context=True)
                truth_fn_s = self._make_supervision_truth_lookup_u_fn(X_all[int(s)], U_all[int(s)])
                stage_P = stage_values[int(s)]
                base_stage = f"{base_stem}_s{int(s) + 1}"
                _emit(
                    base_stage + "_samepipe_pinn.png",
                    u_eval_fn,
                    params_s,
                    stage_P,
                    f"{title_prefix} [samepipe PINN stage={int(s) + 1}]",
                )
                _emit(
                    base_stage + "_samepipe_fem.png",
                    truth_fn_s,
                    {},
                    stage_P,
                    f"{title_prefix} [samepipe FEM stage={int(s) + 1}]",
                )

        return exported

    def _fixed_viz_preload_cases(self) -> List[Dict[str, np.ndarray]]:
        """生成固定拧紧角案例以避免可视化阶段的随机性."""

        nb = int(getattr(self, "_preload_dim", 0) or len(self.cfg.preload_specs) or 1)
        lo = float(self.cfg.preload_min)
        hi = float(self.cfg.preload_max)
        mid = 0.5 * (lo + hi)

        def _make_case(P_list: Sequence[float], order: Sequence[int]) -> Dict[str, np.ndarray]:
            P_arr = np.asarray(P_list, dtype=np.float32).reshape(-1)
            if P_arr.size != nb:
                raise ValueError(f"固定可视化需要 {nb} 维角度输入，收到 {P_arr.size} 维。")
            case: Dict[str, np.ndarray] = {"P": P_arr}
            if not self.cfg.preload_use_stages:
                return case
            order_norm = self._normalize_order(order, nb)
            if order_norm is None:
                return case
            case["order"] = order_norm
            case.update(self._build_stage_case(P_arr, order_norm))
            return case

        cases: List[Dict[str, np.ndarray]] = []

        # 单螺母: 仅一个达到 hi，其余为 lo
        for i in range(nb):
            arr = [lo] * nb
            arr[i] = hi
            cases.append(_make_case(arr, order=list(range(nb))))

        # 等幅: 全部为 mid，并给出两种顺序（若 nb>=2）
        cases.append(_make_case([mid] * nb, order=list(range(nb))))
        if nb >= 2:
            cases.append(_make_case([mid] * nb, order=list(reversed(range(nb)))))

        return cases

    def _resolve_viz_cases(self, n_samples: int) -> List[Dict[str, np.ndarray]]:
        """Resolve visualization cases with deterministic defaults.

        By default we use fixed, reproducible cases so exported results can be
        compared against reference datasets across runs. The legacy behavior of
        using the last sampled training case is available via
        ``viz_use_last_training_case=True``.
        """

        use_last = bool(getattr(self.cfg, "viz_use_last_training_case", False))
        if use_last and self._last_preload_case is not None:
            print("[viz] Using last training tightening case for visualization.")
            return [copy.deepcopy(self._last_preload_case)]

        fixed_cases = self._fixed_viz_preload_cases()
        if fixed_cases:
            print("[viz] Using fixed tightening cases for reproducible visualization.")
            return fixed_cases

        if self._last_preload_case is not None:
            print("[viz] Fixed cases unavailable, fallback to last training case.")
            return [copy.deepcopy(self._last_preload_case)]

        return [self._sample_preload_case() for _ in range(n_samples)]

    def _restore_checkpoint_for_export(self, ckpt_path: str) -> None:
        ckpt = getattr(self, "ckpt", None)
        if ckpt is None or not ckpt_path:
            return
        status = ckpt.restore(str(ckpt_path))
        try:
            status.expect_partial()
        except Exception:
            pass

    def _resolve_visual_export_targets(self) -> List[Dict[str, Optional[str]]]:
        out_dir = str(getattr(self.cfg, "out_dir", "") or "").strip() or "."
        if not bool(getattr(self.cfg, "viz_export_final_and_best", False)):
            return [{"tag": "current", "out_dir": out_dir, "ckpt_path": None}]

        final_ckpt = str(getattr(self, "_final_ckpt_path", "") or "").strip()
        best_ckpt = str(getattr(self, "_best_ckpt_path", "") or "").strip()
        targets: List[Dict[str, Optional[str]]] = [
            {
                "tag": "final",
                "out_dir": os.path.join(out_dir, "final"),
                "ckpt_path": final_ckpt or None,
            }
        ]
        if best_ckpt:
            if (not final_ckpt) or (os.path.abspath(best_ckpt) != os.path.abspath(final_ckpt)):
                targets.append(
                    {
                        "tag": "best",
                        "out_dir": os.path.join(out_dir, "best"),
                        "ckpt_path": best_ckpt,
                    }
                )
        return targets

    def _visualize_after_training(self, n_samples: int = 5):
        targets = self._resolve_visual_export_targets()
        if len(targets) == 1 and str(targets[0].get("tag", "")) == "current":
            self._visualize_current_state_to_out_dir(n_samples=n_samples)
            return

        original_out_dir = self.cfg.out_dir
        final_ckpt = str(getattr(self, "_final_ckpt_path", "") or "").strip()
        try:
            for target in targets:
                ckpt_path = str(target.get("ckpt_path", "") or "").strip()
                if ckpt_path:
                    self._restore_checkpoint_for_export(ckpt_path)
                self.cfg.out_dir = str(target.get("out_dir", original_out_dir) or original_out_dir)
                print(f"[viz] export target={target.get('tag', 'current')} -> {self.cfg.out_dir}")
                self._visualize_current_state_to_out_dir(n_samples=n_samples)
            if final_ckpt:
                self._restore_checkpoint_for_export(final_ckpt)
        finally:
            self.cfg.out_dir = original_out_dir

    def _visualize_current_state_to_out_dir(self, n_samples: int = 5):
        if self.asm is None or self.model is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        cases = self._resolve_viz_cases(n_samples)
        n_total = len(cases) if cases else n_samples
        print(
            f"[trainer] Generating {n_total} deflection maps for '{self.cfg.mirror_surface_name}' ..."
        )
        iter_cases = cases if cases else [self._sample_preload_case() for _ in range(n_samples)]
        viz_records: List[Dict[str, Any]] = []
        for i, preload_case in enumerate(iter_cases):
            P = preload_case["P"]
            order_display = None
            if self.cfg.preload_use_stages and "order" in preload_case:
                order_display = "-".join(
                    str(int(o) + 1) for o in preload_case["order"].tolist()
                )
            unit = str(getattr(self.cfg.tightening_cfg, "angle_unit", "deg") or "deg")
            angle_txt = ",".join(f"{float(x):.2f}" for x in P.tolist())
            title = f"{self.cfg.viz_title_prefix}  theta=[{angle_txt}]{unit}"
            if order_display:
                title += f"  (order={order_display})"
            suffix = f"_{order_display.replace('-', '')}" if order_display else ""
            save_path = os.path.join(
                self.cfg.out_dir, f"deflection_{i+1:02d}{suffix}.png"
            )
            params_full = self._make_preload_params(preload_case)
            params_eval = self._extract_final_stage_params(params_full, keep_context=True)

            # Write a compact tightening report next to the figure.
            if self.tightening is not None and save_path:
                try:
                    report_path = os.path.splitext(save_path)[0] + "_tightening.txt"
                    stage_rows = []
                    if (
                        self.cfg.preload_use_stages
                        and isinstance(preload_case, dict)
                        and "stages" in preload_case
                    ):
                        stages_np = np.asarray(preload_case.get("stages"), dtype=np.float32)
                        if stages_np.ndim == 2 and stages_np.shape[0] > 0:
                            for s in range(int(stages_np.shape[0])):
                                params_s = self._extract_stage_params(params_full, s, keep_context=True)
                                _, st = self.tightening.energy(self.model.u_fn, params_s)
                                stage_rows.append(
                                    np.asarray(st.get("tightening", {}).get("rms", []))
                                )
                    _, st_final = self.tightening.energy(self.model.u_fn, params_eval)
                    final_row = np.asarray(st_final.get("tightening", {}).get("rms", []))

                    with open(report_path, "w", encoding="utf-8") as fp:
                        fp.write(f"theta = {P.tolist()}  [{unit}]\n")
                        if self.cfg.preload_use_stages and "order" in preload_case:
                            fp.write(f"order = {preload_case['order'].tolist()}  (0-based)\n")
                        fp.write("rms = [r1, r2, ...]\n")
                        for s, row in enumerate(stage_rows, start=1):
                            fp.write(f"stage_{s}: {row.tolist()}\n")
                        fp.write(f"final: {final_row.tolist()}\n")
                except Exception as exc:
                    print(f"[viz] tightening report skipped: {exc}")
            try:
                _, _, data_path = self._call_viz(P, params_eval, save_path, title)
                if self.cfg.viz_surface_enabled:
                    if not os.path.exists(save_path):
                        try:
                            import matplotlib.pyplot as plt
                            plt.savefig(save_path, dpi=200, bbox_inches="tight")
                            plt.close()
                        except Exception:
                            pass
                    if order_display:
                        print(f"[viz] saved -> {save_path}  (order={order_display})")
                    else:
                        print(f"[viz] saved -> {save_path}")
                    if data_path:
                        print(f"[viz] displacement data -> {data_path}")
                aligned_path = None
                if data_path:
                    try:
                        aligned_path = self._write_viz_reference_alignment(str(data_path))
                    except Exception as exc:
                        print(f"[viz] reference alignment skipped: {exc}")
                viz_records.append(
                    {
                        "index": i + 1,
                        "P": np.asarray(P, dtype=np.float64).reshape(-1),
                        "order": None if "order" not in preload_case else preload_case.get("order"),
                        "order_display": order_display,
                        "png_path": save_path,
                        "data_path": data_path,
                        "mesh_path": (
                            os.path.splitext(save_path)[0] + "_surface.ply"
                            if self.cfg.viz_write_surface_mesh and save_path
                            else None
                        ),
                        "aligned_path": aligned_path,
                    }
                )
            except TypeError as e:
                print("[viz] signature mismatch:", e)
            except Exception as e:
                print("[viz] error:", e)

            # Optional: plot each preload stage to make tightening order visible.
            stage_viz_records: List[Dict[str, Any]] = []
            if (
                self.cfg.viz_plot_stages
                and self.cfg.preload_use_stages
                and isinstance(preload_case, dict)
                and "stages" in preload_case
            ):
                try:
                    stages_np = np.asarray(preload_case.get("stages"), dtype=np.float32)
                    if stages_np.ndim == 2 and stages_np.shape[0] > 1:
                        stage_indices = self._resolve_stage_plot_indices(preload_case, int(stages_np.shape[0]))
                        n_plot = int(len(stage_indices))
                        for rank, s in enumerate(stage_indices, start=1):
                            P_stage = stages_np[s]
                            title_s = f"{self.cfg.viz_title_prefix}  P=[{int(P_stage[0])},{int(P_stage[1])},{int(P_stage[2])}]N"
                            if order_display:
                                title_s += f"  (order={order_display})"
                            title_s += f"  (stage={rank}/{n_plot})"
                            save_path_s = os.path.join(
                                self.cfg.out_dir, f"deflection_{i+1:02d}{suffix}_s{s+1}.png"
                            )
                            params_s = self._extract_stage_params(params_full, s, keep_context=True)
                            _, _, data_path_s = self._call_viz(P_stage, params_s, save_path_s, title_s)
                            stage_viz_records.append(
                                {
                                    "stage_rank": int(rank),
                                    "stage_index": int(s),
                                    "P": np.asarray(P_stage, dtype=np.float64).reshape(-1),
                                    "png_path": save_path_s,
                                    "data_path": data_path_s,
                                }
                            )
                except Exception as exc:
                    print(f"[viz] stage plots skipped: {exc}")
            if (
                stage_viz_records
                and len(stage_viz_records) > 1
                and bool(getattr(self.cfg, "viz_stage_compare_outputs", True))
            ):
                try:
                    report_path = self._write_stage_comparison_for_case(save_path, stage_viz_records)
                    if report_path:
                        print(f"[viz] stage comparison -> {report_path}")
                except Exception as exc:
                    print(f"[viz] stage comparison skipped: {exc}")
            if bool(getattr(self.cfg, "viz_same_pipeline_supervision_debug", False)):
                try:
                    samepipe_paths = self._write_same_pipeline_supervision_debug_exports(
                        case_index=i + 1,
                        preload_case=preload_case,
                        params_full=params_full,
                        suffix=suffix,
                        title_prefix=title,
                    )
                    for path in samepipe_paths:
                        print(f"[viz] same-pipeline debug -> {path}")
                except Exception as exc:
                    print(f"[viz] same-pipeline debug skipped: {exc}")

        # Additional comparison outputs: common-scale maps and delta maps between cases.
        if cases and viz_records and len(viz_records) > 1 and bool(getattr(self.cfg, "viz_compare_cases", False)):
            try:
                self._write_viz_comparison(viz_records)
            except Exception as exc:
                print(f"[viz] comparison skipped: {exc}")

        try:
            exported = self._write_supervision_eval_outputs()
            for path in exported:
                print(f"[viz] supervision eval -> {path}")
        except Exception as exc:
            print(f"[viz] supervision eval skipped: {exc}")

    def _resolve_stage_plot_indices(self, preload_case: Dict[str, Any], stage_count: int) -> List[int]:
        if stage_count <= 0:
            return []
        indices = list(range(int(stage_count)))
        if not bool(getattr(self.cfg, "viz_skip_release_stage_plot", False)):
            return indices
        stage_last = preload_case.get("stage_last")
        if stage_last is None:
            return indices
        try:
            stage_last_np = np.asarray(stage_last, dtype=np.float32)
        except Exception:
            return indices
        if stage_last_np.ndim != 2 or stage_last_np.shape[0] != stage_count:
            return indices
        keep = [
            i
            for i in range(stage_count)
            if bool(np.any(np.abs(stage_last_np[i]) > 1.0e-8))
        ]
        return keep if keep else indices

    def _resolve_viz_reference_path(self) -> Optional[str]:
        raw = str(getattr(self.cfg, "viz_reference_truth_path", "auto") or "").strip()
        if not raw:
            return None

        low = raw.lower()
        if low in {"none", "off", "false", "0", "disable", "disabled"}:
            return None

        candidates: List[str] = []
        out_dir = str(getattr(self.cfg, "out_dir", "") or "").strip()
        if low == "auto":
            if out_dir:
                candidates.append(os.path.join(out_dir, "3.txt"))
            candidates.append(os.path.join(os.getcwd(), "results", "3.txt"))
        else:
            candidates.append(raw)
            if out_dir and not os.path.isabs(raw):
                candidates.append(os.path.join(out_dir, raw))
            if not os.path.isabs(raw):
                candidates.append(os.path.join(os.getcwd(), raw))

        for cand in candidates:
            path = os.path.abspath(os.path.expanduser(str(cand)))
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _read_reference_truth_samples(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None

        node_ids: List[int] = []
        umag: List[float] = []
        parsed_rows = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                cols = re.split(r"[,\s\t]+", line)
                if len(cols) < 2:
                    continue
                try:
                    nid = int(float(cols[0]))
                    u = float(cols[1])
                except Exception:
                    continue
                parsed_rows += 1
                node_ids.append(nid)
                umag.append(u)

        if not node_ids:
            return None

        return {
            "node_id": np.asarray(node_ids, dtype=np.int64),
            "umag": np.asarray(umag, dtype=np.float64),
            "parsed_rows": int(parsed_rows),
            "path": path,
        }

    def _get_asm_node_id_set(self) -> set[int]:
        if self._asm_node_ids is not None:
            return self._asm_node_ids

        ids: set[int] = set()
        asm = getattr(self, "asm", None)
        nodes = getattr(asm, "nodes", None) if asm is not None else None
        if isinstance(nodes, dict):
            for nid in nodes.keys():
                try:
                    ids.add(int(nid))
                except Exception:
                    continue

        if not ids and asm is not None:
            for part in getattr(asm, "parts", {}).values():
                for nid in getattr(part, "node_ids", []) or []:
                    try:
                        ids.add(int(nid))
                    except Exception:
                        continue

        self._asm_node_ids = ids
        return ids

    def _load_viz_reference_truth(self) -> Optional[Dict[str, Any]]:
        path = self._resolve_viz_reference_path()
        if path is None:
            return None
        if self._viz_reference_cache is not None and self._viz_reference_cache_path == path:
            return self._viz_reference_cache

        loaded = self._read_reference_truth_samples(path)
        if loaded is None:
            self._viz_reference_cache_path = None
            self._viz_reference_cache = None
            return None

        self._viz_reference_cache_path = path
        self._viz_reference_cache = loaded
        return loaded

    def _write_viz_reference_alignment(self, pred_data_path: str) -> Optional[str]:
        if not bool(getattr(self.cfg, "viz_write_reference_aligned", False)):
            return None

        pred = self._read_viz_samples(str(pred_data_path))
        if pred is None:
            return None
        ref = self._load_viz_reference_truth()
        if ref is None:
            return None

        valid_node_ids = self._get_asm_node_id_set()
        if not valid_node_ids:
            print("[viz] reference alignment skipped: assembly node ids unavailable.")
            return None

        ref_ids = np.asarray(ref["node_id"], dtype=np.int64).reshape(-1)
        ref_u = np.asarray(ref["umag"], dtype=np.float64).reshape(-1)
        node_mask = np.asarray([int(nid) in valid_node_ids for nid in ref_ids], dtype=bool)

        ref_node_ids = ref_ids[node_mask]
        ref_node_u = ref_u[node_mask]
        nonnode_count = int(ref_ids.size - ref_node_ids.size)
        if ref_node_ids.size == 0:
            print("[viz] reference alignment skipped: no valid node rows in reference.")
            return None

        ref_map: Dict[int, float] = {}
        for nid, val in zip(ref_node_ids.tolist(), ref_node_u.tolist()):
            ref_map[int(nid)] = float(val)

        pred_ids = np.asarray(pred["node_id"], dtype=np.int64).reshape(-1)
        pred_u = np.asarray(pred["umag"], dtype=np.float64).reshape(-1)
        common_mask = np.asarray([int(nid) in ref_map for nid in pred_ids], dtype=bool)
        common_ids = pred_ids[common_mask]
        if common_ids.size == 0:
            print("[viz] reference alignment skipped: no overlapping node ids.")
            return None

        ref_common = np.asarray([ref_map[int(nid)] for nid in common_ids.tolist()], dtype=np.float64)
        pred_common = pred_u[common_mask]
        diff = pred_common - ref_common

        denom = np.where(np.abs(ref_common) > 1.0e-30, ref_common, np.nan)
        ratio = np.divide(pred_common, denom)
        ratio_abs = np.abs(np.divide(pred_common, np.where(np.abs(ref_common) > 1.0e-30, ref_common, np.nan)))
        ratio_med = float(np.nanmedian(ratio_abs)) if ratio_abs.size else float("nan")

        out_path = os.path.splitext(str(pred_data_path))[0] + "_aligned.txt"
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("# Aligned mirror displacement: prediction vs reference\n")
            fp.write(f"# reference_path={ref.get('path', '')}\n")
            fp.write(f"# reference_rows_total={int(ref_ids.size)}\n")
            fp.write(f"# reference_rows_node_only={int(ref_node_ids.size)}\n")
            fp.write(f"# reference_rows_nonnode={nonnode_count}\n")
            fp.write(f"# predicted_rows={int(pred_ids.size)}\n")
            fp.write(f"# common_rows={int(common_ids.size)}\n")
            fp.write("# columns: node_id u_ref u_pred diff(pred-ref) ratio(pred/ref)\n")
            for nid, u_ref, u_pred, du, rt in zip(
                common_ids.tolist(),
                ref_common.tolist(),
                pred_common.tolist(),
                diff.tolist(),
                ratio.tolist(),
            ):
                fp.write(
                    f"{int(nid):10d} {float(u_ref): .8e} {float(u_pred): .8e} "
                    f"{float(du): .8e} {float(rt): .8e}\n"
                )

        print(
            "[viz] aligned displacement -> "
            f"{out_path} (common={int(common_ids.size)}, nonnode_ref={nonnode_count}, "
            f"median|pred/ref|={ratio_med:.3e})"
        )
        return out_path

    @staticmethod
    def _read_viz_samples(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None

        node_ids: List[int] = []
        x: List[float] = []
        y: List[float] = []
        z: List[float] = []
        ux: List[float] = []
        uy: List[float] = []
        uz: List[float] = []
        umag: List[float] = []
        u_plane: List[float] = []
        v_plane: List[float] = []
        rigid_line: Optional[str] = None

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "rigid_body_removed" in line:
                        rigid_line = line
                    continue
                cols = line.split()
                if len(cols) < 10:
                    continue
                try:
                    node_ids.append(int(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                    z.append(float(cols[3]))
                    ux.append(float(cols[4]))
                    uy.append(float(cols[5]))
                    uz.append(float(cols[6]))
                    umag.append(float(cols[7]))
                    u_plane.append(float(cols[8]))
                    v_plane.append(float(cols[9]))
                except Exception:
                    continue

        if not node_ids:
            return None

        node_arr = np.asarray(node_ids, dtype=np.int64)
        order = np.argsort(node_arr)
        return {
            "node_id": node_arr[order],
            "x": np.asarray(x, dtype=np.float64)[order],
            "y": np.asarray(y, dtype=np.float64)[order],
            "z": np.asarray(z, dtype=np.float64)[order],
            "ux": np.asarray(ux, dtype=np.float64)[order],
            "uy": np.asarray(uy, dtype=np.float64)[order],
            "uz": np.asarray(uz, dtype=np.float64)[order],
            "umag": np.asarray(umag, dtype=np.float64)[order],
            "u_plane": np.asarray(u_plane, dtype=np.float64)[order],
            "v_plane": np.asarray(v_plane, dtype=np.float64)[order],
            "rigid_line": rigid_line,
        }

    @staticmethod
    def _align_viz_sample_to_nodes(sample: Dict[str, Any], node_ids: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        src_ids = np.asarray(sample.get("node_id", []), dtype=np.int64).reshape(-1)
        if src_ids.size == 0:
            return None
        idx = np.searchsorted(src_ids, node_ids)
        valid = (
            (idx >= 0)
            & (idx < src_ids.size)
            & (src_ids[idx] == node_ids)
        )
        if not np.all(valid):
            return None
        out: Dict[str, np.ndarray] = {"node_id": node_ids}
        for key in ("x", "y", "z", "ux", "uy", "uz", "umag", "u_plane", "v_plane"):
            arr = np.asarray(sample.get(key, []), dtype=np.float64).reshape(-1)
            if arr.size != src_ids.size:
                return None
            out[key] = arr[idx]
        return out

    def _write_stage_comparison_for_case(
        self, base_png_path: str, stage_records: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Write stage common-scale and delta maps for one preload case."""

        if not base_png_path or not stage_records:
            return None

        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.tri import Triangulation

        parsed: List[Dict[str, Any]] = []
        for rec in stage_records:
            data_path = rec.get("data_path")
            if not data_path:
                continue
            sample = self._read_viz_samples(str(data_path))
            if sample is None:
                continue
            entry = dict(rec)
            entry["sample"] = sample
            parsed.append(entry)
        if len(parsed) < 2:
            return None

        parsed.sort(key=lambda r: int(r.get("stage_rank", r.get("stage_index", 0)) or 0))

        common_ids = np.asarray(parsed[0]["sample"]["node_id"], dtype=np.int64).reshape(-1)
        for rec in parsed[1:]:
            rec_ids = np.asarray(rec["sample"]["node_id"], dtype=np.int64).reshape(-1)
            common_ids = np.intersect1d(common_ids, rec_ids, assume_unique=False)
            if common_ids.size == 0:
                return None
        if common_ids.size < 3:
            return None

        aligned: List[Dict[str, Any]] = []
        for rec in parsed:
            aligned_sample = self._align_viz_sample_to_nodes(rec["sample"], common_ids)
            if aligned_sample is None:
                return None
            aligned.append({**rec, "aligned": aligned_sample})

        rendered: List[Dict[str, Any]] = []
        max_umag = 0.0
        tri_ref = None
        for rec in aligned:
            render_state = self._build_stage_comparison_render_state(common_ids, rec["aligned"])
            if render_state is None:
                return None
            tri_cur, d_plot = render_state
            if tri_ref is None:
                tri_ref = tri_cur
            else:
                if (
                    tri_ref.x.shape != tri_cur.x.shape
                    or tri_ref.triangles.shape != tri_cur.triangles.shape
                ):
                    return None
            max_umag = max(max_umag, float(np.nanmax(d_plot)))
            rendered.append({**rec, "tri": tri_cur, "d_plot": d_plot})
        max_umag = float(max_umag) + 1.0e-16

        base_stem = os.path.splitext(str(base_png_path))[0]
        common_cmap = str(getattr(self.cfg, "viz_colormap", "turbo") or "turbo")
        delta_cmap = str(
            getattr(
                self.cfg,
                "viz_stage_compare_cmap",
                getattr(self.cfg, "viz_compare_cmap", "coolwarm"),
            )
            or "coolwarm"
        )
        units = str(getattr(self.cfg, "viz_units", "mm") or "mm")
        title_prefix = str(
            getattr(self.cfg, "viz_title_prefix", "Total Deformation (trained PINN)")
        )

        report_path = base_stem + "_stage_compare.txt"
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write("Stage comparison report (same preload case)\n")
            fp.write(f"base_png={base_png_path}\n")
            fp.write(f"n_common_nodes={int(common_ids.size)}\n")
            fp.write(f"common_umag_max={max_umag:.8e}\n\n")
            fp.write("Stages:\n")

            for rec in rendered:
                stage_rank = int(rec.get("stage_rank", rec.get("stage_index", 0)) or 0)
                stage_u = np.asarray(rec["d_plot"], dtype=np.float64)
                common_path = f"{base_stem}_s{stage_rank}_common.png"
                fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                sc = ax.tripcolor(
                    rec["tri"],
                    stage_u,
                    shading="gouraud",
                    cmap=common_cmap,
                    norm=colors.Normalize(vmin=0.0, vmax=max_umag),
                    edgecolors="none",
                )
                cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                cbar.set_label(f"Total displacement magnitude [{units}] (common scale)")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("u (best-fit plane)")
                ax.set_ylabel("v (best-fit plane)")
                ax.set_title(f"{title_prefix} | stage {stage_rank} (common scale)")
                fig.savefig(common_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                fp.write(
                    f"- stage {stage_rank}: min={float(np.nanmin(stage_u)):.8e} "
                    f"max={float(np.nanmax(stage_u)):.8e} mean={float(np.nanmean(stage_u)):.8e}\n"
                )

            fp.write("\nConsecutive deltas:\n")
            for prev, cur in zip(rendered[:-1], rendered[1:]):
                prev_rank = int(prev.get("stage_rank", prev.get("stage_index", 0)) or 0)
                cur_rank = int(cur.get("stage_rank", cur.get("stage_index", 0)) or 0)
                du = np.asarray(cur["d_plot"], dtype=np.float64) - np.asarray(prev["d_plot"], dtype=np.float64)
                rms = float(np.sqrt(np.mean(du * du)))
                mean_abs = float(np.mean(np.abs(du)))
                max_abs = float(np.max(np.abs(du)))
                delta_path = f"{base_stem}_s{cur_rank}_minus_s{prev_rank}.png"
                vlim = max(max_abs, 1.0e-16)

                fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                sc = ax.tripcolor(
                    cur["tri"],
                    du,
                    shading="gouraud",
                    cmap=delta_cmap,
                    norm=colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
                    edgecolors="none",
                )
                cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                cbar.set_label(f"Delta |u| [{units}]")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("u (best-fit plane)")
                ax.set_ylabel("v (best-fit plane)")
                ax.set_title(f"{title_prefix} | stage {cur_rank} - stage {prev_rank}")
                fig.savefig(delta_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                fp.write(
                    f"- s{cur_rank}-s{prev_rank}: rms={rms:.8e} "
                    f"mean|du|={mean_abs:.8e} max|du|={max_abs:.8e}\n"
                )

        return report_path

    def _write_supervision_eval_outputs(self) -> List[str]:
        dataset = getattr(self, "_supervision_dataset", None)
        supervision_cfg = getattr(self.cfg, "supervision", None)
        if dataset is None or self.model is None or supervision_cfg is None:
            return []
        if not bool(getattr(supervision_cfg, "enabled", False)):
            return []
        if not bool(getattr(supervision_cfg, "export_eval_reports", True)):
            return []

        split_names = tuple(getattr(supervision_cfg, "eval_splits", ()) or ())
        if not split_names:
            return []

        u_eval_fn = self.model.u_fn
        if bool(getattr(self.cfg, "viz_force_pointwise", False)) and hasattr(self.model, "u_fn_pointwise"):
            u_eval_fn = self.model.u_fn_pointwise

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        exported: List[str] = []
        compare_enabled = bool(getattr(self.cfg, "viz_supervision_compare_enabled", False))
        compare_split = str(getattr(self.cfg, "viz_supervision_compare_split", "test") or "").strip()
        processed: Dict[str, Dict[str, Any]] = {}
        ordered_splits: List[str] = []
        for split in split_names:
            name = str(split).strip()
            if name and name not in ordered_splits:
                ordered_splits.append(name)
        if compare_enabled and compare_split and compare_split not in ordered_splits:
            ordered_splits.append(compare_split)

        for split in ordered_splits:
            cases = list(getattr(dataset, "cases_by_split", {}).get(split, []) or [])
            if not cases:
                continue
            rows = self._build_supervision_eval_rows(str(split), cases, u_eval_fn)
            processed[str(split)] = {"rows": rows, "cases": cases}
            if str(split) in split_names:
                csv_path = self._write_supervision_eval_split_artifacts(str(split), rows)
                if csv_path:
                    exported.append(csv_path)
        if compare_enabled and compare_split and compare_split in processed:
            exported.extend(
                self._write_supervision_compare_artifacts(
                    compare_split,
                    processed[compare_split]["rows"],
                    processed[compare_split]["cases"],
                    u_eval_fn,
                )
            )
        return exported

    def _build_supervision_eval_rows(
        self,
        split: str,
        cases: List[Dict[str, Any]],
        u_eval_fn,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for case in cases:
            case_local = copy.deepcopy(case)
            if self.cfg.preload_use_stages and "stages" not in case_local:
                case_local.update(self._build_stage_case(case_local["P"], case_local["order"]))
            params_full = self._make_preload_params(case_local)
            X_all = np.asarray(case_local.get("X_obs"), dtype=np.float32)
            U_all = np.asarray(case_local.get("U_obs"), dtype=np.float32)
            if X_all.ndim != 3 or U_all.ndim != 3 or X_all.shape != U_all.shape:
                raise ValueError(
                    f"supervision case {case_local.get('case_id')!r} must carry staged X_obs/U_obs "
                    f"with matching shape (S,N,3); got {X_all.shape} vs {U_all.shape}"
                )

            order = np.asarray(case_local.get("order", []), dtype=np.int32).reshape(-1)
            order_display = "-".join(str(int(x) + 1) for x in order.tolist()) if order.size else ""

            for stage_idx in range(int(U_all.shape[0])):
                params_stage = self._extract_stage_params(params_full, stage_idx, keep_context=True)
                X_stage = tf.convert_to_tensor(X_all[stage_idx], dtype=tf.float32)
                U_true = np.asarray(U_all[stage_idx], dtype=np.float32)
                U_pred_tf = u_eval_fn(X_stage, params=params_stage)
                U_pred = np.asarray(
                    U_pred_tf.numpy() if hasattr(U_pred_tf, "numpy") else U_pred_tf,
                    dtype=np.float32,
                )
                if U_pred.shape != U_true.shape:
                    raise ValueError(
                        f"supervision prediction shape mismatch for case={case_local.get('case_id')!r} "
                        f"stage={stage_idx + 1}: pred {U_pred.shape} vs true {U_true.shape}"
                    )

                err = U_pred - U_true
                err_vec = np.linalg.norm(err, axis=1)
                pred_vec = np.linalg.norm(U_pred, axis=1)
                true_vec = np.linalg.norm(U_true, axis=1)
                rows.append(
                    {
                        "split": str(split),
                        "case_id": str(case_local.get("case_id", "")),
                        "base_id": str(case_local.get("base_id", "")),
                        "source": str(case_local.get("source", "")),
                        "job_name": str(case_local.get("job_name", "")),
                        "stage_rank": int(stage_idx + 1),
                        "obs_count": int(U_true.shape[0]),
                        "theta_1_deg": float(case_local["P"][0]),
                        "theta_2_deg": float(case_local["P"][1]),
                        "theta_3_deg": float(case_local["P"][2]),
                        "order_1": int(order[0] + 1) if order.size >= 1 else 0,
                        "order_2": int(order[1] + 1) if order.size >= 2 else 0,
                        "order_3": int(order[2] + 1) if order.size >= 3 else 0,
                        "order_str": order_display,
                        "rmse_x_mm": float(np.sqrt(np.mean(np.square(err[:, 0])))),
                        "rmse_y_mm": float(np.sqrt(np.mean(np.square(err[:, 1])))),
                        "rmse_z_mm": float(np.sqrt(np.mean(np.square(err[:, 2])))),
                        "rmse_vec_mm": float(np.sqrt(np.mean(np.square(err_vec)))),
                        "mae_vec_mm": float(np.mean(np.abs(err_vec))),
                        "max_vec_mm": float(np.max(np.abs(err_vec))),
                        "pred_rms_vec_mm": float(np.sqrt(np.mean(np.square(pred_vec)))),
                        "true_rms_vec_mm": float(np.sqrt(np.mean(np.square(true_vec)))),
                    }
                )
        return rows

    @staticmethod
    def _select_representative_supervision_rows(
        rows: List[Dict[str, Any]],
        sources: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        final_rows_by_case: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id", ""))
            stage_rank = int(row.get("stage_rank", 0) or 0)
            prev = final_rows_by_case.get(case_id)
            if prev is None or stage_rank > int(prev.get("stage_rank", 0) or 0):
                final_rows_by_case[case_id] = dict(row)

        selected: List[Dict[str, Any]] = []
        for source in sources:
            src = str(source).strip()
            if not src:
                continue
            candidates = [
                row for row in final_rows_by_case.values() if str(row.get("source", "")).strip() == src
            ]
            if not candidates:
                continue
            values = np.asarray([float(row.get("rmse_vec_mm", np.nan)) for row in candidates], dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            median = float(np.median(values[finite]))
            candidates.sort(
                key=lambda row: (
                    abs(float(row.get("rmse_vec_mm", np.inf)) - median),
                    float(row.get("rmse_vec_mm", np.inf)),
                    str(row.get("case_id", "")),
                )
            )
            selected.append(candidates[0])
        return selected

    def _write_supervision_compare_artifacts(
        self,
        split: str,
        rows: List[Dict[str, Any]],
        cases: List[Dict[str, Any]],
        u_eval_fn,
    ) -> List[str]:
        if not rows or not cases:
            return []

        import csv
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.tri import Triangulation

        sources = tuple(getattr(self.cfg, "viz_supervision_compare_sources", ()) or ())
        selected_rows = self._select_representative_supervision_rows(rows, sources)
        if not selected_rows:
            return []

        case_map = {str(case.get("case_id", "")): case for case in cases}
        exported: List[str] = []
        summary_rows: List[Dict[str, Any]] = []
        units = str(getattr(self.cfg, "viz_units", "mm") or "mm")
        main_cmap = str(getattr(self.cfg, "viz_colormap", "turbo") or "turbo")

        for row in selected_rows:
            case_id = str(row.get("case_id", ""))
            case = case_map.get(case_id)
            if case is None:
                continue

            case_local = copy.deepcopy(case)
            if self.cfg.preload_use_stages and "stages" not in case_local:
                case_local.update(self._build_stage_case(case_local["P"], case_local["order"]))
            params_full = self._make_preload_params(case_local)
            stage_rank = max(1, int(row.get("stage_rank", 1) or 1))
            stage_idx = stage_rank - 1
            X_all = np.asarray(case_local.get("X_obs"), dtype=np.float32)
            U_all = np.asarray(case_local.get("U_obs"), dtype=np.float32)
            if stage_idx >= int(X_all.shape[0]) or stage_idx >= int(U_all.shape[0]):
                continue
            params_stage = self._extract_stage_params(params_full, stage_idx, keep_context=True)
            X_stage = np.asarray(X_all[stage_idx], dtype=np.float32)
            U_true = np.asarray(U_all[stage_idx], dtype=np.float32)
            U_pred_tf = u_eval_fn(tf.convert_to_tensor(X_stage, dtype=tf.float32), params=params_stage)
            U_pred = np.asarray(
                U_pred_tf.numpy() if hasattr(U_pred_tf, "numpy") else U_pred_tf,
                dtype=np.float32,
            )
            if U_pred.shape != U_true.shape:
                continue

            node_ids = np.asarray(case_local.get("node_ids", []), dtype=np.int64).reshape(-1)
            pred_render = None
            true_render = None
            if node_ids.size == int(X_stage.shape[0]):
                pred_render = self._build_supervision_compare_render_state(node_ids, X_stage, U_pred)
                true_render = self._build_supervision_compare_render_state(node_ids, X_stage, U_true)

            if pred_render is not None and true_render is not None:
                tri_pred, u_plot_pred, pred_mag = pred_render
                tri_true, u_plot_true, true_mag = true_render
                if (
                    tri_pred.x.shape != tri_true.x.shape
                    or tri_pred.triangles.shape != tri_true.triangles.shape
                ):
                    continue
                tri = tri_pred
                uv = np.stack([np.asarray(tri.x, dtype=np.float64), np.asarray(tri.y, dtype=np.float64)], axis=1)
                err_mag = np.linalg.norm(
                    np.asarray(u_plot_pred, dtype=np.float64) - np.asarray(u_plot_true, dtype=np.float64),
                    axis=1,
                )
            else:
                try:
                    c, e1, e2, _ = _fit_plane_basis(np.asarray(X_stage, dtype=np.float64))
                    uv = _project_to_plane(np.asarray(X_stage, dtype=np.float64), c, e1, e2)
                    tri = Triangulation(uv[:, 0], uv[:, 1])
                except Exception:
                    tri = None
                    uv = np.asarray(X_stage[:, :2], dtype=np.float64)

                pred_mag = np.linalg.norm(U_pred, axis=1)
                true_mag = np.linalg.norm(U_true, axis=1)
                err_mag = np.linalg.norm(U_pred - U_true, axis=1)
            common_vmax = max(float(np.nanmax(pred_mag)), float(np.nanmax(true_mag)), 1.0e-12)
            err_vmax = max(float(np.nanmax(err_mag)), 1.0e-12)

            fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
            panels = (
                ("PINN |u|", pred_mag, main_cmap, colors.Normalize(vmin=0.0, vmax=common_vmax)),
                ("FEM |u|", true_mag, main_cmap, colors.Normalize(vmin=0.0, vmax=common_vmax)),
                ("|PINN-FEM|", err_mag, "magma", colors.Normalize(vmin=0.0, vmax=err_vmax)),
            )
            for ax, (title, values, cmap, norm) in zip(axes, panels):
                if tri is not None and tri.triangles is not None and len(tri.triangles) > 0:
                    artist = ax.tripcolor(
                        tri,
                        np.asarray(values, dtype=np.float64),
                        shading="gouraud",
                        cmap=cmap,
                        norm=norm,
                        edgecolors="none",
                    )
                else:
                    artist = ax.scatter(
                        uv[:, 0],
                        uv[:, 1],
                        c=np.asarray(values, dtype=np.float64),
                        cmap=cmap,
                        norm=norm,
                        s=16.0,
                    )
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)
                cbar = fig.colorbar(artist, ax=ax, shrink=0.90, pad=0.02)
                cbar.set_label(units)

            fig.suptitle(
                " | ".join(
                    [
                        f"case={case_id}",
                        f"source={row.get('source', '')}",
                        f"stage={stage_rank}",
                        f"theta=[{float(row.get('theta_1_deg', 0.0)):.2f},{float(row.get('theta_2_deg', 0.0)):.2f},{float(row.get('theta_3_deg', 0.0)):.2f}]deg",
                        f"order={row.get('order_str', '')}",
                        f"rmse={float(row.get('rmse_vec_mm', 0.0)):.6f}{units}",
                    ]
                ),
                fontsize=10,
            )

            source_tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(row.get("source", "") or "unknown")).strip("_")
            case_tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", case_id).strip("_") or "case"
            out_path = os.path.join(self.cfg.out_dir, f"supervision_compare_{source_tag}_{case_tag}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            exported.append(out_path)
            summary_rows.append(
                {
                    "split": str(split),
                    "source": str(row.get("source", "")),
                    "case_id": case_id,
                    "base_id": str(row.get("base_id", "")),
                    "stage_rank": stage_rank,
                    "rmse_vec_mm": float(row.get("rmse_vec_mm", np.nan)),
                    "figure_path": out_path,
                }
            )

        if not summary_rows:
            return exported

        summary_path = os.path.join(self.cfg.out_dir, "supervision_compare_selected_cases.csv")
        with open(summary_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["split", "source", "case_id", "base_id", "stage_rank", "rmse_vec_mm", "figure_path"],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        exported.append(summary_path)
        return exported

    def _write_supervision_eval_split_artifacts(
        self, split: str, rows: List[Dict[str, Any]]
    ) -> Optional[str]:
        if not rows:
            return None

        import csv

        split_tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(split or "eval")).strip("_") or "eval"
        csv_path = os.path.join(self.cfg.out_dir, f"supervision_eval_{split_tag}.csv")
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                str(item.get("case_id", "")),
                int(item.get("stage_rank", 0) or 0),
            ),
        )
        fieldnames = [
            "split",
            "case_id",
            "base_id",
            "source",
            "job_name",
            "stage_rank",
            "obs_count",
            "theta_1_deg",
            "theta_2_deg",
            "theta_3_deg",
            "order_1",
            "order_2",
            "order_3",
            "order_str",
            "rmse_x_mm",
            "rmse_y_mm",
            "rmse_z_mm",
            "rmse_vec_mm",
            "mae_vec_mm",
            "max_vec_mm",
            "pred_rms_vec_mm",
            "true_rms_vec_mm",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

        supervision_cfg = getattr(self.cfg, "supervision", None)
        if bool(getattr(supervision_cfg, "export_eval_plots", True)):
            self._plot_supervision_eval_heatmap(split_tag, rows_sorted)

        return csv_path

    def _plot_supervision_eval_heatmap(self, split_tag: str, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        import matplotlib.pyplot as plt

        case_ids = sorted({str(row.get("case_id", "")) for row in rows})
        stage_ranks = sorted({int(row.get("stage_rank", 0) or 0) for row in rows})
        if not case_ids or not stage_ranks:
            return None

        case_to_idx = {case_id: idx for idx, case_id in enumerate(case_ids)}
        stage_to_idx = {stage: idx for idx, stage in enumerate(stage_ranks)}
        heat = np.full((len(case_ids), len(stage_ranks)), np.nan, dtype=np.float64)
        for row in rows:
            i = case_to_idx[str(row.get("case_id", ""))]
            j = stage_to_idx[int(row.get("stage_rank", 0) or 0)]
            heat[i, j] = float(row.get("rmse_vec_mm", np.nan))

        vmax = np.nanmax(heat) if np.any(np.isfinite(heat)) else 0.0
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0.0 else 1.0
        fig_h = max(4.0, min(14.0, 0.22 * len(case_ids) + 1.5))
        fig, ax = plt.subplots(figsize=(7.2, fig_h), constrained_layout=True)
        im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label(f"Stage RMSE |du| [{getattr(self.cfg, 'viz_units', 'mm')}]")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Case")
        ax.set_xticks(np.arange(len(stage_ranks)))
        ax.set_xticklabels([str(x) for x in stage_ranks])
        if len(case_ids) <= 40:
            ax.set_yticks(np.arange(len(case_ids)))
            ax.set_yticklabels(case_ids)
        else:
            tick_idx = np.linspace(0, len(case_ids) - 1, num=min(12, len(case_ids)), dtype=int)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([case_ids[i] for i in tick_idx])
        ax.set_title(f"Supervision eval RMSE by case/stage ({split_tag})")
        out_path = os.path.join(self.cfg.out_dir, f"supervision_eval_{split_tag}_rmse_vec.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _write_viz_comparison(self, records: List[Dict[str, Any]]) -> None:
        """
        Generate:
        - common-scale |u| maps to make amplitude comparable
        - delta maps (vector displacement difference) to highlight subtle differences
        - a text report with quantitative metrics
        """
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from matplotlib import colors

        def _read_surface_ply_mesh(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            if not path or not os.path.exists(path):
                return None
            n_vert = None
            n_face = None
            header_done = False
            node_ids: List[int] = []
            tris: List[Tuple[int, int, int]] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if not header_done:
                        if s.startswith("element vertex"):
                            parts = s.split()
                            if len(parts) >= 3:
                                n_vert = int(parts[2])
                        elif s.startswith("element face"):
                            parts = s.split()
                            if len(parts) >= 3:
                                n_face = int(parts[2])
                        elif s == "end_header":
                            header_done = True
                            break
                if not header_done or n_vert is None or n_face is None:
                    return None

                for _ in range(int(n_vert)):
                    row = f.readline()
                    if not row:
                        return None
                    cols = row.strip().split()
                    if len(cols) < 4:
                        return None
                    node_ids.append(int(cols[3]))

                for _ in range(int(n_face)):
                    row = f.readline()
                    if not row:
                        break
                    cols = row.strip().split()
                    if len(cols) < 4:
                        continue
                    try:
                        n = int(cols[0])
                    except Exception:
                        continue
                    if n < 3:
                        continue
                    # Expect triangles; if not, take the first three vertices as a fallback.
                    i0, i1, i2 = int(cols[1]), int(cols[2]), int(cols[3])
                    tris.append((i0, i1, i2))

            if not node_ids or not tris:
                return None
            return (
                np.asarray(node_ids, dtype=np.int64),
                np.asarray(tris, dtype=np.int32),
            )

        samples: List[Dict[str, Any]] = []
        for rec in records:
            data_path = rec.get("data_path")
            if not data_path:
                continue
            s = self._read_viz_samples(str(data_path))
            if s is None:
                continue
            s["record"] = rec
            samples.append(s)

        if len(samples) < 2:
            return

        # Use the first sample as the geometric base for triangulation/mapping.
        geom_base = samples[0]
        geom_base_rec = geom_base["record"]
        base_nodes = geom_base["node_id"]

        # Common scale across all cases (for |u| maps)
        global_umax = 0.0
        for s in samples:
            global_umax = max(global_umax, float(np.nanmax(s["umag"])))
        global_umax = float(global_umax) + 1e-16

        # Triangulation in (u,v) plane for diff plots: prefer FE connectivity from the surface PLY.
        u = np.asarray(geom_base["u_plane"], dtype=np.float64)
        v = np.asarray(geom_base["v_plane"], dtype=np.float64)
        tri = None
        vertex_pos: Optional[np.ndarray] = None
        mesh_info = _read_surface_ply_mesh(str(geom_base_rec.get("mesh_path") or ""))
        if mesh_info is not None:
            mesh_nodes, mesh_tris = mesh_info
            pos = np.searchsorted(base_nodes, mesh_nodes)
            ok = (
                (pos >= 0)
                & (pos < base_nodes.shape[0])
                & (base_nodes[pos] == mesh_nodes)
            )
            if np.all(ok):
                u_vert = u[pos]
                v_vert = v[pos]
                tri = Triangulation(u_vert, v_vert, triangles=mesh_tris)
                vertex_pos = pos
        if tri is None:
            tri = Triangulation(u, v)
            cu, cv = float(np.mean(u)), float(np.mean(v))
            r = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
            r_inner = float(np.nanmin(r)) * 1.02
            r_outer = float(np.nanmax(r)) * 0.98
            tris = np.asarray(tri.triangles, dtype=np.int64)
            uc = u[tris].mean(axis=1)
            vc = v[tris].mean(axis=1)
            rc = np.sqrt((uc - cu) ** 2 + (vc - cv) ** 2)
            tri.set_mask((rc < r_inner) | (rc > r_outer))

        # Report
        report_path = os.path.join(self.cfg.out_dir, "deflection_compare.txt")
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write("Deflection comparison report (PINN)\n")
            fp.write(f"triangulation_base = deflection_{geom_base_rec.get('index', 1):02d}\n\n")
            fp.write("Cases:\n")
            for s in samples:
                rec = s["record"]
                idx = int(rec.get("index", 0))
                P = rec.get("P")
                order_disp = rec.get("order_display") or "-"
                fp.write(
                    f"- {idx:02d} P={P.tolist() if hasattr(P, 'tolist') else P} order={order_disp}"
                )
                if s.get("rigid_line"):
                    fp.write(f" | {s['rigid_line'].lstrip('#').strip()}")
                fp.write("\n")
            fp.write("\nDiffs (grouped by identical P):\n")

            # Common-scale maps (optional)
            if self.cfg.viz_compare_common_scale:
                for s in samples:
                    rec = s["record"]
                    idx = int(rec.get("index", 0))
                    out_name = f"deflection_{idx:02d}_common.png"
                    out_path = os.path.join(self.cfg.out_dir, out_name)
                    umag_plot = (
                        s["umag"] if vertex_pos is None else s["umag"][vertex_pos]
                    )
                    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                    sc = ax.tripcolor(
                        tri,
                        umag_plot,
                        shading="gouraud",
                        cmap=str(self.cfg.viz_colormap or "turbo"),
                        norm=colors.Normalize(vmin=0.0, vmax=global_umax),
                        edgecolors="none",
                    )
                    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                    cbar.set_label(f"Total displacement magnitude [{self.cfg.viz_units}] (common scale)")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("u (best-fit plane)")
                    ax.set_ylabel("v (best-fit plane)")
                    title = f"{self.cfg.viz_title_prefix} | common scale"
                    P = rec.get("P")
                    if P is not None and len(P) >= 3:
                        title += f"  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
                    od = rec.get("order_display")
                    if od:
                        title += f" (order={od})"
                    ax.set_title(title)
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)

            # Delta plots/metrics: compare within each identical-P group so tightening order is directly visible.
            def _key_from_P(rec: Dict[str, Any]) -> Tuple[int, ...]:
                P = rec.get("P")
                if P is None:
                    return tuple()
                arr = np.asarray(P, dtype=np.float64).reshape(-1)
                return tuple(int(round(float(x))) for x in arr.tolist())

            groups: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
            for s in samples:
                rec = s["record"]
                key = _key_from_P(rec)
                groups.setdefault(key, []).append(s)

            for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
                if len(group) < 2:
                    continue
                group = sorted(group, key=lambda s: int(s["record"].get("index", 0)))
                base = group[0]
                base_rec = base["record"]
                base_idx = int(base_rec.get("index", 0))
                fp.write(f"\nP={list(key)} base={base_idx:02d}:\n")

                for s in group[1:]:
                    rec = s["record"]
                    idx = int(rec.get("index", 0))
                    nodes = s["node_id"]
                    if nodes.shape != base_nodes.shape or not np.all(nodes == base_nodes):
                        fp.write(f"- {idx:02d}: node mismatch, skipped\n")
                        continue

                    dux = s["ux"] - base["ux"]
                    duy = s["uy"] - base["uy"]
                    duz = s["uz"] - base["uz"]
                    du = np.sqrt(dux * dux + duy * duy + duz * duz)
                    rms = float(np.sqrt(np.mean(du * du)))
                    maxv = float(np.max(du))
                    dmag = s["umag"] - base["umag"]
                    max_abs_dmag = float(np.max(np.abs(dmag)))
                    arg = int(np.argmax(du))
                    node_max = int(nodes[arg])
                    u_max = float(u[arg])
                    v_max = float(v[arg])
                    fp.write(
                        f"- {idx:02d}: rms|du|={rms:.3e} max|du|={maxv:.3e} "
                        f"max|Δ|u||={max_abs_dmag:.3e} @node={node_max} (u,v)=({u_max:.3f},{v_max:.3f})\n"
                    )

                    dmag_plot = dmag if vertex_pos is None else dmag[vertex_pos]
                    vlim = float(np.max(np.abs(dmag_plot))) + 1e-16
                    out_name = f"deflection_diff_{idx:02d}_minus_{base_idx:02d}.png"
                    out_path = os.path.join(self.cfg.out_dir, out_name)
                    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
                    sc = ax.tripcolor(
                        tri,
                        dmag_plot,
                        shading="gouraud",
                        cmap=str(self.cfg.viz_compare_cmap or "coolwarm"),
                        norm=norm,
                        edgecolors="none",
                    )
                    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                    cbar.set_label(f"Δ|u| [{self.cfg.viz_units}]")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("u (best-fit plane)")
                    ax.set_ylabel("v (best-fit plane)")
                    title = f"Δ|u| vs base ({base_idx:02d})"
                    P = rec.get("P")
                    if P is not None and len(P) >= 3:
                        title += f"  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
                    od = rec.get("order_display")
                    if od:
                        title += f" (order={od})"
                    ax.set_title(title)
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)

        print(f"[viz] comparison report -> {report_path}")


