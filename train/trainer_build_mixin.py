# -*- coding: utf-8 -*-
"""Build/setup mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from inp_io.inp_parser import AssemblyModel
from inp_io.cdb_parser import load_cdb
from mesh.volume_quadrature import build_volume_points
from mesh.contact_pairs import ContactPairSpec, build_contact_map
from physics.material_lib import MaterialLibrary
from model.pinn_model import create_displacement_model, _knn_to_adj
from physics.elasticity_residual import ElasticityResidual
from physics.contact.contact_operator import ContactOperator
from physics.tightening_model import NutTighteningPenalty, NutSpec
from train.ansys_supervision import load_ansys_supervision_dataset


class TrainerBuildMixin:
    def _maybe_prebuild_global_graph(self):
        cfg = self.cfg
        field_cfg = getattr(getattr(cfg, "model_cfg", None), "field", None)
        if field_cfg is None:
            return
        if not bool(getattr(field_cfg, "graph_precompute", False)):
            print("[graph] Graph precompute disabled; dynamic adjacency will be used.")
            return
        if not hasattr(self, "elasticity") or self.elasticity is None:
            return

        try:
            X_nodes_np = getattr(self.elasticity, "X_nodes", None)
            if X_nodes_np is None:
                X_nodes_np = self.elasticity.X_nodes_tf.numpy()
            n_nodes = int(X_nodes_np.shape[0])
            k = int(getattr(field_cfg, "graph_k", 0) or 0)
            if n_nodes <= 0 or k <= 0:
                return

            cache_path = None
            loaded = False
            if cfg.graph_cache_enabled:
                cache_path = self._graph_cache_path(n_nodes)
                if os.path.exists(cache_path):
                    data = np.load(cache_path)
                    knn_idx = data.get("knn_idx")
                    if knn_idx is not None and knn_idx.shape == (n_nodes, k):
                        knn_tf = tf.convert_to_tensor(knn_idx, dtype=tf.int32)
                        self.model.field._global_knn_idx = knn_tf
                        self.model.field._global_knn_n = n_nodes
                        self.model.field._global_adj = _knn_to_adj(knn_tf, n_nodes)
                        loaded = True
                        print(f"[graph] Loaded cached kNN: {cache_path}")
            if not loaded:
                self.model.field.prebuild_adjacency(X_nodes_np)
                print(f"[graph] Pre-built kNN: N={n_nodes} k={k}")
                if cfg.graph_cache_enabled and cache_path:
                    try:
                        knn_idx_np = self.model.field._global_knn_idx.numpy()
                        np.savez_compressed(cache_path, knn_idx=knn_idx_np)
                        print(f"[graph] Saved kNN cache: {cache_path}")
                    except Exception as exc:
                        print(f"[graph] Cache save failed: {exc}")
        except Exception as exc:
            print(f"[trainer] WARNING: 预构建全局邻接失败，将退回动态构图：{exc}")

    def _supervision_load_splits(self) -> Tuple[str, ...]:
        sup_cfg = getattr(self.cfg, "supervision", None)
        if sup_cfg is None:
            return ("train",)

        ordered: List[str] = []
        for raw in (
            getattr(sup_cfg, "train_splits", ("train",)) or ("train",),
            getattr(sup_cfg, "eval_splits", ()) or (),
        ):
            for split in raw:
                name = str(split).strip()
                if name and name not in ordered:
                    ordered.append(name)
        if bool(getattr(self.cfg, "viz_supervision_compare_enabled", False)):
            name = str(getattr(self.cfg, "viz_supervision_compare_split", "test") or "").strip()
            if name and name not in ordered:
                ordered.append(name)
        return tuple(ordered) if ordered else ("train",)

    def _autoguess_contacts_from_inp(self, asm: AssemblyModel) -> List[Dict[str, str]]:
        candidates = []
        try:
            # 0) 直接读取 asm.contact_pairs（通常是 ContactPair dataclass 列表）
            raw = getattr(asm, "contact_pairs", None)
            cand = self._normalize_pairs(raw)
            if cand:
                return cand

            # 1) 若模型实现了 autoguess_contact_pairs()
            if hasattr(asm, "autoguess_contact_pairs") and callable(asm.autoguess_contact_pairs):
                pairs = asm.autoguess_contact_pairs()
                cand = self._normalize_pairs(pairs)
                if cand:
                    return cand

            # 2) 兜底：常见属性名
            for attr in ["contacts", "contact_pairs", "interactions", "contact", "pairs"]:
                obj = getattr(asm, attr, None)
                cand = self._normalize_pairs(obj)
                if cand:
                    candidates.extend(cand)

            # 去重
            unique, seen = [], set()
            for d in candidates:
                key = (d.get("master_key"), d.get("slave_key"))
                if key not in seen and all(key):
                    unique.append(d);
                    seen.add(key)
            return unique
        except Exception:
            return []

    @staticmethod
    def _normalize_pairs(obj: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if obj is None:
            return out
        # 统一成可迭代
        seq = obj
        if isinstance(obj, dict):
            seq = [obj]
        elif not isinstance(obj, (list, tuple)):
            seq = [obj]

        for item in seq:
            # 1) 显式 (master, slave)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                m, s = item[0], item[1]
                out.append({"master_key": str(m), "slave_key": str(s)})
                continue
            # 2) dict
            if isinstance(item, dict):
                keys = {k.lower(): v for k, v in item.items()}
                m = keys.get("master_key") or keys.get("master") or keys.get("a")
                s = keys.get("slave_key") or keys.get("slave") or keys.get("b")
                if m and s:
                    out.append({"master_key": str(m), "slave_key": str(s)})
                continue
            # 3) dataclass / 任意对象：有 .master / .slave 属性即可
            m = getattr(item, "master", None)
            s = getattr(item, "slave", None)
            if m is not None and s is not None:
                out.append({"master_key": str(m), "slave_key": str(s)})
                continue
        return out

    # ----------------- Build -----------------
    def build(self):
        cfg = self.cfg

        def _raise_vol_error(enum_names, X_vol, w_vol, mat_id):
            enum_str = ", ".join(f"{i}->{n}" for i, n in enumerate(enum_names))
            shapes = dict(
                X_vol=None if X_vol is None else tuple(getattr(X_vol, "shape", [])),
                w_vol=None if w_vol is None else tuple(getattr(w_vol, "shape", [])),
                mat_id=None if mat_id is None else tuple(getattr(mat_id, "shape", [])),
            )
            msg = (
                "\n[trainer] ERROR: build_volume_points 未返回有效体积分点；训练终止。\n"
                f"  - 材料枚举(按 part2mat 顺序)：{enum_str}\n"
                f"  - 返回 shapes: {shapes}\n"
                "  - 常见原因：\n"
                "      * INP 中的零件名与 part2mat 的键不一致（大小写/空格）。\n"
                "      * 材料名不在 materials 字典里。\n"
                "      * 网格上没有体积分点（或被过滤为空）。\n"
                "  - 建议：检查 INP 的 part2mat 配置与网格数据，确保体积分点和材料映射正确生成。\n"
            )
            raise RuntimeError(msg)

        steps = [
            "Load Mesh", "Volume/Materials", "Elasticity",
            "Contact", "Tightening", "Ties/BCs",
            "Model/Opt", "Checkpoint"
        ]

        print(f"[INFO] Build.start  mesh_path={cfg.inp_path}")

        pb_kwargs = dict(
            total=len(steps),
            desc="Build",
            leave=True,
            disable=not (self._tqdm_enabled and self.cfg.build_bar_enabled),
        )
        if cfg.build_bar_color:
            pb_kwargs["colour"] = cfg.build_bar_color
        with tqdm(**pb_kwargs) as pb:
            # 1) Mesh
            ext = os.path.splitext(cfg.inp_path)[1].lower()
            if ext == ".cdb":
                self.asm = load_cdb(cfg.inp_path)
                mesh_tag = "CDB"
            elif ext == ".inp":
                from inp_io.inp_parser import load_inp
                self.asm = load_inp(cfg.inp_path)
                mesh_tag = "INP"
            else:
                raise ValueError(f"[trainer] Only .cdb or .inp supported, got: {cfg.inp_path}")
            print(f"[INFO] Loaded {mesh_tag}: surfaces={len(self.asm.surfaces)} "
                  f"elsets={len(self.asm.elsets)} contact_pairs(raw)={len(getattr(self.asm, 'contact_pairs', []))}")
            pb.update(1)

            self._supervision_dataset = None
            sup_cfg = getattr(cfg, "supervision", None)
            if bool(getattr(sup_cfg, "enabled", False)):
                self._supervision_dataset = load_ansys_supervision_dataset(
                    case_table_path=str(getattr(sup_cfg, "case_table_path", "") or ""),
                    stage_dir=str(getattr(sup_cfg, "stage_dir", "") or ""),
                    asm=self.asm,
                    splits=self._supervision_load_splits(),
                    stage_count=int(getattr(sup_cfg, "stage_count", 3) or 3),
                    single_case_id=(None if getattr(sup_cfg, "single_case_id", None) in (None, "") else str(getattr(sup_cfg, "single_case_id")).strip()),
                    single_case_stages=tuple(int(x) for x in (getattr(sup_cfg, "single_case_stages", ()) or ())),
                    feature_mode=str(getattr(sup_cfg, "feature_mode", "cartesian") or "cartesian"),
                    target_frame=str(getattr(sup_cfg, "target_frame", "cartesian") or "cartesian"),
                    annulus_center=getattr(sup_cfg, "annulus_center", None),
                    annulus_r_in=getattr(sup_cfg, "annulus_r_in", None),
                    annulus_r_out=getattr(sup_cfg, "annulus_r_out", None),
                    annulus_fourier_order=int(getattr(sup_cfg, "annulus_fourier_order", 0) or 0),
                    shuffle=bool(getattr(sup_cfg, "shuffle", True)),
                    seed=int(getattr(sup_cfg, "seed", cfg.seed)),
                    split_group_key=str(getattr(sup_cfg, "split_group_key", "base_id") or "base_id"),
                    split_stratify_key=getattr(sup_cfg, "split_stratify_key", "source"),
                    test_group_quotas=getattr(sup_cfg, "test_group_quotas", None),
                    cv_n_folds=int(getattr(sup_cfg, "cv_n_folds", 5)),
                    cv_fold_index=int(getattr(sup_cfg, "cv_fold_index", 0)),
                )
                print(f"[supervision] loaded staged ANSYS cases: {self._supervision_dataset.counts()}")

            # 2) 体积分点 & 材料映射（严格检查）
            self.matlib = MaterialLibrary(cfg.materials)
            X_vol, w_vol, mat_id = build_volume_points(self.asm, cfg.part2mat, self.matlib)

            enum_names = list(dict.fromkeys(cfg.part2mat.values()))
            enum_str = ", ".join(f"{i}->{n}" for i, n in enumerate(enum_names))
            print(f"[trainer] Material enum (from part2mat order): {enum_str}")

            # —— 严格检查
            if X_vol is None or w_vol is None or mat_id is None:
                _raise_vol_error(enum_names, X_vol, w_vol, mat_id)

            n = getattr(X_vol, "shape", [0])[0]
            if getattr(w_vol, "shape", [0])[0] != n or getattr(mat_id, "shape", [0])[0] != n or n == 0:
                _raise_vol_error(enum_names, X_vol, w_vol, mat_id)

            # —— 暴露到 Trainer
            self.X_vol = X_vol
            self.w_vol = w_vol
            self.mat_id = mat_id
            self.enum_names = enum_names
            def _extract_E_nu(tag: str, spec: Any) -> Tuple[float, float]:
                if isinstance(spec, (tuple, list)) and len(spec) >= 2:
                    return float(spec[0]), float(spec[1])
                if isinstance(spec, dict):
                    return float(spec["E"]), float(spec["nu"])
                raise TypeError(
                    f"[trainer] Material '{tag}' spec must be (E, nu) or dict with keys 'E'/'nu', got {type(spec)}"
                )

            self.id2props_map = {
                i: _extract_E_nu(name, cfg.materials[name]) for i, name in enumerate(enum_names)
            }

            pb.update(1)

            # 3) Elasticity (residual)
            self.elasticity = ElasticityResidual(
                asm=self.asm,
                X_vol=X_vol,
                w_vol=w_vol,
                mat_id=mat_id,
                matlib=self.matlib,
                materials=cfg.materials,
                cfg=cfg.elas_cfg,
            )
            if hasattr(self.elasticity, "set_sample_metrics_cache_enabled"):
                self.elasticity.set_sample_metrics_cache_enabled(
                    False
                )
            print("[trainer] Elasticity: residual (volume points)")
            pb.update(1)

            # 4) 接触（优先使用 cfg；否则尝试自动探测）
            self._cp_specs = []
            contact_source = ""
            if cfg.contact_pairs:
                try:
                    self._cp_specs = [ContactPairSpec(**d) for d in cfg.contact_pairs]
                except TypeError:
                    norm = self._normalize_pairs(cfg.contact_pairs)
                    self._cp_specs = [ContactPairSpec(**d) for d in norm] if norm else []
                contact_source = "配置"
            else:
                auto_pairs = self._autoguess_contacts_from_inp(self.asm)
                if auto_pairs:
                    self._cp_specs = [ContactPairSpec(**d) for d in auto_pairs]
                    contact_source = "自动识别"

            self.contact = None
            if self._cp_specs:
                try:
                    cmap = build_contact_map(
                        self.asm,
                        self._cp_specs,
                        cfg.n_contact_points_per_pair,
                        seed=cfg.contact_seed,
                        two_pass=cfg.contact_two_pass,
                        mode=cfg.contact_mode,
                        mortar_gauss=cfg.contact_mortar_gauss,
                        mortar_max_points=cfg.contact_mortar_max_points,
                    )
                    cat = cmap.concatenate()
                    self.contact = ContactOperator(cfg.contact_cfg)
                    self.contact.build_from_cat(cat, extra_weights=None, auto_orient=True)
                    self._current_contact_cat = cat
                    self._init_contact_hardening()
                    total_pts = len(cmap)
                    src_txt = f"（{contact_source}）" if contact_source else ""
                    print(
                        f"[contact] 已加载 {len(self._cp_specs)} 对接触面{src_txt}，"
                        f"采样 {total_pts} 个点。"
                    )
                except Exception as exc:
                    print(f"[contact] 构建接触失败：{exc}")
                    self.contact = None
            else:
                print("[contact] 未找到接触信息，训练将不启用接触。")

            pb.update(1)

            # 5) 螺母拧紧（旋转角）
            if cfg.preload_specs:
                try:
                    specs = [NutSpec(**d) for d in cfg.preload_specs]
                    self.tightening = NutTighteningPenalty(cfg.tightening_cfg)
                    self.tightening.build_from_specs(
                        self.asm,
                        specs,
                        n_points_each=cfg.preload_n_points_each,
                        seed=cfg.seed,
                    )
                    print(f"[tightening] 已配置 {len(specs)} 个螺母表面样本。")
                except Exception as exc:
                    print(f"[tightening] 构建拧紧样本失败：{exc}")
                    self.tightening = None
            else:
                self.tightening = None
                print("[tightening] 未提供螺母拧紧配置。")
            pb.update(1)

            # 6) Ties/BCs（如需，可在 cfg 里填充）
            self.bcs_ops = []
            pb.update(1)

            # 6.5) 根据预紧特征维度统一 ParamEncoder 输入形状，避免 staged 特征长度变化
            self._warmup_case = self._make_warmup_case()
            self._warmup_params = self._make_preload_params(self._warmup_case)
            feat_dim = self._infer_preload_feat_dim(self._warmup_params)
            if feat_dim:
                old_dim = getattr(cfg.model_cfg.encoder, "in_dim", None)
                if old_dim != feat_dim:
                    print(
                        f"[model] 预紧特征维度 {old_dim} -> {feat_dim}，统一 ParamEncoder 输入。"
                    )
                    cfg.model_cfg.encoder.in_dim = feat_dim

            # 7) 模型 & 优化器
            if getattr(cfg.model_cfg.field.fourier, "seed", None) is None:
                cfg.model_cfg.field.fourier.seed = int(cfg.seed)
            if cfg.mixed_precision:
                cfg.model_cfg.mixed_precision = cfg.mixed_precision
            self.model = create_displacement_model(cfg.model_cfg)

            self._maybe_prebuild_global_graph()

            # Attach engineering semantics (optional, default-off).
            if bool(getattr(cfg.model_cfg.field, "use_engineering_semantics", False)):
                try:
                    sem_feat = self._build_node_semantic_features(
                        self.asm,
                        sorted_node_ids=self.elasticity.sorted_node_ids,
                        part2mat=cfg.part2mat,
                        mirror_surface_name=cfg.mirror_surface_name,
                    )
                    expected_dim = int(getattr(cfg.model_cfg.field, "semantic_feat_dim", 0) or 0)
                    if expected_dim > 0 and sem_feat.shape[1] != expected_dim:
                        if sem_feat.shape[1] > expected_dim:
                            sem_feat = sem_feat[:, :expected_dim]
                        else:
                            pad = np.zeros((sem_feat.shape[0], expected_dim - sem_feat.shape[1]), dtype=np.float32)
                            sem_feat = np.concatenate([sem_feat, pad], axis=1)
                    self.model.field.set_node_semantic_features(sem_feat)
                    print(
                        f"[model] attached engineering semantic features: "
                        f"{sem_feat.shape[0]}x{sem_feat.shape[1]}"
                    )
                except Exception as exc:
                    print(f"[trainer] WARNING: 语义特征构建/挂载失败，将继续无语义特征训练：{exc}")
            base_optimizer = tf.keras.optimizers.Adam(cfg.lr)
            mp_policy = str(cfg.mixed_precision or "").strip().lower()
            use_loss_scale = mp_policy.startswith("mixed_")
            if use_loss_scale:
                base_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                print("[trainer] 已启用 LossScaleOptimizer 以配合混合精度训练。")
            self.optimizer = base_optimizer
            self._apply_gradients_kwargs = self._compute_apply_gradients_kwargs(self.optimizer)
            pb.update(1)

            # 8) checkpoint
            self.ckpt = tf.train.Checkpoint(
                encoder=self.model.encoder,
                field=self.model.field,
                opt=self.optimizer,
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, directory=cfg.ckpt_dir, max_to_keep=int(cfg.ckpt_max_to_keep)
            )
            pb.update(1)

        # 预热网络，确保所有权重在进入梯度带之前已创建，从而可以被显式 watch
        try:
            warmup_n = min(2048, int(self.X_vol.shape[0])) if hasattr(self, "X_vol") else 0
        except Exception:
            warmup_n = 0
        if warmup_n > 0:
            X_sample = tf.convert_to_tensor(self.X_vol[:warmup_n], dtype=tf.float32)
            params = self._warmup_params or self._make_preload_params(self._make_warmup_case())
            eval_params = self._extract_final_stage_params(params)
            # 调用一次前向以创建所有变量；忽略实际输出
            _ = self.model.u_fn(X_sample, eval_params)

        raw_train_vars = (
            list(self.model.encoder.trainable_variables)
            + list(self.model.field.trainable_variables)
        )
        self._train_vars = self._apply_trainable_scope(raw_train_vars)
        if not self._train_vars:
            raise RuntimeError(
                "[trainer] 未发现可训练权重，请检查模型创建/预热流程是否成功。"
            )
        trainable_scope = str(getattr(self.cfg, "trainable_scope", "all") or "all")
        print(f"[trainer] trainable scope = {trainable_scope} ({len(self._train_vars)} vars)")

        print(f"[trainer] GPU allocator = {os.environ.get('TF_GPU_ALLOCATOR', '(default)')}")
        print(
            f"[contact] 状态：{'已启用' if self.contact is not None else '未启用'}"
        )
        print(
            f"[tightening] 状态：{'已启用' if self.tightening is not None else '未启用'}"
        )

    # ----------------- 组装总能量 -----------------
