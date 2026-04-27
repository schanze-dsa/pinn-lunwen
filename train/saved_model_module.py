# -*- coding: utf-8 -*-
"""SavedModel export module extracted from trainer.py."""

from __future__ import annotations

import tensorflow as tf

from model.pinn_model import DisplacementModel


def ensure_partial_restore_compat(status):
    """Allow partial checkpoint restores for backward-compatible mixed upgrades."""

    expect_partial = getattr(status, "expect_partial", None)
    if callable(expect_partial):
        try:
            expect_partial()
        except Exception:
            pass
    return status


class _SavedModelModule(tf.Module):
    """TensorFlow module exposing the PINN forward pass for SavedModel export."""

    @tf.autograph.experimental.do_not_convert
    def __init__(
        self,
        model: DisplacementModel,
        use_stages: bool,
        append_release_stage: bool,
        shift: float,
        scale: float = 1.0,
        n_bolts: int = 3,
    ):
        # Avoid zero-arg super() here: some TF/AutoGraph versions may attempt to
        # convert __init__ and fail to resolve the implicit __class__ cell.
        tf.Module.__init__(self, name="pinn_saved_model")
        # 1. 显式追踪子模块 (关键修复)
        # 将 DisplacementModel 的核心子层挂载到 self 上，确保 TF 能追踪到变量
        self.encoder = model.encoder
        self.field = model.field
        
        # 2. 保留原始模型的引用 (用于调用 u_fn)
        # 注意：直接用 self._model.u_fn 可能会导致追踪路径断裂
        # 我们需要确保 u_fn 使用的 encoder/field 就是上面挂载的这两个
        self._model = model

        self._use_stages = bool(use_stages)
        self._append_release_stage = bool(append_release_stage)
        self._shift = tf.constant(shift, dtype=tf.float32)
        self._scale = tf.constant(scale, dtype=tf.float32)
        self._n_bolts = int(max(1, n_bolts))

    @tf.function(
        autograph=False,
        input_signature=[
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="x"),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name="p"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="order"),
        ]
    )
    def run(self, x, p, order):
        # 准备参数
        params = self._prepare_params(p, order)
        
        # 调用模型的前向传播
        # 由于 self._model.encoder 就是 self.encoder，变量是共享且被追踪的
        return self._model.u_fn(x, params)

    def _prepare_params(self, P, order):
        # 确保 P 是 1D
        P = tf.reshape(P, (self._n_bolts,))
        
        # 如果不启用分阶段，直接返回 P
        if not self._use_stages:
            return {"P": P}
            
        # 归一化顺序
        order = self._normalize_order(order)
        
        # 构建阶段张量 (包含 P_hat 特征)
        stage_P, stage_feat, rank_vec, stage_count_total = self._build_stage_tensors(P, order)
        
        # 返回最后一个阶段的数据，并补充与训练可视化一致的上下文字段。
        return {
            "P": stage_P[-1],
            "P_hat": stage_feat[-1],
            "stage_order": order,
            "stage_rank": rank_vec,
            "stage_count": tf.cast(stage_count_total, tf.int32),
        }

    def _normalize_order(self, order):
        order = tf.reshape(order, (self._n_bolts,))
        default = tf.range(self._n_bolts, dtype=tf.int32)
        
        # 检查是否全部 >= 0
        cond = tf.reduce_all(order >= 0)
        order = tf.where(cond, order, default)
        
        # 检查是否需要从 1-based 转 0-based
        minv = tf.reduce_min(order)
        maxv = tf.reduce_max(order)

        def _one_based():
            return order - 1

        order = tf.cond(
            tf.logical_and(tf.greater_equal(minv, 1), tf.less_equal(maxv, self._n_bolts)),
            _one_based,
            lambda: order,
        )
        return order

    def _build_stage_tensors(self, P, order):
        stage_count_bolts = self._n_bolts
        stage_count_total = stage_count_bolts + (1 if self._append_release_stage else 0)
        cumulative = tf.zeros_like(P)
        mask = tf.zeros_like(P)
        
        # 使用 TensorArray 动态构建序列
        loads_ta = tf.TensorArray(tf.float32, size=stage_count_bolts)
        masks_ta = tf.TensorArray(tf.float32, size=stage_count_bolts)
        last_ta = tf.TensorArray(tf.float32, size=stage_count_bolts)

        def body(i, cum, mask_vec, loads, masks, lasts):
            # 获取当前步骤要拧的螺栓索引
            bolt = tf.gather(order, i)
            bolt = tf.clip_by_value(bolt, 0, self._n_bolts - 1)
            
            # 获取该螺栓的力
            load_val = tf.gather(P, bolt)
            idx = tf.reshape(bolt, (1, 1))
            
            # 更新累积载荷 (cumulative)
            cum = tf.tensor_scatter_nd_update(cum, idx, tf.reshape(load_val, (1,)))
            
            # 更新掩码 (mask)
            mask_vec = tf.tensor_scatter_nd_update(
                mask_vec, idx, tf.ones((1,), dtype=tf.float32)
            )
            
            # 记录到 Array
            loads = loads.write(i, cum)
            masks = masks.write(i, mask_vec)
            
            # 构建 last_active (当前操作的螺栓)
            last_vec = tf.zeros_like(P)
            last_vec = tf.tensor_scatter_nd_update(
                last_vec, idx, tf.ones((1,), dtype=tf.float32)
            )
            lasts = lasts.write(i, last_vec)
            
            return i + 1, cum, mask_vec, loads, masks, lasts

        _, cumulative, mask, loads_ta, masks_ta, last_ta = tf.while_loop(
            lambda i, *_: tf.less(i, stage_count_bolts),
            body,
            (0, cumulative, mask, loads_ta, masks_ta, last_ta),
        )

        stage_P = loads_ta.stack()
        stage_masks = masks_ta.stack()
        stage_last = last_ta.stack()

        if self._append_release_stage:
            # Final post-tightening stage: all bolts locked, no active force control.
            stage_P = tf.concat([stage_P, tf.expand_dims(cumulative, axis=0)], axis=0)
            stage_masks = tf.concat([stage_masks, tf.expand_dims(mask, axis=0)], axis=0)
            stage_last = tf.concat(
                [stage_last, tf.expand_dims(tf.zeros_like(P), axis=0)], axis=0
            )

        # 构建 Rank 矩阵
        indices = tf.reshape(order, (-1, 1))
        ranks = tf.cast(tf.range(stage_count_bolts), tf.float32)
        rank_vec = tf.tensor_scatter_nd_update(
            tf.zeros((self._n_bolts,), tf.float32), indices, ranks
        )
        if stage_count_bolts > 1:
            rank_vec = rank_vec / tf.cast(stage_count_bolts - 1, tf.float32)
        else:
            rank_vec = tf.zeros_like(rank_vec)

        # 拼接最终特征 P_hat。
        # 必须与 Trainer._make_preload_params 的 staged 特征定义一致：
        # [norm, mask, last, rank, t_stage, delta_t]
        feats_ta = tf.TensorArray(tf.float32, size=stage_count_total)
        tighten_time = rank_vec
        for i in range(stage_count_total):
            # 归一化 P
            norm = (stage_P[i] - self._shift) / self._scale
            if stage_count_total > 1:
                t_stage = tf.cast(i, tf.float32) / tf.cast(stage_count_total - 1, tf.float32)
            else:
                t_stage = tf.cast(0.0, tf.float32)
            delta_t = tf.maximum(tf.cast(0.0, tf.float32), t_stage - tighten_time)
            feat = tf.concat(
                [norm, stage_masks[i], stage_last[i], rank_vec, tf.reshape(t_stage, (1,)), delta_t],
                axis=0,
            )
            feats_ta = feats_ta.write(i, feat)

        stage_feat = feats_ta.stack()
        return stage_P, stage_feat, rank_vec, stage_count_total


