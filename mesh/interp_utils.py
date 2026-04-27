# -*- coding: utf-8 -*-
"""
mesh/interp_utils.py

Small interpolation helpers used by physics operators.

The key design constraint for this repo is that the displacement network contains a
GCN backbone, so evaluating ``u_fn`` on different point sets can produce different
graphs and therefore an inconsistent displacement field.  To make the field
well-defined, we evaluate the network once on *mesh nodes* and interpolate values to
arbitrary query points using barycentric weights on surface triangles.
"""

from __future__ import annotations

import tensorflow as tf


def interp_bary_tf(
    u_nodes: tf.Tensor,
    tri_node_idx: tf.Tensor,
    bary: tf.Tensor,
) -> tf.Tensor:
    """
    Barycentric interpolation of nodal values on triangles.

    Parameters
    ----------
    u_nodes : (N_nodes, C) tensor
        Nodal field values (e.g. displacement with C=3).
    tri_node_idx : (N_pts, 3) int tensor
        For each point, indices of the 3 triangle vertices in the nodal array.
    bary : (N_pts, 3) tensor
        Barycentric weights (sum to ~1).

    Returns
    -------
    u_pts : (N_pts, C) tensor
        Interpolated values at query points.
    """

    u_nodes = tf.convert_to_tensor(u_nodes)
    tri_node_idx = tf.cast(tf.convert_to_tensor(tri_node_idx), tf.int32)
    bary = tf.cast(tf.convert_to_tensor(bary), u_nodes.dtype)

    u_tri = tf.gather(u_nodes, tri_node_idx)  # (N_pts, 3, C)
    return tf.reduce_sum(u_tri * bary[:, :, None], axis=1)

