#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-facing facade for mixed residual outer-inner coupling."""

from __future__ import annotations

from model.normal_contact_coupling import (
    assemble_normal_contact_coupling as _assemble_normal_contact_coupling,
)


def assemble_mixed_residual_coupling(**kwargs):
    return _assemble_normal_contact_coupling(**kwargs)
