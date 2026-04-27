#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-facing facade for the normal-contact-first training protocol."""

from __future__ import annotations


def resolve_normal_contact_runtime_settings(controller, *, route_mode: str):
    return controller._resolve_coupling_tightening_runtime_settings(route_mode=route_mode)


def resolve_normal_contact_protocol_stats(controller, *, route_mode: str, refinement_steps: int):
    return controller._resolve_coupling_tightening_stats(
        route_mode=route_mode,
        refinement_steps=refinement_steps,
    )
