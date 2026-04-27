#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-facing facade for the differentiable inner contact layer."""

from __future__ import annotations

from physics.contact.contact_implicit_backward import (
    _validate_normal_contact_linearization as _validate_normal_contact_linearization,
)
from physics.contact.contact_implicit_backward import (
    attach_normal_contact_implicit_backward as _attach_normal_contact_implicit_backward,
)
from physics.contact.contact_inner_solver import solve_contact_inner as _solve_contact_inner
from physics.contact.local_error_bound_analysis import (
    analyze_local_error_bounds as _analyze_local_error_bounds,
)


def solve_differentiable_inner_contact(*args, **kwargs):
    return _solve_contact_inner(*args, **kwargs)


def validate_normal_contact_first_contract(linearization):
    return _validate_normal_contact_linearization(linearization)


def attach_normal_contact_implicit_backward(flat_state, flat_inputs, linearization):
    return _attach_normal_contact_implicit_backward(flat_state, flat_inputs, linearization)


def analyze_local_error_bounds(linearization, residual_perturbation):
    return _analyze_local_error_bounds(linearization, residual_perturbation)
