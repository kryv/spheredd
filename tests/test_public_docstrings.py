"""Checks for the detailed public API documentation."""

from __future__ import annotations

import inspect

from spheredd import evaluate_moments, spherical_design


def test_spherical_design_docstring_documents_contract() -> None:
    doc = inspect.getdoc(spherical_design)
    assert doc is not None
    for section in ("Parameters", "Returns", "Raises", "Notes", "Examples"):
        assert section in doc
    assert "(radius**2 / r) * retained_covariance" in doc
    assert "actual_n_points" in doc
    assert "base_design_strength" in doc


def test_evaluate_moments_docstring_documents_scaling() -> None:
    doc = inspect.getdoc(evaluate_moments)
    assert doc is not None
    for section in ("Parameters", "Returns", "Raises", "Notes", "Examples"):
        assert section in doc
    assert "sqrt(source_rank) / radius" in doc
    assert "radius=None" in doc
    assert "dimension**k" in doc
