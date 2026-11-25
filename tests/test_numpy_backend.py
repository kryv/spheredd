from __future__ import annotations

import numpy as np
import pytest

from spheredd import (
    evaluate_moments,
    evaluate_moments2,
    evaluate_moments4,
    evaluate_moments6,
    spherical_2_design,
    spherical_4_design,
    spherical_6_design,
    spherical_design,
)


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("rank", [2, 3, 4, 5, 6])
def test_numpy_generation_and_covariance_recovery(order: int, rank: int) -> None:
    diagonal = np.linspace(0.5, 2.0, rank)
    covariance = np.diag(diagonal)
    mean = np.linspace(-0.2, 0.2, rank)

    points, weights, info = spherical_design(
        covariance,
        n_points=1,
        mean=mean,
        design_order=order,
    )

    assert isinstance(points, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert info["backend"] == "numpy"
    assert info["rank"] == rank
    assert info["actual_n_points"] == info["base_size"]
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights @ points, mean, atol=2e-12)

    moments = evaluate_moments(
        points,
        weights,
        max_order=order,
        radius=info["radius"],
        source_rank=info["rank"],
    )
    assert np.allclose(moments["covariance"], covariance, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize(
    ("wrapper", "helper", "order"),
    [
        (spherical_2_design, evaluate_moments2, 2),
        (spherical_4_design, evaluate_moments4, 4),
        (spherical_6_design, evaluate_moments6, 6),
    ],
)
def test_wrappers(wrapper, helper, order: int) -> None:
    points, weights, info = wrapper(np.eye(4), 1)
    result = helper(points, weights, radius=1.0, source_rank=4)
    assert info["design_order"] == order
    assert max(result["moments"]) == order


def test_complete_block_rounding_and_seed_reproducibility() -> None:
    first = spherical_design(np.eye(5), 88, design_order=6, seed=123)
    second = spherical_design(np.eye(5), 88, design_order=6, seed=123)
    points, weights, info = first

    assert info["base_size"] == 87
    assert info["copies"] == 2
    assert info["actual_n_points"] == 174
    assert np.array_equal(points, second[0])
    assert np.array_equal(weights, second[1])

    result = evaluate_moments6(points, weights, radius=1.0, source_rank=5)
    assert np.allclose(result["covariance"], np.eye(5), atol=5e-11, rtol=5e-11)


def test_numerical_rank_can_be_smaller_than_ambient_dimension() -> None:
    rng = np.random.default_rng(4)
    factor = rng.normal(size=(9, 6))
    covariance = factor @ factor.T
    points, weights, info = spherical_design(covariance, 1, design_order=6)

    assert points.shape == (146, 9)
    assert weights.shape == (146,)
    assert info["rank"] == 6


def test_radius_sqrt_rank_matches_input_covariance_directly() -> None:
    rank = 4
    covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    points, weights, _ = spherical_design(
        covariance,
        1,
        design_order=4,
        radius=np.sqrt(rank),
    )
    centroid = weights @ points
    centered = points - centroid
    observed = np.einsum("n,ni,nj->ij", weights, centered, centered)
    assert np.allclose(observed, covariance, atol=2e-12, rtol=2e-12)


def test_float32_is_supported() -> None:
    covariance = np.eye(4, dtype=np.float32)
    points, weights, info = spherical_design(covariance, 1, design_order=6)
    assert points.dtype == np.float32
    assert weights.dtype == np.float32
    assert info["dtype"] == "float32"


def test_zero_covariance_and_validation_errors() -> None:
    points, weights, info = spherical_design(np.zeros((3, 3)), 4, design_order=6)
    assert points.shape == (4, 3)
    assert np.all(points == 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert info["rank"] == 0

    with pytest.raises(ValueError, match="square"):
        spherical_design(np.zeros((2, 3)), 1)
    with pytest.raises(ValueError, match="symmetric"):
        spherical_design(np.array([[1.0, 1.0], [0.0, 1.0]]), 1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        spherical_design(np.diag([1.0, -1.0]), 1)
    with pytest.raises(ValueError, match="2, 4, or 6"):
        spherical_design(np.eye(2), 1, design_order=3)
    with pytest.raises(ValueError, match="device"):
        spherical_design(np.eye(2), 1, device="cpu")
