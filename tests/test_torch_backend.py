from __future__ import annotations

import numpy as np
import pytest

import spheredd


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("rank", [2, 3, 4, 5, 6])
def test_torch_auto_dispatch_and_covariance_recovery(order: int, rank: int) -> None:
    covariance = torch.diag(torch.linspace(0.5, 2.0, rank, dtype=torch.float64))
    points, weights, info = spheredd.spherical_design(
        covariance,
        1,
        design_order=order,
    )

    assert torch.is_tensor(points)
    assert torch.is_tensor(weights)
    assert points.dtype == torch.float64
    assert points.device == covariance.device
    assert info["backend"] == "torch"
    assert info["rank"] == rank

    moments = spheredd.evaluate_moments(
        points,
        weights,
        max_order=order,
        radius=1.0,
        source_rank=rank,
    )
    assert torch.allclose(moments["covariance"], covariance, atol=4e-11, rtol=4e-11)


def test_explicit_torch_backend_accepts_numpy_input() -> None:
    points, weights, info = spheredd.spherical_design(
        np.eye(3),
        1,
        design_order=6,
        backend="torch",
        dtype=torch.float64,
        device="cpu",
    )
    assert torch.is_tensor(points)
    assert torch.is_tensor(weights)
    assert points.shape == (24, 3)
    assert info["device"] == "cpu"


def test_torch_float32_and_seeded_multiple_blocks() -> None:
    covariance = torch.eye(5, dtype=torch.float32)
    first = spheredd.spherical_design(covariance, 88, design_order=6, seed=17)
    second = spheredd.spherical_design(covariance, 88, design_order=6, seed=17)
    assert first[0].dtype == torch.float32
    assert first[0].shape == (174, 5)
    assert torch.equal(first[0], second[0])


def test_torch_autograd_reaches_covariance_eigenvalues() -> None:
    diagonal = torch.tensor([0.7, 1.3, 2.1], dtype=torch.float64, requires_grad=True)
    covariance = torch.diag(diagonal)
    points, weights, _ = spheredd.spherical_design(covariance, 1, design_order=4)
    loss = torch.sum(weights[:, None] * points.square())
    loss.backward()

    assert diagonal.grad is not None
    assert torch.all(torch.isfinite(diagonal.grad))
    assert torch.allclose(diagonal.grad, torch.full_like(diagonal, 1.0 / 3.0), atol=2e-12)


def test_torch_available() -> None:
    assert spheredd.torch_available()
