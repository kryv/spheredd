"""Minimal optional PyTorch example."""

import torch

from spheredd import evaluate_moments6, spherical_design

covariance = torch.eye(6, dtype=torch.float64)
points, weights, info = spherical_design(
    covariance,
    n_points=1,
    design_order=6,
)

moments = evaluate_moments6(
    points,
    weights,
    radius=info["radius"],
    source_rank=info["rank"],
)

print(info)
print(moments["covariance"])
