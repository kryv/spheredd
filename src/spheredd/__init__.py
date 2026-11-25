"""Equal-weight spherical 2-, 4-, and 6-designs for NumPy and PyTorch."""

from ._api import (
    evaluate_moments,
    evaluate_moments2,
    evaluate_moments4,
    evaluate_moments6,
    spherical_2_design,
    spherical_4_design,
    spherical_6_design,
    spherical_design,
    torch_available,
)

__all__ = [
    "evaluate_moments",
    "evaluate_moments2",
    "evaluate_moments4",
    "evaluate_moments6",
    "spherical_2_design",
    "spherical_4_design",
    "spherical_6_design",
    "spherical_design",
    "torch_available",
]

__version__ = "0.1.2"
