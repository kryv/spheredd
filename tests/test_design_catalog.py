from __future__ import annotations

import numpy as np
import pytest

from spheredd._designs import get_base_design
from spheredd._validation import spherical_design_max_error


EXPECTED_SIZES = {
    2: {2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
    4: {2: 5, 3: 12, 4: 20, 5: 29, 6: 27},
    6: {2: 7, 3: 24, 4: 42, 5: 87, 6: 146},
}


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("rank", [2, 3, 4, 5, 6])
def test_base_sizes_and_design_conditions(order: int, rank: int) -> None:
    design = get_base_design(rank, order)
    assert design.points.shape == (EXPECTED_SIZES[order][rank], rank)
    assert not design.points.flags.writeable
    assert design.strength >= order
    assert spherical_design_max_error(design.points, order) < 5e-13
    assert np.allclose(np.linalg.norm(design.points, axis=1), 1.0, atol=5e-13)


def test_rank3_order6_is_a_7_design_but_not_an_8_design() -> None:
    design = get_base_design(3, 6)
    assert design.strength == 7
    assert spherical_design_max_error(design.points, 7) < 5e-13
    assert spherical_design_max_error(design.points, 8) > 1e-4


def test_numerical_design_metadata() -> None:
    assert not get_base_design(5, 4).exact
    assert not get_base_design(5, 6).exact
    assert not get_base_design(6, 6).exact
    assert get_base_design(6, 4).exact
    assert "minimum unknown" in get_base_design(5, 6).cardinality_status
