import pytest

from enet import reference


@pytest.mark.parametrize(
    "z, gamma, expected",
    [
        (3.0, 1.0, 2.0),
        (-3.0, 1.0, -2.0),
        (0.5, 1.0, 0.0),
        (-0.5, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ],
)
def test_soft_threshold(z, gamma, expected):
    assert reference.soft_threshold(z, gamma) == pytest.approx(expected)
