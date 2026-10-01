import pytest

from stats.average import mean


def test_empty_is_zero():
    assert mean([]) == 0.0


def test_normal_cases_unchanged():
    assert mean([1, 2, 3]) == 2
    assert mean([2.5]) == pytest.approx(2.5)
