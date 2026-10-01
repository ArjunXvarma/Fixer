"""Hidden oracle: the agent never sees this file."""

from testpkg.tribonacci import tribonacci

# 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149
EXPECTED = [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]


def test_documented_sequence():
    assert [tribonacci(n) for n in range(len(EXPECTED))] == EXPECTED


def test_the_reported_bug():
    assert tribonacci(0) == 0
    assert tribonacci(3) == 2
