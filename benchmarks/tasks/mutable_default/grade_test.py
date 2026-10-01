from listutils.collecting import collect


def test_calls_are_independent():
    assert collect("a") == ["a"]
    assert collect("b") == ["b"]
    assert collect("c") == ["c"]


def test_explicit_target_still_works():
    target = ["x"]
    assert collect("y", target) == ["x", "y"]
    assert target == ["x", "y"]
