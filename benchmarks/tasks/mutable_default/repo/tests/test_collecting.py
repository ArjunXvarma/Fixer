from listutils.collecting import collect


def test_single_call():
    assert collect("a") == ["a"]
