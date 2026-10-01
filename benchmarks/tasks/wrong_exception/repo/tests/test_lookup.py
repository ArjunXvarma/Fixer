from stats.lookup import lookup


def test_found():
    assert lookup({"a": 1}, "a") == 1
