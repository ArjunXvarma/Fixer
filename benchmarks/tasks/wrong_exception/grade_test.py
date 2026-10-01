import pytest

from stats.lookup import lookup


def test_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        lookup({}, "missing")


def test_found_still_works():
    assert lookup({"a": 1}, "a") == 1
