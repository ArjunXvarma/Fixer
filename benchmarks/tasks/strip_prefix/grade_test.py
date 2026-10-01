from textutils.prefixes import strip_prefix


def test_removes_one_occurrence_only():
    assert strip_prefix("aab", "a") == "ab"
    assert strip_prefix("xxy", "x") == "xy"


def test_absent_prefix_is_unchanged():
    assert strip_prefix("hello", "abc") == "hello"
    assert strip_prefix("banana", "na") == "banana"


def test_ordinary_case_still_works():
    assert strip_prefix("test_foo", "test_") == "foo"
