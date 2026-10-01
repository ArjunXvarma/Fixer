from textutils.prefixes import strip_prefix


def test_removes_prefix():
    assert strip_prefix("test_foo", "test_") == "foo"
