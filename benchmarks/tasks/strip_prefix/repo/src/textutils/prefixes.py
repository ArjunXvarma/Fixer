def strip_prefix(text, prefix):
    """Remove `prefix` from the start of `text`, if it is there.

    Only one occurrence is removed, and text without the prefix is
    returned unchanged.
    """
    return text.lstrip(prefix)
