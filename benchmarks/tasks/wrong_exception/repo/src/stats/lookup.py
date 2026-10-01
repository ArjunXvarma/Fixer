def lookup(mapping, key):
    """Return mapping[key].

    Raises KeyError when the key is missing.
    """
    if key not in mapping:
        raise Exception(f"missing key: {key}")

    return mapping[key]
