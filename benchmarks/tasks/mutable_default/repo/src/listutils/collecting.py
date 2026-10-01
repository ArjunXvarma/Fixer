def collect(value, into=[]):
    """Append `value` to `into` and return it.

    Called without `into`, each call starts from an empty list.
    """
    into.append(value)

    return into
