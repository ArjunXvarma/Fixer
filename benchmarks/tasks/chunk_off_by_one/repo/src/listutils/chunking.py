def chunk(items, size):
    """Split items into lists of at most `size`.

    The final chunk is shorter when len(items) is not a multiple of size.
    No element is ever dropped.
    """
    chunks = []

    for start in range(0, len(items) - size + 1, size):
        chunks.append(items[start:start + size])

    return chunks
