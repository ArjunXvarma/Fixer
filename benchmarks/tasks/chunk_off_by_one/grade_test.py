from listutils.chunking import chunk


def test_keeps_the_partial_chunk():
    assert chunk([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]]


def test_nothing_is_dropped():
    for length in range(1, 12):
        items = list(range(length))
        for size in (1, 2, 3, 5):
            flat = [item for part in chunk(items, size) for item in part]
            assert flat == items, (length, size)


def test_exact_multiple_still_works():
    assert chunk([1, 2, 3, 4, 5, 6], 3) == [[1, 2, 3], [4, 5, 6]]
