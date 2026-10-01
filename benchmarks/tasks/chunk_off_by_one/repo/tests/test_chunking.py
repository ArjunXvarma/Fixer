from listutils.chunking import chunk


def test_exact_multiple():
    assert chunk([1, 2, 3, 4, 5, 6], 3) == [[1, 2, 3], [4, 5, 6]]
