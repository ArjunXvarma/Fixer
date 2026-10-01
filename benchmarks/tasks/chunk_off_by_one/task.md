# chunk() silently drops the last items

Splitting a list of 7 into chunks of 3 loses the tail:

```python
from listutils.chunking import chunk

chunk([1, 2, 3, 4, 5, 6, 7], 3)
# [[1, 2, 3], [4, 5, 6]]   -- where did 7 go?
```

The docstring says no element is ever dropped.
