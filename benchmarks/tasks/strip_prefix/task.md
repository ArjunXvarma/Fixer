# strip_prefix() removes too much

```python
from textutils.prefixes import strip_prefix

strip_prefix("aab", "a")       # "b"    -- expected "ab"
strip_prefix("xxy", "x")       # "y"    -- expected "xy"
```

It should remove one occurrence of the prefix, nothing more.
