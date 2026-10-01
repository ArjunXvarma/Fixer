# collect() remembers values from previous calls

```python
from listutils.collecting import collect

collect("a")   # ["a"]
collect("b")   # ["a", "b"]  -- expected ["b"]
```

The docstring says each call without `into` starts from an empty list.
