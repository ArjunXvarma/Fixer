# lookup() raises a bare Exception, not KeyError

Callers cannot catch the failure properly:

```python
from stats.lookup import lookup

try:
    lookup({}, "missing")
except KeyError:
    print("handled")     # never runs
```

The docstring says it raises KeyError.
