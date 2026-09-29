"""
Somewhere disposable for the agent to work.

    SANDBOXES["local"]()   # temp copy, no isolation, instant
"""

from fixer.sandbox.base import Sandbox
from fixer.sandbox.local import LocalSandbox

SANDBOXES = {
    "local": LocalSandbox,
}

__all__ = ["Sandbox", "LocalSandbox", "SANDBOXES"]
