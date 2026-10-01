"""Hidden oracle: the agent never sees this file."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "existing_lint_error.py"


def load():
    spec = importlib.util.spec_from_file_location("existing_lint_error", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_imports_at_all():
    load()


def test_division_works():
    assert load().division(123, 15) == pytest.approx(8.2)
