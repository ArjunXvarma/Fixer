import shutil
from pathlib import Path

from fixer.eval.tasks import Task
from fixer.sandbox.base import Sandbox
from fixer.tools.testing import run_tests

# A directory the agent never saw, so its own tests cannot grade it.
GRADING_DIR = "_grading"


def score(box: Sandbox, task: Task, diff: str) -> dict:
    """
    resolved        the hidden oracle passes against the agent's code
    no_regression   the repository's original tests still pass
    touched_tests   the agent edited tests, which makes any pass suspect
    """

    workdir = Path(box.workdir)

    # Read the diff before anything is restored.
    touched_tests = any(
        line.startswith(("+++ b/tests/", "--- a/tests/")) for line in diff.splitlines()
    )

    grading = workdir / GRADING_DIR
    grading.mkdir(exist_ok=True)
    shutil.copy(task.oracle, grading / "test_oracle.py")

    oracle = run_tests(box.workdir, GRADING_DIR)

    box.run(["git", "checkout", "--", "tests"])
    baseline = run_tests(box.workdir, "tests")

    shutil.rmtree(grading, ignore_errors=True)

    return {
        "resolved": oracle["status"] == "passed",
        "oracle_status": oracle["status"],
        "oracle_phase": oracle["phase"],
        "oracle_failed": oracle["tests_failed"],
        "no_regression": baseline["status"] in ("passed", "no_tests"),
        "baseline_status": baseline["status"],
        "touched_tests": touched_tests,
        "diff_lines": len([line for line in diff.splitlines() if line[:1] in "+-"]),
    }
