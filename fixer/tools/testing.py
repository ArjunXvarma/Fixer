import re
import shutil
import sys
from pathlib import Path

from fixer.tools.shell import run_command


def run_tests(
    repo_path: str,
    test_path: str | None = None,
    timeout: int = 120,
) -> dict:
    """
    Run the repository's tests with pytest.

    Finds a pytest that actually works and runs it from the repository root.
    run_command puts the repository's src/ directory on PYTHONPATH, so
    src-layout packages import correctly.

    Args:
        repo_path: Root of the target repository.
        test_path: Optional test file/directory, relative to repo_path.
        timeout: Maximum execution time in seconds.
    """

    repo = Path(repo_path).resolve()

    if test_path:
        target = (repo / test_path).resolve()

        if not target.is_relative_to(repo):
            return {
                "status": "error",
                "phase": "setup",
                "stderr": "Test path must stay inside the repository.",
            }

        if not target.exists():
            return {
                "status": "error",
                "phase": "setup",
                "stderr": (
                    f"Test path does not exist: {test_path}. "
                    "Paths are relative to the repository root."
                ),
            }

    pytest_command = _find_pytest(repo)

    if pytest_command is None:
        return {
            "status": "error",
            "phase": "setup",
            "stderr": "pytest is not installed in the repository venv, "
            f"in {sys.executable}, or on PATH.",
        }

    # -rA reports every outcome; no:cacheprovider keeps the repo clean.
    command = [*pytest_command, "-rA", "-p", "no:cacheprovider"]

    if test_path:
        command.append(test_path)

    print(f"[TEST] {' '.join(command)}")

    result = run_command(repo_path, command, timeout)

    counts = _parse_output(result["stdout"])
    status, phase = _classify(result["return_code"], counts)

    print(f"[TEST] {status} (phase={phase}) {counts}")

    return {
        "status": status,
        "phase": phase,
        "return_code": result["return_code"],
        **counts,
        "command": command,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _find_pytest(repo: Path) -> list[str] | None:
    """Return a working pytest command, or None if there isn't one."""

    venv_python = repo / ".venv" / "bin" / "python"

    if venv_python.is_file() and _has_pytest(repo, str(venv_python)):
        return [str(venv_python), "-m", "pytest"]

    if _has_pytest(repo, sys.executable):
        return [sys.executable, "-m", "pytest"]

    # pytest often lives in a different environment than python3.
    found = shutil.which("pytest")

    return [found] if found else None


def _has_pytest(repo: Path, python: str) -> bool:
    check = run_command(str(repo), [python, "-c", "import pytest"])

    return check["return_code"] == 0


def _parse_output(output: str) -> dict:
    """Pull test counts out of pytest's output."""

    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    # The final summary line, e.g. "==== 1 failed, 2 passed in 0.10s ===="
    summary = [line for line in output.splitlines() if re.search(r"=+.* in [\d.]+s", line)]

    if summary:
        for number, word in re.findall(r"(\d+) (\w+)", summary[-1]):
            key = "errors" if word.startswith("error") else word

            if key in counts:
                counts[key] += int(number)

    # "collected 3 items" or "collected 0 items / 1 error"
    collected = re.search(r"collected (\d+) items?(?: / (\d+) error)?", output)

    return {
        "tests_collected": int(collected.group(1)) if collected else 0,
        "tests_passed": counts["passed"],
        "tests_failed": counts["failed"],
        "tests_errors": counts["errors"],
        "tests_skipped": counts["skipped"],
        "collection_errors": int(collected.group(2) or 0) if collected else 0,
    }


def _classify(return_code: int | None, counts: dict) -> tuple[str, str]:
    """
    Turn pytest's exit code into a status and the phase it stopped in.

    "collection" means no test ever ran, so a failure there says nothing about
    the code under test. "run" means tests actually executed.
    """

    ran = counts["tests_passed"] + counts["tests_failed"] + counts["tests_skipped"]

    if return_code is None:
        return "error", "setup"

    if counts["collection_errors"] and ran == 0:
        return "error", "collection"

    if return_code == 0:
        return "passed", "run"

    if return_code == 1:
        return "failed", "run"

    if return_code == 5:
        return "no_tests", "collection"

    return "error", "run"
