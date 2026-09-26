import os
import subprocess
from pathlib import Path


def run_command(
    repo_path: str,
    command: list[str],
    timeout: int = 30,
) -> dict:
    """
    Run a command inside the repository.

    Commands are passed as an argument list rather than a shell string.
    Example:
        ["ls", "-la"]
        ["git", "status"]
        ["python3", "-c", "import testpkg"]

    Runs from the repository root, with the repository's own packages
    importable (see _python_path).
    """

    repo = Path(repo_path).resolve()

    # Validate repository path
    if not repo.is_dir():
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": f"Repository path is not a directory: {repo}",
        }

    if not command:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": "Command cannot be empty.",
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = _python_path(repo, env.get("PYTHONPATH"))

    try:
        result = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
            env=env,
        )

        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds.",
        }

    except FileNotFoundError:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": f"Command not found: {command[0]}",
        }

    except Exception as exc:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
        }


def _python_path(repo: Path, existing: str | None) -> str:
    """
    Make the repository's own packages importable.

    A src/ layout keeps packages out of the repository root, so running
    anything from the root fails with ModuleNotFoundError until src/ is on the
    import path.
    """

    roots = [str(repo)]

    if (repo / "src").is_dir():
        roots.insert(0, str(repo / "src"))

    if existing:
        roots.append(existing)

    return os.pathsep.join(roots)
