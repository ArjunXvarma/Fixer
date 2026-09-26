from pathlib import Path
import subprocess

IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
}


def list_files(repo_path: str) -> list[str]:
    root = Path(repo_path)
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)

        # Check the relative parts, so a directory above the repo that happens
        # to be named "build" doesn't hide the whole repository.
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        files.append(str(relative))

    return sorted(files)


def git_diff(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout
