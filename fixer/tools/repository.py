from pathlib import Path
import subprocess

IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", "build", "dist"}


def list_files(repo_path: str) -> list[str]:
    root = Path(repo_path)
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        files.append(str(path.relative_to(root)))

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
