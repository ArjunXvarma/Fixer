import subprocess
import json
from pathlib import Path


def search_code(
    repo_path: str,
    pattern: str,
    max_results: int = 50,
) -> dict:
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return {
            "matches": [],
            "truncated": False,
            "error": f"Repository path does not exist: {repo}",
        }

    try:
        result = subprocess.run(
            ["rg", "--json", pattern, str(repo)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode not in (0, 1):
            return {
                "matches": [],
                "truncated": False,
                "error": result.stderr.strip() or "ripgrep failed",
            }

        matches = []

        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "match":
                continue

            match_data = data["data"]

            absolute_path = Path(match_data["path"]["text"])
            try:
                relative_path = absolute_path.relative_to(repo)
            except ValueError:
                relative_path = absolute_path

            matches.append(
                {
                    "file": str(relative_path),
                    "line": match_data["line_number"],
                    "text": match_data["lines"]["text"].rstrip(),
                }
            )

            if len(matches) >= max_results:
                break

        truncated = len(matches) >= max_results

        return {
            "matches": matches,
            "truncated": truncated,
            "error": None,
        }

    except FileNotFoundError:
        return {
            "matches": [],
            "truncated": False,
            "error": "ripgrep ('rg') is not installed. Install it with 'brew install ripgrep'.",
        }

    except Exception as exc:
        return {
            "matches": [],
            "truncated": False,
            "error": str(exc),
        }
