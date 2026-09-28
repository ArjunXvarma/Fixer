import logging
from pathlib import Path

from patch_ng import fromstring


class PatchLog(logging.Handler):
    """Collect patch_ng's log, the only place it says why a hunk failed."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def apply_patch(
    repo_path: str,
    diff_text: str,
    strip: int = 1,
) -> dict:
    """
    Apply a unified diff inside the repository.

    Paths in the diff are relative to the repository root. strip=1 removes the
    leading a/ and b/ components that git writes.

    Args:
        repo_path: Root directory of the target repository.
        diff_text: The unified diff to apply.
        strip: Leading path components to remove from the diff's paths.

    Returns:
        status: "applied", "unchanged" (already applied), "failed" (the diff
            did not match the files) or "error".
        phase: "validation", "parse" or "apply".
        files: the files the diff targets.
        diagnostics: what patch_ng reported.
        error: the reason, or None.
    """

    repo = Path(repo_path).resolve()

    if not repo.is_dir():
        return {
            "status": "error",
            "phase": "validation",
            "files": [],
            "diagnostics": "",
            "error": f"Repository path is not a directory: {repo}",
        }

    log = PatchLog()
    logger = logging.getLogger("patch_ng")
    logger.addHandler(log)

    files: list[str] = []
    status, phase, error = "error", "parse", None

    try:
        patch = fromstring(diff_text.encode())

        if not patch:
            error = "Diff could not be parsed as a unified diff."
        else:
            phase = "apply"

            files = [patch.decode_clean(item.source, "a/") for item in patch.items]

            if not patch.apply(root=str(repo), strip=strip):
                status = "failed"
                error = "The diff's context or line numbers do not match the files."
            elif any("already patched" in line for line in log.lines):
                status = "unchanged"
                error = "Nothing changed; the repository already has this change."
            else:
                status = "applied"

    except Exception as exc:
        error = str(exc)

    finally:
        logger.removeHandler(log)

    print(f"[PATCH] {status} (phase={phase}) {', '.join(files) or '<no files>'}")

    if error:
        print(f"[PATCH] {error}")

    return {
        "status": status,
        "phase": phase,
        "files": files,
        "diagnostics": "\n".join(log.lines),
        "error": error,
    }
