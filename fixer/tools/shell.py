import subprocess


DEFAULT_TIMEOUT = 30


def run_command(
    repo_path: str,
    command: list[str],
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:

    result = subprocess.run(
        command,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
