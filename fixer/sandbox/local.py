import shutil
import tempfile
from pathlib import Path

from fixer.sandbox.base import Sandbox
from fixer.tools.shell import run_command

IGNORED = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".venv", ".mypy_cache"
)


class LocalSandbox(Sandbox):
    def __init__(self, keep: bool = False):
        self.workdir = ""
        self.keep = keep

    def create(self):
        self.workdir = tempfile.mkdtemp(prefix="fixer-sandbox-")

        print(f"[SANDBOX] local {self.workdir}")

        return self.workdir

    def copy_repo(self, repo_path: str):
        source = Path(repo_path).resolve()

        shutil.copytree(source, self.workdir, dirs_exist_ok=True, ignore=IGNORED)

        self._git("init", "-q")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "sandbox baseline")

        print(
            f"[SANDBOX] copied {source.name} ({len(list(Path(self.workdir).rglob('*')))} entries)"
        )

    def run(self, command: list[str], timeout: int = 30) -> dict:
        return run_command(self.workdir, command, timeout)

    def diff(self) -> str:
        self._git("add", "-A")

        return self._git("diff", "--cached")["stdout"]

    def destroy(self):
        if self.keep:
            print(f"[SANDBOX] kept {self.workdir}")
            return

        shutil.rmtree(self.workdir, ignore_errors=True)

        print(f"[SANDBOX] removed {self.workdir}")

    def _git(self, *args: str) -> dict:
        return run_command(
            self.workdir,
            ["git", "-c", "user.email=fixer@local", "-c", "user.name=fixer", *args],
        )
