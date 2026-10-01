from dataclasses import dataclass
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks" / "tasks"
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "test-repo"


@dataclass(frozen=True)
class Task:
    name: str
    repo: Path
    statement: Path
    oracle: Path

    def prompt(self) -> str:
        return self.statement.read_text()


def _task(name: str) -> Task:
    own_repo = BENCHMARKS / name / "repo"

    return Task(
        name=name,
        repo=own_repo if own_repo.is_dir() else DEFAULT_REPO,
        statement=BENCHMARKS / name / "task.md",
        oracle=BENCHMARKS / name / "grade_test.py",
    )


TASKS = {
    path.name: _task(path.name)
    for path in sorted(BENCHMARKS.iterdir())
    if (path / "task.md").is_file()
}
