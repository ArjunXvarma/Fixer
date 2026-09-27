"""
Run one architecture against one repository.

    python -m fixer.run
    python -m fixer.run --arch react --repo ../some-other-repo
"""

import argparse
import time
from collections import Counter
from pathlib import Path

from fixer.agent.state import AgentState, Plan
from fixer.architectures import ARCHITECTURES

DEFAULT_REPO = str(Path(__file__).resolve().parents[1] / "test-repo")

DEFAULT_TASK = """
Investigate the failing Tribonacci tests.

Do not modify any files.

Determine:
1. Which tests are failing.
2. Which implementation is responsible.
3. What the expected behavior appears to be.
4. What the likely root cause is.
"""


def run(arch: str, repo_path: str, task: str) -> dict:
    agent_loop = ARCHITECTURES[arch]()

    state = AgentState(
        task=task,
        repo_path=str(Path(repo_path).resolve()),
        messages=[{"role": "user", "content": task}],
    )

    print(f"[FIXER] architecture={arch} repo={state.repo_path}")

    start = time.perf_counter()

    # Generous: each architecture is expected to stop itself.
    result = agent_loop.invoke(state, config={"recursion_limit": 100})

    result["seconds"] = round(time.perf_counter() - start, 1)

    return result


def report(arch: str, result: dict) -> None:
    """The numbers to compare architectures on."""

    calls = [
        call["name"]
        for message in result["messages"]
        for call in getattr(message, "tool_calls", None) or []
    ]

    print(f"\n{'=' * 70}")
    print(
        f"[{arch}] {len(calls)} tool calls, {result['iteration']} iterations, "
        f"{result['seconds']}s"
    )

    for name, count in Counter(calls).most_common():
        print(f"    {count} x {name}")

    print("=" * 70)
    print(result["messages"][-1].text or "[no answer]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Fixer architecture.")
    parser.add_argument("--arch", default="react", choices=sorted(ARCHITECTURES))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--task", default=DEFAULT_TASK)

    args = parser.parse_args()

    report(args.arch, run(args.arch, args.repo, args.task))


if __name__ == "__main__":
    main()
