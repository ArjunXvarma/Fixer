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
from fixer.sandbox import SANDBOXES

DEFAULT_REPO = str(Path(__file__).resolve().parents[1] / "test-repo")

DEFAULT_TASK = """
Fix the Tribonacci implementation so that all tests pass.

Determine:
1. Which tests are failing.
2. Which implementation is responsible.
3. What the expected behavior appears to be.
4. What the likely root cause is.
5. What files to change
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

    result = agent_loop.invoke(state, config={"recursion_limit": 100})

    result["seconds"] = round(time.perf_counter() - start, 1)

    return result


def run_sandboxed(arch: str, repo_path: str, task: str, sandbox: str) -> dict:
    with SANDBOXES[sandbox]() as box:
        box.copy_repo(repo_path)

        result = run(arch, box.workdir, task)
        result["diff"] = box.diff()

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

    if result.get("diff"):
        print(f"\n{'=' * 70}")
        print("[FIXER] the agent changed:")
        print("=" * 70)
        print(result["diff"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Fixer architecture.")
    parser.add_argument("--arch", default="react", choices=sorted(ARCHITECTURES))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--sandbox",
        default="local",
        choices=["none", *sorted(SANDBOXES)],
        help="work in a disposable copy (default) or directly in --repo",
    )

    args = parser.parse_args()

    if args.sandbox == "none":
        result = run(args.arch, args.repo, args.task)
    else:
        result = run_sandboxed(args.arch, args.repo, args.task, args.sandbox)

    report(args.arch, result)


if __name__ == "__main__":
    main()
