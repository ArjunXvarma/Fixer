"""
python -m fixer.eval --arch autonomous_agent --repeats 3
python -m fixer.eval --arch none                     # no-op baseline
python -m fixer.eval --arch react --dry-run          # stub model, no quota
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fixer.agent.llm import LLM
from fixer.agent.state import AgentState
from fixer.architectures import ARCHITECTURES
from fixer.eval.score import score
from fixer.eval.tasks import TASKS
from fixer.sandbox import LocalSandbox

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results.jsonl"


def once(arch: str, task, dry_run: bool) -> dict:
    """One sandboxed attempt at one task."""

    row = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task.name,
        "arch": arch,
        "dry_run": dry_run,
    }

    with LocalSandbox() as box:
        box.copy_repo(task.repo)

        start = time.perf_counter()
        diff = ""

        if arch == "none":
            row |= {"status": "skipped", "iterations": 0, "tool_calls": 0}
        else:
            state = AgentState(
                task=task.prompt(),
                repo_path=box.workdir,
                messages=[{"role": "user", "content": task.prompt()}],
            )

            try:
                result = ARCHITECTURES[arch]().invoke(
                    state, config={"recursion_limit": 100}
                )
                tool_calls = [
                    call
                    for message in result["messages"]
                    for call in getattr(message, "tool_calls", None) or []
                ]
                calls = [call["name"] for call in tool_calls]

                # The call sequence with its first argument: without this a
                # repeated-call loop is invisible after the run.
                trace = [
                    f"{call['name']}({next(iter(call['args'].values()), '')!s:.60})"
                    for call in tool_calls
                ]
                row |= {
                    "status": result.get("status", "unknown"),
                    "iterations": result.get("iteration", 0),
                    "tool_calls": len(calls),
                    "by_tool": dict(Counter(calls)),
                    "trace": trace,
                    "errors": result.get("errors", []),
                }
            except Exception as exc:
                row |= {
                    "status": "infra_error",
                    "infra_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "iterations": 0,
                    "tool_calls": 0,
                }

            diff = box.diff()

        row["seconds"] = round(time.perf_counter() - start, 1)
        row |= score(box, task, diff)
        row["diff"] = diff

    return row


def summarise(rows: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"{'task':22} {'arch':18} {'resolved':>9} {'regress':>8} {'calls':>6} {'secs':>7}"
    )
    print("=" * 78)

    for row in rows:
        flag = " TESTS!" if row.get("touched_tests") else ""
        print(
            f"{row['task']:22} {row['arch']:18} "
            f"{str(row['resolved']):>9} {str(row['no_regression']):>8} "
            f"{row['tool_calls']:>6} {row['seconds']:>7}{flag}"
        )

    scored = [r for r in rows if r["status"] != "infra_error"]
    infra = len(rows) - len(scored)

    print("-" * 78)
    print(
        f"resolved {sum(r['resolved'] for r in scored)}/{len(scored)} scored runs"
        + (f", {infra} infra errors excluded" if infra else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a Fixer architecture.")
    parser.add_argument(
        "--arch", default="none", choices=["none", *sorted(ARCHITECTURES)]
    )
    parser.add_argument("--task", default="all", choices=["all", *sorted(TASKS)])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="replace the model with a scripted stub: no API calls, no quota",
    )

    args = parser.parse_args()

    if args.dry_run:
        from fixer.eval.stub import install_stub

        install_stub()

    tasks = list(TASKS.values()) if args.task == "all" else [TASKS[args.task]]

    if args.arch != "none" and not args.dry_run:
        print(f"[EVAL] model: {LLM()}")

    rows = []

    for attempt in range(args.repeats):
        for task in tasks:
            print(f"\n[EVAL] {task.name} | {args.arch} | attempt {attempt + 1}")
            row = once(args.arch, task, args.dry_run)
            rows.append(row)

            with RESULTS.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    summarise(rows)
    print(f"[EVAL] rows appended to {RESULTS}")


if __name__ == "__main__":
    main()
