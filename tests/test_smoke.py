import json

import pytest

from fixer.agent.state import AgentState, Plan, VerificationResult
from fixer.agent.tools import TOOLS
from fixer.architectures import ARCHITECTURES
from fixer.eval.tasks import TASKS
from fixer.sandbox import SANDBOXES


# --- architectures ----------------------------------------------------------


def test_every_architecture_compiles(monkeypatch):
    """build() must return a compiled graph without touching a provider."""

    from fixer.eval.stub import install_stub

    install_stub()

    for name, build in ARCHITECTURES.items():
        graph = build()
        assert graph.get_graph().nodes, f"{name} compiled with no nodes"


def test_architectures_are_registered():
    assert set(ARCHITECTURES) == {"react", "simple_planner", "autonomous_agent"}


# --- tools ------------------------------------------------------------------


def test_tool_names_are_unique():
    names = [tool.name for tool in TOOLS]
    assert len(names) == len(set(names))


def test_repo_path_is_hidden_from_the_model():
    """
    repo_path is injected from graph state. If it leaks into the schema the
    model sees, the model starts choosing repository paths again.
    """

    for tool in TOOLS:
        visible = tool.tool_call_schema.model_json_schema().get("properties", {})
        assert "repo_path" not in visible, f"{tool.name} exposes repo_path"


def test_every_tool_has_a_description():
    for tool in TOOLS:
        assert tool.description.strip(), f"{tool.name} has no description"


# --- state ------------------------------------------------------------------


def test_agent_state_builds_from_the_three_required_fields():
    state = AgentState(task="t", repo_path="/tmp", messages=[])

    assert state.plan == Plan(goal="", steps=[])
    assert state.verification is None
    assert state.iteration == 0


def test_verification_verdicts():
    result = VerificationResult(
        verdict="replan",
        reason="r",
        evidence=["e"],
        missing_evidence=[],
        contradiction="observed X, but step 2 assumes Y",
        invalidated_steps=[2],
    )

    assert result.verdict == "replan"

    with pytest.raises(ValueError):
        VerificationResult(
            verdict="not-a-verdict", reason="r", evidence=[], missing_evidence=[]
        )


# --- benchmark suite --------------------------------------------------------


def test_tasks_are_discovered():
    assert len(TASKS) == 8


def test_every_task_is_complete_and_self_contained():
    for name, task in TASKS.items():
        assert task.statement.is_file(), f"{name} has no task.md"
        assert task.oracle.is_file(), f"{name} has no grade_test.py"
        assert task.repo.is_dir(), f"{name} has no fixture repo"
        assert task.repo.name == "repo", f"{name} depends on an external repository"
        assert task.prompt().strip(), f"{name} has an empty statement"


def test_oracles_are_not_inside_the_fixture_the_agent_sees():
    """An oracle the agent can read is an oracle the agent can rewrite."""

    for name, task in TASKS.items():
        assert task.oracle.parent == task.repo.parent, name
        assert not (task.repo / task.oracle.name).exists(), name


# --- sandbox ----------------------------------------------------------------


def test_sandbox_copies_runs_and_tears_down():
    task = TASKS["mutable_default"]

    with SANDBOXES["local"]() as box:
        box.copy_repo(task.repo)
        workdir = box.workdir

        assert box.workdir and box.run(["git", "status"])["return_code"] == 0
        assert box.diff() == "", "a fresh copy should have no diff"

    import os

    assert not os.path.exists(workdir), "the sandbox copy outlived the context"


def test_sandbox_reports_the_diff_it_made():
    from pathlib import Path

    task = TASKS["mean_of_empty"]

    with SANDBOXES["local"]() as box:
        box.copy_repo(task.repo)

        target = Path(box.workdir) / "src/stats/average.py"
        target.write_text(target.read_text() + "\n# touched\n")

        assert "# touched" in box.diff()


# --- results file -----------------------------------------------------------


def test_recorded_results_are_valid_jsonl():
    from fixer.eval.runner import RESULTS

    if not RESULTS.exists():
        pytest.skip("no runs recorded yet")

    required = {"task", "arch", "status", "resolved", "no_regression"}

    for line in RESULTS.read_text().splitlines():
        row = json.loads(line)
        assert required <= set(row), f"row missing fields: {required - set(row)}"
