import json
import time
from functools import partial

from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from fixer.agent.llm import build_model, build_structured_model
from fixer.agent.prompts import (
    EVIDENCE_RULES,
    PLAN_OBJECTIVE,
    PLAN_OUTPUT_FORMAT,
    PLAN_QUALITY_RULES,
    PLAN_STEP_RULES,
    PLANNING_RULES,
    REPLANNER_SYSTEM_PROMPT,
    REPORT_FORMAT,
    REPORT_LIMITATIONS,
    TOOL_AWARENESS,
    TOOL_RULES,
    VERIFICATION_RULES,
    VERIFIER_ROLE,
    get_planner_step_prompt,
)
from fixer.agent.state import AgentState, Plan, VerificationResult
from fixer.agent.tools import TOOLS
from fixer.tools.testing import run_tests

MAX_ITERATIONS = 30
MAX_TOOL_CALLS = 40
MAX_STEP_RETRIES = 2
MAX_REPLANS = 1


ROLE = """
You are Fixer, an autonomous software engineering agent working inside a Git
repository. You investigate, and when the task calls for a change you make it.

apply_patch_tool is the only way you change the repository. Never edit files
through run_command_tool. Read a file before you patch it, and build the diff
from what you actually read.

After every step the runtime runs the test suite for you. Do not claim a fix
works; the test result decides that.
"""

PLANNER_ROLE = """
You are Fixer's planning agent, working on a task in a Git repository.

You do not inspect the repository and you do not call tools. You write the plan
that an executor will carry out. That executor can list files, read them,
search the code, run commands, run the test suite, and change files by applying
a patch.

When the task requires a code change, make that change one of the steps, and
follow it with a step that verifies it.
"""

PLANNER_SYSTEM_PROMPT = "\n".join(
    [
        PLANNER_ROLE,
        PLAN_OBJECTIVE,
        PLANNING_RULES,
        PLAN_STEP_RULES,
        TOOL_AWARENESS,
        PLAN_QUALITY_RULES,
        PLAN_OUTPUT_FORMAT,
    ]
)

VERIFIER_SYSTEM_PROMPT = "\n".join(
    [
        VERIFIER_ROLE,
        VERIFICATION_RULES,
        EVIDENCE_RULES,
    ]
)

ANSWER_NOW = (
    "The run is over. Write your final answer now, using only the evidence above."
)


def tool_calls_so_far(state: AgentState) -> int:
    """Count every tool call the run has made."""

    return sum(
        len(getattr(message, "tool_calls", None) or []) for message in state.messages
    )


def patched_files(messages: list) -> list[str]:
    """
    The files apply_patch_tool actually changed, read back from its results.

    The graph cannot see inside ToolNode, so the record of what was written
    comes from the tool messages it produced.
    """

    files: list[str] = []

    for message in messages:
        if getattr(message, "name", None) != "apply_patch_tool":
            continue

        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue

        if result.get("status") == "applied":
            files.extend(result.get("files", []))

    return sorted(set(files))


def test_summary(state: AgentState, stdout_chars: int = 1500) -> str:
    """The latest test run, short enough to put in a prompt."""

    if not state.test_results:
        return "No test run yet."

    result = state.test_results[-1]

    return (
        f"status={result['status']} phase={result['phase']} "
        f"collected={result['tests_collected']} passed={result['tests_passed']} "
        f"failed={result['tests_failed']} errors={result['tests_errors']}\n"
        f"{(result['stdout'] or result['stderr'])[-stdout_chars:]}"
    )


def api_error(state: AgentState, where: str, exc: Exception) -> dict:
    """
    Record an unreachable model and mark the run for an early finish.

    The free tier returns 503 often enough that a long run will meet one. This
    architecture has already edited files by then, so the run must end with a
    report rather than a traceback.
    """

    print(f"[{where.upper()}] Gemini unavailable: {str(exc)[:160]}")

    return {
        "errors": [*state.errors, f"{where}: {exc}"],
        "status": "api_error",
    }


def offline_report(state: AgentState) -> str:
    """The report to fall back on when even the finalize call cannot be made."""

    steps = state.plan["steps"]
    done = [step["description"] for step in steps if step["status"] == "completed"]

    return "\n".join(
        [
            "The run ended early because the model API was unavailable.",
            "",
            f"Steps completed: {len(done)} of {len(steps)}",
            *(f"  - {description}" for description in done),
            "",
            f"Files changed: {', '.join(state.files_modified) or 'none'}",
            "",
            f"Last test run: {test_summary(state, stdout_chars=600)}",
            "",
            "Errors:",
            *(f"  - {error}" for error in state.errors),
            "",
            "Nothing here is verified beyond that test run.",
        ]
    )


def with_status(plan: Plan, index: int, status: str) -> Plan:
    """A copy of the plan with one step's status changed."""

    steps = [
        {**step, "status": status} if position == index else step
        for position, step in enumerate(plan["steps"])
    ]

    return {**plan, "steps": steps}


def planner(state: AgentState, model) -> dict:
    """
    Turn the task into an ordered plan.

    Reads state.task. Returns the plan, current_step reset to 0, and the
    message index the first step's evidence starts from.
    """

    print("\n" + "=" * 70)
    print("[PLANNER] Creating execution plan")
    print("=" * 70)

    start = time.perf_counter()

    try:
        response = model.invoke([("system", PLANNER_SYSTEM_PROMPT), *state.messages])
    except GoogleAPIError as exc:
        return api_error(state, "planner", exc)

    print(f"[PLANNER] Gemini responded in {time.perf_counter() - start:.2f}s")
    print(f"[PLANNER] Goal: {response['goal']}")

    for step in response["steps"]:
        print(f"  [{step['id']}] {step['description']}")

    return {
        "plan": response,
        "current_step": 0,
        "step_start_message_index": len(state.messages),
        "status": "executing",
    }


def agent(state: AgentState, model) -> dict:
    """
    Execute the current plan step, one model turn at a time.

    Reads plan["steps"][current_step] and the message history. Returns the
    model's response plus the incremented iteration. The response either
    requests tools — including apply_patch_tool, which is how this
    architecture edits files — or states the step's result in text, which is
    the signal that the step is finished.
    """

    steps = state.plan["steps"]
    index = state.current_step
    step = steps[index]

    contract = get_planner_step_prompt(
        step["id"],
        len(steps),
        step["description"],
        step["purpose"],
        step["expected_result"],
        state.plan["goal"],
        index,
    )

    iteration = state.iteration + 1
    retrying = state.verification is not None

    print(f"\n{'=' * 70}")
    print(f"[AGENT] Iteration {iteration} | step {step['id']}/{len(steps)}", end="")
    print(f" | attempt {state.step_retries + 1}" if retrying else "")
    print(f"{'=' * 70}")

    history = list(state.messages)

    if retrying:
        missing = "\n".join(f"- {item}" for item in state.verification.missing_evidence)

        history.append(
            (
                "human",
                "This step did not pass verification.\n\n"
                f"Reason: {state.verification.reason}\n\n"
                f"Missing evidence:\n{missing}\n\n"
                f"Latest test run:\n{test_summary(state)}\n\n"
                "Address that, then state the result in one sentence.",
            )
        )

    try:
        response = model.invoke(
            [("system", "\n".join([ROLE, TOOL_RULES, contract])), *history]
        )
    except GoogleAPIError as exc:
        return {**api_error(state, "agent", exc), "iteration": iteration}

    if not response.tool_calls and not (response.text or "").strip():
        response = AIMessage(content=f"Step {step['id']} needed no further work.")

    for call in response.tool_calls:
        print(f" -> {call['name']} {call['args']}")

    if not response.tool_calls:
        print(f"[AGENT] {response.text.strip()[:200]}")

    update = {"messages": [response], "iteration": iteration}

    if step["status"] == "pending":
        print(f"[PLAN] Step {step['id']} pending -> running")
        update["plan"] = with_status(state.plan, index, "running")

    return update


def test(state: AgentState) -> dict:
    """
    Run the repository's test suite and record the outcome.

    Runs after the executor finishes a step, so the verifier can judge a step
    against test evidence rather than the executor's own claim. Takes no
    model: whether the tests pass is not a judgement call.
    """

    result = run_tests(state.repo_path)

    changed = patched_files(state.messages)

    print(f"[TESTS] {result['status']} (phase={result['phase']})")

    if changed:
        print(f"[TESTS] files changed so far: {', '.join(changed)}")

    return {
        "test_results": [*state.test_results, result],
        "files_modified": changed,
    }


def verifier(state: AgentState, model) -> dict:
    """
    Decide whether the current step achieved its objective.

    Reads the current step, this step's messages (from
    step_start_message_index) and the latest test result. Returns the
    verdict in state.verification, which route_verification then acts on.
    The executor stopping is a claim; this node is what makes it a finding.
    """

    steps = state.plan["steps"]
    index = state.current_step
    step = steps[index]

    step_messages = state.messages[state.step_start_message_index :]

    previous_results = [
        {
            "id": previous["id"],
            "description": previous["description"],
            "status": previous["status"],
        }
        for previous in steps[:index]
    ]

    remaining_steps = [
        {
            "id": upcoming["id"],
            "description": upcoming["description"],
            "expected_result": upcoming["expected_result"],
        }
        for upcoming in steps[index + 1 :]
    ]

    context = f"""
Original task:
{state.task}

Overall goal:
{state.plan["goal"]}

Current step:
{step["description"]}

Expected result:
{step["expected_result"]}

Previously completed steps:
{previous_results}

Steps still to come (a contradiction is an observation that makes one of
these pointless or impossible):
{remaining_steps}

Files this run has changed:
{state.files_modified or "none"}

Test run after this step:
{test_summary(state)}

Evaluate the current step using the execution messages, the tool observations
and the test result above. A change that applied but left tests failing has
not achieved its objective.
"""

    print(f"\n[VERIFIER] Checking step {step['id']}")

    start = time.perf_counter()

    try:
        result = model.invoke(
            [("system", VERIFIER_SYSTEM_PROMPT), ("user", context), *step_messages]
        )
    except GoogleAPIError as exc:
        # route_verification already sends a missing verdict to finalize.
        return {**api_error(state, "verifier", exc), "verification": None}

    if result.verdict == "replan" and not (
        result.contradiction and result.invalidated_steps
    ):
        print("[VERIFIER] replan without a named contradiction -> incomplete")
        result = result.model_copy(update={"verdict": "incomplete"})

    print(f"[VERIFIER] {result.verdict} ({time.perf_counter() - start:.2f}s)")
    print(f"[VERIFIER] Reason: {result.reason}")

    for item in result.missing_evidence:
        print(f"[VERIFIER] Missing: {item}")

    return {"verification": result}


def advance(state: AgentState) -> dict:
    """
    Mark the current step completed and start the next one clean.

    Returns the plan with this step's status set to completed, current_step
    incremented, and the per-step fields — verification, retry counters and
    step_start_message_index — reset so the next step starts without the
    previous step's feedback.
    """

    index = state.current_step

    print(f"[PLAN] Step {state.plan['steps'][index]['id']} running -> completed")

    return {
        "plan": with_status(state.plan, index, "completed"),
        "current_step": index + 1,
        "verification": None,
        "step_retries": 0,
        "step_start_message_index": len(state.messages),
    }


def repair(state: AgentState) -> dict:
    """
    Send a failed step back to the executor with the failure named.

    Reached when verification says the step did not achieve its objective —
    typically a patch that applied but left tests failing. Returns the
    incremented attempt counter, so the loop can give up rather than retry
    the same edit forever.
    """

    attempt = state.step_retries + 1

    print(f"[REPAIR] Attempt {attempt} of {MAX_STEP_RETRIES} on this step.")

    return {"step_retries": attempt, "status": "repairing"}


def replanner(state: AgentState, model) -> dict:
    """
    Rewrite the remaining steps after an observation contradicted the plan.

    Keeps the completed steps that the contradiction did not invalidate,
    asks the model for the steps that should follow instead, and returns the
    new plan with current_step pointing at the first of them.
    """

    result = state.verification
    invalid = set(result.invalidated_steps)

    kept = [
        step
        for step in state.plan["steps"]
        if step["status"] == "completed" and step["id"] not in invalid
    ]

    request = [
        ("system", REPLANNER_SYSTEM_PROMPT),
        (
            "user",
            f"""
Original task:
{state.task}

Previous goal:
{state.plan["goal"]}

Steps already completed (do not repeat these):
{[step["description"] for step in kept]}

Files this run has already changed:
{state.files_modified or "none"}

What contradicted the old plan:
{result.contradiction}

The evidence behind that contradiction:
{result.evidence}

Write the remaining steps only, starting from where the evidence now points.
""",
        ),
    ]

    try:
        response = model.invoke(request)
    except GoogleAPIError as exc:
        return api_error(state, "replanner", exc)

    new_steps = [{**step, "status": "pending"} for step in response["steps"]]

    steps = [
        {**step, "id": position + 1} for position, step in enumerate(kept + new_steps)
    ]

    print(f"[REPLAN] {result.contradiction}")
    print(f"[REPLAN] kept {len(kept)} steps, {len(new_steps)} new")

    return {
        "plan": {"goal": response["goal"], "steps": steps},
        "current_step": len(kept),
        "verification": None,
        "step_retries": 0,
        "replans": state.replans + 1,
        "step_start_message_index": len(state.messages),
    }


def finalize(state: AgentState, model) -> dict:
    """
    Write the final report and end the run.

    The only node that reports. Reads the whole history plus the plan's step
    statuses, and says what was changed, what the tests showed, and what
    could not be established when the plan did not finish.
    """

    steps = state.plan["steps"]
    done = sum(1 for step in steps if step["status"] == "completed")
    unfinished = not steps or done < len(steps)

    print(
        f"\n[FINALIZE] {done}/{len(steps)} steps completed, "
        f"{tool_calls_so_far(state)} tool calls."
    )

    if state.files_modified:
        print(f"[FINALIZE] changed: {', '.join(state.files_modified)}")

    sections = [ROLE, REPORT_FORMAT]

    if unfinished:
        print("[FINALIZE] Plan did not finish: reporting with limitations.")
        sections.append(REPORT_LIMITATIONS)

    outcome = f"""
Files you changed in this run:
{state.files_modified or "none"}

Final test run:
{test_summary(state)}

Say plainly whether that test run verifies the change.
"""

    try:
        response = model.invoke(
            [
                ("system", "\n".join(sections)),
                *state.messages,
                ("human", outcome + "\n" + ANSWER_NOW),
            ]
        )
    except GoogleAPIError as exc:
        print(f"[FINALIZE] Gemini unavailable: {str(exc)[:160]}")
        response = AIMessage(content=offline_report(state))

    return {
        "messages": [response],
        "status": "blocked" if unfinished else "completed",
    }


def after_planner(state: AgentState) -> str:
    """
    Start executing, or stop.

    Returns "agent" when the planner produced steps, "finalize" when it did
    not — the executor would index past an empty plan.
    """

    if state.status == "api_error":
        return "finalize"

    if state.plan["steps"]:
        return "agent"

    print("\n[ROUTER] Planner produced no steps. Finalising.")
    return "finalize"


def route_agent(state: AgentState) -> str:
    """
    Decide whether the executor wants tools, has finished the step, or has
    spent the run's budget.

    Returns "tools" when the last message requests tool calls, "done" when the
    executor answered in text, which sends the work to the test node, and
    "finalize" when the iteration or tool-call budget is gone.
    """

    if state.status == "api_error":
        return "finalize"

    if not getattr(state.messages[-1], "tool_calls", None):
        print("\n[ROUTER] Executor stopped. Testing.")
        return "done"

    if state.iteration >= MAX_ITERATIONS:
        print(f"\n[ROUTER] Hit the {MAX_ITERATIONS} iteration limit.")
        return "finalize"

    if tool_calls_so_far(state) >= MAX_TOOL_CALLS:
        print(f"\n[ROUTER] Hit the {MAX_TOOL_CALLS} tool-call limit.")
        return "finalize"

    print("\n[ROUTER] Tool call detected.")
    return "tools"


def route_verification(state: AgentState) -> str:
    """
    Act on the verifier's verdict.

    Returns "complete" to advance to the next step, "incomplete" to repair and
    retry this one, "replan" when the evidence falsified an assumption the
    remaining steps depend on, or "blocked" when a budget is spent and the run
    has to report what it has.
    """

    result = state.verification

    if result is None or result.verdict == "blocked":
        print("[ROUTER] Blocked. Finalising with limitations.")
        return "blocked"

    if result.verdict == "complete":
        return "complete"

    if result.verdict == "replan":
        if state.replans >= MAX_REPLANS:
            print(f"[ROUTER] Replan budget ({MAX_REPLANS}) spent.")
            return "blocked"

        return "replan"

    if state.step_retries >= MAX_STEP_RETRIES:
        print(f"[ROUTER] Repair budget ({MAX_STEP_RETRIES}) spent on this step.")
        return "blocked"

    if state.iteration >= MAX_ITERATIONS:
        print(f"[ROUTER] Hit the {MAX_ITERATIONS} iteration limit.")
        return "blocked"

    return "incomplete"


def after_advance(state: AgentState) -> str:
    """
    Continue the plan, or stop.

    Returns "agent" while steps remain, "finalize" once the plan is done.
    """

    if state.current_step < len(state.plan["steps"]):
        return "agent"

    print("\n[ROUTER] All plan steps complete.")
    return "finalize"


def after_replan(state: AgentState) -> str:
    """
    Execute the rewritten plan, or stop.

    Returns "agent" when the replanner produced steps, "finalize" when it
    produced none because the evidence already answers the task or the model
    was unreachable.
    """

    if state.status == "api_error":
        return "finalize"

    if state.current_step < len(state.plan["steps"]):
        return "agent"

    print("\n[ROUTER] Replan left nothing to execute. Finalising.")
    return "finalize"


def build():
    graph = StateGraph(AgentState)

    graph.add_node(
        "planner",
        partial(planner, model=build_structured_model(Plan)),
    )
    graph.add_node("agent", partial(agent, model=build_model()))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("test", test)
    graph.add_node(
        "verifier",
        partial(
            verifier,
            model=build_structured_model(VerificationResult),
        ),
    )
    graph.add_node(
        "replanner",
        partial(replanner, model=build_structured_model(Plan)),
    )
    graph.add_node("advance", advance)
    graph.add_node("repair", repair)
    graph.add_node("finalize", partial(finalize, model=build_model(tool_choice="none")))

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner", after_planner, {"agent": "agent", "finalize": "finalize"}
    )

    graph.add_conditional_edges(
        "agent",
        route_agent,
        {"tools": "tools", "done": "test", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("test", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_verification,
        {
            "complete": "advance",
            "incomplete": "repair",
            "replan": "replanner",
            "blocked": "finalize",
        },
    )

    graph.add_conditional_edges(
        "advance", after_advance, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("repair", "agent")
    graph.add_conditional_edges(
        "replanner", after_replan, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()
