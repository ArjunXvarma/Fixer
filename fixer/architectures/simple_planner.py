import time
from functools import partial

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from fixer.agent.llm import build_model
from fixer.agent.prompts import (
    EVIDENCE_RULES,
    PLAN_OBJECTIVE,
    PLAN_OUTPUT_FORMAT,
    PLAN_QUALITY_RULES,
    PLAN_STEP_RULES,
    PLANNING_RULES,
    REPORT_FORMAT,
    REPORT_LIMITATIONS,
    ROLE,
    TOOL_AWARENESS,
    TOOL_RULES,
    VERIFICATION_RULES,
    VERIFIER_ROLE,
    get_planner_step_prompt,
    REPLANNER_SYSTEM_PROMPT,
)
from fixer.agent.state import AgentState, Plan, VerificationResult
from fixer.agent.tools import TOOLS

# A step needs a tool turn plus a closing turn, and a retry costs another pair.
MAX_ITERATIONS = 20
MAX_TOOL_CALLS = 25
MAX_STEP_RETRIES = 2
MAX_REPLANS = 2

PLANNER_SYSTEM_PROMPT = "\n".join(
    [
        ROLE,
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
    "The plan is complete. Write your final answer now, using only the evidence above."
)


def tool_calls_so_far(state: AgentState) -> int:
    return sum(
        len(getattr(message, "tool_calls", None) or []) for message in state.messages
    )


def with_status(plan: Plan, index: int, status: str) -> Plan:
    """A copy of the plan with one step's status changed."""

    steps = [
        {**step, "status": status} if position == index else step
        for position, step in enumerate(plan["steps"])
    ]

    return {**plan, "steps": steps}


def planner(state: AgentState, model) -> dict:
    print("\n" + "=" * 70)
    print("[PLANNER] Creating execution plan")
    print("=" * 70)

    start = time.perf_counter()

    response = model.invoke(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            *state.messages,
        ]
    )

    print(f"[PLANNER] Gemini responded in {time.perf_counter() - start:.2f}s")
    print(f"[PLANNER] Goal: {response['goal']}")

    for step in response["steps"]:
        print(f"  [{step['id']}] {step['description']} ({step['status']})")

    return {
        "plan": response,
        "current_step": 0,
        # Step 1's evidence starts after everything already in the history.
        "step_start_message_index": len(state.messages),
    }


def agent(state: AgentState, model) -> dict:
    plan = state.plan
    steps = plan["steps"]

    if not steps:
        raise ValueError("Planner produced no steps.")

    index = state.current_step
    step = steps[index]
    total = len(steps)

    contract = get_planner_step_prompt(
        step["id"],
        total,
        step["description"],
        step["purpose"],
        step["expected_result"],
        plan["goal"],
        index,
    )

    iteration = state.iteration + 1
    retrying = state.verification is not None

    print(f"\n{'=' * 70}")
    print(f"[AGENT] Iteration {iteration} | step {step['id']}/{total}", end="")
    print(f" | retry {state.step_retries}/{MAX_STEP_RETRIES}" if retrying else "")
    print(f"{'=' * 70}")

    history = list(state.messages)

    if retrying:
        missing = "\n".join(f"- {item}" for item in state.verification.missing_evidence)

        history.append(
            (
                "human",
                "This step was judged incomplete.\n\n"
                f"Reason: {state.verification.reason}\n\n"
                f"Missing evidence:\n{missing}\n\n"
                "Obtain that evidence, then state the result in one sentence.",
            )
        )

    response = model.invoke(
        [("system", "\n".join([ROLE, TOOL_RULES, contract])), *history]
    )

    if not response.tool_calls and not (response.text or "").strip():
        response = AIMessage(content=f"Step {step['id']} needed no further work.")

    for call in response.tool_calls:
        print(f"  → {call['name']} {call['args']}")

    if not response.tool_calls:
        print(f"[AGENT] {response.text.strip()[:200]}")

    update = {"messages": [response], "iteration": iteration}

    if step["status"] == "pending":
        print(f"[PLAN] Step {step['id']} pending → running")
        update["plan"] = with_status(plan, index, "running")

    return update


def verifier(state: AgentState, model) -> dict:
    step = state.plan["steps"][state.current_step]

    step_messages = state.messages[state.step_start_message_index :]

    previous_results = [
        {
            "id": previous["id"],
            "description": previous["description"],
            "status": previous["status"],
        }
        for previous in state.plan["steps"][: state.current_step]
    ]

    # A contradiction is measured against the steps still to come, so the
    # verifier has to be shown them.
    remaining_steps = [
        {
            "id": upcoming["id"],
            "description": upcoming["description"],
            "expected_result": upcoming["expected_result"],
        }
        for upcoming in state.plan["steps"][state.current_step + 1 :]
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

Evaluate the current step using the execution messages
and tool observations provided.
"""

    print(f"\n[VERIFIER] Checking step {step['id']}")

    start = time.perf_counter()

    result = model.invoke(
        [
            ("system", VERIFIER_SYSTEM_PROMPT),
            ("user", context),
            *step_messages,
        ]
    )

    print(f"[VERIFIER] {result.verdict} ({time.perf_counter() - start:.2f}s)")
    print(f"[VERIFIER] Reason: {result.reason}")

    for item in result.missing_evidence:
        print(f"[VERIFIER] Missing: {item}")

    if result.verdict == "replan" and not (
        result.contradiction and result.invalidated_steps
    ):
        print(
            f"[VERIFIER] replan without a named contradiction -> treating as incomplete"
        )
        result = result.model_copy(update={"verdict": "incomplete"})

    return {
        "verification": result,
        "step_retries": state.step_retries + (result.verdict == "incomplete"),
    }


def advance(state: AgentState) -> dict:
    """Mark the current step completed and start the next one clean."""

    index = state.current_step
    step = state.plan["steps"][index]

    print(f"[PLAN] Step {step['id']} running → completed")

    return {
        "plan": with_status(state.plan, index, "completed"),
        "current_step": index + 1,
        "verification": None,
        "step_retries": 0,
        "step_start_message_index": len(state.messages),
    }


def finalize(state: AgentState, model) -> dict:
    steps = state.plan["steps"]
    done = sum(1 for step in steps if step["status"] == "completed")
    # An empty plan means a replan discarded everything: nothing was verified,
    # so the report must not present itself as a finished investigation.
    unfinished = not steps or done < len(steps)

    print(
        f"\n[FINALIZE] {done}/{len(steps)} steps completed, "
        f"{tool_calls_so_far(state)} tool calls."
    )

    sections = [ROLE, REPORT_FORMAT]

    if unfinished:
        print("[FINALIZE] Plan did not finish: reporting with limitations.")
        sections.append(REPORT_LIMITATIONS)

    response = model.invoke(
        [
            ("system", "\n".join(sections)),
            *state.messages,
            ("human", ANSWER_NOW),
        ]
    )

    return {"messages": [response], "status": "blocked" if unfinished else "completed"}


def replanner(state: AgentState, model) -> dict:
    result = state.verification
    invalid = set(result.invalidated_steps)

    completed_steps = [
        s
        for s in state.plan["steps"]
        if s["status"] == "completed" and s["id"] not in invalid
    ]
    response = model.invoke(
        [
            ("system", REPLANNER_SYSTEM_PROMPT),
            (
                "user",
                f"""
Original task:
{state.task}

Previous goal:
{state.plan["goal"]}

Steps already completed (do not repeat these):
{[step["description"] for step in completed_steps]}

What contradicted the old plan:
{result.contradiction}

The evidence behind that contradiction:
{result.evidence}

Write the remaining steps only, starting from where the evidence now points.
""",
            ),
        ]
    )

    new_steps = [{**step, "status": "pending"} for step in response["steps"]]

    # Renumber everything, so ids stay 1..n and a dropped completed step
    # cannot leave two steps sharing an id.
    steps = [
        {**step, "id": position + 1}
        for position, step in enumerate(completed_steps + new_steps)
    ]

    print(f"[REPLAN] {result.contradiction}")
    print(f"[REPLAN] kept {len(completed_steps)} steps, {len(new_steps)} new")
    return {
        "plan": {"goal": response["goal"], "steps": steps},
        "current_step": len(completed_steps),
        "verification": None,
        "step_retries": 0,
        "replans": state.replans + 1,
        "step_start_message_index": len(state.messages),
    }


def decide_tool_call(state: AgentState) -> str:
    if not getattr(state.messages[-1], "tool_calls", None):
        print("\n[ROUTER] Executor stopped. Verifying.")
        return "verify"

    if state.iteration >= MAX_ITERATIONS:
        print(f"\n[ROUTER] Hit the {MAX_ITERATIONS} iteration limit.")
        return "finalize"

    if tool_calls_so_far(state) >= MAX_TOOL_CALLS:
        print(f"\n[ROUTER] Hit the {MAX_TOOL_CALLS} tool-call limit.")
        return "finalize"

    print("\n[ROUTER] Tool call detected.")
    return "tools"


def route_verification(state: AgentState) -> str:
    result = state.verification

    if result is None or result.verdict == "blocked":
        print("[ROUTER] Step blocked. Finalising with limitations.")
        return "blocked"

    if result.verdict == "complete":
        return "advance"

    if result.verdict == "replan":
        if state.replans >= MAX_REPLANS:
            print("[ROUTER] Replan budget exhausted.")
            return "blocked"
        return "replan"

    if state.step_retries > MAX_STEP_RETRIES:
        print(f"[ROUTER] Step retry budget ({MAX_STEP_RETRIES}) exhausted.")
        return "blocked"

    if state.iteration >= MAX_ITERATIONS:
        print(f"[ROUTER] Hit the {MAX_ITERATIONS} iteration limit.")
        return "blocked"

    print(f"[ROUTER] Retrying step (attempt {state.step_retries + 1}).")
    return "retry"


def after_replan(state: AgentState) -> str:
    if state.current_step < len(state.plan["steps"]):
        return "agent"

    # Replanning rule 9 allows an empty remainder when the evidence already
    # answers the task. Going to the agent here would index past the plan.
    print("\n[ROUTER] Replan left nothing to execute. Finalising.")
    return "finalize"


def after_advance(state: AgentState) -> str:
    if state.current_step < len(state.plan["steps"]):
        return "agent"

    print("\n[ROUTER] All plan steps complete.")
    return "finalize"


def build():
    graph = StateGraph(AgentState)

    graph.add_node(
        "planner",
        partial(planner, model=build_model(tools=None).with_structured_output(Plan)),
    )
    graph.add_node("agent", partial(agent, model=build_model()))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node(
        "verifier",
        partial(
            verifier,
            model=build_model(tools=None).with_structured_output(
                VerificationResult, method="json_schema"
            ),
        ),
    )
    graph.add_node(
        "replanner",
        partial(replanner, model=build_model(tools=None).with_structured_output(Plan)),
    )
    graph.add_node("advance", advance)
    graph.add_node("finalize", partial(finalize, model=build_model(tool_choice="none")))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")

    graph.add_conditional_edges(
        "agent",
        decide_tool_call,
        {"tools": "tools", "verify": "verifier", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")

    graph.add_conditional_edges(
        "verifier",
        route_verification,
        {
            "advance": "advance",
            "retry": "agent",
            "blocked": "finalize",
            "replan": "replanner",
        },
    )

    graph.add_conditional_edges(
        "replanner", after_replan, {"agent": "agent", "finalize": "finalize"}
    )

    graph.add_conditional_edges(
        "advance", after_advance, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()
