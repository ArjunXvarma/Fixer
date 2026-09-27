import time
from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from fixer.agent.llm import build_model
from fixer.agent.prompts import REPORT_FORMAT, ROLE, TOOL_RULES
from fixer.agent.state import AgentState
from fixer.agent.tools import TOOLS

MAX_ITERATIONS = 40

WORKFLOW = """
# How to investigate

Follow this order. A normal investigation takes 4-6 tool calls in total.

1. list_files_tool — see the layout. Once.
2. run_tests_tool — run the tests the task names, or the whole suite. Once.
3. read_file_tool — read the test that fails.
4. read_file_tool — read the implementation that test exercises.
5. run_command_tool — only if you must observe runtime behaviour the code
   does not make obvious. One call.
6. Write your diagnosis.

Skip any step the task does not need.
"""

STOP = """
# When to stop

Stop calling tools as soon as the task's questions can be answered from
evidence you already have.
"""

SYSTEM_PROMPT = "\n".join([ROLE, WORKFLOW, TOOL_RULES, STOP, REPORT_FORMAT])


def agent(state: AgentState, model) -> dict:
    iteration = state.iteration + 1

    print(f"\n{'=' * 70}")
    print(f"[AGENT] Iteration {iteration}")
    print(f"{'=' * 70}")

    start = time.perf_counter()

    response = model.invoke([("system", SYSTEM_PROMPT), *state.messages])

    print(f"[AGENT] Gemini responded in {time.perf_counter() - start:.2f}s")

    for call in response.tool_calls:
        print(f"  → {call['name']} {call['args']}")

    if not response.tool_calls:
        print("[AGENT] Final response received.")

    return {"messages": [response], "iteration": iteration}


def decide_tool_call(state: AgentState) -> str:
    if not getattr(state.messages[-1], "tool_calls", None):
        print("\n[ROUTER] No tool call detected. Ending agent.")
        return "end"

    if state.iteration >= MAX_ITERATIONS:
        print(f"\n[ROUTER] Hit the {MAX_ITERATIONS} iteration limit. Ending agent.")
        return "end"

    print("\n[ROUTER] Tool call detected.")
    return "tool"


def build():
    model = build_model()

    graph = StateGraph(AgentState)

    graph.add_node("agent", partial(agent, model=model))
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", decide_tool_call, {"tool": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
