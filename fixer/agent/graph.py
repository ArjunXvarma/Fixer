import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from fixer.agent.state import AgentState
from fixer.agent.tools import TOOLS

load_dotenv()

# Stops a confused run from looping forever.
MAX_ITERATIONS = 40

SYSTEM_PROMPT = """

You are Fixer, an investigation agent. You diagnose problems in a Git
repository using tools. You never modify files.

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

# Rules

Before every tool call, check whether the answer is already in this
conversation. If it is, use it instead of calling the tool again.

One probe is enough. If you need runtime values, print them all in a single
command:

["python3", "-c", "from pkg.mod import f; print([f(n) for n in range(6)])"]

Never call a tool to re-print a value you have already seen.

Do not write your own reference implementation to compare against. Derive the
expected behaviour from the docstring, the tests, or the task.

Do not read files that cannot change your answer.

The test tool owns the test environment. Never run pytest through
run_command_tool, never set PYTHONPATH, never investigate how pytest is
installed.

Check the test result's "phase" before diagnosing:
- "collection" means no test ran. The problem is imports or configuration, and
  you have learned nothing about the implementation.
- "run" means tests executed, so failures are real outcomes.

# When to stop

Stop calling tools as soon as the task's questions can be answered from
evidence you already have. Then reply with:

1. What you ran or read.
2. The file and line responsible.
3. The evidence, quoted from tool output.
4. Your diagnosis.
5. Anything you could not determine.

State what you observed as fact and everything else as a hypothesis. Never say
tests pass unless the runner reported it.
"""

# bind_tools on Gemini builds its function schemas from the tool's full
# signature, which would show the model the injected repo_path. These specs
# leave injected arguments out; ToolNode still runs the real tools.
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite").bind_tools(
    [convert_to_openai_tool(tool) for tool in TOOLS]
)


def agent(state: AgentState) -> dict:
    iteration = state.iteration + 1

    print(f"\n{'=' * 70}")
    print(f"[AGENT] Iteration {iteration}")
    print(f"{'=' * 70}")

    start = time.perf_counter()

    response = model.invoke(
        [
            ("system", SYSTEM_PROMPT),
            *state.messages,
        ]
    )

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


graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tool_node", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent", decide_tool_call, {"tool": "tool_node", "end": END}
)
graph.add_edge("tool_node", "agent")

agent_loop = graph.compile()


if __name__ == "__main__":
    REPO_PATH = str(Path(__file__).resolve().parents[2] / "test-repo")

    TASK = """
    Investigate the failing Tribonacci tests.

    Do not modify any files.

    Determine:
    1. Which tests are failing.
    2. Which implementation is responsible.
    3. What the expected behavior appears to be.
    4. What the likely root cause is.
    """

    initial_state = AgentState(
        task=TASK,
        repo_path=REPO_PATH,
        messages=[{"role": "user", "content": TASK}],
    )

    result = agent_loop.invoke(
        initial_state, config={"recursion_limit": MAX_ITERATIONS * 2 + 5}
    )

    print(f"\n{'=' * 70}")
    print(f"[FIXER] Final response after {result['iteration']} iterations")
    print("=" * 70)
    print(result["messages"][-1].text or "[no answer: hit the iteration limit]")
