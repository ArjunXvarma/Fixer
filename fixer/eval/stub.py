import difflib
from pathlib import Path

from langchain_core.messages import AIMessage

from fixer.eval.tasks import DEFAULT_REPO
from fixer.agent.state import Plan, VerificationResult

BUGGY_TRIBONACCI = """    trib_history = [1, 1, 2, None]
    if n < 3:
        return trib_history[n-1]
    for _ in range(n-3):
        trib_history[3] = sum(trib_history[:3])
        trib_history[:3] = trib_history[1:]
    return trib_history[3]"""

FIXED_TRIBONACCI = """    trib_history = [0, 1, 1]
    if n < 3:
        return trib_history[n]
    for _ in range(n - 2):
        next_value = sum(trib_history)
        trib_history = [trib_history[1], trib_history[2], next_value]
    return trib_history[2]"""

FIXES = {
    "tribonacci": (
        "src/testpkg/tribonacci.py",
        lambda text: text.replace(BUGGY_TRIBONACCI, FIXED_TRIBONACCI),
    ),
    "missing_colon": (
        "tests/missing_colon.py",
        lambda text: text.replace(
            "def division(a: float, b: float) -> float\n",
            "def division(a: float, b: float) -> float:\n",
        ),
    ),
}


def patch_for(path: str, fix) -> str:
    """
    A unified diff built from the fixture, so it always applies.

    Built with lineterm="" rather than keepends: these fixtures have no
    trailing newline, and keepends then glues the last - and + lines together
    into a diff that will not parse.
    """

    text = (DEFAULT_REPO / path).read_text()

    lines = difflib.unified_diff(
        text.splitlines(),
        fix(text).splitlines(),
        f"a/{path}",
        f"b/{path}",
        lineterm="",
    )

    return "\n".join(lines) + "\n"


def which_task(messages) -> str | None:
    text = " ".join(str(getattr(m, "content", m)) for m in messages).lower()

    for name in FIXES:
        if name in text or name.replace("_", " ") in text:
            return name

    return None


class StubModel:
    def __init__(self, kind="text"):
        self.kind = kind
        self.patched = False

    def with_structured_output(self, schema, **kwargs):
        return StubModel("verifier" if schema is VerificationResult else "plan")

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        if self.kind == "plan":
            return Plan(
                goal="Fix the reported bug.",
                steps=[
                    {
                        "id": 1,
                        "description": "Apply the fix.",
                        "purpose": "Resolve the reported bug.",
                        "expected_result": "The patch applies.",
                        "status": "pending",
                    }
                ],
            )

        if self.kind == "verifier":
            return VerificationResult(
                verdict="complete",
                reason="stub verifier",
                evidence=["the patch applied"],
                missing_evidence=[],
            )

        task = which_task(messages)

        if task and not self.patched:
            self.patched = True
            path, fix = FIXES[task]

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "apply_patch_tool",
                        "args": {"patch": patch_for(path, fix)},
                        "id": "stub-1",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content="Stub run finished.")


def install_stub() -> None:
    """Point every architecture at the stub instead of a provider."""

    import fixer.architectures.autonomous_agent as autonomous_agent
    import fixer.architectures.react as react
    import fixer.architectures.simple_planner as simple_planner

    for module in (react, simple_planner, autonomous_agent):
        module.build_model = lambda tools=None, tool_choice=None: StubModel()

        if hasattr(module, "build_structured_model"):
            module.build_structured_model = lambda schema: (
                StubModel().with_structured_output(schema)
            )

    print("[EVAL] dry run: stub model, no API calls")
