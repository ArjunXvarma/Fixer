from dataclasses import dataclass, field
from typing import List, Dict, Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanStep(TypedDict):
    id: Annotated[int, "The id of the step"]
    description: Annotated[str, "What to do in the step"]
    purpose: Annotated[str, "Why this step is necessary"]
    expected_result: Annotated[str, "What evidence or result should be obtained"]
    status: Annotated[str, "pending, running, completed, or failed"]


class Plan(TypedDict):
    """The complete plan structure"""

    goal: Annotated[str, "The end goal the plan should aim to achieve"]
    steps: Annotated[
        list[PlanStep], "All the steps involved in the plan to achieve the goal"
    ]


class VerificationResult(BaseModel):
    verdict: Literal["complete", "incomplete", "blocked"]

    reason: str = Field(description="Why this verdict is justified")

    evidence: list[str] = Field(
        description="Specific observations supporting the verdict"
    )

    missing_evidence: list[str] = Field(description="Evidence still required, if any")


@dataclass
class AgentState:
    task: str
    repo_path: str
    messages: Annotated[list, add_messages]
    verification: VerificationResult | None = None
    step_start_message_index: int = 0
    step_retries: int = 0
    plan: Plan = field(default_factory=lambda: Plan(goal="", steps=[]))
    files_inspected: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    command_history: List[str] = field(default_factory=list)
    test_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "initialised"
    current_step: int = 0
