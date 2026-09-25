from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class AgentState:
    task: str
    repo_path: str
    plan: List[str] = field(default_factory=list)
    files_inspected: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    command_history: List[str] = field(default_factory=list)
    test_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "initialised"
