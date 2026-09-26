"""
The tools the LLM can call.

repo_path is injected from the graph state, so the model never sees it and
never chooses it. Everything the model passes is relative to the repo root.
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from fixer.tools.filesystem import read_file
from fixer.tools.repository import list_files, git_diff
from fixer.tools.search import search_code
from fixer.tools.shell import run_command
from fixer.tools.testing import run_tests

# Filled in by ToolNode from AgentState.repo_path; invisible to the model.
RepoPath = Annotated[str, InjectedState("repo_path")]


@tool
def list_files_tool(repo_path: RepoPath) -> list[str]:
    """List the files in the repository, as paths relative to its root."""

    print("\n[TOOL] list_files")

    return list_files(repo_path)


@tool
def read_file_tool(file_path: str, repo_path: RepoPath) -> str:
    """
    Read a file from the repository.

    Args:
        file_path: Path relative to the repository root,
            e.g. "src/testpkg/tribonacci.py".
    """

    print(f"\n[TOOL] read_file {file_path}")

    return read_file(repo_path, file_path)


@tool
def run_tests_tool(
    repo_path: RepoPath,
    test_path: str = "",
    timeout: int = 120,
) -> dict:
    """
    Run the repository's tests. This handles the test environment and import
    paths for you: never run pytest through run_command_tool.

    Args:
        test_path: Optional test file or directory relative to the repository
            root, e.g. "tests/test_tribonaccy.py". Omit to run everything.

    The result's "phase" tells you where the run stopped: "collection" means no
    test ever ran (an import or config problem, which says nothing about the
    code under test), "run" means tests actually executed.
    """

    print("\n[TOOL] run_tests")

    return run_tests(repo_path, test_path or None, timeout)


@tool
def search_code_tool(pattern: str, repo_path: RepoPath, max_results: int = 50) -> dict:
    """Search the repository's source code for a regular expression."""

    print(f"\n[TOOL] search_code {pattern}")

    return search_code(repo_path, pattern, max_results)


@tool
def git_diff_tool(repo_path: RepoPath) -> str:
    """Return the repository's uncommitted git diff."""

    print("\n[TOOL] git_diff")

    return git_diff(repo_path)


@tool
def run_command_tool(
    command: list[str], repo_path: RepoPath, timeout: int = 30
) -> dict:
    """
    Run a single diagnostic command inside the repository.

    Use this only when no specialized repository tool can answer
    the question.

    Prefer:
    - list_files_tool for repository structure
    - read_file_tool for file contents
    - search_code_tool for code search
    - run_tests_tool for running tests
    - git_diff_tool for repository changes

    Do not use this tool to repeat information already available
    from another tool.
    """
    print(f"\n[TOOL] run_command {' '.join(command)}")

    result = run_command(repo_path, command, timeout)

    print(f"[TOOL] return_code={result['return_code']}")

    return result


TOOLS = [
    list_files_tool,
    read_file_tool,
    search_code_tool,
    git_diff_tool,
    run_command_tool,
    run_tests_tool,
]
