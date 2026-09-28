from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from fixer.tools.filesystem import read_file
from fixer.tools.repository import list_files, git_diff
from fixer.tools.search import search_code
from fixer.tools.shell import run_command
from fixer.tools.testing import run_tests
from fixer.tools.file_modification import apply_patch

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


@tool
def apply_patch_tool(patch: str, repo_path: RepoPath) -> dict:
    """
    Apply a unified diff to the repository.

    Read a file before patching it, and write the diff against exactly what you
    read: a/ and b/ path prefixes, unchanged context lines copied verbatim, and
    hunk headers whose start lines and counts match the file.

    Args:
        patch: The unified diff, for example

            --- a/src/testpkg/tribonacci.py
            +++ b/src/testpkg/tribonacci.py
            @@ -10,3 +10,3 @@
                 trib_history = [1, 1, 2, None]
            -    if n < 3:
            +    if n < 3 and n > 0:

    The result's "status" says what happened:
    - "applied": the files changed.
    - "unchanged": the repository already contained this change.
    - "failed": the diff did not match the files. Re-read the file and rebuild
      the diff from its actual contents rather than guessing again.
    - "error": the diff could not be parsed as a unified diff.

    "diagnostics" names the hunk that failed and what it expected.

    Applying a patch is not evidence that it works. Run run_tests_tool after.
    """

    print(f"\n[TOOL] apply_patch for diff \n{patch}\n")

    return apply_patch(repo_path, patch)


TOOLS = [
    list_files_tool,
    read_file_tool,
    search_code_tool,
    git_diff_tool,
    run_command_tool,
    run_tests_tool,
    apply_patch_tool,
]
