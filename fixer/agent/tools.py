from fixer.tools.filesystem import read_file
from fixer.tools.repository import list_files, git_diff
from fixer.tools.search import search_code
from fixer.tools.shell import run_command

TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "git_diff": git_diff,
    "run_command": run_command,
}
