from fixer.tools.repository import list_files, git_diff
from fixer.tools.filesystem import read_file
from fixer.tools.shell import run_command


REPO = "test-repo"


def main():
    print("=== FILES ===")

    files = list_files(REPO)

    for file in files[:30]:
        print(file)

    print("\n=== FILE CONTENT ===")

    if files:
        print(read_file(REPO, files[0]))

    print("\n=== GIT STATUS ===")

    result = run_command(
        REPO,
        ["git", "status", "--short"],
    )

    print(result["stdout"])

    print("\n=== DIFF ===")

    print(git_diff(REPO))


if __name__ == "__main__":
    main()
