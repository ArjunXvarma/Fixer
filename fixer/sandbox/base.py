from abc import ABC, abstractmethod


class Sandbox(ABC):
    """
    A disposable copy of a repository for the agent to work in.

    Use it as a context manager so the copy is always cleaned up:

        with LocalSandbox() as box:
            box.copy_repo("test-repo")
            ...                        # agent works in box.workdir
            print(box.diff())          # what it changed
    """

    # Where the repository lives from the agent's point of view. Pass this as
    # the run's repo_path and every tool works unchanged.
    workdir: str

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def copy_repo(self, repo_path: str):
        pass

    @abstractmethod
    def run(self, command: list[str], timeout: int = 30) -> dict:
        pass

    @abstractmethod
    def diff(self) -> str:
        pass

    @abstractmethod
    def destroy(self):
        pass
