from abc import ABC, abstractmethod


class Sandbox(ABC):
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
