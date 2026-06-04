from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    async def generate(self, prompt: str):
        pass

    @abstractmethod
    async def stream(self, prompt: str):
        pass
