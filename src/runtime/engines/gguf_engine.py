from llama_cpp import Llama
from .base import BaseEngine


class GGUFEngine(BaseEngine):
    def __init__(self, model_path: str):
        self.model = Llama(model_path=model_path, n_ctx=4096)

    async def generate(self, prompt: str):
        out = self.model(prompt, max_tokens=128)
        return out["choices"][0]["text"]

    async def stream(self, prompt: str):
        out = self.model(prompt, max_tokens=128, stream=True)
        for token in out:
            yield token["choices"][0]["text"]
