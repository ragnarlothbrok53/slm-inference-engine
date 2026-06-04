from mlx_lm import load, generate
from .base import BaseEngine
from mlx_lm import stream_generate


class MLXEngine(BaseEngine):
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)

    async def generate(self, prompt: str):
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=128)

    async def stream(self, prompt: str):
        for response in stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=128
        ):
            yield response.text
