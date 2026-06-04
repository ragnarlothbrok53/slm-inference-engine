from runtime.engines.gguf_engine import GGUFEngine
from runtime.engines.mlx_engine import MLXEngine

class EngineManager:
    def __init__(self, model_path, engine='gguf'):
        if engine == 'gguf':
            self.engine = GGUFEngine(model_path)
        elif engine == 'mlx':
            self.engine = MLXEngine(model_path)

    def get_engine(self):
        return self.engine
