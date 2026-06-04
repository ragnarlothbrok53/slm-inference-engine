from fastapi import APIRouter
from runtime.models.request import CompletionRequest
from runtime.runtime.engine_manager import EngineManager

router = APIRouter()
engine_manager = None


def initialize_engine(model_path, engine):
    global engine_manager
    engine_manager = EngineManager(model_path, engine=engine)


@router.post("/v1/completions")
async def completion(req: CompletionRequest):
    engine = engine_manager.get_engine()
    text = await engine.generate(req.prompt)
    return {"text": text}
