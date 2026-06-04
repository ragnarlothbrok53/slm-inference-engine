from fastapi import FastAPI
from runtime.api.routes import router, initialize_engine

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def load_model():
    import os
    model_path = os.getenv("MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
    engine = os.getenv("ENGINE", "gguf")
    initialize_engine(model_path, engine)


app.include_router(router)
