import typer
import uvicorn
import os

app = typer.Typer(help="ML Service Runtime Support CLI")

@app.callback()
def main():
    """ML Service CLI Management Tool."""
    pass

@app.command()
def serve(
    model: str = typer.Option("models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf", help="Path to model"),
    engine: str = typer.Option("gguf", help="Engine to use: gguf, mlx")
):
    """Start the ML Service runtime server."""
    os.environ["MODEL_PATH"] = model
    os.environ["ENGINE"] = engine
    
    print(f"Starting ML Service runtime server with {engine} engine...")
    uvicorn.run("runtime.api.server:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    app()
