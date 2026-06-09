# SLM Inference Engine

## Project Overview

This project provides a high-performance inference engine for Small Language Models (SLMs), designed for efficient deployment and serving of AI models. It supports various backend engines, including `llama-cpp-python` and `mlx-lm`, to leverage different hardware capabilities and model formats. The engine is built with FastAPI for a robust API interface and includes features for hardware detection, performance profiling, and streaming responses.

## Key Features

*   **Flexible Backend Support:** Integrates with `llama-cpp-python` for GGUF models and `mlx-lm` for Apple Silicon optimization.
*   **High-Performance API:** Built with FastAPI for fast and scalable model serving.
*   **Hardware Detection and Profiling:** Automatically detects available hardware and provides tools for performance profiling.
*   **Streaming Responses:** Supports streaming of generated tokens for real-time interaction.
*   **Observability:** Includes Prometheus metrics and OpenTelemetry for monitoring and tracing.
*   **Modular Design:** A well-structured codebase that allows for easy extension and integration of new model engines.

## Technical Stack

*   **Backend Framework:** FastAPI
*   **LLM Inference:** `llama-cpp-python`, `mlx-lm`
*   **Logging:** `loguru`
*   **Observability:** `prometheus-client`, `opentelemetry`
*   **CLI:** `typer`
*   **Language:** Python 3.11+

## Getting Started

To set up and run the SLM Inference Engine, follow these steps:

### Prerequisites

*   Python 3.11+
*   `uv` (recommended for dependency management)

### Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ragnarlothbrok53/slm-inference-engine.git
    cd slm-inference-engine
    ```
2.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```
3.  **Run the inference engine:**
    ```bash
    python -m src.runtime.cli serve
    ```
    This will start the FastAPI server, making the SLM inference API available.

## Usage Examples

### Simple Chat

An example of a simple chat application using the inference engine can be found in `examples/simple_chat.py`.

## Project Structure

*   **`src/runtime/api/`**: Defines the FastAPI routes for model inference.
*   **`src/runtime/engines/`**: Contains implementations for different LLM backend engines (e.g., GGUF, MLX).
*   **`src/runtime/hardware/`**: Modules for hardware detection and profiling.
*   **`examples/`**: Sample scripts demonstrating how to use the inference engine.
*   **`tests/`**: Unit and integration tests.

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` (if available) for guidelines.

## License

This project is licensed under the MIT license - see the `LICENSE` file for details.
