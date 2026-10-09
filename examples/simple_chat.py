"""Minimal interactive chat client that streams tokens from a running slm-runtime server.

    uv run slm-runtime serve            # in one terminal
    uv run python examples/simple_chat.py

The server exposes a raw completions API, so the client applies the chat template itself. The
template below is ChatML (Qwen2.5 and many other instruction-tuned models); change it to match
your model.
"""

from __future__ import annotations

import json
import sys

import httpx

URL = "http://127.0.0.1:8000/v1/completions"
SYSTEM = "You are a concise, helpful assistant."


def chatml(history: list[tuple[str, str]]) -> str:
    parts = [f"<|im_start|>system\n{SYSTEM}<|im_end|>"]
    parts += [f"<|im_start|>{role}\n{text}<|im_end|>" for role, text in history]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def main() -> None:
    history: list[tuple[str, str]] = []
    print("Chatting with slm-runtime. Ctrl-C to exit.")
    with httpx.Client(timeout=120) as client:
        while True:
            try:
                user = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not user:
                continue
            history.append(("user", user))
            payload = {
                "prompt": chatml(history),
                "max_tokens": 256,
                "temperature": 0.7,
                "stop": ["<|im_end|>"],
                "stream": True,
            }
            reply = ""
            print("bot> ", end="", flush=True)
            with client.stream("POST", URL, json=payload) as r:
                if r.status_code != 200:
                    r.read()
                    print(f"[HTTP {r.status_code}] {r.text}")
                    history.pop()
                    continue
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        print(f"\n[error] {event['error']['message']}", file=sys.stderr)
                        break
                    text = event["choices"][0]["text"]
                    reply += text
                    print(text, end="", flush=True)
            print()
            history.append(("assistant", reply))


if __name__ == "__main__":
    main()
