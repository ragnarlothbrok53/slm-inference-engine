"""Best-effort host description, recorded with benchmark results and shown by ``slm-runtime info``.

Uses only the standard library and psutil; the previous version imported torch (~GBs) just to
check for CUDA/MPS.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil

_WIN_CPU_KEY = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":  # (not platform.system(): mypy only narrows sys.platform)
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WIN_CPU_KEY) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return platform.processor() or platform.machine()


def _nvidia_gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def detect_hardware() -> dict[str, Any]:
    gpus = _nvidia_gpus()
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "nvidia_gpus": gpus,
        "accelerator": "cuda" if gpus else ("metal" if apple_silicon else "cpu"),
    }
