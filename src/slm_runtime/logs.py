"""Logging setup: human-readable by default, one JSON object per line with ``SLM_LOG_JSON=true``."""

from __future__ import annotations

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, httpx) through loguru so all output shares one format."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    logger.remove()
    if json:
        logger.add(sys.stderr, level=level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss.SSS}</green> <level>{level: <7}</level> "
            "{message} <dim>{extra}</dim>",
        )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False
    # The app emits its own structured access log; uvicorn's would be a duplicate.
    logging.getLogger("uvicorn.access").disabled = True
