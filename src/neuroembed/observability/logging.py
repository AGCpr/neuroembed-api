"""Observability — structlog JSON logging + Prometheus metrics."""
from __future__ import annotations

import logging
import sys

import structlog

from neuroembed.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog + stdlib logging for JSON-line output to stdout.

    The format is one JSON object per line so a downstream log shipper
    (Loki, Datadog, ELK) can index fields without a parser. For local dev
    we use a console-friendly renderer instead.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger. Use this everywhere instead of stdlib logging."""
    result = structlog.get_logger(name)
    # structlog's return type is wide; the BoundLogger branch is the one we
    # get at runtime. Cast through Any to satisfy strict return types.
    return result  # type: ignore[no-any-return]
