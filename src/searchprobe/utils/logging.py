"""Structured logging utilities for SearchProbe."""

import logging
import sys

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a structured logger with rich formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"searchprobe.{name}")

    if not logger.handlers:
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for SearchProbe.

    Args:
        level: Logging level for the searchprobe namespace
    """
    root_logger = logging.getLogger("searchprobe")

    if not root_logger.handlers:
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    root_logger.setLevel(level)
