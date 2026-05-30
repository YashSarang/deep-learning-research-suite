"""
Logger — lightweight colored console output with timestamps.
Used throughout the pipeline for consistent log formatting.
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime


# ── ANSI colour codes ─────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _CYAN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BOLD + _RED,
}


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        ts     = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"{_GREY}[{ts}]{_RESET} {colour}{record.levelname:8s}{_RESET}"
        msg    = super().format(record)
        return f"{prefix} {msg}"


def get_logger(name: str = "mcq_solver", level: int = logging.INFO) -> logging.Logger:
    """
    Get (or create) a named logger with coloured console output.

    Usage:
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Pipeline started")
        log.warning("Low confidence: %.2f", conf)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColourFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Convenience singleton
log = get_logger()
