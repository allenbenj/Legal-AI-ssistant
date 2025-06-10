"""Utility for initializing application-wide logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    JSONHandler,
)


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def init_logging(
    level: int = logging.INFO, log_file: Optional[Path] = None
) -> logging.Logger:
    """Configure root logging and return a detailed logger for the app.

    Parameters
    ----------
    level:
        Logging level for the root logger.
    log_file:
        Optional path for a standard log file.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if log_file is None:
        log_file = LOG_DIR / "app.log"

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        )
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

        # Structured JSON logs
        json_log_path = log_file.with_suffix(".jsonl")
        json_handler = JSONHandler(json_log_path)
        json_handler.setLevel(level)
        logging.getLogger().addHandler(json_handler)

    # Ensure our detailed logger for the application exists
    logger = get_detailed_logger("LegalAISystem", LogCategory.SYSTEM)
    logger.info(
        "Logging initialized",
        parameters={"log_file": str(log_file)},
    )
    return logger


__all__ = ["init_logging", "LOG_DIR"]
