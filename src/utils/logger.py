"""Academic-grade structured logging utility for reproducible ML experiments.

This module provides a standardised logger that outputs to both console
(coloured, human-readable) and a rotating file (structured, machine-parseable).
It replaces all ``print()`` statements throughout the codebase with proper
log-level routing.

Usage::

    from src.utils.logger import setup_logger

    logger = setup_logger(__name__)
    logger.info("Training started with config: %s", config)
    logger.warning("Dataset fraction < 0.01 — PPL may be unstable")

Do **not** call ``setup_logger`` at module level in every file; instead rely
on the root logger configured once in the entrypoint (e.g., ``train.py``).
Import ``logging.getLogger(__name__)`` directly when the root config has
already been applied.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default format strings
# ----------------------
# Console: concise, coloured (via a custom StreamHandler below)
CONSOLE_FMT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
CONSOLE_DATE_FMT: str = "%H:%M:%S"

# File: verbose, machine-parseable
FILE_FMT: str = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(filename)s:%(lineno)d | %(message)s"
FILE_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

# Default log level
DEFAULT_LEVEL: int = logging.INFO

# Maximum size per log file before rotation (10 MB)
MAX_BYTES: int = 10 * 1024 * 1024

# Number of backup log files to retain
BACKUP_COUNT: int = 3


class _ColourFormatter(logging.Formatter):
    """Lightweight ANSI-coloured console formatter for log-level emphasis.

    Maps WARNING and ERROR levels to yellow and red respectively so that
    important messages stand out in the terminal while keeping INFO/DEBUG
    at the default colour.
    """

    # ANSI escape codes
    _RESET: str = "\033[0m"
    _COLOURS: dict[int, str] = {
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }

    def format(self, record: logging.LogRecord) -> str:
        """Apply ANSI colour wrapping before delegating to the parent formatter."""
        colour = self._COLOURS.get(record.levelno, "")
        if colour:
            record.levelname = f"{colour}{record.levelname}{self._RESET}"
            record.msg = f"{colour}{record.msg}{self._RESET}"
        return super().format(record)


def _resolve_log_dir(log_dir: Optional[str]) -> Path:
    """Determine the logging directory, creating it if necessary.

    Args:
        log_dir: Explicit directory path. If ``None``, defaults to
            ``<project_root>/output/logs``.

    Returns:
        A ``pathlib.Path`` pointing to an existing, writable directory.
    """
    if log_dir is not None:
        path = Path(log_dir)
    else:
        # Default: <project_root>/output/logs
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "output" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_file_handler(
    log_dir: Optional[str],
    experiment_name: str,
) -> RotatingFileHandler:
    """Create a rotating file handler with a timestamped filename.

    The filename format is ``<experiment_name>_YYYY-MM-DD_HHMMSS.log`` so
    that each run produces a uniquely identifiable log artifact.

    Args:
        log_dir: Directory for log files (see ``_resolve_log_dir``).
        experiment_name: Human-readable experiment identifier.

    Returns:
        A configured ``RotatingFileHandler`` instance.
    """
    directory = _resolve_log_dir(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sanitised_name = experiment_name.replace(" ", "_").replace("/", "-")
    filename = f"{sanitised_name}_{timestamp}.log"
    filepath = directory / filename

    handler = RotatingFileHandler(
        str(filepath),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(DEFAULT_LEVEL)
    handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=FILE_DATE_FMT))
    return handler


def _create_console_handler() -> logging.StreamHandler:
    """Create a console (stderr) handler with coloured formatting.

    Returns:
        A ``StreamHandler`` writing to ``sys.stderr``.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(DEFAULT_LEVEL)
    handler.setFormatter(_ColourFormatter(CONSOLE_FMT, datefmt=CONSOLE_DATE_FMT))
    return handler


def setup_logger(
    name: Optional[str] = None,
    *,
    log_dir: Optional[str] = None,
    experiment_name: str = "gpt2-code-finetuning",
    level: int = DEFAULT_LEVEL,
) -> logging.Logger:
    """Configure and return a root-level logger for academic ML experiments.

    This function should be called **once** at the top of the main entrypoint
    (``train.py`` or ``cli.py``).  Subsequent modules can safely use
    ``logging.getLogger(__name__)`` and will inherit the root configuration.

    Args:
        name: Logger name. Pass ``__name__`` from the calling module.
            If ``None``, configures the root logger.
        log_dir: Directory to store rotating log files. Defaults to
            ``<project_root>/output/logs``.
        experiment_name: Prefix for the log filename.
        level: Minimum log level to capture.

    Returns:
        A configured ``logging.Logger`` instance ready for use.

    Example:
        >>> logger = setup_logger(__name__, experiment_name="ablation-run-001")
        >>> logger.info("Starting training with PPL target.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid attaching duplicate handlers if already configured
    if not logger.handlers:
        console_handler = _create_console_handler()
        file_handler = _create_file_handler(log_dir, experiment_name)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Suppress overly verbose loggers from third-party libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.INFO)

    return logger
