"""Logging-Einrichtung: verstaendliche Konsolen-Ausgabe + persistente
Log-Datei fuer jeden Lauf.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "autocut"


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Richtet einen Logger ein, der gleichzeitig in die Konsole und in
    eine Datei unter ``log_dir/run_<timestamp>.log`` schreibt.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Verhindert doppelte Handler, falls setup_logging mehrfach aufgerufen wird
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug("Logging eingerichtet, Log-Datei: %s", log_file)
    return logger


def get_logger() -> logging.Logger:
    """Gibt den bereits eingerichteten Logger zurueck (oder einen
    Standard-Logger, falls setup_logging noch nicht aufgerufen wurde)."""
    return logging.getLogger(LOGGER_NAME)
