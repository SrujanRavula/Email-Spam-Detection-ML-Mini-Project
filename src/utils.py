"""
Utility functions for logging, timing, and formatting.
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def setup_logger(name: str = "spam_detection") -> logging.Logger:
    """
    Configures and returns a standardized logger writing to both
    console (stdout) and logs/spam_detection.log.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timer(description: str = "Task", logger: logging.Logger = None) -> Generator[None, None, None]:
    """
    Context manager to accurately measure execution duration.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        msg = f"{description} completed in {elapsed:.3f} seconds."
        if logger:
            logger.info(msg)
        else:
            print(msg)


def print_banner(title: str, width: int = 70) -> None:
    """Prints a styled terminal banner."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)
