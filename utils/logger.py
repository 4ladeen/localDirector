"""
utils/logger.py – Centralised logging for Director-Local.

Every module writes through this logger so that all messages land in both
the console *and* a ``process.log`` file with ISO-8601 timestamps.
"""

import logging
import os
import sys
import time
from functools import wraps

LOG_FILE = "process.log"
_LOGGER_NAME = "director"

# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return (or create) the shared application logger."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def log_timing(label: str):
    """Decorator that logs how long a function took to run."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            minutes, seconds = divmod(int(elapsed), 60)
            logger.info("%s complete: %dm %02ds", label, minutes, seconds)
            return result
        return wrapper
    return decorator
