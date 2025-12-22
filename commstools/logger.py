"""
Logging utilities for the CommsTools library.

This module provides a colorized logger to facilitate debugging and monitoring
of signal processing workflows.
"""

import logging
import sys


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI color codes to the log levels.
    """

    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s [%(levelname)s] [%(name)s/%(filename)s] %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with ANSI color codes based on the log level.
        """
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{log_color}{self.FORMAT}{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def get_logger(name: str = "commstools") -> logging.Logger:
    """
    Returns a logger instance for the CommsTools library.

    If no handlers are present, it adds a StreamHandler with a colorized formatter.

    Args:
        name: Name of the logger.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)

    return logger


# Create a default logger for the package
logger = get_logger()


def set_log_level(level):
    """
    Sets the log level for the CommsTools logger.

    Args:
        level: logging.DEBUG, logging.INFO, etc. or string "DEBUG", "INFO", etc.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
