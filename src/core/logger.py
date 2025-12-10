"""ALC Logger - Centralized logging for ALC-Algo
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ALCLogger:
    """Centralized logging system for ALC-Algo.
    Maintains audit trail with TJH/RT identifiers.
    """

    def __init__(self, name: str = "ALC", log_dir: str = "data/logs"):
        """Initialize logger.

        Args:
        ----
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup file and console handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (daily rotation with machine isolation)
        today = datetime.now().strftime("%Y-%m-%d")
        machine_id = platform.node().replace(" ", "_").replace(".", "_")
        log_file = self.log_dir / f"alc_algo_{today}_{machine_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str, user_id: Optional[str] = "TJH"):
        """Log info message."""
        self.logger.info(f"[{user_id}] {message}")

    def warning(self, message: str, user_id: Optional[str] = "TJH"):
        """Log warning message."""
        self.logger.warning(f"[{user_id}] {message}")

    def error(self, message: str, user_id: Optional[str] = "TJH"):
        """Log error message."""
        self.logger.error(f"[{user_id}] {message}")

    def debug(self, message: str, user_id: Optional[str] = "TJH"):
        """Log debug message."""
        self.logger.debug(f"[{user_id}] {message}")

    def critical(self, message: str, user_id: Optional[str] = "TJH"):
        """Log critical message."""
        self.logger.critical(f"[{user_id}] {message}")


# Global logger instance
alc_logger = ALCLogger()

