"""Shared environment and command-line logging setup."""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def configure_logging() -> None:
    load_environment()
    level = os.getenv("VEGAS_LOG_LEVEL", "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Invalid VEGAS_LOG_LEVEL: {level}")
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(level)
