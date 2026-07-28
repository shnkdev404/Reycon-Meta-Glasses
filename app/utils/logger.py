"""
Structured logger for system telemetry, spatial fusion events, and threat alerts.
Handles UTF-8 stream encoding safely on Windows consoles.
"""
import logging
import sys

# Configure UTF-8 encoding for Windows stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(f"perception.{name}")
