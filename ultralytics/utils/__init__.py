# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import logging
import os
import platform
import sys
from pathlib import Path

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ultralytics package root
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"

# Logging
LOGGER = logging.getLogger("ultralytics")


def set_logging(name="ultralytics", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support."""
    level = logging.DEBUG if verbose else logging.ERROR

    # Configure handler with UTF-8 encoding for Windows compatibility
    formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    if platform.system() == "Windows":
        with contextlib.suppress(Exception):
            handler.stream.reconfigure(encoding="utf-8")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def is_colab():
    """Check if the current script is running inside a Google Colab notebook."""
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """Check if the current script is running inside a Kaggle kernel."""
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_docker():
    """Determine if the script is running inside a Docker container."""
    with contextlib.suppress(Exception):
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    return False


def get_ubuntu_version():
    """Retrieve the Ubuntu version if the OS is Ubuntu, otherwise return None."""
    with contextlib.suppress(Exception):
        with open("/etc/os-release") as f:
            content = f.read()
            if "ID=ubuntu" in content:
                for line in content.splitlines():
                    if line.startswith("VERSION_ID"):
                        return line.split("=")[1].strip().strip('"')
    return None


def colorstr(*input):
    """Apply ANSI color codes to a string for terminal output.

    Args:
        *input: Variable length arguments. If multiple, the last is the string,
                preceding ones are color/style names.

    Returns:
        str: Colorized string.

    Example:
        >>> colorstr('blue', 'bold', 'hello world')
        '\033[34m\033[1mhello world\033[0m'
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors.get(x, "") for x in args) + f"{string}" + colors["end"]


# Initialize logger
LOGGER = set_logging("ultralytics", verbose=True)
