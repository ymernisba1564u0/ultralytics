# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Utility functions for checking system requirements and package dependencies."""

import importlib
import platform
import sys
from pathlib import Path
from typing import Optional

from ultralytics.utils import LOGGER


def check_python(minimum: str = "3.8.0") -> bool:
    """Check if current Python version meets the minimum required version.

    Args:
        minimum: Minimum Python version string, e.g. '3.8.0'.

    Returns:
        True if current version meets minimum, False otherwise.
    """
    current = platform.python_version()
    if _compare_versions(current, minimum) < 0:
        LOGGER.warning(
            f"WARNING ⚠️ Python {minimum} is required, but Python {current} is currently installed."
        )
        return False
    return True


def check_version(
    current: str,
    minimum: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
) -> bool:
    """Check if a version string meets the minimum required version.

    Args:
        current: Current version string.
        minimum: Minimum required version string.
        name: Name of the package/component being checked (for logging).
        hard: If True, raise an exception instead of returning False.

    Returns:
        True if current version meets minimum requirement.

    Raises:
        ModuleNotFoundError: If hard=True and version check fails.
    """
    result = _compare_versions(current, minimum) >= 0
    if not result:
        msg = f"{name} version {current} does not meet minimum requirement of {minimum}."
        if hard:
            raise ModuleNotFoundError(msg)
        LOGGER.warning(f"WARNING ⚠️ {msg}")
    return result


def check_requirements(requirements, exclude: tuple = (), install: bool = True) -> bool:
    """Check and optionally install missing Python package requirements.

    Args:
        requirements: A string, Path to requirements.txt, or list of requirement strings.
        exclude: Tuple of package names to exclude from checks.
        install: If True, attempt to install missing packages.

    Returns:
        True if all requirements are satisfied.
    """
    import pkg_resources

    if isinstance(requirements, Path):
        requirements = requirements.read_text().splitlines()
    elif isinstance(requirements, str):
        requirements = [requirements]

    missing = []
    for req in requirements:
        req = req.strip()
        if not req or req.startswith("#"):
            continue
        pkg = req.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
        if pkg in exclude:
            continue
        try:
            pkg_resources.require(req)
        except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
            missing.append(req)

    if missing:
        pkgs_str = " ".join(f'"{m}"' for m in missing)
        LOGGER.info(f"Missing packages detected: {missing}")
        if install:
            import subprocess
            LOGGER.info(f"Attempting to install missing packages...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *missing],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                LOGGER.warning(f"Failed to install packages: {result.stderr}")
                return False
            LOGGER.info("Packages installed successfully.")
        else:
            LOGGER.warning(f"Please install required packages: pip install {pkgs_str}")
            return False
    return True


def check_imshow(warn: bool = False) -> bool:
    """Check if cv2.imshow() can be used in the current environment."""
    try:
        import cv2
        cv2.imshow("test", __import__("numpy").zeros((1, 1, 3), dtype="uint8"))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow(): {e}")
        return False


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    from packaging.version import Version
    a, b = Version(v1), Version(v2)
    return 0 if a == b else (1 if a > b else -1)
