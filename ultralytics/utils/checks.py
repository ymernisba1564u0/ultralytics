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
            # Use --quiet flag to reduce pip output noise during auto-install
            LOGGER.info(f"Attempting to auto-install missing packages: {missing}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                LOGGER.warning(f"WARNING ⚠️ Failed to install some packages: {result.stderr.strip()}")
                return False
            LOGGER.info("Auto-install complete.")
        else:
            return False
    return True
