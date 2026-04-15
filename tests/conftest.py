# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def solution_assets():
    """
    Session-scoped fixture to cache solution test assets.

    Downloads videos and other assets once to persistent directory (WEIGHTS_DIR/test_assets).
    Returns a dict mapping asset names to cached paths.
    """
    from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
    from ultralytics.utils.downloads import safe_download

    # Use persistent directory alongside weights
    cache_dir = WEIGHTS_DIR / "test_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Define all assets needed for solution tests
    assets = {
        # Videos
        "demo_video": "solutions_ci_demo.mp4",
        "crop_video": "decelera_landscape_min.mov",
        "pose_video": "solution_ci_pose_demo.mp4",
        "parking_video": "solution_ci_parking_demo.mp4",
        "vertical_video": "solution_vertical_demo.mp4",
        # Parking manager files
        "parking_areas": "solution_ci_parking_areas.json",
        "parking_model": "solutions_ci_parking_model.pt",
    }

    asset_paths = {}
    for name, filename in assets.items():
        asset_path = cache_dir / filename
        if not asset_path.exists():
            safe_download(url=f"{ASSETS_URL}/{filename}", dir=cache_dir)
        asset_paths[name] = asset_path

    return asset_paths


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    """Modify the list of test items to exclude tests marked as slow if the --slow option is not specified.

    Args:
        config: The pytest configuration object that provides access to command-line options.
        items (list): The list of collected pytest item objects to be modified based on the presence of --slow option.
    """
    if not config.getoption("--slow"):
        # Remove the item entirely from the list of test items if it's marked as 'slow'
        items[:] = [item for item in items if "slow" not in item.keywords]


def pytest_sessionstart(session):
    """Initialize session configurations for pytest.

    This function is automatically called by pytest after the 'Session' object has been created but before performing
    test collection. It sets the initial seeds for the test session.

    Args:
        session: The pytest session object.
    """
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Cleanup operations after pytest session.

    This function is automatically called by pytest at the end of the entire test session. It removes certain files and
    directories used during testing.

    Args:
        terminalreporter: The terminal reporter object used for terminal output.
        exitstatus (int): The exit status of the test run.
        config: The pytest config object.
    """
    from ultralytics.utils import WEIGHTS_DIR

    # Remove files
    models = [path for x in {"*.onnx", "*.torchscript"} for path in WEIGHTS_DIR.rglob(x)]
    for file in ["decelera_portrait_min.mov", "bus.jpg", "yolo26n.onnx", "yolo26n.torchscript", *models]:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    models = [path for x in {"*.mlpackage", "*_openvino_model"} for path in WEIGHTS_DIR.rglob(x)]
    for directory in [WEIGHTS_DIR / "test_assets", WEIGHTS_DIR / "path with spaces", *models]:
        shutil.rmtree(directory, ignore_errors=True)
