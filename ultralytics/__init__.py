# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.0.0"

from ultralytics.models import YOLO
from ultralytics.utils import SETTINGS

# Personal fork - studying YOLO architecture and training pipeline
__author_note__ = "Fork maintained for personal learning and experimentation"

# Default to verbose=False for cleaner output during experiments
SETTINGS.update({
    "verbose": False,
    "runs_dir": "./runs",  # keep experiment outputs in local ./runs directory
})

__all__ = ["__version__", "YOLO", "SETTINGS"]
