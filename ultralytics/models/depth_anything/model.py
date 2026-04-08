# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth Anything V2 model interface for Ultralytics.

Provides zero-shot monocular depth estimation using pretrained Depth Anything V2 models
with DINOv2 encoder + DPT decoder architecture.

Examples:
    >>> from ultralytics import DepthAnything
    >>> model = DepthAnything("depth_anything_v2_vits.pth")
    >>> results = model.predict("image.jpg")
    >>> depth_map = results[0].depth  # numpy array (H, W)
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .predict import DepthAnythingPredictor


class DepthAnything(Model):
    """Depth Anything V2 model for zero-shot monocular depth estimation.

    This class provides an interface to pretrained Depth Anything V2 models within the
    Ultralytics framework. It supports ViT-S, ViT-B, and ViT-L encoder variants.

    Attributes:
        model (torch.nn.Module): The loaded Depth Anything V2 model.
        task (str): The task type, set to "depth".

    Examples:
        >>> model = DepthAnything("depth_anything_v2_vits.pth")
        >>> results = model.predict("image.jpg", imgsz=518)
        >>> results[0].depth  # relative depth map as numpy array
    """

    def __init__(self, model: str = "depth_anything_v2_vits.pth") -> None:
        """Initialize Depth Anything V2 model.

        Args:
            model: Path to pretrained weights (.pt or .pth).
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("Depth Anything requires pre-trained *.pt or *.pth model.")
        super().__init__(model=model, task="depth")

    def _load(self, weights: str, task=None):
        """Load pretrained Depth Anything V2 weights.

        Args:
            weights: Path to the weights file.
            task: Task name (unused, always "depth").
        """
        from .build import build_depth_anything

        self.model = build_depth_anything(weights)

    def predict(self, source, stream=False, **kwargs):
        """Run depth estimation on the given source.

        Args:
            source: Image path, PIL Image, numpy array, or video source.
            stream: If True, enables streaming mode.
            **kwargs: Additional arguments passed to the predictor.

        Returns:
            List of Results objects with .depth attribute.
        """
        overrides = dict(conf=0.25, task="depth", mode="predict", imgsz=518)
        kwargs = {**overrides, **kwargs}
        return super().predict(source, stream, **kwargs)

    def info(self, detailed=False, verbose=True):
        """Log model information."""
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        """Map 'depth' task to DepthAnythingPredictor."""
        return {"depth": {"predictor": DepthAnythingPredictor}}
