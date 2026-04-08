# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build and load Depth Anything V2 models with pretrained weights."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import torch_load

from .modules import DepthAnythingV2

# Map checkpoint filename suffixes → encoder size
MODEL_MAP = {
    "depth-anything-v2-vits.pt": "vits",
    "depth-anything-v2-vitb.pt": "vitb",
    "depth-anything-v2-vitl.pt": "vitl",
    # Aliases for common naming
    "depth_anything_v2_vits.pth": "vits",
    "depth_anything_v2_vitb.pth": "vitb",
    "depth_anything_v2_vitl.pth": "vitl",
}

# HuggingFace URLs for raw checkpoints
HF_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}


def _infer_encoder(checkpoint: str) -> str:
    """Infer encoder size from checkpoint filename."""
    ckpt = str(checkpoint).lower()
    for suffix, enc in MODEL_MAP.items():
        if ckpt.endswith(suffix):
            return enc
    # Fallback: look for vits/vitb/vitl in filename
    match = re.search(r"vit([sbl])", ckpt)
    if match:
        return f"vit{match.group(1)}"
    raise ValueError(f"Cannot infer encoder from checkpoint: {checkpoint}. Use a filename containing 'vits', 'vitb', or 'vitl'.")


def build_depth_anything(checkpoint: str = "depth_anything_v2_vits.pth") -> DepthAnythingV2:
    """Build a Depth Anything V2 model and load pretrained weights.

    Args:
        checkpoint: Path to .pt/.pth checkpoint file. If not found locally,
                    attempts to download from HuggingFace.

    Returns:
        DepthAnythingV2 model in eval mode with loaded weights.
    """
    encoder_name = _infer_encoder(checkpoint)

    # Build model (DINOv2 encoder loaded via torchhub with random weights)
    model = DepthAnythingV2(encoder_name=encoder_name)

    # Download checkpoint if needed
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        # Try downloading from HuggingFace
        url = HF_URLS.get(encoder_name)
        if url:
            checkpoint = attempt_download_asset(url)
        else:
            checkpoint = attempt_download_asset(checkpoint)

    # Load weights
    state_dict = torch_load(checkpoint)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
