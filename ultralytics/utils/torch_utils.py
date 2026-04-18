# Ultralytics YOLO 🚀, AGPL-3.0 license
"""PyTorch utility functions for model operations."""

import os
import platform
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER


def select_device(device="", batch=0, verbose=True):
    """Select the appropriate compute device (CPU, CUDA, or MPS).

    Args:
        device (str): Device string, e.g. 'cpu', 'cuda:0', '0', 'mps'.
        batch (int): Batch size for validation.
        verbose (bool): Whether to log device info.

    Returns:
        torch.device: Selected device.
    """
    if isinstance(device, torch.device):
        return device

    device = str(device).lower().strip()
    cpu = device == "cpu"
    mps = device == "mps"

    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif mps:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this device.")
    else:
        if device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA not available, falling back to CPU.")
            return torch.device("cpu")

    if mps:
        selected = torch.device("mps")
    elif cpu or not torch.cuda.is_available():
        selected = torch.device("cpu")
    else:
        selected = torch.device("cuda:0")

    if verbose:
        LOGGER.info(f"Using device: {selected}")
    return selected


def get_num_params(model):
    """Return total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_num_gradients(model):
    """Return number of parameters with gradients in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(model, detailed=False, verbose=True):
    """Print model summary including layer count and parameter count.

    Args:
        model (nn.Module): PyTorch model.
        detailed (bool): Whether to print per-layer details.
        verbose (bool): Whether to print output.
    """
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules()))

    if verbose:
        LOGGER.info(f"Model summary: {n_l} layers, {n_p:,} parameters, {n_g:,} gradients")
        if detailed:
            for name, p in model.named_parameters():
                LOGGER.info(f"  {name}: {list(p.shape)}, grad={p.requires_grad}")
    return n_l, n_p, n_g


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from object b to object a, with optional include/exclude filters."""
    for k, v in b.__dict__.items():
        if (include and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


def strip_optimizer(f, s=""):
    """Strip optimizer from saved checkpoint to reduce file size.

    Args:
        f (str | Path): Path to checkpoint file.
        s (str): Output path; if empty, overwrites input.
    """
    x = torch.load(f, map_location=torch.device("cpu"))
    if "optimizer" in x:
        x["optimizer"] = None
    if "best_fitness" in x:
        x["best_fitness"] = None
    torch.save(x, s or f)
    mb = Path(s or f).stat().st_size / 1e6
    LOGGER.info(f"Optimizer stripped from '{f}', saved to '{s or f}' ({mb:.1f} MB)")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes wait for rank-0 to complete a task in distributed training."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        torch.distributed.barrier()
    yield
    if initialized and local_rank == 0:
        torch.distributed.barrier()
