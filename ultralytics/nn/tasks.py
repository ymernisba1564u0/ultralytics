# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model task definitions for detection, segmentation, and classification."""

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C3, SPPF, Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import model_info, select_device


def parse_model(d, ch):
    """Parse a model definition dict and return the model layers and output channel list."""
    LOGGER.info(f"{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc = d.get('nc', 80)
    depth_multiple = d.get('depth_multiple', 1.0)
    width_multiple = d.get('width_multiple', 1.0)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d.get('backbone', []) + d.get('head', [])):
        m = eval(m) if isinstance(m, str) else m  # eval module string
        n = max(round(n * depth_multiple), 1) if n > 1 else n

        if m in (Conv, C3, SPPF):
            c1, c2 = ch[f], args[0]
            c2 = int(c2 * width_multiple)
            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)
                n = 1
        elif m is Detect:
            args.append([ch[x] for x in f])

        module = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in module.parameters())
        module.i, module.f, module.type, module.np = i, f, t, np
        LOGGER.info(f"{i:>3}{str(f):>18}{n:>3}{np:>10}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    """Base class for all YOLO models."""

    def forward(self, x):
        """Forward pass through the model."""
        return self._forward_once(x)

    def _forward_once(self, x):
        """Run forward pass, caching outputs for skip connections."""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def info(self, verbose=False):
        """Print model information."""
        model_info(self, verbose=verbose)

    def load(self, weights):
        """Load weights into the model, transferring only matching layers."""
        csd = weights.float().state_dict()
        # Only transfer weights where both key name and tensor shape match
        csd = {k: v for k, v in csd.items() if k in self.state_dict() and
               self.state_dict()[k].shape == v.shape}
        self.load_state_dict(csd, strict=False)
        LOGGER.info(f"Transferred {len(csd)}/{len(self.state_dict())} items from pretrained weights")


class DetectionModel(BaseModel):
    """YOLO detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None):
        """Initialize DetectionModel with conf
