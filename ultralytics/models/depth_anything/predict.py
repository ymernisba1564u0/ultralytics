# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth Anything V2 predictor for monocular depth estimation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results


class DepthAnythingPredictor(BasePredictor):
    """Predictor for Depth Anything V2 monocular depth estimation.

    Produces per-pixel relative depth maps from RGB images. The output is
    affine-invariant (scale and shift are arbitrary); use least-squares
    alignment against ground truth for evaluation.

    Examples:
        >>> from ultralytics import DepthAnything
        >>> model = DepthAnything("depth_anything_v2_vits.pth")
        >>> results = model.predict("image.jpg")
        >>> depth_map = results[0].depth  # numpy array (H, W)
    """

    def __init__(self, cfg="default.yaml", overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"

    def preprocess(self, im):
        """Normalize images with ImageNet stats for DINOv2 encoder.

        Args:
            im: torch.Tensor (B, 3, H, W) or list of numpy arrays.

        Returns:
            Preprocessed tensor (B, 3, H, W) with H,W divisible by 14.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255.0

        # ImageNet normalization (required for DINOv2)
        im = (im - self.mean) / self.std

        # Ensure dimensions are divisible by 14 (ViT patch size)
        _, _, h, w = im.shape
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        if new_h != h or new_w != w:
            im = F.interpolate(im, size=(new_h, new_w), mode="bilinear", align_corners=True)

        return im

    def inference(self, im, *args, **kwargs):
        """Run depth estimation inference."""
        return self.model.model(im)

    def postprocess(self, preds, img, orig_imgs):
        """Convert depth predictions to Results objects.

        Args:
            preds: Model output tensor (B, H, W) of relative depth values.
            img: Preprocessed input tensor.
            orig_imgs: Original images (list of numpy arrays or tensor).

        Returns:
            List of Results, each with a .depth attribute (numpy array H×W).
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = [orig_imgs] if isinstance(orig_imgs, np.ndarray) else orig_imgs.cpu().numpy()
            if orig_imgs.ndim == 4:
                orig_imgs = list(orig_imgs)
            else:
                orig_imgs = [orig_imgs]

        results = []
        for i, orig_img in enumerate(orig_imgs):
            depth = preds[i] if preds.ndim == 3 else preds
            orig_h, orig_w = orig_img.shape[:2]

            # Resize depth map to original image size
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=True,
            ).squeeze().cpu().numpy()

            result = Results(
                orig_img=orig_img,
                path=self.batch[0][i] if self.batch else "",
                names={0: "depth"},
            )
            result.depth = depth  # Store depth map as attribute
            results.append(result)

        return results

    def setup_model(self, model, verbose=True):
        """Set up Depth Anything model for inference."""
        from ultralytics.utils.torch_utils import select_device

        device = select_device(self.args.device, verbose=verbose)
        model = model.to(device)
        model = model.half() if self.args.half else model.float()
        model.eval()

        self.model = model
        self.device = device

        # ImageNet normalization constants for DINOv2
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        if self.args.half:
            self.mean = self.mean.half()
            self.std = self.std.half()

        # Compatibility attributes
        self.model.format = "depth_anything"
        self.model.stride = 14  # ViT patch size
        self.model.fp16 = self.args.half
        self.done_warmup = True

    def pre_transform(self, im):
        """Resize images to target size (imgsz) for inference.

        Args:
            im: List of numpy arrays (H, W, 3).

        Returns:
            List of resized numpy arrays.
        """
        target = self.imgsz[0] if isinstance(self.imgsz, (list, tuple)) else self.imgsz
        result = []
        for img in im:
            h, w = img.shape[:2]
            scale = target / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            # Round to nearest multiple of 14
            new_h = max(14, (new_h // 14) * 14)
            new_w = max(14, (new_w // 14) * 14)
            import cv2
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            result.append(resized)
        return result
