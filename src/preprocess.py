"""Stream extraction for CrossModalFusionNet.

The model takes four tensors, all derived from a single input image:
    rgb   — the image itself
    ela   — Error Level Analysis (JPEG recompression residual)
    noise — high-frequency noise residual
    dct   — log-magnitude DCT of the grayscale image

Extracted from RealTimeMultiModal_Fixed.ipynb, sections 1 and 14A.

Changed from the notebook: ELA is computed entirely in memory instead of
writing to a fixed path under /tmp. The notebook version is fine in a
single-threaded Colab session, but a web server handling two concurrent
requests would have them overwrite each other's temp file.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

__all__ = ["eval_transform", "extract_streams", "get_ela_modality",
           "get_noise_residual", "get_dct_image"]

# Must match the evaluation transform used in the notebook exactly.
# Any divergence here silently degrades accuracy at serving time.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_ela_modality(img: Image.Image, quality: int = 95) -> Image.Image:
    """Error Level Analysis as a 3-channel image, computed in memory."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    ela = np.abs(
        np.array(img, dtype=np.float64) - np.array(resaved, dtype=np.float64)
    )
    ela_gray = np.mean(ela, axis=2)
    spread = ela_gray.max() - ela_gray.min()
    ela_norm = ((ela_gray - ela_gray.min()) / (spread + 1e-10) * 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(ela_norm, cv2.COLOR_GRAY2RGB))


def get_noise_residual(img: Image.Image) -> Image.Image:
    """High-frequency noise residual via Gaussian blur subtraction."""
    arr = np.array(img).astype(np.float32)
    residual = np.clip(arr - cv2.GaussianBlur(arr, (5, 5), 0), 0, 255)
    return Image.fromarray(residual.astype(np.uint8))


def get_dct_image(img: Image.Image) -> Image.Image:
    """Log-magnitude DCT of the grayscale image, as a 3-channel image."""
    gray = np.float32(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)) / 255.0
    dct = cv2.normalize(
        np.log(np.abs(cv2.dct(gray)) + 1e-6), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(dct, cv2.COLOR_GRAY2RGB))


def extract_streams(
    img: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all four input streams. Each returned tensor is (1, 3, 224, 224)."""
    img = img.convert("RGB")
    return (
        eval_transform(img).unsqueeze(0),
        eval_transform(get_ela_modality(img)).unsqueeze(0),
        eval_transform(get_noise_residual(img)).unsqueeze(0),
        eval_transform(get_dct_image(img)).unsqueeze(0),
    )