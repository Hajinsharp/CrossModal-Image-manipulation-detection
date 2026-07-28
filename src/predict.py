"""Single-image inference for CrossModalFusionNet (3-class)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from model import CrossModalFusionNet
from preprocess import extract_streams

CLASS_NAMES = ["Authentic", "Manipulated", "AI-Generated"]
HF_REPO = "Roy407/crossmodalfusionnet"
WEIGHTS_FILE = "weights.pth"

# The Docker image bakes weights.pth in at build time (see Dockerfile) so a
# cold start loads a local file instead of hitting HF Hub over the network
# -- avoids both cold-start latency and rate-limit exposure on every
# scale-from-zero event. Falls back to the Hub download for local dev,
# where this file won't exist.
WEIGHTS_LOCAL_PATH = Path(__file__).resolve().parent / WEIGHTS_FILE

_model: torch.nn.Module | None = None


def load_model(device: str = "cpu") -> torch.nn.Module:
    """Load the model once per process, downloading weights on first call
    unless a baked-in local copy is already present."""
    global _model
    if _model is None:
        weights_path = (
            WEIGHTS_LOCAL_PATH
            if WEIGHTS_LOCAL_PATH.exists()
            else hf_hub_download(repo_id=HF_REPO, filename=WEIGHTS_FILE)
        )
        model = CrossModalFusionNet(num_classes=len(CLASS_NAMES), pretrained=False)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval().to(device)
        _model = model
    return _model


def predict(image: Image.Image, device: str = "cpu") -> dict[str, float]:
    """Return {class_name: probability} for one image."""
    model = load_model(device)
    rgb, ela, noise, dct = (t.to(device) for t in extract_streams(image))
    with torch.no_grad():
        probs = torch.softmax(model(rgb, ela, noise, dct), dim=1)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify one image.")
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    scores = predict(Image.open(args.image))
    top = max(scores, key=scores.get)
    print(f"Prediction: {top} ({scores[top]:.1%})\n")
    for name, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {p:.4f}")