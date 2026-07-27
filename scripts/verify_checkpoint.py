"""Verify a CrossModalFusionNet checkpoint against the extracted model code.

Usage:
    python scripts/verify_checkpoint.py --weights best_cross_modal_3class.pth
    python scripts/verify_checkpoint.py --weights best_cross_modal_3class.pth --image assets/sample.jpg

Checks, in order:
  1. What the checkpoint file actually contains
  2. How many classes the saved head has
  3. Whether the keys match the model definition exactly
  4. Whether a forward pass runs and produces valid probabilities
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import CrossModalFusionNet  # noqa: E402
from preprocess import extract_streams  # noqa: E402

CLASS_NAMES_2 = ["Authentic", "Manipulated"]
CLASS_NAMES_3 = ["Authentic", "Manipulated", "AI-Generated"]


def load_state_dict(path: Path) -> dict:
    """Unwrap the common checkpoint container formats."""
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and not any(
        k.endswith(".weight") or k.endswith(".bias") for k in obj
    ):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj:
                print(f"  Checkpoint is a container; using obj['{key}']")
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise SystemExit(f"Unexpected checkpoint type: {type(obj)}")

    # Saved from nn.DataParallel -> every key prefixed with 'module.'
    if all(k.startswith("module.") for k in obj):
        print("  Stripping 'module.' prefix (saved from nn.DataParallel)")
        obj = {k[len("module."):]: v for k, v in obj.items()}

    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    print(f"\n[1] Reading {args.weights}")
    print(f"  Size: {args.weights.stat().st_size / 1e6:.1f} MB")
    state = load_state_dict(args.weights)
    print(f"  Tensors: {len(state)}")
    print(f"  Parameters: {sum(v.numel() for v in state.values()) / 1e6:.1f}M")

    print("\n[2] Inferring output head")
    head_key = "classifier.3.weight"
    if head_key not in state:
        candidates = [k for k in state if k.startswith("classifier")]
        raise SystemExit(
            f"  '{head_key}' not found. classifier keys present: {candidates}"
        )
    num_classes = state[head_key].shape[0]
    names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_2
    print(f"  {num_classes} classes -> {names}")
    if num_classes == 2:
        print("  NOTE: this is the binary CASIA2 checkpoint, not the 3-class model.")

    print("\n[3] Loading into CrossModalFusionNet")
    model = CrossModalFusionNet(num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"  MISSING ({len(missing)}) — model expects, checkpoint lacks:")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}) — checkpoint has, model lacks:")
        for k in unexpected[:10]:
            print(f"    {k}")
        if len(unexpected) > 10:
            print(f"    ... and {len(unexpected) - 10} more")
    if not missing and not unexpected:
        print("  Exact match. Every tensor loaded.")

    model.eval()

    print("\n[4] Forward pass")
    if args.image:
        from PIL import Image

        rgb, ela, noise, dct = extract_streams(Image.open(args.image))
        print(f"  Input: {args.image.name}")
    else:
        rgb = ela = noise = dct = torch.randn(1, 3, 224, 224)
        print("  Input: random tensors (pass --image for a real check)")

    with torch.no_grad():
        logits = model(rgb, ela, noise, dct)
        probs = torch.softmax(logits, dim=1)[0]

    print(f"  Logits shape: {tuple(logits.shape)}  (expected (1, {num_classes}))")
    print(f"  Probabilities sum to {probs.sum():.6f}")
    print()
    for name, p in sorted(zip(names, probs.tolist()), key=lambda kv: -kv[1]):
        bar = "#" * int(p * 40)
        print(f"  {name:<16} {p:.4f}  {bar}")

    ok = not missing and not unexpected and logits.shape == (1, num_classes)
    print(f"\n{'PASS' if ok else 'REVIEW THE WARNINGS ABOVE'}\n")


if __name__ == "__main__":
    main()