"""Evaluate a CrossModalFusionNet checkpoint on a held-out test split.

Usage (2-class CASIA2 checkpoint):
    python scripts/evaluate.py --weights best_cross_modal.pth \
        --casia-auth /path/to/CASIA2/Au --casia-manip /path/to/CASIA2/Tp

Usage (3-class checkpoint, adds AI-Generated):
    python scripts/evaluate.py --weights best_cross_modal_3class.pth \
        --casia-auth /path/to/CASIA2/Au --casia-manip /path/to/CASIA2/Tp \
        --ai-generated /path/to/cifake/train/FAKE

The number of classes is read from the checkpoint itself (classifier.3.weight
shape), same as scripts/verify_checkpoint.py.

Split: deterministic, class-stratified train/val/test partition of the
combined image pool — each class is split independently in the same
proportions (--train-frac/--val-frac, default 0.70/0.15, so test is the
remaining 0.15) and seeded (--seed, default 42), then concatenated. Only
the test partition is ever loaded here — train and val indices are
computed so they can be excluded, never touched otherwise. This mirrors
the protocol used by Sardhara et al. 2026 ("DeepForgeryNet", Frontiers in
AI) on the same dataset — 70/15/15 stratified, seed 42 — so results are
comparable split-for-split, not just dataset-for-dataset.

Caveat for 3-class checkpoints: the fine-tuning stage in
RealTimeMultiModal_Fixed.ipynb (Section 14) only ever carved an 80/20
train/val split of the combined CASIA2+CIFAKE pool — it never held out a
test set, so "best" was picked by validation accuracy alone. The split
computed here is freshly derived from the raw source directories and is
NOT guaranteed disjoint from whatever the fine-tuning run trained or
selected on. Report this alongside the numbers; don't present them as a
clean held-out result without the caveat.

Also note: directory listings are sorted here for a reproducible split.
The notebook's datasets use raw, unsorted os.listdir(), and an unstratified
random_split — so this script's split will NOT match the exact
test5_dataset partition from notebook Cell 34 index-for-index, even with
the same seed and fractions. That's a deliberate change (stratified >
unstratified for a class-imbalanced set), not an oversight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import CrossModalFusionNet  # noqa: E402
from preprocess import extract_streams  # noqa: E402

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_NAMES_2 = ["Authentic", "Manipulated"]
CLASS_NAMES_3 = ["Authentic", "Manipulated", "AI-Generated"]


class ForensicsTestDataset(Dataset):
    """5-stream (rgb, ela, noise, dct, label) dataset for evaluation.

    Reuses preprocess.extract_streams so evaluation is computed exactly the
    same way as production inference (predict.py), not a re-derived copy.
    """

    def __init__(self, samples: list[tuple[str, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        rgb, ela, noise, dct = extract_streams(img)
        return rgb.squeeze(0), ela.squeeze(0), noise.squeeze(0), dct.squeeze(0), label


def list_images(d: Path) -> list[str]:
    return sorted(
        str(d / f) for f in os.listdir(d) if f.lower().endswith(VALID_EXT)
    )


def build_samples(
    auth_dir: Path, manip_dir: Path, ai_dir: Path | None
) -> list[tuple[str, int]]:
    samples = [(p, 0) for p in list_images(auth_dir)]
    samples += [(p, 1) for p in list_images(manip_dir)]
    if ai_dir is not None:
        samples += [(p, 2) for p in list_images(ai_dir)]
    return samples


def cap_per_class(
    samples: list[tuple[str, int]], max_per_class: int | None, seed: int
) -> list[tuple[str, int]]:
    """Randomly subsample each class down to at most max_per_class images
    (seeded, so reproducible). Mirrors the thesis's own 3-class balancing
    strategy (Table 16: cap each class at the smallest class's size) —
    useful when one class (e.g. the full 60k-image CIFAKE pool) dwarfs the
    others and would otherwise blow up eval runtime for no rigor benefit."""
    if max_per_class is None:
        return samples

    by_class: dict[int, list[tuple[str, int]]] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    g = torch.Generator().manual_seed(seed)
    capped: list[tuple[str, int]] = []
    for label in sorted(by_class):
        cls_samples = by_class[label]
        if len(cls_samples) > max_per_class:
            perm = torch.randperm(len(cls_samples), generator=g).tolist()[:max_per_class]
            capped.extend(cls_samples[i] for i in perm)
        else:
            capped.extend(cls_samples)
    return capped


def test_split(
    samples: list[tuple[str, int]], train_frac: float, val_frac: float, seed: int
) -> list[tuple[str, int]]:
    """Class-stratified train/val/test split: each class's indices are
    permuted and sliced independently (same seeded generator, consumed in
    class order), then the test slices are concatenated. Keeps the test
    set's class balance matching the full pool regardless of imbalance —
    same protocol as the DeepForgeryNet baseline (Sardhara et al. 2026)."""
    by_class: dict[int, list[tuple[str, int]]] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    g = torch.Generator().manual_seed(seed)
    test_samples: list[tuple[str, int]] = []
    for label in sorted(by_class):
        cls_samples = by_class[label]
        n = len(cls_samples)
        perm = torch.randperm(n, generator=g).tolist()
        train_sz = int(train_frac * n)
        val_sz = int(val_frac * n)
        test_samples.extend(cls_samples[i] for i in perm[train_sz + val_sz:])
    return test_samples


def load_state_dict(path: Path) -> dict:
    """Unwrap the common checkpoint container formats (mirrors verify_checkpoint.py)."""
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and not any(
        k.endswith(".weight") or k.endswith(".bias") for k in obj
    ):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj:
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise SystemExit(f"Unexpected checkpoint type: {type(obj)}")

    if all(k.startswith("module.") for k in obj):
        obj = {k[len("module."):]: v for k, v in obj.items()}

    return obj


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary ROC curve (fpr, tpr, thresholds), sklearn-compatible convention:
    thresholds are sorted descending, and fpr/tpr start at (0, 0)."""
    order = np.argsort(-scores, kind="stable")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    P = labels.sum()
    N = len(labels) - P

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    # Keep only the last point for each run of equal scores (distinct thresholds).
    distinct = np.where(np.diff(scores_sorted))[0]
    distinct = np.r_[distinct, len(scores_sorted) - 1]

    tpr = tps[distinct] / P if P > 0 else np.zeros(len(distinct))
    fpr = fps[distinct] / N if N > 0 else np.zeros(len(distinct))
    thresholds = scores_sorted[distinct]

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float, float]:
    """Youden's J statistic: argmax(TPR - FPR). Returns (threshold, tpr, fpr) at that point."""
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def compute_metrics(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict:
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    per_class = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append(
            {"class": class_names[i], "precision": prec, "recall": rec, "f1": f1, "support": support}
        )

    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0
    macro = {
        "precision": float(np.mean([m["precision"] for m in per_class])),
        "recall": float(np.mean([m["recall"] for m in per_class])),
        "f1": float(np.mean([m["f1"] for m in per_class])),
    }
    weighted = {
        "precision": sum(m["precision"] * m["support"] for m in per_class) / total if total else 0.0,
        "recall": sum(m["recall"] * m["support"] for m in per_class) / total if total else 0.0,
        "f1": sum(m["f1"] * m["support"] for m in per_class) / total if total else 0.0,
    }
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "confusion_matrix": cm.tolist(),
    }


def print_report(metrics: dict, class_names: list[str]) -> None:
    print(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
    header = f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print(header)
    print("-" * len(header))
    for m in metrics["per_class"]:
        print(f"{m['class']:<16}{m['precision']:>10.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['support']:>10d}")
    print("-" * len(header))
    ma = metrics["macro_avg"]
    wa = metrics["weighted_avg"]
    print(f"{'macro avg':<16}{ma['precision']:>10.4f}{ma['recall']:>10.4f}{ma['f1']:>10.4f}")
    print(f"{'weighted avg':<16}{wa['precision']:>10.4f}{wa['recall']:>10.4f}{wa['f1']:>10.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm = metrics["confusion_matrix"]
    colw = max(len(n) for n in class_names) + 2
    print(" " * colw + "".join(f"{n:>{colw}}" for n in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:<{colw}}" + "".join(f"{v:>{colw}}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--casia-auth", required=True, type=Path, help="CASIA2 authentic image dir")
    parser.add_argument("--casia-manip", required=True, type=Path, help="CASIA2 tampered image dir")
    parser.add_argument("--ai-generated", type=Path, default=None, help="AI-generated image dir (3-class checkpoints only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Randomly subsample each class down to at most N images before splitting "
             "(seeded). Use this to cap a much larger class (e.g. full CIFAKE) so eval "
             "runtime stays proportional — mirrors the thesis's own balancing strategy.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the JSON report")
    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("--train-frac + --val-frac must be < 1.0 so a non-empty test split remains")

    state = load_state_dict(args.weights)
    head_key = "classifier.3.weight"
    if head_key not in state:
        raise SystemExit(f"'{head_key}' not found in checkpoint; is this a CrossModalFusionNet weights file?")
    num_classes = state[head_key].shape[0]
    class_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_2

    if num_classes == 3 and args.ai_generated is None:
        raise SystemExit("Checkpoint has 3 classes — pass --ai-generated <dir> for the AI-Generated class")
    if num_classes == 2 and args.ai_generated is not None:
        raise SystemExit("Checkpoint has 2 classes — remove --ai-generated, it won't be used")

    print(f"Checkpoint:  {args.weights}")
    print(f"Classes:     {num_classes} -> {class_names}")
    print(f"CASIA2 Au:   {args.casia_auth}")
    print(f"CASIA2 Tp:   {args.casia_manip}")
    if args.ai_generated:
        print(f"AI-Gen dir:  {args.ai_generated}")
    print(f"Split:       train={args.train_frac:.2f} val={args.val_frac:.2f} "
          f"test={1 - args.train_frac - args.val_frac:.2f}  seed={args.seed}")

    if num_classes == 3:
        print(
            "\nWARNING: this checkpoint was fine-tuned (notebook Section 14) using only an "
            "80/20 train/val split with no held-out test set — model selection used the same "
            "data being scored here may partially overlap with. Treat the numbers below as "
            "indicative, not a clean generalization estimate, until retrained with a true "
            "3-way split.\n"
        )

    samples = build_samples(args.casia_auth, args.casia_manip, args.ai_generated)
    pool_size_before_cap = len(samples)
    samples = cap_per_class(samples, args.max_per_class, args.seed)
    if args.max_per_class is not None:
        print(f"Capped each class at {args.max_per_class} images "
              f"(pool: {pool_size_before_cap} -> {len(samples)})")
    test_samples = test_split(samples, args.train_frac, args.val_frac, args.seed)

    counts = np.bincount([lbl for _, lbl in test_samples], minlength=num_classes)
    print(f"\nTest set: {len(test_samples)} images")
    for name, c in zip(class_names, counts):
        print(f"  {name}: {int(c)}")

    model = CrossModalFusionNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval().to(args.device)

    loader = DataLoader(ForensicsTestDataset(test_samples), batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for rgb, ela, noise, dct, labels in tqdm(loader, desc="Evaluating"):
            logits = model(
                rgb.to(args.device), ela.to(args.device), noise.to(args.device), dct.to(args.device)
            )
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("\n=== Argmax threshold (standard, no tuning) ===")
    metrics = compute_metrics(all_labels, all_preds, class_names)
    print_report(metrics, class_names)

    roc_block = None
    if num_classes == 2:
        # Positive class = "Manipulated" (index 1), matching the thesis's convention.
        pos_scores = all_probs[:, 1]
        pos_labels = (all_labels == 1).astype(int)
        fpr, tpr, thresholds = roc_curve(pos_labels, pos_scores)
        auc = roc_auc(fpr, tpr)
        tau_star, tpr_at_tau, fpr_at_tau = optimal_threshold(fpr, tpr, thresholds)

        preds_at_tau = (pos_scores >= tau_star).astype(int)
        metrics_at_tau = compute_metrics(all_labels, preds_at_tau, class_names)

        print(f"\n=== ROC analysis ===")
        print(f"AUC: {auc:.4f}")
        print(f"Optimal threshold (Youden's J): tau*={tau_star:.4f}  TPR={tpr_at_tau:.4f}  FPR={fpr_at_tau:.4f}")

        print(f"\n=== ROC-optimal threshold (tau*={tau_star:.4f}) — isolates threshold-policy effect on THIS split ===")
        print_report(metrics_at_tau, class_names)

        roc_block = {
            "auc": auc,
            "optimal_threshold": tau_star,
            "tpr_at_optimal": tpr_at_tau,
            "fpr_at_optimal": fpr_at_tau,
            "metrics_at_optimal_threshold": metrics_at_tau,
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights),
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset_paths": {
            "casia_auth": str(args.casia_auth),
            "casia_manip": str(args.casia_manip),
            "ai_generated": str(args.ai_generated) if args.ai_generated else None,
        },
        "split": {
            "stratified": True,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": round(1 - args.train_frac - args.val_frac, 4),
            "max_per_class": args.max_per_class,
            "pool_size_before_cap": pool_size_before_cap,
            "total_pool_size": len(samples),
            "test_size": len(test_samples),
            "test_class_counts": {name: int(c) for name, c in zip(class_names, counts)},
        },
        "metrics": metrics,
        "roc": roc_block,
    }

    output = args.output or Path(__file__).parent / "eval_results" / f"{args.weights.stem}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved full report to {output}")


if __name__ == "__main__":
    main()
