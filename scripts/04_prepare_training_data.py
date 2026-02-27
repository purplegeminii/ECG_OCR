#!/usr/bin/env python3
"""
04_prepare_training_data.py
===========================
Prepare tesstrain-ready training data from preprocessed images + GT files.

Files are written directly into tesstrain's expected directory structure:

    tesstrain/data/<MODEL_NAME>-ground-truth/   ← all .tif + .gt.txt pairs

tesstrain's Makefile then handles LSTMF generation internally when
05_run_training.sh calls `make training`. There is no separate
training_data/ or eval_data/ directory — tesstrain owns that entirely.

The train/eval split is recorded in results/test_set.txt so 06_evaluate.py
knows which images were held out, but tesstrain itself manages the split
via its RATIO variable (default 0.9 train / 0.1 eval).

What this script does:
  1. Pairs .tif images with their .gt.txt ground truth files
  2. Validates each pair (image quality, GT content)
  3. Copies files into tesstrain/data/<MODEL_NAME>-ground-truth/
  4. Records the held-out test stems in results/test_set.txt
  5. Reports dataset statistics

Usage:
    python scripts/04_prepare_training_data.py
    python scripts/04_prepare_training_data.py --source augmented/ --gt ground_truth/
    python scripts/04_prepare_training_data.py --stats-only
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table

from utils import (
    setup_logging, load_config, list_images,
    load_image_gray, read_ground_truth, write_ground_truth,
    pair_images_with_gt,
)

console = Console()


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_pair(img_path: Path, gt_path: Path) -> tuple[bool, str]:
    """
    Validate an image+GT pair. Returns (is_valid, reason).
    """
    # Check GT content
    try:
        text = read_ground_truth(gt_path)
    except Exception as e:
        return False, f"Cannot read GT: {e}"

    if not text.strip():
        return False, "Empty GT"
    if len(text.strip()) < 3:
        return False, f"GT too short ({len(text)} chars)"

    # Check image
    try:
        img = load_image_gray(img_path)
    except Exception as e:
        return False, f"Cannot read image: {e}"

    h, w = img.shape
    if w < 100 or h < 20:
        return False, f"Image too small ({w}×{h})"

    # Check image is not blank
    mean_px = img.mean()
    if mean_px > 250:
        return False, f"Image appears blank (mean pixel={mean_px:.0f})"
    if mean_px < 5:
        return False, f"Image appears all-black (mean pixel={mean_px:.0f})"

    return True, "OK"


# ─── Split logic ──────────────────────────────────────────────────────────────

def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split pairs into (train, validation, test) sets.
    Shuffle deterministically using seed.
    """
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = pairs[:n_train]
    val   = pairs[n_train:n_train + n_val]
    test  = pairs[n_train + n_val:]

    return train, val, test


# ─── File preparation ─────────────────────────────────────────────────────────

def copy_pairs_to_tesstrain(
    pairs: list[tuple[Path, Path]],
    ground_truth_dir: Path,
) -> list[str]:
    """
    Copy image+GT pairs into tesstrain's ground-truth directory.

    tesstrain expects all training files — both train and eval — in one flat
    directory: tesstrain/data/<MODEL_NAME>-ground-truth/
    It handles the train/eval split itself via its RATIO Makefile variable.

    Files are copied as-is (no renaming needed; tesstrain accepts any stem).
    Returns list of stems copied.
    """
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    stems = []

    for img_path, gt_path in tqdm(pairs, desc="Copying to tesstrain", unit="pair"):
        dest_img = ground_truth_dir / img_path.name
        dest_gt  = ground_truth_dir / gt_path.name

        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        if not dest_gt.exists():
            shutil.copy2(gt_path, dest_gt)

        stems.append(img_path.stem)

    return stems


# ─── Statistics ───────────────────────────────────────────────────────────────

def print_dataset_stats(
    pairs: list[tuple[Path, Path]],
    train: list, val: list, test: list,
) -> None:
    """Print dataset statistics table."""
    table = Table(title="Dataset Statistics", show_lines=True)
    table.add_column("Split",  style="cyan bold")
    table.add_column("Count",  style="white")
    table.add_column("% Total", style="dim")

    total = len(pairs)
    for name, subset in [("Train", train), ("Validation", val), ("Test", test)]:
        pct = 100 * len(subset) / total if total else 0
        table.add_row(name, str(len(subset)), f"{pct:.1f}%")
    table.add_row("[bold]TOTAL[/bold]", str(total), "100%")

    console.print(table)

    # GT text length distribution
    lengths = []
    for _, gt_path in pairs:
        try:
            text = read_ground_truth(gt_path)
            lengths.append(len(text))
        except Exception:
            pass

    if lengths:
        console.print(
            f"\n[dim]GT text length — "
            f"min: {min(lengths)}, max: {max(lengths)}, "
            f"mean: {sum(lengths)/len(lengths):.1f} chars[/dim]"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def prepare_training_data(
    source_dir: str,
    gt_dir: str,
    tesstrain_dir: str,
    cfg: dict,
    stats_only: bool = False,
) -> dict:
    """
    Full pipeline: validate → copy into tesstrain/data/<MODEL>-ground-truth/

    tesstrain owns the train/eval split. We only record held-out test stems
    separately in results/test_set.txt for use by 06_evaluate.py.
    """
    source_path    = Path(source_dir)
    gt_path        = Path(gt_dir)
    tesstrain_path = Path(tesstrain_dir)

    model_name  = cfg.get("model", {}).get("name", "ecg_meter")
    seed        = cfg.get("training", {}).get("random_seed", 42)
    test_ratio  = 1.0 - cfg.get("training", {}).get("split", {}).get("train", 0.80) \
                      - cfg.get("training", {}).get("split", {}).get("validation", 0.10)
    min_samples = cfg.get("training", {}).get("min_samples_for_training", 100)

    # tesstrain ground-truth directory convention
    gt_out_dir = tesstrain_path / "data" / f"{model_name}-ground-truth"

    # ── Find pairs ────────────────────────────────────────────────────────────
    logger.info(f"Finding image-GT pairs in {source_path} + {gt_path}")
    all_pairs = pair_images_with_gt(source_path, gt_path)
    logger.info(f"Found {len(all_pairs)} pairs")

    if not all_pairs:
        logger.error(
            "No image-GT pairs found. "
            "Ensure preprocessed images and .gt.txt files have matching stems."
        )
        return {}

    # ── Validate pairs ────────────────────────────────────────────────────────
    logger.info("Validating pairs...")
    valid_pairs, invalid = [], []
    for img_p, gt_p in tqdm(all_pairs, desc="Validating", unit="pair"):
        ok, reason = validate_pair(img_p, gt_p)
        if ok:
            valid_pairs.append((img_p, gt_p))
        else:
            invalid.append((img_p.name, reason))

    if invalid:
        logger.warning(f"{len(invalid)} invalid pairs:")
        for name, reason in invalid[:10]:
            logger.warning(f"  {name}: {reason}")
        if len(invalid) > 10:
            logger.warning(f"  ...and {len(invalid)-10} more")

    logger.success(f"{len(valid_pairs)} valid pairs ready for training")

    if len(valid_pairs) < min_samples:
        logger.warning(
            f"Only {len(valid_pairs)} valid samples. "
            f"Minimum recommended: {min_samples}. "
            f"Consider collecting more images or increasing augmentation factor."
        )

    # ── Carve out a test set (for 06_evaluate.py) before handing to tesstrain ─
    # tesstrain handles its own train/eval split from whatever we give it.
    # We just hold back ~10% as a completely unseen test set.
    random.seed(seed)
    shuffled = list(valid_pairs)
    random.shuffle(shuffled)
    n_test   = max(1, int(len(shuffled) * test_ratio))
    test_pairs  = shuffled[:n_test]
    train_pairs = shuffled[n_test:]

    # Fake val count for stats display (tesstrain handles real split internally)
    val_display = int(len(train_pairs) * 0.10)
    train_display = len(train_pairs) - val_display

    if stats_only:
        print_dataset_stats(valid_pairs, train_pairs, test_pairs, test_pairs)
        console.print(
            f"\n[dim]Note: tesstrain manages its own train/eval split from "
            f"the {len(train_pairs)} non-test pairs. "
            f"The {n_test} test pairs are held back for 06_evaluate.py.[/dim]"
        )
        return {
            "total": len(valid_pairs),
            "to_tesstrain": len(train_pairs),
            "held_out_test": n_test,
        }

    print_dataset_stats(valid_pairs, train_pairs, test_pairs, test_pairs)
    console.print(
        f"\n[cyan]tesstrain ground-truth dir:[/cyan] {gt_out_dir}\n"
        f"[dim]tesstrain will internally split these {len(train_pairs)} pairs "
        f"into ~90% train / ~10% eval via its RATIO variable.[/dim]\n"
        f"[dim]{n_test} pairs held back as unseen test set → results/test_set.txt[/dim]"
    )

    # ── Copy training pairs into tesstrain/data/<model>-ground-truth/ ─────────
    if not tesstrain_path.exists():
        logger.error(
            f"tesstrain submodule not found at {tesstrain_path}. "
            f"Run: git submodule update --init"
        )
        return {}

    logger.info(f"Copying {len(train_pairs)} pairs to {gt_out_dir} ...")
    copy_pairs_to_tesstrain(train_pairs, gt_out_dir)

    # ── Copy test pairs to eval_data/ for unbiased evaluation ─────────────────
    eval_data_dir = Path(cfg.get("paths", {}).get("eval_data", "eval_data"))
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying {len(test_pairs)} test pairs to {eval_data_dir} ...")
    copy_pairs_to_tesstrain(test_pairs, eval_data_dir)
    
    # Also save test set list for reference
    test_list_path = Path("results/test_set.txt")
    test_list_path.parent.mkdir(exist_ok=True)
    test_list_path.write_text(
        "\n".join(str(img_p) for img_p, _ in test_pairs) + "\n"
    )
    logger.info(f"Test set: {len(test_pairs)} images in {eval_data_dir}")

    logger.success("Training data preparation complete!")
    return {
        "total": len(valid_pairs),
        "invalid": len(invalid),
        "to_tesstrain": len(train_pairs),
        "held_out_test": n_test,
        "ground_truth_dir": str(gt_out_dir),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tesstrain training data")
    parser.add_argument("--source",        default="augmented/",    help="Source images dir")
    parser.add_argument("--gt",            default="ground_truth/", help="Ground truth .gt.txt dir")
    parser.add_argument("--tesstrain-dir", default="tesstrain/",    help="Path to tesstrain submodule")
    parser.add_argument("--config",        default="config/config.yaml")
    parser.add_argument("--stats-only",    action="store_true",     help="Show stats without copying files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/prepare.log")
    cfg = load_config(args.config)

    stats = prepare_training_data(
        source_dir=args.source,
        gt_dir=args.gt,
        tesstrain_dir=args.tesstrain_dir,
        cfg=cfg,
        stats_only=args.stats_only,
    )
    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
