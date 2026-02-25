#!/usr/bin/env python3
"""
08_iterative_correction.py
==========================
Iterative Error Correction Loop (dshea89-inspired)

After initial fine-tuning, this script helps identify and correct
persistent OCR errors through repeated targeted retraining cycles.

Workflow:
  1. Run evaluation → identify worst-performing images
  2. Present failures for human correction review
  3. Add corrected samples to training set
  4. Trigger incremental retraining
  5. Re-evaluate and compare

Usage:
    python scripts/08_iterative_correction.py --eval-dir eval_data/ --rounds 3
    python scripts/08_iterative_correction.py --find-errors --threshold 0.1
    python scripts/08_iterative_correction.py --add-corrections corrections/ --retrain
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm

from utils import (
    setup_logging, load_config, list_images,
    load_image_gray, read_ground_truth, write_ground_truth,
    pair_images_with_gt,
)

try:
    from jiwer import cer as compute_cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


def basic_cer(ground_truth: str, prediction: str) -> float:
    """Calculate basic Character Error Rate for fallback when jiwer unavailable."""
    if not ground_truth:
        return 1.0
    
    s1, s2 = ground_truth.lower(), prediction.lower()
    matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
    errors = abs(len(s1) - len(s2)) + (len(s1) - matches)
    return errors / len(s1) if s1 else 0.0


console = Console()


# ─── Error identification ─────────────────────────────────────────────────────

def find_high_error_samples(
    eval_dir: str,
    gt_dir: str,
    model_tessdata: str | None,
    lang: str,
    cfg: dict,
    cer_threshold: float = 0.1,
) -> list[dict]:
    """
    Run OCR on eval set and return samples with CER above threshold.
    These are candidates for targeted correction.
    """
    import pytesseract

    eval_path = Path(eval_dir)
    gt_path   = Path(gt_dir)
    pairs = pair_images_with_gt(eval_path, gt_path)

    if not pairs:
        logger.error("No pairs found")
        return []

    ocr_cfg = cfg.get("ocr", {})
    psm = ocr_cfg.get("psm", 6)
    oem = ocr_cfg.get("oem", 1)
    custom_config = f"--oem {oem} --psm {psm}"
    if model_tessdata:
        custom_config += f" --tessdata-dir {model_tessdata}"

    high_error = []
    logger.info(f"Scanning {len(pairs)} samples for high-error cases (threshold={cer_threshold})...")

    for img_path, gt_path_item in tqdm(pairs, desc="Scanning"):
        gt_text = read_ground_truth(gt_path_item)

        try:
            img = cv2.imread(str(img_path))
            pred_text = pytesseract.image_to_string(img, lang=lang, config=custom_config).strip()
        except Exception as e:
            logger.warning(f"OCR failed: {img_path.name}: {e}")
            continue

        if HAS_JIWER:
            cer = compute_cer(gt_text, pred_text) if gt_text else 1.0
        else:
            cer = basic_cer(gt_text, pred_text)

        if cer > cer_threshold:
            high_error.append({
                "image_path": str(img_path),
                "gt_path": str(gt_path_item),
                "gt_text": gt_text,
                "predicted": pred_text,
                "cer": round(cer, 4),
            })

    high_error.sort(key=lambda x: x["cer"], reverse=True)
    logger.info(f"Found {len(high_error)} samples with CER > {cer_threshold}")
    return high_error


# ─── Correction collection ────────────────────────────────────────────────────

def collect_corrections(
    high_error_samples: list[dict],
    corrections_dir: str,
    gt_dir: str,
) -> int:
    """
    Interactive CLI session to review and correct high-error samples.
    Saves corrected .gt.txt files for retraining.
    """
    corrections_path = Path(corrections_dir)
    gt_path = Path(gt_dir)
    corrections_path.mkdir(parents=True, exist_ok=True)

    corrected = 0

    console.print(Panel.fit(
        f"[bold]Iterative Correction Session[/bold]\n\n"
        f"Reviewing {len(high_error_samples)} high-error samples.\n"
        f"For each, confirm the GT text or type a correction.\n"
        f"Commands: [yellow]OK[/yellow] (accept GT), "
        f"[yellow]SKIP[/yellow], [yellow]QUIT[/yellow]",
    ))

    for i, sample in enumerate(high_error_samples):
        img_path = Path(sample["image_path"])
        console.print(
            f"\n[bold cyan][{i+1}/{len(high_error_samples)}][/bold cyan] "
            f"{img_path.name} — CER: [red]{sample['cer']:.3f}[/red]"
        )
        console.print(f"  GT text:   [green]{sample['gt_text']}[/green]")
        console.print(f"  Predicted: [red]{sample['predicted']}[/red]")

        while True:
            try:
                user_input = input("  Correct text (or OK/SKIP/QUIT): ").strip()
            except (EOFError, KeyboardInterrupt):
                logger.info(f"Session ended. Corrected {corrected} samples.")
                return corrected

            if user_input.upper() == "QUIT":
                return corrected

            if user_input.upper() == "SKIP":
                break

            if user_input.upper() == "OK":
                # Accept existing GT — copy image + GT to corrections dir
                dest_img = corrections_path / img_path.name
                dest_gt  = corrections_path / (img_path.stem + ".gt.txt")
                shutil.copy2(img_path, dest_img)
                write_ground_truth(sample["gt_text"], dest_gt)
                corrected += 1
                console.print("  [green]✓ Added to correction set[/green]")
                break

            if user_input:
                # User provided corrected text
                dest_img = corrections_path / img_path.name
                dest_gt  = corrections_path / (img_path.stem + ".gt.txt")
                shutil.copy2(img_path, dest_img)
                write_ground_truth(user_input, dest_gt)

                # Also update the original GT
                original_gt = gt_path / (img_path.stem + ".gt.txt")
                write_ground_truth(user_input, original_gt)

                corrected += 1
                console.print("  [green]✓ Saved correction[/green]")
                break
            else:
                console.print("  [dim]Empty input ignored[/dim]")

    console.print(f"\n[green bold]Correction session complete: {corrected} samples corrected[/green bold]")
    return corrected


# ─── Incremental retraining ───────────────────────────────────────────────────

def trigger_incremental_retrain(
    corrections_dir: str,
    cfg: dict,
    dry_run: bool = False,
) -> bool:
    """
    Add corrections to training set and trigger retraining.
    """
    corrections_path = Path(corrections_dir)
    train_dir = Path(cfg.get("paths", {}).get("training_data", "training_data/"))

    # Copy corrections to training set
    images = list_images(corrections_path)
    if not images:
        logger.warning("No images in corrections dir")
        return False

    logger.info(f"Adding {len(images)} corrected samples to training set...")

    for img_path in images:
        gt_path = corrections_path / (img_path.stem + ".gt.txt")
        if not gt_path.exists():
            logger.warning(f"No GT for correction: {img_path.name}")
            continue

        dest_stem = f"correction_{datetime.now().strftime('%Y%m%d')}_{img_path.stem}"
        shutil.copy2(img_path, train_dir / f"{dest_stem}.tif")
        shutil.copy2(gt_path,  train_dir / f"{dest_stem}.gt.txt")

    logger.success(f"Added {len(images)} corrections to {train_dir}")

    if dry_run:
        logger.info("[DRY RUN] Would now run: bash scripts/05_run_training.sh --resume")
        return True

    # Trigger retraining
    logger.info("Triggering incremental retraining...")
    try:
        result = subprocess.run(
            ["bash", "scripts/05_run_training.sh", "--resume"],
            capture_output=False,
            check=True,
        )
        logger.success("Retraining complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e}")
        return False


# ─── Iteration tracking ───────────────────────────────────────────────────────

def save_iteration_log(
    round_num: int,
    stats: dict,
    log_path: str = "logs/iteration_log.csv",
) -> None:
    """Append iteration stats to a running log."""
    p = Path(log_path)
    p.parent.mkdir(exist_ok=True)

    row = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        **stats,
    }

    write_header = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Iteration log updated: {log_path}")


def print_iteration_progress(log_path: str = "logs/iteration_log.csv") -> None:
    """Display progress across all correction rounds."""
    p = Path(log_path)
    if not p.exists():
        logger.info("No iteration log found yet")
        return

    with open(p) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    table = Table(title="Correction Round Progress", show_lines=True)
    table.add_column("Round",  style="cyan")
    table.add_column("Mean CER", style="white")
    table.add_column("Corrected Samples", style="white")
    table.add_column("Timestamp", style="dim")

    for row in rows:
        cer_val = float(row.get("mean_cer", 1.0))
        color = "green" if cer_val < 0.02 else "yellow" if cer_val < 0.05 else "red"
        table.add_row(
            row.get("round", "?"),
            f"[{color}]{cer_val:.4f}[/{color}]",
            row.get("corrected_samples", "?"),
            row.get("timestamp", "?")[:19],
        )

    console.print(table)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative OCR Error Correction")
    parser.add_argument("--eval-dir",     default="raw_images/new/",     help="Evaluation images dir")
    parser.add_argument("--gt-dir",       default="ground_truth/",  help="Ground truth dir")
    parser.add_argument("--corrections",  default="corrections/",   help="Corrections output dir")
    parser.add_argument("--model-dir",    help="Custom model tessdata dir")
    parser.add_argument("--lang",         default="eng")
    parser.add_argument("--threshold",    type=float, default=0.10,  help="CER threshold for flagging")
    parser.add_argument("--rounds",       type=int,   default=3,     help="Number of correction rounds")
    parser.add_argument("--find-errors",  action="store_true",       help="Find and report errors only")
    parser.add_argument("--retrain",      action="store_true",       help="Trigger retraining after corrections")
    parser.add_argument("--dry-run",      action="store_true",       help="Simulate retraining (no actual training)")
    parser.add_argument("--progress",     action="store_true",       help="Show round progress")
    parser.add_argument("--config",       default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/correction.log")
    cfg = load_config(args.config)

    if args.progress:
        print_iteration_progress()
        return

    if args.find_errors:
        samples = find_high_error_samples(
            args.eval_dir, args.gt_dir, args.model_dir,
            args.lang, cfg, args.threshold,
        )
        console.print(f"\n[bold]{len(samples)} high-error samples found:[/bold]")
        for s in samples[:20]:
            console.print(f"  {Path(s['image_path']).name}: CER={s['cer']:.3f}")
        return

    # Full correction rounds
    for round_num in range(1, args.rounds + 1):
        console.print(Panel.fit(
            f"[bold]Correction Round {round_num}/{args.rounds}[/bold]",
            style="blue",
        ))

        # Find errors
        high_error = find_high_error_samples(
            args.eval_dir, args.gt_dir, args.model_dir,
            args.lang, cfg, args.threshold,
        )

        if not high_error:
            console.print("[green]No high-error samples found — model is performing well![/green]")
            break

        # Collect corrections
        round_corrections_dir = f"{args.corrections}/round_{round_num:02d}"
        corrected = collect_corrections(high_error, round_corrections_dir, args.gt_dir)

        # Trigger retraining
        if args.retrain and corrected > 0:
            trigger_incremental_retrain(round_corrections_dir, cfg, dry_run=args.dry_run)

        # Log this round
        save_iteration_log(round_num, {
            "high_error_found": len(high_error),
            "corrected_samples": corrected,
            "mean_cer": high_error[0]["cer"] if high_error else 0,
        })

        console.print(f"Round {round_num} complete: {corrected} samples corrected\n")

    print_iteration_progress()


if __name__ == "__main__":
    main()
