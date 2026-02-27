#!/usr/bin/env python3
"""
add_to_eval_data.py
===================
Add new images to eval_data/ for testing.

This script:
  1. Takes raw meter images (any format)
  2. Preprocesses them (same pipeline as 01_preprocess.py)
  3. Converts to .tif format
  4. Prompts for ground truth text (or reads from file)
  5. Saves to eval_data/ with matching .gt.txt files

This is useful when you want to test your model on completely new images
that weren't part of your original dataset.

Usage:
    # Process single image interactively
    python scripts/add_to_eval_data.py --image meter_photo.jpg

    # Process directory of images
    python scripts/add_to_eval_data.py --input new_meters/ --interactive

    # Use existing ground truth files
    python scripts/add_to_eval_data.py --input new_meters/ --gt-dir labels/

    # Just preprocess without annotation
    python scripts/add_to_eval_data.py --input new_meters/ --no-annotation
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from loguru import logger
from rich.console import Console
from rich.prompt import Prompt

from utils import (
    setup_logging, load_config, list_images,
    load_image_gray, read_ground_truth, write_ground_truth,
)


def _fallback_preprocess(img, cfg):
    """Basic fallback preprocessing if 01_preprocess.py import fails."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    target_width = cfg.get("preprocessing", {}).get("target_width", 1000)
    if img.shape[1] < target_width:
        scale = target_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Adaptive thresholding
    block_size = cfg.get("preprocessing", {}).get("adaptive_thresh_block_size", 11)
    c = cfg.get("preprocessing", {}).get("adaptive_thresh_c", 2)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )
    
    return img


# Import preprocessing function from 01_preprocess.py
# (can't use regular import because filename starts with a number)
preprocess_image: Callable[[Any, dict], Any]
try:
    spec = importlib.util.spec_from_file_location(
        "scripts.01_preprocess",
        Path(__file__).parent / "01_preprocess.py"
    )
    if spec and spec.loader:
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)
        preprocess_image = preprocess_module.preprocess_image
        logger.info("Loaded preprocess_image from 01_preprocess.py")
    else:
        preprocess_image = _fallback_preprocess
        logger.warning("Using fallback preprocessing (spec/loader not available)")
except Exception as e:
    preprocess_image = _fallback_preprocess
    logger.warning(f"Using fallback preprocessing (import failed: {e})")

console = Console()


def get_ground_truth_interactive(img_path: Path) -> str | None:
    """Prompt user to enter ground truth text for an image."""
    console.print(f"\n[cyan]Image:[/cyan] {img_path.name}")
    console.print("[dim]Enter the text visible on the meter (or 'skip' to skip this image)[/dim]")
    
    text = Prompt.ask("Ground truth text")
    
    if text.lower() in ["skip", "s", ""]:
        return None
    
    return text.strip()


def process_and_add_image(
    img_path: Path,
    eval_data_dir: Path,
    gt_dir: Path | None,
    cfg: dict,
    interactive: bool = False,
    no_annotation: bool = False,
) -> bool:
    """
    Process a single image and add it to eval_data/.
    
    Returns True if successful, False if skipped.
    """
    # Generate output filename with timestamp to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{timestamp}_{img_path.stem}"
    
    out_img_path = eval_data_dir / f"{stem}.tif"
    out_gt_path  = eval_data_dir / f"{stem}.gt.txt"
    
    # Check if already exists
    if out_img_path.exists():
        logger.warning(f"File already exists: {out_img_path.name}, adding suffix")
        import random
        stem = f"{stem}_{random.randint(100,999)}"
        out_img_path = eval_data_dir / f"{stem}.tif"
        out_gt_path  = eval_data_dir / f"{stem}.gt.txt"
    
    # Load and preprocess image
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Cannot read image: {img_path}")
            return False
        
        processed = preprocess_image(img, cfg)
        
        # Save as TIFF
        cv2.imwrite(str(out_img_path), processed)
        logger.info(f"Saved preprocessed image: {out_img_path.name}")
        
    except Exception as e:
        logger.error(f"Failed to process {img_path.name}: {e}")
        return False
    
    # Get ground truth
    gt_text = None
    
    if no_annotation:
        # Save without ground truth
        logger.info(f"Skipping annotation (--no-annotation)")
        return True
    
    if gt_dir:
        # Try to find matching GT file
        gt_file = gt_dir / f"{img_path.stem}.gt.txt"
        if gt_file.exists():
            try:
                gt_text = read_ground_truth(gt_file)
                logger.info(f"Found ground truth: {gt_file.name}")
            except Exception as e:
                logger.warning(f"Cannot read GT file {gt_file}: {e}")
        else:
            logger.warning(f"No GT file found: {gt_file.name}")
    
    if gt_text is None and interactive:
        gt_text = get_ground_truth_interactive(img_path)
        if gt_text is None:
            logger.info(f"Skipped by user: {img_path.name}")
            out_img_path.unlink()  # Remove the image we just saved
            return False
    
    if gt_text is None and not interactive and not gt_dir:
        logger.warning(f"No ground truth available for {img_path.name}")
        logger.warning(f"Use --interactive or --gt-dir to provide ground truth")
        return True  # Keep image but no GT
    
    # Save ground truth
    if gt_text:
        write_ground_truth(gt_text, out_gt_path)
        logger.success(f"Added to eval_data: {stem}")
    
    return True


def add_images_to_eval_data(
    input_path: str | None,
    image_path: str | None,
    eval_data_dir: str,
    gt_dir: str | None,
    cfg: dict,
    interactive: bool = False,
    no_annotation: bool = False,
) -> dict:
    """
    Main processing logic.
    """
    eval_path = Path(eval_data_dir)
    eval_path.mkdir(parents=True, exist_ok=True)
    
    gt_path = Path(gt_dir) if gt_dir else None
    
    # Collect images to process
    images = []
    if image_path:
        images = [Path(image_path)]
    elif input_path:
        input_p = Path(input_path)
        if input_p.is_file():
            images = [input_p]
        else:
            images = list_images(input_p)
    
    if not images:
        logger.error("No images found to process")
        return {"processed": 0, "skipped": 0}
    
    logger.info(f"Found {len(images)} image(s) to process")
    
    processed = 0
    skipped = 0
    
    for img_path in images:
        success = process_and_add_image(
            img_path, eval_path, gt_path, cfg, interactive, no_annotation
        )
        if success:
            processed += 1
        else:
            skipped += 1
    
    console.print(
        f"\n[green bold]✓ Complete:[/green bold] "
        f"{processed} images added to {eval_path}, "
        f"{skipped} skipped"
    )
    
    return {"processed": processed, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add new images to eval_data/ for testing"
    )
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Directory of images to add"
    )
    input_group.add_argument(
        "--image",
        help="Single image file to add"
    )
    
    # Ground truth options
    parser.add_argument(
        "--gt-dir",
        help="Directory containing .gt.txt files (matching image stems)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for ground truth text for each image"
    )
    parser.add_argument(
        "--no-annotation",
        action="store_true",
        help="Add images without ground truth (for inference-only testing)"
    )
    
    # Output
    parser.add_argument(
        "--eval-data-dir",
        default="eval_data/",
        help="Output directory (default: eval_data/)"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/add_eval.log")
    cfg = load_config(args.config)
    
    if not args.interactive and not args.gt_dir and not args.no_annotation:
        console.print(
            "[yellow]Warning:[/yellow] No ground truth source specified.\n"
            "Use --interactive, --gt-dir, or --no-annotation"
        )
        console.print("Proceeding with --no-annotation mode...")
        args.no_annotation = True
    
    stats = add_images_to_eval_data(
        input_path=args.input,
        image_path=args.image,
        eval_data_dir=args.eval_data_dir,
        gt_dir=args.gt_dir,
        cfg=cfg,
        interactive=args.interactive,
        no_annotation=args.no_annotation,
    )
    
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
