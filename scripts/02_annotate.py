#!/usr/bin/env python3
"""
02_annotate.py
==============
Annotation helper for ECG Meter OCR ground truth labeling.

Two modes:
  1. `--launch`  : Start a Label Studio server for GUI annotation
  2. `--convert` : Convert Label Studio JSON export → .gt.txt files
  3. `--manual`  : Simple CLI tool to type ground truth for images quickly
  4. `--validate`: Check all .gt.txt files for common errors

Usage:
    python scripts/02_annotate.py --launch --images preprocessed/
    python scripts/02_annotate.py --convert --export ls_export.json --output ground_truth/
    python scripts/02_annotate.py --manual --images preprocessed/ --output ground_truth/
    python scripts/02_annotate.py --validate --gt ground_truth/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from utils import (
    setup_logging, load_config, list_images,
    read_ground_truth, write_ground_truth, pair_images_with_gt,
)

console = Console()


# ─── Label Studio launcher ────────────────────────────────────────────────────

LABEL_STUDIO_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <Labels name="label" toName="image">
    <Label value="meter_reading" background="red"/>
    <Label value="account_number" background="blue"/>
    <Label value="meter_serial" background="green"/>
    <Label value="date" background="orange"/>
    <Label value="other_text" background="gray"/>
  </Labels>
  <TextArea name="transcription" toName="image"
            placeholder="Type the FULL text visible on the meter..."
            rows="4" editable="true" maxSubmissions="1"/>
</View>
"""

def launch_label_studio(images_dir: str) -> None:
    """Launch Label Studio with pre-configured project for meter annotation."""
    images_dir_path = Path(images_dir).resolve()
    project_dir = Path("label_studio_project")
    project_dir.mkdir(exist_ok=True)

    # Write labeling config
    config_file = project_dir / "labeling_config.xml"
    config_file.write_text(LABEL_STUDIO_CONFIG)

    # Check if images are .tif (not supported by Label Studio)
    tif_images = list(images_dir_path.glob("*.tif")) + list(images_dir_path.glob("*.tiff"))
    
    if tif_images:
        logger.warning(f"Found {len(tif_images)} .tif images - Label Studio doesn't support TIFF format")
        console.print("[yellow]Converting .tif → .png for Label Studio...[/yellow]")
        
        # Create PNG directory
        png_dir = images_dir_path.parent / f"{images_dir_path.name}_png"
        png_dir.mkdir(exist_ok=True)
        
        # Convert using ImageMagick mogrify
        try:
            # Copy tif files to png directory
            for tif_file in tqdm(tif_images, desc="Copying TIFs"):
                shutil.copy2(tif_file, png_dir / tif_file.name)
            
            # Convert to PNG in place
            logger.info(f"Converting {len(tif_images)} images to PNG format...")
            result = subprocess.run(
                ["mogrify", "-format", "png", "*.tif"],
                cwd=png_dir,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.error(f"ImageMagick conversion failed: {result.stderr}")
                console.print(
                    "[red]ImageMagick not found or conversion failed.[/red]\n"
                    "[yellow]Install ImageMagick:[/yellow] brew install imagemagick\n"
                    "[yellow]Or use manual mode:[/yellow] python scripts/02_annotate.py --manual"
                )
                return
            
            # Remove .tif files from png directory
            for tif_file in png_dir.glob("*.tif"):
                tif_file.unlink()
            for tif_file in png_dir.glob("*.tiff"):
                tif_file.unlink()
            
            logger.success(f"Converted images saved to: {png_dir}")
            images_dir_path = png_dir  # Use PNG directory for Label Studio
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            console.print(
                "[red]Could not convert images to PNG.[/red]\n"
                "[yellow]Use manual mode instead:[/yellow] python scripts/02_annotate.py --manual"
            )
            return

    console.print(Panel.fit(
        "[bold green]Label Studio Annotation Setup[/bold green]\n\n"
        "1. Label Studio will open in your browser at http://localhost:8080/\n"
        "2. Create a new project.\n"
        "3. Import images from: [cyan]" + str(images_dir_path) + "[/cyan]\n"
        "4. Use the labeling config from: [cyan]" + str(config_file) + "[/cyan]\n"
        "5. For each image, type the FULL meter text in the transcription box.\n"
        "6. When done, export as JSON and run:\n"
        "   [yellow]python scripts/02_annotate.py --convert --export export.json --output ground_truth/[/yellow]\n",
        title="Instructions",
    ))

    try:
        subprocess.run(["label-studio", "start", "--no-browser"], check=False)
    except FileNotFoundError:
        logger.error("Label Studio not found. Install with: pip install label-studio")
        logger.info("Alternative: use --manual mode for CLI-based annotation")


# ─── Label Studio JSON converter ──────────────────────────────────────────────

def convert_label_studio_export(
    export_file: str,
    output_dir: str,
) -> dict:
    """
    Convert Label Studio JSON export to .gt.txt files.

    Label Studio exports each annotation as a task. We extract the
    transcription text and save it as a .gt.txt file matching the image stem.
    """
    export_path = Path(export_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(export_path) as f:
        data = json.load(f)

    converted, skipped = 0, 0

    for task in tqdm(data, desc="Converting annotations"):
        # Get original filename from file_upload or data fields
        filename = task.get("file_upload", "")
        if not filename:
            # Fallback to data.image or data.ocr
            image_url = task.get("data", {}).get("image", "") or task.get("data", {}).get("ocr", "")
            filename = Path(image_url.split("/")[-1]).name if image_url else ""
        
        # Remove UUID prefix if present (Label Studio adds them)
        # Format: "de5198c7-postpaidmeter_img1.png" -> "postpaidmeter_img1.png"
        if "-" in filename and len(filename.split("-")[0]) == 8:
            filename = "-".join(filename.split("-")[1:])
        
        stem = Path(filename).stem

        if not stem:
            skipped += 1
            continue

        # Extract transcription text
        annotations = task.get("annotations", [])
        if not annotations:
            logger.warning(f"No annotation for: {stem}")
            skipped += 1
            continue

        text = ""
        for ann in annotations:
            for result in ann.get("result", []):
                if result.get("type") == "textarea" and result.get("from_name") == "transcription":
                    text_array = result.get("value", {}).get("text", [])
                    if text_array:
                        text = text_array[0].strip()
                        break
            if text:
                break

        if not text:
            logger.warning(f"Empty transcription for: {stem}")
            skipped += 1
            continue

        gt_file = output_path / f"{stem}.gt.txt"
        write_ground_truth(text, gt_file)
        converted += 1

    logger.success(f"Converted {converted} annotations, skipped {skipped}")
    return {"converted": converted, "skipped": skipped}


# ─── Manual CLI annotation ────────────────────────────────────────────────────

def manual_annotate(images_dir: str, output_dir: str, overwrite: bool = False) -> None:
    """
    Simple CLI-based annotation tool.
    Displays image info and prompts user to type the ground truth.
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = list_images(images_path)
    if not images:
        logger.error(f"No images found in {images_dir}")
        return

    # Filter already-annotated
    if not overwrite:
        images = [
            p for p in images
            if not (output_path / (p.stem + ".gt.txt")).exists()
        ]

    if not images:
        console.print("[green]All images are already annotated![/green]")
        return

    console.print(Panel.fit(
        f"[bold]Manual Annotation Mode[/bold]\n"
        f"Images to annotate: [cyan]{len(images)}[/cyan]\n\n"
        "For each image, type the COMPLETE text visible on the meter.\n"
        "Include: meter reading (kWh), account number, date, serial number.\n"
        "Type [yellow]SKIP[/yellow] to skip, [yellow]QUIT[/yellow] to stop.\n"
        "Separate fields with a space or newline as they appear on the meter."
    ))

    annotated = 0
    for i, img_path in enumerate(images):
        console.print(f"\n[bold cyan]Image {i+1}/{len(images)}:[/bold cyan] {img_path.name}")

        # Try to display image dimensions
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                console.print(f"  Size: {w}×{h}px")
        except Exception:
            pass

        while True:
            try:
                text = input("  Ground truth: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Annotation session ended.[/yellow]")
                logger.info(f"Annotated {annotated} images")
                return

            if text.upper() == "QUIT":
                console.print("[yellow]Quitting annotation session.[/yellow]")
                logger.info(f"Annotated {annotated} images")
                return

            if text.upper() == "SKIP":
                console.print("  [dim]Skipped[/dim]")
                break

            if text:
                gt_file = output_path / (img_path.stem + ".gt.txt")
                write_ground_truth(text, gt_file)
                console.print(f"  [green]✓ Saved:[/green] {gt_file.name}")
                annotated += 1
                break
            else:
                console.print("  [red]Empty input. Type text or SKIP.[/red]")

    console.print(f"\n[green bold]Annotation complete: {annotated} images labeled.[/green bold]")


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_ground_truth(gt_dir: str, image_dir: str | None = None) -> dict:
    """
    Check .gt.txt files for common issues:
    - Empty files
    - Files with no digit (suspicious for meter reading labels)
    - Files matching no image (orphaned)
    - Images with no GT file
    """
    gt_path = Path(gt_dir)
    gt_files = sorted(gt_path.glob("*.gt.txt"))

    if not gt_files:
        logger.warning(f"No .gt.txt files found in {gt_dir}")
        return {}

    issues = []
    stats = {
        "total": len(gt_files),
        "empty": 0,
        "no_digits": 0,
        "very_short": 0,
        "ok": 0,
    }

    table = Table(title=f"Ground Truth Validation — {gt_dir}", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Content", style="dim")

    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8").strip()
        status = "✓ OK"
        color = "green"

        if not text:
            status = "✗ EMPTY"
            color = "red"
            stats["empty"] += 1
            issues.append(gt_file.name)
        elif len(text) < 4:
            status = "⚠ VERY SHORT"
            color = "yellow"
            stats["very_short"] += 1
        elif not re.search(r"\d", text):
            status = "⚠ NO DIGITS"
            color = "yellow"
            stats["no_digits"] += 1
        else:
            stats["ok"] += 1

        table.add_row(
            gt_file.name,
            f"[{color}]{status}[/{color}]",
            text[:60] + ("..." if len(text) > 60 else ""),
        )

    console.print(table)

    # Check for images without GT
    if image_dir:
        images = list_images(Path(image_dir))
        missing_gt = [
            p for p in images
            if not (gt_path / (p.stem + ".gt.txt")).exists()
        ]
        if missing_gt:
            console.print(f"\n[red]{len(missing_gt)} images have no GT file:[/red]")
            for p in missing_gt[:10]:
                console.print(f"  {p.name}")
            if len(missing_gt) > 10:
                console.print(f"  ... and {len(missing_gt) - 10} more")

    console.print(f"\n[bold]Summary:[/bold] {stats}")
    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECG OCR Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch",   action="store_true", help="Launch Label Studio server")
    group.add_argument("--convert",  action="store_true", help="Convert Label Studio JSON export")
    group.add_argument("--manual",   action="store_true", help="CLI-based manual annotation")
    group.add_argument("--validate", action="store_true", help="Validate existing .gt.txt files")

    parser.add_argument("--images",    default="preprocessed/",  help="Images directory")
    parser.add_argument("--output",    default="ground_truth/",  help="Output .gt.txt directory")
    parser.add_argument("--export",    help="Label Studio JSON export file (for --convert)")
    parser.add_argument("--gt",        default="ground_truth/",  help="GT directory (for --validate)")
    parser.add_argument("--overwrite", action="store_true",       help="Overwrite existing .gt.txt files")
    parser.add_argument("--config",    default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/annotate.log")

    if args.launch:
        launch_label_studio(args.images)

    elif args.convert:
        if not args.export:
            logger.error("--export required with --convert")
            sys.exit(1)
        convert_label_studio_export(args.export, args.output)

    elif args.manual:
        manual_annotate(args.images, args.output, overwrite=args.overwrite)

    elif args.validate:
        validate_ground_truth(
            args.gt,
            image_dir=args.images if os.path.exists(args.images) else None,
        )


if __name__ == "__main__":
    main()
