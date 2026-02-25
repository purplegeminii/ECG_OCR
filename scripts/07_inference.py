#!/usr/bin/env python3
"""
07_inference.py
===============
Production-Ready ECG Meter Reading Inference Pipeline

Features:
  - Batch processing of meter images
  - Domain-specific post-processing (extract readings, account numbers, dates)
  - Confidence thresholding with human-review flagging
  - Output to CSV, JSON, or console
  - Error handling and retry logic
  - Processing statistics

Usage:
    python scripts/07_inference.py --input raw_images/new/
    python scripts/07_inference.py --input image.jpg --single
    python scripts/07_inference.py --input raw_images/ --output results/batch_001.csv --format csv
    python scripts/07_inference.py --input raw_images/ --output results/ --format json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table

from utils import (
    setup_logging, load_config, load_image_bgr,
    list_images, list_images_recursive, validate_meter_reading,
)

console = Console()


def preprocess_image_in_memory(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply preprocessing to an already-loaded image.
    Inline version of the preprocessing pipeline for inference-time use.
    """
    pre_cfg = cfg.get("preprocessing", {})
    block_size   = pre_cfg.get("adaptive_thresh_block_size", 11)
    thresh_c     = pre_cfg.get("adaptive_thresh_c", 2)
    denoise_str  = pre_cfg.get("denoise_strength", 10)
    target_width = pre_cfg.get("target_width", 1000)

    # Upscale if small
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=denoise_str)

    # Adaptive threshold
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, thresh_c,
    )

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


# ─── OCR Engine ───────────────────────────────────────────────────────────────

class ECGOCREngine:
    """
    Tesseract OCR engine configured for ECG meter images.
    Handles model selection, preprocessing, and post-processing.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ocr_cfg = cfg.get("ocr", {})
        self.psm = ocr_cfg.get("psm", 6)
        self.oem = ocr_cfg.get("oem", 1)
        self.whitelist = ocr_cfg.get("whitelist", "")
        self.conf_threshold = ocr_cfg.get("confidence_threshold", 60)

        # Model setup
        model_cfg = cfg.get("model", {})
        self.model_name = model_cfg.get("name", "eng")
        models_dir = Path(cfg.get("paths", {}).get("models", "models/"))
        tessdata_dir = models_dir / self.model_name / "tessdata"

        if tessdata_dir.exists() and (tessdata_dir / f"{self.model_name}.traineddata").exists():
            self.tessdata_dir = str(tessdata_dir)
            self.lang = self.model_name
            logger.info(f"Using custom model: {self.model_name}")
        else:
            self.tessdata_dir = None
            self.lang = "eng"
            logger.warning(f"Custom model not found — falling back to 'eng'")

    def build_config(self) -> str:
        config = f"--oem {self.oem} --psm {self.psm}"
        if self.whitelist:
            config += f" -c tessedit_char_whitelist={self.whitelist}"
        if self.tessdata_dir:
            config += f" --tessdata-dir {self.tessdata_dir}"
        return config

    def ocr_image(self, img: np.ndarray) -> dict:
        """
        Run OCR on a preprocessed image.
        Returns dict with text, confidence, words, and flagged status.
        """
        config = self.build_config()

        try:
            # Get detailed word-level data
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            # Filter words with confidence
            words = []
            confs = []
            low_conf_words = []

            for i, (word, conf) in enumerate(zip(data["text"], data["conf"])):
                word = str(word).strip()
                conf = int(conf)
                if not word:
                    continue
                words.append(word)
                if conf >= 0:
                    confs.append(conf)
                    if conf < self.conf_threshold:
                        low_conf_words.append((word, conf))

            full_text = " ".join(words)
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            flagged = mean_conf < self.conf_threshold or bool(low_conf_words)

            return {
                "text": full_text,
                "mean_confidence": round(mean_conf, 1),
                "low_confidence_words": low_conf_words,
                "flagged_for_review": flagged,
                "word_count": len(words),
            }

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract not installed or not found in PATH")
            raise
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {
                "text": "",
                "mean_confidence": 0.0,
                "low_confidence_words": [],
                "flagged_for_review": True,
                "word_count": 0,
                "error": str(e),
            }


# ─── Single image processing ──────────────────────────────────────────────────

def process_image(
    img_path: Path,
    engine: ECGOCREngine,
    cfg: dict,
    preprocess: bool = True,
) -> dict:
    """
    Full pipeline for a single meter image:
    load → preprocess → OCR → validate → return structured result
    """
    start_time = time.time()

    try:
        img = load_image_bgr(img_path)
    except Exception as e:
        return {
            "filename": img_path.name,
            "status": "error",
            "error": f"Could not load image: {e}",
            "timestamp": datetime.now().isoformat(),
        }

    # Preprocessing
    if preprocess:
        try:
            processed = preprocess_image_in_memory(img, cfg)
        except Exception as e:
            logger.warning(f"Preprocessing failed for {img_path.name}: {e}")
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    else:
        processed = img

    # OCR
    ocr_result = engine.ocr_image(processed)

    # Domain validation and field extraction
    validation = validate_meter_reading(ocr_result["text"])

    elapsed = round(time.time() - start_time, 3)

    return {
        "filename": img_path.name,
        "filepath": str(img_path),
        "status": "ok" if ocr_result["text"] else "empty",
        "raw_text": ocr_result["text"],
        "meter_readings": validation["meter_readings"],
        "account_numbers": validation["account_numbers"],
        "meter_serials": validation["meter_serials"],
        "dates": validation["dates"],
        "mean_confidence": ocr_result["mean_confidence"],
        "flagged_for_review": ocr_result["flagged_for_review"] or validation["flagged"],
        "low_confidence_words": ocr_result["low_confidence_words"],
        "processing_time_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


# ─── Batch processing ─────────────────────────────────────────────────────────

def process_batch(
    input_dir: Path,
    engine: ECGOCREngine,
    cfg: dict,
    recursive: bool = False,
    preprocess: bool = True,
) -> list[dict]:
    """Process all images in a directory."""
    if recursive:
        images = list_images_recursive(input_dir)
    else:
        images = list_images(input_dir)

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return []

    logger.info(f"Processing {len(images)} images from {input_dir}")
    results = []

    for img_path in tqdm(images, desc="Running OCR", unit="img"):
        result = process_image(img_path, engine, cfg, preprocess=preprocess)
        results.append(result)

    return results


# ─── Output writers ───────────────────────────────────────────────────────────

def save_csv(results: list[dict], output_path: Path) -> None:
    """Save results to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten list fields
    flat_results = []
    for r in results:
        flat = dict(r)
        flat["meter_readings"]  = "; ".join(r.get("meter_readings", []))
        flat["account_numbers"] = "; ".join(r.get("account_numbers", []))
        flat["meter_serials"]   = "; ".join(r.get("meter_serials", []))
        flat["dates"]           = "; ".join(r.get("dates", []))
        flat["low_confidence_words"] = str(r.get("low_confidence_words", []))
        flat_results.append(flat)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
        writer.writeheader()
        writer.writerows(flat_results)

    logger.success(f"CSV saved: {output_path}")


def save_json(results: list[dict], output_path: Path) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"JSON saved: {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print a rich console summary of batch results."""
    total = len(results)
    flagged = sum(1 for r in results if r.get("flagged_for_review"))
    errors  = sum(1 for r in results if r.get("status") == "error")
    empty   = sum(1 for r in results if r.get("status") == "empty")
    ok      = total - flagged - errors - empty

    table = Table(title="Inference Summary", show_lines=True)
    table.add_column("Metric",  style="cyan")
    table.add_column("Count",   style="white")
    table.add_column("%",       style="dim")

    def pct(n):
        return f"{100*n/total:.1f}%" if total else "—"

    table.add_row("[green]OK / High confidence[/green]",   str(ok),      pct(ok))
    table.add_row("[yellow]Flagged for review[/yellow]",   str(flagged), pct(flagged))
    table.add_row("[dim]Empty result[/dim]",               str(empty),   pct(empty))
    table.add_row("[red]Errors[/red]",                     str(errors),  pct(errors))
    table.add_row("[bold]TOTAL[/bold]",                    str(total),   "100%")

    console.print(table)

    # Sample flagged items
    flagged_items = [r for r in results if r.get("flagged_for_review")][:5]
    if flagged_items:
        console.print("\n[yellow]Flagged for human review (first 5):[/yellow]")
        for r in flagged_items:
            console.print(f"  {r['filename']} (conf={r.get('mean_confidence', 0):.0f}%)")
            console.print(f"    Text: {r.get('raw_text', '')[:60]}")

    # Average processing speed
    times = [r.get("processing_time_s", 0) for r in results if r.get("processing_time_s")]
    if times:
        avg_time = sum(times) / len(times)
        console.print(f"\n[dim]Average processing time: {avg_time:.3f}s/image[/dim]")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECG Meter OCR Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/07_inference.py --input raw_images/
  python scripts/07_inference.py --input image.jpg --single
  python scripts/07_inference.py --input raw_images/ --output results/output.csv
  python scripts/07_inference.py --input raw_images/ --format json --output results/
        """,
    )
    parser.add_argument("--input",   "-i", required=True, help="Input image or directory")
    parser.add_argument("--output",  "-o", default="results/",  help="Output file or directory")
    parser.add_argument("--format",  "-f", choices=["csv", "json", "console"], default="csv")
    parser.add_argument("--single",  action="store_true",  help="Process a single file")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--config",  default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/inference.log")
    cfg = load_config(args.config)

    engine = ECGOCREngine(cfg)
    preprocess = not args.no_preprocess

    if args.single:
        img_path = Path(args.input)
        result = process_image(img_path, engine, cfg, preprocess=preprocess)
        console.print_json(json.dumps(result, indent=2))
        return

    results = process_batch(
        Path(args.input), engine, cfg,
        recursive=args.recursive,
        preprocess=preprocess,
    )

    print_summary(results)

    if not results:
        return

    out = Path(args.output)

    if args.format == "csv":
        if out.is_dir() or not out.suffix:
            out = out / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_csv(results, out)

    elif args.format == "json":
        if out.is_dir() or not out.suffix:
            out = out / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(results, out)

    elif args.format == "console":
        for r in results:
            console.print(f"\n[bold]{r['filename']}[/bold]")
            console.print(f"  Text: {r['raw_text']}")
            console.print(f"  Readings: {r['meter_readings']}")
            console.print(f"  Accounts: {r['account_numbers']}")
            console.print(f"  Conf: {r['mean_confidence']:.1f}%")
            if r["flagged_for_review"]:
                console.print("  [yellow]⚠ FLAGGED FOR REVIEW[/yellow]")


if __name__ == "__main__":
    main()
