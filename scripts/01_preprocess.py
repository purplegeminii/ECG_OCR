#!/usr/bin/env python3
"""
01_preprocess.py
================
ECG Meter Image Preprocessing Pipeline

Converts raw meter photos into clean, normalised images ready for
Tesseract OCR training and inference.

Steps applied to each image:
  1. Load & validate
  2. Resize to minimum width (upscale only)
  3. Convert to grayscale
  4. Deskew / straighten
  5. Perspective correction (if significant tilt detected)
  6. Adaptive thresholding (handles glare and uneven lighting)
  7. Denoising
  8. ROI extraction (isolate meter display region)
  9. Final padding & save as TIFF

Usage:
    python scripts/01_preprocess.py --input raw_images/ --output preprocessed/
    python scripts/01_preprocess.py --input raw_images/ --output preprocessed/ --debug
    python scripts/01_preprocess.py --input raw_images/ --output preprocessed/ --no-roi
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

from utils import (
    load_config, setup_logging, list_images,
    load_image_bgr, save_image, resize_to_width, deskew,
)


# ─── ROI Extraction ───────────────────────────────────────────────────────────

def extract_meter_roi(image: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Attempt to extract the meter display ROI (region of interest) using contour detection.

    ECG meters have a rectangular LCD/digital display that forms a large
    dark rectangle. We try to find this. Falls back to full image if not found.
    """
    h, w = image.shape[:2]

    # Work on a small downscaled copy for speed
    scale = 0.5
    small = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Blur + edge detect
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("No contours found for ROI — using full image")
        return image

    # Sort by area descending; look for a rectangular candidate
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.05 * (small.shape[0] * small.shape[1]):
            continue  # Too small

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) in (4, 5, 6):  # Roughly rectangular
            x, y, rw, rh = cv2.boundingRect(cnt)
            # Scale back up
            x = int(x / scale)
            y = int(y / scale)
            rw = int(rw / scale)
            rh = int(rh / scale)

            # Sanity: ROI should be at least 20% of image area
            if rw * rh < 0.2 * w * h:
                continue

            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + rw + padding)
            y2 = min(h, y + rh + padding)

            roi = image[y1:y2, x1:x2]
            logger.debug(f"ROI extracted: ({x1},{y1}) → ({x2},{y2})")
            return roi

    logger.debug("Could not identify meter ROI — using full image")
    return image


# ─── Perspective correction ───────────────────────────────────────────────────

def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct perspective distortion using corner detection.
    Returns corrected image or original if correction not possible.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        return image  # Need exactly 4 corners

    pts = approx.reshape(4, 2).astype(np.float32)

    # Order: top-left, top-right, bottom-right, bottom-left
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))
    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    if width < 50 or height < 50:
        return image

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    logger.debug("Perspective correction applied")
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


# ─── Core preprocessing ───────────────────────────────────────────────────────

def preprocess_image(
    img_path: Path,
    output_dir: Path,
    cfg: dict,
    extract_roi: bool = True,
    debug: bool = False,
) -> Path | None:
    """
    Full preprocessing pipeline for a single meter image.
    Returns the output path on success, None on failure.
    """
    pre_cfg = cfg.get("preprocessing", {})
    target_width   = pre_cfg.get("target_width", 1000)
    block_size     = pre_cfg.get("adaptive_thresh_block_size", 11)
    thresh_c       = pre_cfg.get("adaptive_thresh_c", 2)
    denoise_str    = pre_cfg.get("denoise_strength", 10)
    do_deskew      = pre_cfg.get("deskew", True)
    roi_padding    = pre_cfg.get("roi_padding", 20)
    min_width      = pre_cfg.get("min_image_width", 400)

    try:
        # ── Load ──────────────────────────────────────────────────────────────
        img = load_image_bgr(img_path)
        orig_h, orig_w = img.shape[:2]

        if debug:
            _show_debug(img, f"1. Original ({orig_w}x{orig_h})")

        # ── Minimum size check ────────────────────────────────────────────────
        if orig_w < min_width:
            logger.warning(f"{img_path.name}: width {orig_w}px below minimum {min_width}px")

        # ── Upscale if needed ─────────────────────────────────────────────────
        img = resize_to_width(img, target_width)
        if debug:
            _show_debug(img, f"2. Resized ({img.shape[1]}x{img.shape[0]})")

        # ── Perspective correction (colour image) ──────────────────────────────
        img = correct_perspective(img)
        if debug:
            _show_debug(img, "3. Perspective corrected")

        # ── Grayscale ─────────────────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Deskew ────────────────────────────────────────────────────────────
        if do_deskew:
            gray = deskew(gray)
            if debug:
                _show_debug(gray, "4. Deskewed")

        # ── Denoise ───────────────────────────────────────────────────────────
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_str)
        if debug:
            _show_debug(denoised, "5. Denoised")

        # ── Adaptive threshold ────────────────────────────────────────────────
        # block_size must be odd
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            thresh_c,
        )
        if debug:
            _show_debug(binary, "6. Adaptive threshold")

        # ── Morphological cleanup ─────────────────────────────────────────────
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if debug:
            _show_debug(cleaned, "7. Morphological cleanup")

        # ── ROI extraction ────────────────────────────────────────────────────
        if extract_roi:
            # Run ROI on the colour image (better contrast detection)
            roi_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_img = load_image_bgr(img_path)  # reload for colour ROI detection
            roi_img = resize_to_width(roi_img, target_width)
            roi_region = extract_meter_roi(roi_img, padding=roi_padding)
            # If ROI was extracted, match crop in cleaned binary
            if roi_region.shape != roi_img.shape:
                # We need to find and apply the same crop to binary
                # Simple approach: just re-threshold the ROI directly
                roi_gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
                roi_denoised = cv2.fastNlMeansDenoising(roi_gray, h=denoise_str)
                cleaned = cv2.adaptiveThreshold(
                    roi_denoised, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, thresh_c
                )
            if debug:
                _show_debug(cleaned, "8. ROI extracted")

        # ── Add border padding ────────────────────────────────────────────────
        final = cv2.copyMakeBorder(
            cleaned, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=255  # white border
        )

        # ── Save as TIFF (Tesseract preferred format) ──────────────────────────
        out_path = output_dir / (img_path.stem + ".tif")
        save_image(final, out_path)
        return out_path

    except Exception as e:
        logger.error(f"Failed to preprocess {img_path.name}: {e}")
        return None


def _show_debug(img: np.ndarray, title: str) -> None:
    """Show intermediate image in debug mode."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Batch processing ─────────────────────────────────────────────────────────

def preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: dict,
    extract_roi: bool = True,
    debug: bool = False,
) -> dict:
    """Process all images in input_dir, save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info(f"Preprocessing {len(images)} images from {input_dir}")

    success, failed = 0, 0
    failed_files = []

    for img_path in tqdm(images, desc="Preprocessing", unit="img"):
        result = preprocess_image(img_path, output_dir, cfg,
                                  extract_roi=extract_roi, debug=debug)
        if result:
            success += 1
        else:
            failed += 1
            failed_files.append(img_path.name)

    logger.success(f"Preprocessing complete: {success} OK, {failed} failed")
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")

    return {
        "total": len(images),
        "success": success,
        "failed": failed,
        "failed_files": failed_files,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECG Meter Image Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True, help="Input image directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed images")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI extraction")
    parser.add_argument("--debug",  action="store_true", help="Show intermediate images (requires display)")
    parser.add_argument("--single", "-s", help="Process a single image file (for testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/preprocess.log")
    cfg = load_config(args.config)

    if args.single:
        img_path = Path(args.single)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = preprocess_image(
            img_path, output_dir, cfg,
            extract_roi=not args.no_roi,
            debug=args.debug,
        )
        if result:
            logger.success(f"Saved to: {result}")
        else:
            logger.error("Preprocessing failed")
    else:
        stats = preprocess_batch(
            Path(args.input),
            Path(args.output),
            cfg,
            extract_roi=not args.no_roi,
            debug=args.debug,
        )
        logger.info(f"Summary: {stats}")


if __name__ == "__main__":
    main()
