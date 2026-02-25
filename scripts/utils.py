"""
utils.py — Shared utilities for the ECG OCR pipeline.
"""
from __future__ import annotations

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


# ─── Config loader ────────────────────────────────────────────────────────────

_CONFIG = None


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and cache the YAML config file."""
    global _CONFIG
    if _CONFIG is None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Run: cp config/config.example.yaml config/config.yaml"
            )
        with open(path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_cfg(key_path: str, default: Any = None) -> Any:
    """
    Access nested config with dot notation.
    e.g. get_cfg('model.learning_rate')
    """
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """Configure loguru logger with file + console output."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )


# ─── Image utilities ──────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(directory: str | Path) -> list[Path]:
    """Return all supported image files in a directory (non-recursive)."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def list_images_recursive(directory: str | Path) -> list[Path]:
    """Return all supported image files in directory tree."""
    d = Path(directory)
    return sorted(
        p for p in d.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load image as BGR numpy array, raise if not found."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_image_gray(path: str | Path) -> np.ndarray:
    """Load image as grayscale numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Save numpy image to file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def resize_to_width(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target_width, maintaining aspect ratio."""
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)


def estimate_dpi(img: np.ndarray, physical_width_mm: float = 80.0) -> float:
    """
    Rough DPI estimate based on assumed physical meter display width.
    Default: ~80mm wide meter display is typical for ECG meters.
    """
    h, w = img.shape[:2]
    dpi = w / (physical_width_mm / 25.4)
    return dpi


# ─── Deskew ───────────────────────────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in a grayscale or binary image using Hough line detection.
    Returns the rotated image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)

    if lines is None:
        return image  # Can't detect lines, return as-is

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image  # Already straight enough

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug(f"Deskewed by {median_angle:.2f}°")
    return rotated


# ─── Domain validation ────────────────────────────────────────────────────────

def validate_meter_reading(raw_text: str) -> Dict[str, Any]:
    """
    Apply ECG domain rules to extracted OCR text.
    Returns structured dict of found fields plus a validation status.
    """
    cfg = load_config().get("domain", {})

    reading_pat = cfg.get("meter_reading_pattern", r"\b\d{4,6}(?:\.\d{1,2})?\b")
    account_pat = cfg.get("account_number_pattern", r"\b\d{10,13}\b")
    serial_pat  = cfg.get("meter_serial_pattern",   r"[A-Z0-9]{8,15}")
    date_pat    = cfg.get("date_pattern",            r"\d{2}[/-]\d{2}[/-]\d{2,4}")

    readings = re.findall(reading_pat, raw_text)
    accounts = re.findall(account_pat, raw_text)
    serials  = re.findall(serial_pat,  raw_text)
    dates    = re.findall(date_pat,    raw_text)

    min_r = cfg.get("min_valid_reading", 0)
    max_r = cfg.get("max_valid_reading", 999999)

    valid_readings = []
    for r in readings:
        try:
            val = float(r)
            if min_r <= val <= max_r:
                valid_readings.append(r)
        except ValueError:
            pass

    return {
        "raw_text": raw_text,
        "meter_readings": valid_readings,
        "account_numbers": accounts,
        "meter_serials": serials,
        "dates": dates,
        "valid": bool(valid_readings or accounts),
        "flagged": not bool(valid_readings),
    }


# ─── Ground truth I/O ─────────────────────────────────────────────────────────

def read_ground_truth(gt_file: str | Path) -> str:
    """Read a .gt.txt ground truth file."""
    with open(gt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_ground_truth(text: str, gt_file: str | Path) -> None:
    """Write ground truth text to a .gt.txt file."""
    p = Path(gt_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def pair_images_with_gt(
    image_dir: str | Path,
    gt_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """
    Return list of (image_path, gt_path) tuples where both files exist.
    GT files are matched by stem name.
    """
    pairs = []
    for img_path in list_images(image_dir):
        gt_path = Path(gt_dir) / (img_path.stem + ".gt.txt")
        if gt_path.exists():
            pairs.append((img_path, gt_path))
        else:
            logger.warning(f"No GT file for: {img_path.name}")
    return pairs
