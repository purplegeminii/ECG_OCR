#!/usr/bin/env python3
"""
03_augment.py
=============
Synthetic Data Augmentation for ECG Meter OCR Training

Expands small datasets by generating realistic variations of meter images
that simulate field conditions: lighting variation, slight angles, camera
noise, ink wear, and focus blur.

Each augmented image gets the same .gt.txt as its source image.

Usage:
    python scripts/03_augment.py --input preprocessed/ --output augmented/ --factor 5
    python scripts/03_augment.py --input preprocessed/ --output augmented/ --factor 3 --gt ground_truth/
    python scripts/03_augment.py --preview --input preprocessed/ --single meter_001.tif
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

from utils import (
    setup_logging, load_config,
    list_images, load_image_bgr, save_image,
    write_ground_truth, read_ground_truth,
)

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    logger.warning("albumentations not installed — using basic augmentations only")


# ─── Augmentation pipeline ────────────────────────────────────────────────────

def build_augmentation_pipeline(cfg: dict):
    """Build a list of augmentation functions from config."""
    aug_cfg = cfg.get("augmentation", {})

    if HAS_ALBUMENTATIONS:
        return _build_albumentations_pipeline(aug_cfg)
    else:
        return _build_basic_pipeline(aug_cfg)


def _build_albumentations_pipeline(aug_cfg: dict):
    """Full albumentations-based pipeline."""
    transforms = A.Compose([
        # Lighting variations (common in field photos)
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get("brightness_limit", 0.3),
            contrast_limit=aug_cfg.get("contrast_limit", 0.3),
            p=0.7,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),

        # Blur (out-of-focus photos, motion)
        A.OneOf([
            A.GaussianBlur(blur_limit=aug_cfg.get("blur_limit", 3), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.4),

        # Noise (camera sensor noise, compression artefacts)
        A.GaussNoise(p=0.4),

        # Geometric variations (slight camera angle, holder position)
        A.Rotate(
            limit=aug_cfg.get("rotation_limit", 5),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5,
        ),
        A.Perspective(
            scale=aug_cfg.get("perspective_scale", [0.02, 0.05]),
            p=0.3,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=3,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.3,
        ),

        # Shadows / lighting patches (hand shadow, reflection)
        A.RandomShadow(p=0.2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),

        # JPEG compression artefacts
        A.ImageCompression(quality_range=(60, 100), p=0.3),
    ])
    return transforms


def _build_basic_pipeline(aug_cfg: dict):
    """Fallback pipeline without albumentations."""
    return None  # handled separately


def augment_with_albumentations(
    img: np.ndarray,
    transforms,
    num_variations: int,
) -> List[np.ndarray]:
    """Generate num_variations augmented copies of img."""
    results = []
    for _ in range(num_variations):
        augmented = transforms(image=img)["image"]
        results.append(augmented)
    return results


def augment_basic(img: np.ndarray, num_variations: int, aug_cfg: dict) -> List[np.ndarray]:
    """Basic augmentation without albumentations."""
    results = []
    for i in range(num_variations):
        aug = img.copy()

        # Random brightness
        factor = random.uniform(0.7, 1.3)
        aug = np.clip(aug.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Random rotation
        angle = random.uniform(-5, 5)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random Gaussian blur
        if random.random() < 0.4:
            ksize = random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

        # Random noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 10, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        results.append(aug)
    return results


# ─── Morphological degradation ────────────────────────────────────────────────

def apply_morphological_degradation(img: np.ndarray) -> np.ndarray:
    """
    Simulate ink wear, dot matrix degradation, or aged meter displays.
    Applies subtle erosion or dilation to binary/near-binary images.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    operation = random.choice(["erode", "dilate", "none"])
    if operation == "none":
        return img

    kernel_size = random.choice([2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = random.randint(1, 2)

    if operation == "erode":
        processed = cv2.erode(gray, kernel, iterations=iterations)
    else:
        processed = cv2.dilate(gray, kernel, iterations=iterations)

    if img.ndim == 3:
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return processed


# ─── Main augmentation batch ──────────────────────────────────────────────────

def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    gt_dir: Path | None,
    out_gt_dir: Path | None,
    factor: int,
    cfg: dict,
    copy_originals: bool = True,
) -> dict:
    """
    Augment all images in input_dir by `factor` times.

    If gt_dir is provided, copies matching .gt.txt files for each augmented image.
    If copy_originals, also copies originals to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if out_gt_dir:
        out_gt_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        logger.error(f"No images in {input_dir}")
        return {}

    aug_cfg = cfg.get("augmentation", {})
    transforms = build_augmentation_pipeline(cfg)
    apply_morpho = aug_cfg.get("apply_morphological", True)

    generated = 0
    skipped = 0

    logger.info(f"Augmenting {len(images)} images × {factor} = up to {len(images)*factor} new images")

    for img_path in tqdm(images, desc="Augmenting", unit="img"):
        # Load original
        try:
            img = load_image_bgr(img_path)
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            skipped += 1
            continue

        # Read corresponding GT if available
        gt_text = None
        if gt_dir:
            gt_file = gt_dir / (img_path.stem + ".gt.txt")
            if gt_file.exists():
                gt_text = read_ground_truth(gt_file)
            else:
                logger.warning(f"No GT for {img_path.name}")

        # Copy original to output
        if copy_originals:
            out_original = output_dir / img_path.name
            if not out_original.exists():
                shutil.copy2(img_path, out_original)
            if gt_text and out_gt_dir:
                write_ground_truth(gt_text, out_gt_dir / (img_path.stem + ".gt.txt"))

        # Generate augmented variants
        if HAS_ALBUMENTATIONS and transforms is not None:
            variants = augment_with_albumentations(img, transforms, factor)
        else:
            variants = augment_basic(img, factor, aug_cfg)

        for idx, aug_img in enumerate(variants, start=1):
            # Optionally apply morphological degradation
            if apply_morpho and random.random() < 0.3:
                aug_img = apply_morphological_degradation(aug_img)

            out_stem = f"{img_path.stem}_aug{idx:03d}"
            out_path = output_dir / f"{out_stem}.tif"
            save_image(aug_img, out_path)

            # Copy GT for this augmented image
            if gt_text and out_gt_dir:
                write_ground_truth(gt_text, out_gt_dir / f"{out_stem}.gt.txt")

            generated += 1

    logger.success(
        f"Augmentation complete: {generated} new images generated, "
        f"{skipped} source images skipped"
    )
    return {"original": len(images), "generated": generated, "skipped": skipped}


# ─── Preview mode ─────────────────────────────────────────────────────────────

def preview_augmentation(input_dir: Path, single: str | None, cfg: dict) -> None:
    """Show augmentation effects on a single image using OpenCV display."""
    if single:
        img_path = Path(single)
        if not img_path.exists():
            img_path = input_dir / single
    else:
        images = list_images(input_dir)
        if not images:
            logger.error("No images found")
            return
        img_path = images[0]

    img = load_image_bgr(img_path)
    transforms = build_augmentation_pipeline(cfg)

    logger.info(f"Previewing augmentations on: {img_path.name}")
    logger.info("Press any key to cycle through variants. Press Q to quit.")

    # Show original
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    # Show 6 variants
    for i in range(6):
        if HAS_ALBUMENTATIONS and transforms:
            aug = transforms(image=img)["image"]
        else:
            aug = augment_basic(img, 1, cfg.get("augmentation", {}))[0]
        cv2.imshow(f"Augmented variant {i+1}", aug)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("Q"):
            break

    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG OCR Dataset Augmentation")
    parser.add_argument("--input",  "-i", default="preprocessed/")
    parser.add_argument("--output", "-o", default="augmented/")
    parser.add_argument("--gt",     default="ground_truth/",  help="GT dir for source labels")
    parser.add_argument("--out-gt", default="ground_truth/",  help="Where to write augmented GT files")
    parser.add_argument("--factor", "-f", type=int, default=5, help="Augmentation multiplier")
    parser.add_argument("--no-originals", action="store_true", help="Don't copy originals to output")
    parser.add_argument("--preview", action="store_true", help="Preview mode (requires display)")
    parser.add_argument("--single",  help="Single image to preview (--preview mode)")
    parser.add_argument("--config",  default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/augment.log")
    cfg = load_config(args.config)

    if args.preview:
        preview_augmentation(Path(args.input), args.single, cfg)
        return

    gt_dir = Path(args.gt) if Path(args.gt).exists() else None
    out_gt_dir = Path(args.out_gt) if args.out_gt else None

    stats = augment_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        gt_dir=gt_dir,
        out_gt_dir=out_gt_dir,
        factor=args.factor,
        cfg=cfg,
        copy_originals=not args.no_originals,
    )
    logger.info(f"Augmentation stats: {stats}")


if __name__ == "__main__":
    main()
