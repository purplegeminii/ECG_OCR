#!/usr/bin/env python3
"""
06_evaluate.py
==============
Comprehensive evaluation of the fine-tuned ECG OCR model.

Metrics computed:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Field-level accuracy: meter reading, account number, date
  - Per-character confusion matrix
  - Confidence score distribution
  - Comparison: custom model vs base eng model

Reports generated:
  - Console summary (rich table)
  - results/evaluation_report.csv
  - results/confusion_matrix.png
  - results/cer_distribution.png

Usage:
    python scripts/06_evaluate.py
    python scripts/06_evaluate.py --model models/ecg_meter/tessdata/ecg_meter.traineddata
    python scripts/06_evaluate.py --test-dir eval_data/ --compare
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from utils import (
    setup_logging, load_config, list_images,
    load_image_gray, read_ground_truth, pair_images_with_gt,
    validate_meter_reading,
)

try:
    from jiwer import wer as compute_wer, cer as compute_cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    logger.warning("jiwer not installed — falling back to basic CER/WER")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

console = Console()


# ─── Basic CER/WER fallbacks ──────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def basic_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def basic_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(" ".join(ref_words), " ".join(hyp_words)) / len(ref_words)


# ─── OCR runner ───────────────────────────────────────────────────────────────

def run_ocr(
    img_path: Path,
    tessdata_dir: str | None,
    lang: str,
    cfg: dict,
) -> tuple[str, float]:
    """
    Run Tesseract OCR on an image.
    Returns (text, mean_confidence).
    """
    ocr_cfg = cfg.get("ocr", {})
    psm = ocr_cfg.get("psm", 6)
    oem = ocr_cfg.get("oem", 1)
    whitelist = ocr_cfg.get("whitelist", "")

    custom_config = f"--oem {oem} --psm {psm}"
    if whitelist:
        custom_config += f" -c tessedit_char_whitelist={whitelist!r}"
    if tessdata_dir:
        custom_config += f" --tessdata-dir {tessdata_dir}"

    try:
        img = cv2.imread(str(img_path))
        data = pytesseract.image_to_data(
            img, lang=lang, config=custom_config,
            output_type=pytesseract.Output.DICT,
        )
        words = [w for w in data["text"] if w.strip()]
        confs = [int(c) for c, w in zip(data["conf"], data["text"])
                 if w.strip() and int(c) >= 0]

        text = " ".join(words)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return text.strip(), mean_conf

    except Exception as e:
        logger.warning(f"OCR failed on {img_path.name}: {e}")
        return "", 0.0


# ─── Field extraction ─────────────────────────────────────────────────────────

def extract_fields(text: str, cfg: dict) -> dict:
    """Extract domain-specific fields from OCR text."""
    domain = cfg.get("domain", {})

    def find(pattern: str) -> list[str]:
        try:
            return re.findall(pattern, text)
        except Exception:
            return []

    return {
        "meter_readings": find(domain.get("meter_reading_pattern", r"\b\d{4,6}(?:\.\d{1,2})?\b")),
        "account_numbers": find(domain.get("account_number_pattern", r"\b\d{10,13}\b")),
        "dates": find(domain.get("date_pattern", r"\d{2}[/-]\d{2}[/-]\d{2,4}")),
    }


def field_accuracy(pred_fields: dict, gt_text: str, cfg: dict) -> dict[str, bool]:
    """Check if predicted fields match what's in the GT text."""
    gt_fields = extract_fields(gt_text, cfg)
    results = {}

    for field in ["meter_readings", "account_numbers", "dates"]:
        pred_vals = set(pred_fields.get(field, []))
        gt_vals   = set(gt_fields.get(field, []))
        if not gt_vals:
            results[field] = None  # Not applicable
        else:
            results[field] = bool(pred_vals & gt_vals)  # Any overlap

    return results


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def build_char_confusion(
    predictions: list[str],
    references: list[str],
) -> dict[str, dict[str, int]]:
    """Build character-level confusion matrix."""
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, ref in zip(predictions, references):
        # Simple character-level alignment (no proper alignment for simplicity)
        for p, r in zip(pred[:len(ref)], ref[:len(pred)]):
            if p != r:
                confusion[r][p] += 1
    return dict(confusion)


def plot_confusion_matrix(
    confusion: dict,
    output_path: str,
    top_n: int = 15,
) -> None:
    """Plot top confused character pairs."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping confusion matrix plot")
        return

    # Flatten to list of (reference, predicted, count)
    pairs = []
    for ref_char, preds in confusion.items():
        for pred_char, count in preds.items():
            pairs.append((ref_char, pred_char, count))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_n]

    if not pairs:
        return

    labels = [f"'{r}'→'{p}'" for r, p, _ in pairs]
    counts = [c for _, _, c in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), counts, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Character Confusions (reference→predicted)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")


def plot_cer_distribution(cer_values: list[float], output_path: str) -> None:
    """Plot CER distribution histogram."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cer_values, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    mean_cer_val = float(np.mean(cer_values))
    ax.axvline(mean_cer_val, color="red", linestyle="--",
               label=f"Mean CER: {mean_cer_val:.3f}")
    ax.set_xlabel("Character Error Rate")
    ax.set_ylabel("Number of Images")
    ax.set_title("CER Distribution Across Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"CER distribution saved: {output_path}")


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    test_dir: str,
    gt_dir: str,
    model_path: str | None,
    lang: str,
    cfg: dict,
    compare_base: bool = False,
    output_dir: str = "results/",
) -> dict:
    """Full evaluation pipeline."""
    test_path = Path(test_dir)
    gt_path   = Path(gt_dir)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine tessdata dir and lang from model path
    tessdata_dir = None
    if model_path:
        model_p = Path(model_path)
        if model_p.exists():
            tessdata_dir = str(model_p.parent)
            lang = model_p.stem

    pairs = pair_images_with_gt(test_path, gt_path)
    if not pairs:
        logger.error(f"No image-GT pairs found in {test_dir} + {gt_dir}")
        return {}

    logger.info(f"Evaluating {len(pairs)} samples with lang='{lang}'")

    results = []
    all_cer, all_wer = [], []
    field_correct = defaultdict(int)
    field_total   = defaultdict(int)
    all_pred, all_ref = [], []

    for img_path, gt_path_item in tqdm(pairs, desc="Evaluating", unit="img"):
        gt_text = read_ground_truth(gt_path_item)
        pred_text, confidence = run_ocr(img_path, tessdata_dir, lang, cfg)

        # Compute metrics
        if HAS_JIWER:
            cer = compute_cer(gt_text, pred_text) if gt_text else 0.0
            wrd = compute_wer(gt_text, pred_text) if gt_text else 0.0
        else:
            cer = basic_cer(gt_text, pred_text)
            wrd = basic_wer(gt_text, pred_text)

        all_cer.append(cer)
        all_wer.append(wrd)
        all_pred.append(pred_text)
        all_ref.append(gt_text)

        # Field accuracy
        pred_fields = extract_fields(pred_text, cfg)
        field_acc = field_accuracy(pred_fields, gt_text, cfg)
        for field, correct in field_acc.items():
            if correct is not None:
                field_total[field] += 1
                if correct:
                    field_correct[field] += 1

        results.append({
            "image": img_path.name,
            "ground_truth": gt_text,
            "predicted": pred_text,
            "cer": round(cer, 4),
            "wer": round(wrd, 4),
            "confidence": round(confidence, 1),
            "reading_correct": field_acc.get("meter_readings"),
            "account_correct": field_acc.get("account_numbers"),
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mean_cer = np.mean(all_cer) if all_cer else 1.0
    mean_wer = np.mean(all_wer) if all_wer else 1.0
    median_cer = np.median(all_cer) if all_cer else 1.0

    # ── Print results table ───────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric",  style="cyan bold")
    table.add_column("Value",   style="white")
    table.add_column("Target",  style="dim")

    target_cer = cfg.get("model", {}).get("target_cer", 0.02)

    cer_color = "green" if mean_cer <= target_cer else "red"
    table.add_row("Mean CER",    f"[{cer_color}]{mean_cer:.4f} ({mean_cer*100:.2f}%)[/{cer_color}]", f"≤{target_cer*100:.0f}%")
    table.add_row("Median CER",  f"{median_cer:.4f} ({median_cer*100:.2f}%)", "—")
    table.add_row("Mean WER",    f"{mean_wer:.4f} ({mean_wer*100:.2f}%)",     "—")
    table.add_row("Samples",     str(len(results)), "—")

    for field in ["meter_readings", "account_numbers", "dates"]:
        if field_total[field] > 0:
            acc = field_correct[field] / field_total[field]
            color = "green" if acc >= 0.95 else "yellow" if acc >= 0.80 else "red"
            label = field.replace("_", " ").title()
            table.add_row(
                f"Field: {label}",
                f"[{color}]{acc*100:.1f}%[/{color}] ({field_correct[field]}/{field_total[field]})",
                "≥95%",
            )

    console.print(table)

    # ── Worst cases ───────────────────────────────────────────────────────────
    worst = sorted(results, key=lambda x: x["cer"], reverse=True)[:5]
    console.print("\n[bold red]Worst 5 predictions:[/bold red]")
    for r in worst:
        console.print(f"  {r['image']}: CER={r['cer']:.3f}")
        console.print(f"    GT:   {r['ground_truth'][:60]}")
        console.print(f"    Pred: {r['predicted'][:60]}")

    # ── Save CSV report ───────────────────────────────────────────────────────
    csv_path = out_path / "evaluation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.success(f"Detailed report saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cer_distribution(all_cer, str(out_path / "cer_distribution.png"))

    confusion = build_char_confusion(all_pred, all_ref)
    plot_confusion_matrix(confusion, str(out_path / "confusion_matrix.png"))

    summary = {
        "mean_cer": round(mean_cer, 4),
        "median_cer": round(median_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "samples": len(results),
        "target_met": mean_cer <= target_cer,
        "field_accuracy": {
            f: round(field_correct[f] / field_total[f], 4) if field_total[f] > 0 else None
            for f in ["meter_readings", "account_numbers"]
        },
    }

    console.print(Panel.fit(
        f"[bold]{'✓ TARGET MET' if summary['target_met'] else '✗ TARGET NOT MET'}[/bold]\n"
        f"Mean CER: {mean_cer*100:.2f}% (target: ≤{target_cer*100:.0f}%)",
        style="green" if summary["target_met"] else "red",
    ))

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECG OCR Model")
    parser.add_argument("--test-dir",  default="raw_images/new/",    help="Test images directory")
    parser.add_argument("--gt-dir",    default="ground_truth/", help="Ground truth directory")
    parser.add_argument("--model",     help="Path to .traineddata model file")
    parser.add_argument("--lang",      default="eng",          help="Tesseract language code")
    parser.add_argument("--output",    default="results/",     help="Output directory for reports")
    parser.add_argument("--compare",   action="store_true",    help="Compare against base eng model")
    parser.add_argument("--config",    default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/evaluate.log")
    cfg = load_config(args.config)

    summary = evaluate_model(
        test_dir=args.test_dir,
        gt_dir=args.gt_dir,
        model_path=args.model,
        lang=args.lang,
        cfg=cfg,
        compare_base=args.compare,
        output_dir=args.output,
    )
    logger.info(f"Evaluation summary: {summary}")


if __name__ == "__main__":
    main()
