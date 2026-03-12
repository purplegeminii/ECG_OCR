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
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

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
    from jiwer import wer as _jiwer_wer_fn, cer as _jiwer_cer_fn

    def _jiwer_cer(reference: str, hypothesis: str) -> float:
        """jiwer CER, clamped to [0, 1] and safe for empty inputs."""
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 1.0
        return min(float(_jiwer_cer_fn(reference, hypothesis)), 1.0)

    def _jiwer_wer(reference: str, hypothesis: str) -> float:
        """jiwer WER, clamped to [0, 1] and safe for empty inputs."""
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 1.0
        return min(float(_jiwer_wer_fn(reference, hypothesis)), 1.0)

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    logger.warning("jiwer not installed — falling back to basic CER/WER")

try:
    import matplotlib
    # Backend priority:
    #   1. MPLBACKEND env var – if already set, matplotlib picks it up natively;
    #      do not override it.
    #   2. plotting.backend in config/config.yaml  (default: "Agg")
    if not os.environ.get("MPLBACKEND"):
        try:
            import yaml as _yaml
            _cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(_cfg_path) as _f:
                _mpl_backend: str = (
                    _yaml.safe_load(_f)
                    .get("plotting", {})
                    .get("backend", "Agg")
                )
        except Exception:
            _mpl_backend = "Agg"
        matplotlib.use(_mpl_backend)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

console = Console()


# ─── Basic CER/WER fallbacks ──────────────────────────────────────────────────

def edit_distance(s1: Sequence, s2: Sequence) -> int:
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
    """CER in [0, 1].  Both inputs are plain strings."""
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0  # undefined — treat as 100 % error
    return min(edit_distance(reference, hypothesis) / len(reference), 1.0)


def basic_wer(reference: str, hypothesis: str) -> float:
    """WER in [0, 1].  Computed at word level."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0  # undefined — treat as 100 % error
    return min(edit_distance(ref_words, hyp_words) / len(ref_words), 1.0)


def compute_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
    """
    Return (cer, wer) both as native Python floats in [0, 1].
    Uses jiwer when available, falls back to the basic edit-distance impl.
    """
    if HAS_JIWER:
        cer = _jiwer_cer(reference, hypothesis)
        wer = _jiwer_wer(reference, hypothesis)
    else:
        cer = basic_cer(reference, hypothesis)
        wer = basic_wer(reference, hypothesis)
    return cer, wer


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
        # Do NOT use !r — Tesseract expects the raw string, not a Python repr
        custom_config += f" -c tessedit_char_whitelist={whitelist}"
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
    """
    Build character-level confusion matrix using edit-distance traceback
    so substitutions, deletions and insertions are correctly attributed.
    """
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred, ref in zip(predictions, references):
        m, n = len(ref), len(pred)
        # Build DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
        # Traceback to record substitutions / deletions / insertions
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == pred[j - 1]:
                i -= 1; j -= 1  # match — no confusion
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                confusion[ref[i - 1]][pred[j - 1]] += 1  # substitution
                i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                confusion[ref[i - 1]]["<del>"] += 1  # deletion
                i -= 1
            else:
                confusion["<ins>"][pred[j - 1]] += 1  # insertion
                j -= 1

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

def _batch_ocr_metrics(
    pairs: list[tuple[Path, Path]],
    tessdata_dir: str | None,
    lang: str,
    cfg: dict,
) -> tuple[list[float], list[float], list[str], list[str]]:
    """Run OCR on all pairs; return (cer_list, wer_list, preds, refs)."""
    all_cer, all_wer, preds, refs = [], [], [], []
    for img_path, gt_path_item in pairs:
        gt_text = read_ground_truth(gt_path_item)
        pred_text, _ = run_ocr(img_path, tessdata_dir, lang, cfg)
        cer, wrd = compute_metrics(gt_text, pred_text)
        all_cer.append(cer)
        all_wer.append(wrd)
        preds.append(pred_text)
        refs.append(gt_text)
    return all_cer, all_wer, preds, refs


def evaluate_model(
    test_dir: str,
    gt_dir: str | None,
    model_path: str | None,
    lang: str,
    cfg: dict,
    compare_base: bool = False,
    output_dir: str = "results/",
) -> dict:
    """Full evaluation pipeline."""
    test_path = Path(test_dir)
    out_path  = Path(output_dir)

    # Resolve GT directory:
    #   1. Explicit gt_dir that exists          → use it
    #   2. Explicit gt_dir that doesn't exist   → warn and fall back to test_dir
    #      (handles eval_data/ where .gt.txt files live alongside images)
    #   3. gt_dir is None                       → fall back to test_dir
    if gt_dir and Path(gt_dir).exists():
        gt_path = Path(gt_dir)
    else:
        if gt_dir and not Path(gt_dir).exists():
            logger.warning(
                f"GT directory '{gt_dir}' not found — "
                f"falling back to test directory '{test_dir}' "
                f"(expected for co-located eval sets like eval_data/)."
            )
        gt_path = test_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine tessdata dir and lang from model path
    tessdata_dir = None
    if model_path:
        model_p = Path(model_path)
        if model_p.exists():
            tessdata_dir = str(model_p.parent)
            lang = model_p.stem
        else:
            logger.warning(f"Model file not found: {model_path} — falling back to '{lang}'")

    pairs = pair_images_with_gt(test_path, gt_path)
    if not pairs:
        logger.error(f"No image-GT pairs found in {test_path} / {gt_path}")
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

        # Compute metrics — native Python floats in [0, 1]
        cer, wrd = compute_metrics(gt_text, pred_text)

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
            # Raw [0-1] for programmatic use; *_pct columns are human-readable
            "cer": round(cer, 4),
            "cer_pct": round(cer * 100, 2),
            "wer": round(wrd, 4),
            "wer_pct": round(wrd * 100, 2),
            "confidence": round(confidence, 1),
            "reading_correct": field_acc.get("meter_readings"),
            "account_correct": field_acc.get("account_numbers"),
        })

    # ── Aggregate metrics (always native Python floats in [0, 1]) ─────────────
    mean_cer   = float(np.mean(all_cer))   if all_cer else 1.0
    mean_wer   = float(np.mean(all_wer))   if all_wer else 1.0
    median_cer = float(np.median(all_cer)) if all_cer else 1.0
    std_cer    = float(np.std(all_cer))    if all_cer else 0.0
    min_cer    = float(np.min(all_cer))    if all_cer else 0.0
    max_cer    = float(np.max(all_cer))    if all_cer else 0.0

    # ── Print results table ───────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric",  style="cyan bold")
    table.add_column("Value",   style="white")
    table.add_column("Target",  style="dim")

    target_cer = cfg.get("model", {}).get("target_cer", 0.02)

    cer_color = "green" if mean_cer <= target_cer else "red"
    table.add_row("Mean CER",       f"[{cer_color}]{mean_cer*100:.2f}%[/{cer_color}] ({mean_cer:.4f})",   f"≤{target_cer*100:.0f}%")
    table.add_row("Median CER",     f"{median_cer*100:.2f}% ({median_cer:.4f})",                          "—")
    table.add_row("Std CER",        f"{std_cer*100:.2f}% ({std_cer:.4f})",                                "—")
    table.add_row("Min / Max CER",  f"{min_cer*100:.2f}% / {max_cer*100:.2f}%",                           "—")
    table.add_row("Mean WER",       f"{mean_wer*100:.2f}% ({mean_wer:.4f})",                              "—")
    table.add_row("Samples",        str(len(results)), "—")

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
        console.print(f"  {r['image']}: CER={r['cer_pct']:.1f}%  WER={r['wer_pct']:.1f}%  conf={r['confidence']:.0f}")
        console.print(f"    GT:   {r['ground_truth'][:80]}")
        console.print(f"    Pred: {r['predicted'][:80]}")

    # ── Comparison against base eng model ────────────────────────────────────
    if compare_base:
        logger.info("Running comparison against base 'eng' model...")
        base_cer, base_wer, _, _ = _batch_ocr_metrics(pairs, None, "eng", cfg)
        base_mean_cer = float(np.mean(base_cer)) if base_cer else 1.0
        base_mean_wer = float(np.mean(base_wer)) if base_wer else 1.0
        cmp_table = Table(title="Model Comparison", show_lines=True)
        cmp_table.add_column("Model",    style="cyan bold")
        cmp_table.add_column("Mean CER", style="white")
        cmp_table.add_column("Mean WER", style="white")
        cer_delta = mean_cer - base_mean_cer
        delta_color = "green" if cer_delta < 0 else "red"
        cmp_table.add_row(
            f"Custom ({lang})",
            f"[{delta_color}]{mean_cer*100:.2f}%[/{delta_color}]",
            f"{mean_wer*100:.2f}%",
        )
        cmp_table.add_row(
            "Base (eng)",
            f"{base_mean_cer*100:.2f}%",
            f"{base_mean_wer*100:.2f}%",
        )
        cmp_table.add_row(
            "Δ (custom − base)",
            f"[{delta_color}]{cer_delta*100:+.2f}%[/{delta_color}]",
            f"{(mean_wer - base_mean_wer)*100:+.2f}%",
        )
        console.print(cmp_table)

    # ── Save CSV report ───────────────────────────────────────────────────────
    csv_path = out_path / "evaluation_report.csv"
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.success(f"Detailed report saved: {csv_path}")
    else:
        logger.warning("No results to write to CSV")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cer_distribution(all_cer, str(out_path / "cer_distribution.png"))

    confusion = build_char_confusion(all_pred, all_ref)
    plot_confusion_matrix(confusion, str(out_path / "confusion_matrix.png"))

    # All float values are native Python floats in [0, 1]; *_pct mirrors in %
    summary = {
        "mean_cer":      round(mean_cer,    4),
        "mean_cer_pct":  round(mean_cer * 100, 2),
        "median_cer":    round(median_cer,  4),
        "std_cer":       round(std_cer,     4),
        "min_cer":       round(min_cer,     4),
        "max_cer":       round(max_cer,     4),
        "mean_wer":      round(mean_wer,    4),
        "mean_wer_pct":  round(mean_wer * 100, 2),
        "samples":       len(results),
        "target_met":    bool(mean_cer <= target_cer),
        "field_accuracy": {
            f: round(float(field_correct[f]) / field_total[f], 4) if field_total[f] > 0 else None
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
    parser.add_argument("--test-dir",  default="eval_data/",    help="Test images directory")
    parser.add_argument("--gt-dir",    default="ground_truth/",
                        help="Ground truth .gt.txt directory (default: ground_truth/). "
                             "Falls back to --test-dir when the path does not exist, "
                             "which is useful for eval_data/ where GT files are co-located.")
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
