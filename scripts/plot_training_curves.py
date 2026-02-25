#!/usr/bin/env python3
"""
plot_training_curves.py
=======================
Parse Tesseract LSTM training log and plot CER / loss curves.

The training log contains lines like:
  At iteration 1000, Mean rms=1.234%, delta=0.567%, BCER train=8.901%, ...
  BEST model written to ... with mean error rate of 6.789%

Usage:
    python scripts/plot_training_curves.py --log logs/training_20240101_120000.log
    python scripts/plot_training_curves.py --log logs/training.log --output results/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("matplotlib is required: pip install matplotlib")
    sys.exit(1)


# ─── Log parser ───────────────────────────────────────────────────────────────

def parse_training_log(log_path: str) -> dict:
    """
    Parse Tesseract LSTM training log and extract metrics.

    Handles log format from both lstmtraining and tesstrain output.
    """
    iterations   = []
    train_cer    = []
    eval_cer     = []
    best_cer     = []
    best_iters   = []
    rms_values   = []

    # Patterns
    # "At iteration 1000, Mean rms=1.23%, delta=0.45%, BCER train=8.90%, BCER eval=7.65%..."
    # Also handles "At iteration 92/700/700, mean rms=..." format
    iter_pattern = re.compile(
        r"At iteration\s+(?:\d+/)?(\d+)(?:/\d+)?,\s*"
        r"[Mm]ean rms=([\d.]+)%,\s*"
        r"delta=([\d.]+)%,\s*"
        r"BCER train=([\d.]+)%"
        r"(?:,\s*BCER eval=([\d.]+)%)?"
    )

    # "BEST model written to ... with mean error rate of 6.789%"
    # Also handles "New best BCER = 6.789" format
    best_pattern = re.compile(
        r"BEST model written.*?with mean error rate of ([\d.]+)%\s+at iteration\s+(\d+)"
    )
    new_best_pattern = re.compile(
        r"At iteration\s+(?:\d+/)?(\d+)(?:/\d+)?,.*?New best BCER\s*=\s*([\d.]+)"
    )

    # tesstrain style: "0.234  [1000/10000]"
    tesstrain_pattern = re.compile(r"([\d.]+)\s+\[(\d+)/\d+\]")

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Standard lstmtraining output
            m = iter_pattern.search(line)
            if m:
                iter_n = int(m.group(1))
                rms    = float(m.group(2))
                t_cer  = float(m.group(4))
                e_cer  = float(m.group(5)) if m.group(5) else None

                iterations.append(iter_n)
                rms_values.append(rms)
                train_cer.append(t_cer)
                if e_cer is not None:
                    eval_cer.append((iter_n, e_cer))
                continue

            # Best model line
            m = best_pattern.search(line)
            if m:
                cer  = float(m.group(1))
                itr  = int(m.group(2))
                best_cer.append(cer)
                best_iters.append(itr)
                continue

            # New best BCER format
            m = new_best_pattern.search(line)
            if m:
                itr = int(m.group(1))
                cer = float(m.group(2))
                best_cer.append(cer)
                best_iters.append(itr)
                continue

            # tesstrain compact output
            m = tesstrain_pattern.search(line)
            if m and not iterations:
                cer = float(m.group(1)) * 100
                itr = int(m.group(2))
                iterations.append(itr)
                train_cer.append(cer)

    return {
        "iterations": iterations,
        "train_cer": train_cer,
        "eval_cer": eval_cer,
        "rms_values": rms_values,
        "best_cer": best_cer,
        "best_iters": best_iters,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(data: dict, output_dir: str, log_name: str = "training") -> None:
    """Generate and save training curve plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    iters     = data["iterations"]
    t_cer     = data["train_cer"]
    e_cer_pts = data["eval_cer"]
    rms       = data["rms_values"]
    best_cers = data["best_cer"]
    best_it   = data["best_iters"]

    if not iters:
        logger.error("No training data found in log file")
        return

    # ── Plot 1: CER over iterations ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Tesseract LSTM Training Progress — ECG Meter OCR", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(iters, t_cer, "b-", linewidth=1.5, alpha=0.8, label="Train CER (%)")

    if e_cer_pts:
        e_iters = [p[0] for p in e_cer_pts]
        e_vals  = [p[1] for p in e_cer_pts]
        ax1.plot(e_iters, e_vals, "r-", linewidth=1.5, alpha=0.8, label="Eval CER (%)")

    if best_it and best_cers:
        ax1.scatter(best_it, best_cers, color="green", s=60, zorder=5,
                    label="Best model saved", marker="*")

    # Target line
    ax1.axhline(y=2.0, color="orange", linestyle="--", alpha=0.7, label="Target CER 2%")

    ax1.set_ylabel("Character Error Rate (%)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Smooth trend line
    if len(t_cer) > 10:
        window = max(5, len(t_cer) // 20)
        smoothed = np.convolve(t_cer, np.ones(window)/window, mode="valid")
        smooth_iters = iters[window-1:][:len(smoothed)]
        ax1.plot(smooth_iters, smoothed, "b--", linewidth=2, alpha=0.5, label=f"Smoothed (w={window})")

    # ── Plot 2: RMS error ─────────────────────────────────────────────────────
    ax2 = axes[1]
    if rms:
        ax2.plot(iters[:len(rms)], rms, "purple", linewidth=1.5, alpha=0.8, label="RMS Error (%)")
        ax2.set_ylabel("RMS Error (%)")
        ax2.set_xlabel("Training Iteration")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    else:
        ax2.set_visible(False)
        axes[0].set_xlabel("Training Iteration")

    plt.tight_layout()
    out_path = out / f"{log_name}_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Training curve saved: {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info(f"Training iterations logged: {len(iters)}")
    logger.info(f"Final train CER: {t_cer[-1]:.2f}%")
    if best_cers:
        logger.success(f"Best CER achieved: {min(best_cers):.2f}% at iteration {best_it[best_cers.index(min(best_cers))]}")
    if t_cer[-1] <= 2.0:
        logger.success("✓ Target CER ≤2% achieved!")
    else:
        logger.warning(f"Target CER ≤2% not yet reached (current: {t_cer[-1]:.2f}%)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Tesseract training curves")
    parser.add_argument("--log",    "-l", required=True, help="Training log file")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)

    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        sys.exit(1)

    logger.info(f"Parsing training log: {log_path}")
    data = parse_training_log(str(log_path))
    plot_training_curves(data, args.output, log_name=log_path.stem)


if __name__ == "__main__":
    main()
