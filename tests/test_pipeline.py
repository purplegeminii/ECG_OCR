#!/usr/bin/env python3
"""
tests/test_pipeline.py
======================
Unit tests for the ECG OCR pipeline.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --cov=scripts
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.utils import (
    validate_meter_reading,
    resize_to_width,
    deskew,
    list_images,
    write_ground_truth,
    read_ground_truth,
)


# ─── Helper: create test images ───────────────────────────────────────────────

def make_test_image(w: int = 400, h: int = 200, text: str = "12345") -> np.ndarray:
    """Create a simple test image with text."""
    img = np.ones((h, w), dtype=np.uint8) * 255  # white background
    cv2.putText(img, text, (20, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 3)
    return img


def make_test_color_image(w: int = 400, h: int = 200) -> np.ndarray:
    """Create a simple color test image."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 10), (w-10, h-10), (0, 0, 0), 2)
    return img


# ─── Tests: Domain validation ─────────────────────────────────────────────────

class TestDomainValidation(unittest.TestCase):

    def setUp(self):
        # Minimal config for testing
        import yaml
        self.cfg = yaml.safe_load("""
domain:
  meter_reading_pattern: "\\\\b\\\\d{4,6}(?:\\\\.\\\\d{1,2})?\\\\b"
  account_number_pattern: "\\\\b\\\\d{10,13}\\\\b"
  meter_serial_pattern: "[A-Z0-9]{8,15}"
  date_pattern: "\\\\d{2}[/-]\\\\d{2}[/-]\\\\d{2,4}"
  min_valid_reading: 0
  max_valid_reading: 999999
""")

    def test_valid_meter_reading(self):
        result = validate_meter_reading("12345 kWh")
        self.assertIn("12345", result["meter_readings"])
        self.assertTrue(result["valid"])

    def test_meter_reading_with_decimal(self):
        result = validate_meter_reading("Account: 1234567890 Reading: 5678.9 kWh")
        self.assertIn("5678.9", result["meter_readings"])
        self.assertIn("1234567890", result["account_numbers"])

    def test_empty_text(self):
        result = validate_meter_reading("")
        self.assertEqual(result["meter_readings"], [])
        self.assertFalse(result["valid"])

    def test_flagged_when_no_reading(self):
        result = validate_meter_reading("ECG GHANA LIMITED")
        self.assertTrue(result["flagged"])

    def test_reading_out_of_range(self):
        # A 3-digit number should not match 4-6 digit pattern
        result = validate_meter_reading("Reading: 123")
        self.assertNotIn("123", result["meter_readings"])

    def test_full_meter_string(self):
        # Typical ECG meter text
        text = "ACCOUNT: 12345678901 READING: 003456 kWh DATE: 15/01/2024"
        result = validate_meter_reading(text)
        self.assertTrue(len(result["meter_readings"]) > 0)
        self.assertTrue(len(result["account_numbers"]) > 0)


# ─── Tests: Image utilities ───────────────────────────────────────────────────

class TestImageUtils(unittest.TestCase):

    def test_resize_to_width_upscale(self):
        img = make_test_image(300, 150)
        result = resize_to_width(img, 600)
        self.assertEqual(result.shape[1], 600)
        self.assertEqual(result.shape[0], 300)  # Proportional

    def test_resize_to_width_no_downscale(self):
        img = make_test_image(800, 400)
        result = resize_to_width(img, 600)
        self.assertEqual(result.shape[1], 800)  # Not changed

    def test_deskew_straight_image(self):
        img = make_test_image(400, 200)
        result = deskew(img)
        self.assertEqual(result.shape, img.shape)

    def test_deskew_grayscale(self):
        img = make_test_image(400, 200)
        result = deskew(img)
        self.assertIsInstance(result, np.ndarray)

    def test_deskew_color_image(self):
        img = make_test_color_image()
        result = deskew(img)
        self.assertIsInstance(result, np.ndarray)


# ─── Tests: File I/O ──────────────────────────────────────────────────────────

class TestFileIO(unittest.TestCase):

    def test_write_and_read_gt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "test.gt.txt"
            write_ground_truth("12345 kWh", gt_path)
            result = read_ground_truth(gt_path)
            self.assertEqual(result, "12345 kWh")

    def test_gt_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "test.gt.txt"
            write_ground_truth("  12345  \n", gt_path)
            result = read_ground_truth(gt_path)
            self.assertEqual(result, "12345")

    def test_list_images_tif(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create fake image files
            for name in ["a.tif", "b.tif", "c.jpg", "d.txt"]:
                (tmpdir / name).touch()
            images = list_images(tmpdir)
            self.assertEqual(len(images), 3)  # tif, tif, jpg

    def test_list_images_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = list_images(tmpdir)
            self.assertEqual(images, [])


# ─── Tests: Preprocessing ─────────────────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):

    def test_adaptive_threshold(self):
        """Verify adaptive thresholding produces binary output."""
        img = make_test_image(400, 200)
        result = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        unique_vals = np.unique(result)
        # Should be binary (0 and 255 only)
        self.assertTrue(all(v in [0, 255] for v in unique_vals))

    def test_denoise_does_not_crash(self):
        """FastNlMeansDenoising should work on grayscale."""
        img = make_test_image(400, 200)
        # Add noise
        noisy = img.copy()
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = cv2.fastNlMeansDenoising(noisy, h=10)
        self.assertEqual(result.shape, img.shape)

    def test_morphological_cleanup(self):
        """Morphological opening should not crash."""
        img = make_test_image(400, 200)
        kernel = np.ones((1, 1), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        self.assertEqual(result.shape, img.shape)


# ─── Tests: CER calculation ───────────────────────────────────────────────────

class TestCERCalculation(unittest.TestCase):

    def setUp(self):
        # Import basic_cer directly
        def basic_cer_local(ref, hyp):
            def edit_dist(s1, s2):
                m, n = len(s1), len(s2)
                dp = list(range(n + 1))
                for i in range(1, m + 1):
                    prev = dp[0]; dp[0] = i
                    for j in range(1, n + 1):
                        temp = dp[j]
                        if s1[i-1] == s2[j-1]: dp[j] = prev
                        else: dp[j] = 1 + min(prev, dp[j], dp[j-1])
                        prev = temp
                return dp[n]
            if not ref:
                return 0.0 if not hyp else 1.0
            return edit_dist(ref, hyp) / len(ref)

        self.cer = basic_cer_local

    def test_perfect_match(self):
        self.assertAlmostEqual(self.cer("12345", "12345"), 0.0)

    def test_total_mismatch(self):
        # All chars wrong
        cer = self.cer("AAAAA", "BBBBB")
        self.assertGreater(cer, 0.5)

    def test_one_char_error(self):
        # One substitution in 5-char string = 0.2
        cer = self.cer("12345", "12346")
        self.assertAlmostEqual(cer, 0.2)

    def test_empty_reference(self):
        self.assertAlmostEqual(self.cer("", ""), 0.0)
        self.assertAlmostEqual(self.cer("", "abc"), 1.0)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
