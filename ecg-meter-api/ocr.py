import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path
# import os

# Point to your trained model (relative to this script's location)
TESSDATA_DIR = Path(__file__).parent / "model"
MODEL_NAME   = "ecg_meter"

TESS_CONFIG = (
    f"--oem 1 --psm 6 "
    f"--tessdata-dir {TESSDATA_DIR} "
    # f"-c tessedit_char_whitelist=0123456789.-kWhKWH/ "
    f"-c tessedit_char_whitelist=0123456789.-kWhKWH/ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz "
)

def preprocess(image_bytes: bytes) -> np.ndarray:
    """Clean the uploaded image before OCR."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if small
    h, w = gray.shape
    if w < 1000:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary   = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary

def extract_reading(text: str) -> dict:
    """Pull structured fields out of raw OCR text."""
    return {
        "meter_readings":  re.findall(r"\b\d{4,6}(?:\.\d{1,2})?\b", text),
        "meter_serials":   re.findall(r"[A-Z0-9]{8,15}", text),
        "account_numbers": re.findall(r"\b\d{10,13}\b", text),
        "dates":           re.findall(r"\d{2}[/-]\d{2}[/-]\d{2,4}", text),
    }

def run_ocr(image_bytes: bytes) -> dict:
    processed = preprocess(image_bytes)
    raw_text  = pytesseract.image_to_string(
        processed, lang=MODEL_NAME, config=TESS_CONFIG
    ).strip()

    # Confidence data
    data     = pytesseract.image_to_data(
        processed, lang=MODEL_NAME, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    confs    = [int(c) for c in data["conf"] if int(c) >= 0]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0

    fields   = extract_reading(raw_text)
    flagged  = mean_conf < 60 or not fields["meter_readings"]

    return {
        "raw_text":        raw_text,
        "meter_readings":  fields["meter_readings"],
        "meter_serials":   fields["meter_serials"],
        "account_numbers": fields["account_numbers"],
        "dates":           fields["dates"],
        "confidence":      mean_conf,
        "flagged":         flagged,
    }