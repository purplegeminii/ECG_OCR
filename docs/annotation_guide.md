# Annotation Guide — ECG Meter Images

## Overview

Good ground truth labels are the most important factor in OCR training quality.
This guide covers what to label, how to handle edge cases, and quality standards.

---

## What to Capture in Each Meter Image

For each ECG postpaid meter image, your `.gt.txt` file should contain
**all machine-readable text visible on the meter display**, in reading order
(top-to-bottom, left-to-right).

### Typical ECG Meter Fields

| Field | Example | Notes |
|-------|---------|-------|
| Customer account number | `12345678901` | 11-digit ECG account |
| Meter serial number | `E0040ABC123` | Alphanumeric on meter face |
| Current kWh reading | `003456.5` | Include decimal if shown |
| Date/time of reading | `15/01/2024` | Format varies by meter model |
| Meter brand code | `ACT` or `LG` | Sometimes printed |
| Unit | `kWh` | If printed separately |

### Example Ground Truth Texts

```
# Actaris meter - typical format
12345678901 003456 kWh 15/01/2024

# Landis+Gyr meter - may show more fields
E0040ABC12345 ACCT 12345678901 READ 5432.1 kWh

# Conlog meter - minimal display
003456
```

---

## Annotation Rules

### DO:
- **Include all visible digits** on the meter display
- **Include units** (kWh) if visible on the display
- **Separate fields with a single space** as they appear
- **Be consistent** across images from the same meter type
- **Keep case** as shown (most meters show uppercase)

### DON'T:
- Don't include background text (installation labels, stickers)
- Don't include partially visible characters (cut off by image edge)
- Don't add punctuation not visible on the meter
- Don't guess if a digit is unclear — use `?` for unknowable characters (and then exclude from training set)

---

## Handling Difficult Cases

### Partially Obscured Readings
If part of the display is obscured (hand, glare, shadow):
- Label only the clearly visible portion
- Mark the image as `PARTIAL` in your filename or in a notes file
- These images may be better excluded from training

### Multiple Readings
Some meters cycle through screens. If you have multiple captures of the same
meter in different display states, each image gets its own GT file.

### Faded/Worn Displays
For very degraded displays where you cannot read digits:
- Skip these images for training (they add noise)
- Consider flagging them for a separate "hard examples" validation set

### Date Formats
ECG meters use various date formats. Label exactly as shown:
- `15/01/2024` or `15-01-24` or `01/2024` — whatever is displayed

---

## Quality Standards

A good training dataset should have:

| Criterion | Target |
|-----------|--------|
| Images per meter brand | ≥ 50 |
| GT label accuracy | 100% (no labeling errors) |
| Labels with digit reading | ≥ 95% |
| Average label length | 10-50 characters |
| Variety of lighting conditions | All common conditions represented |
| Variety of image quality | Good, average, and slightly degraded |

---

## Checking Your Labels

Run validation to catch common issues:

```bash
make validate-gt
# or
python scripts/02_annotate.py --validate --gt ground_truth/ --images preprocessed/
```

This will flag:
- Empty labels
- Labels shorter than 4 characters (suspicious)
- Labels with no digits (suspicious for meter readings)
- Images with no corresponding label

---

## Sample Dataset Structure

After labeling, your `ground_truth/` folder should look like:

```
ground_truth/
├── meter_001.gt.txt      # "12345678901 003456 kWh"
├── meter_002.gt.txt      # "23456789012 015234.5 kWh 20/01/2024"
├── meter_003.gt.txt      # "34567890123 087654"
├── meter_001_aug001.gt.txt  # Same as meter_001.gt.txt (augmented copy)
├── meter_001_aug002.gt.txt  # Same as meter_001.gt.txt
...
```
