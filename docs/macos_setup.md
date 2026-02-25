# macOS Setup Guide

The main `install_dependencies.sh` targets Ubuntu/Debian Linux.
For macOS, follow these steps:

## System Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract + tessdata
brew install tesseract
brew install tesseract-lang

# Download tessdata_best (for fine-tuning)
mkdir -p ~/tessdata_best
curl -L -o ~/tessdata_best/eng.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Install imagemagick + ghostscript
brew install imagemagick ghostscript

# Ruby (usually pre-installed on macOS)
gem install unicode-scripts unicode-categories
```

## Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## tesstrain

tesstrain is a git submodule of this project, so it comes with the repo automatically:

```bash
# When cloning the project for the first time:
git clone --recurse-submodules <your-repo-url>

# If you already cloned without --recurse-submodules:
git submodule update --init
```

## Config Adjustments for macOS

In `config/config.yaml`, update tessdata path:

```yaml
model:
  tessdata_best_dir: /usr/local/share/tessdata  # Homebrew tessdata path
  # OR if using tessdata_best downloaded above:
  # tessdata_best_dir: ~/tessdata_best
```

## Verify Installation

```bash
tesseract --version
python3 -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python3 -c "import cv2; print(cv2.__version__)"
```

## Known macOS Issues

**Tesseract path not found by pytesseract:**
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # Homebrew
# or on Apple Silicon:
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
```

**OpenCV display (for --debug mode):**
macOS may require `brew install qt` for OpenCV GUI windows to work.
