#!/usr/bin/env bash
# =============================================================================
# install_dependencies.sh
# System-level dependency installer for ECG OCR Pipeline
# Tested on Ubuntu 20.04 / 22.04
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo "============================================================"
echo "  ECG OCR Pipeline — Dependency Installer"
echo "============================================================"
echo ""

# ── Check OS ──────────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    log_error "This script requires Linux (Ubuntu recommended). For macOS, see docs/macos_setup.md"
fi

# ── System packages ───────────────────────────────────────────────────────────
log_info "Updating package lists..."
sudo apt-get update -qq

log_info "Installing Tesseract OCR and dev libraries..."
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev

# Install tessdata_best (higher quality base models for fine-tuning)
log_info "Installing tessdata_best..."
sudo apt-get install -y tesseract-ocr-eng || true
TESSDATA_BEST_DIR="/usr/share/tesseract-ocr/5/tessdata_best"
sudo mkdir -p "$TESSDATA_BEST_DIR"
if [ ! -f "$TESSDATA_BEST_DIR/eng.traineddata" ]; then
    log_info "Downloading eng.traineddata (best) from GitHub..."
    sudo curl -L -o "$TESSDATA_BEST_DIR/eng.traineddata" \
        "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
    log_success "Downloaded eng.traineddata (best)"
else
    log_success "eng.traineddata (best) already present"
fi

log_info "Installing tesstrain dependencies..."
sudo apt-get install -y \
    make \
    wget \
    bc \
    git \
    python3-pip \
    python3-venv

log_info "Installing image processing libraries..."
sudo apt-get install -y \
    libopencv-dev \
    ghostscript \
    imagemagick \
    libmagickwand-dev \
    ffmpeg  # needed by some OpenCV builds

log_info "Installing Ruby (for dshea89 unicharset generation)..."
sudo apt-get install -y ruby ruby-dev
gem install unicode-scripts unicode-categories 2>/dev/null || \
    log_warn "Ruby gems install failed — dshea89 pipeline step may not work"

# ── Clone tesstrain ────────────────────────────────────────────────────────────
if [ ! -d "tesstrain" ]; then
    log_info "Cloning tesstrain repository..."
    git clone https://github.com/tesseract-ocr/tesstrain.git
    log_success "tesstrain cloned"
else
    log_success "tesstrain already present"
fi

# ── Python virtual environment ────────────────────────────────────────────────
log_info "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip -q

log_info "Installing Python requirements..."
pip install -r requirements.txt -q

log_success "Python dependencies installed"

# ── Verify Tesseract ──────────────────────────────────────────────────────────
log_info "Verifying Tesseract installation..."
TESS_VERSION=$(tesseract --version 2>&1 | head -1)
if [[ "$TESS_VERSION" == *"tesseract"* ]]; then
    log_success "Tesseract: $TESS_VERSION"
else
    log_error "Tesseract not found after installation"
fi

# ── Verify Python packages ────────────────────────────────────────────────────
log_info "Verifying key Python packages..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && log_success "OpenCV OK"
python3 -c "import pytesseract; print('pytesseract OK')" && log_success "pytesseract OK"
python3 -c "import albumentations; print('albumentations OK')" && log_success "albumentations OK"

# ── Copy config ───────────────────────────────────────────────────────────────
if [ ! -f "config/config.yaml" ]; then
    cp config/config.example.yaml config/config.yaml
    log_success "Config file created at config/config.yaml — please review before running"
else
    log_warn "config/config.yaml already exists, not overwriting"
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}Installation complete!${NC}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review config/config.yaml"
echo "  2. Place your meter images in raw_images/"
echo "  3. Run: python scripts/01_preprocess.py --input raw_images/ --output preprocessed/"
echo ""
