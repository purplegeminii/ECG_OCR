#!/usr/bin/env bash
# =============================================================================
# 05_run_training.sh
# Tesseract LSTM Fine-Tuning via tesstrain submodule
#
# Delegates entirely to tesstrain's Makefile, which handles:
#   - Generating .lstmf files from .tif + .gt.txt pairs
#   - Extracting the LSTM layer from the base model
#   - Running lstmtraining with checkpointing
#   - Selecting the best checkpoint
#   - Packaging the final .traineddata
#
# Training data is read from:
#   tesstrain/data/<MODEL_NAME>-ground-truth/
# which was populated by 04_prepare_training_data.py.
#
# The final model is copied from tesstrain's output into:
#   models/<MODEL_NAME>/tessdata/<MODEL_NAME>.traineddata
#
# Prerequisites:
#   - git submodule update --init    (initialise tesstrain)
#   - python scripts/04_prepare_training_data.py  (populate ground-truth dir)
#
# Usage:
#   bash scripts/05_run_training.sh
#   bash scripts/05_run_training.sh --iterations 5000
#   bash scripts/05_run_training.sh --lr 0.00005
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()    { echo -e "${BLUE}[$(date +%H:%M:%S)][INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[$(date +%H:%M:%S)][OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[$(date +%H:%M:%S)][WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[$(date +%H:%M:%S)][ERROR]${NC} $1"; exit 1; }
log_section() { echo -e "\n${CYAN}═══ $1 ═══${NC}\n"; }

# ─── Load config values from YAML ─────────────────────────────────────────────
parse_yaml_value() {
    local key="$1"
    local file="${2:-config/config.yaml}"
    grep -E "^\s*${key}:" "$file" 2>/dev/null | head -1 \
        | sed 's/.*: //' | sed 's/#.*//' | tr -d '"' | tr -d "'" | xargs
}

MODEL_NAME=$(parse_yaml_value "name"           config/config.yaml); MODEL_NAME="${MODEL_NAME:-ecg_meter}"
BASE_MODEL=$(parse_yaml_value "base_model"     config/config.yaml); BASE_MODEL="${BASE_MODEL:-eng}"
MAX_ITER=$(parse_yaml_value   "max_iterations" config/config.yaml); MAX_ITER="${MAX_ITER:-10000}"
LR=$(parse_yaml_value         "learning_rate"  config/config.yaml); LR="${LR:-0.0001}"

# ─── CLI overrides ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) MAX_ITER="$2";  shift 2 ;;
        --lr)         LR="$2";        shift 2 ;;
        --model)      MODEL_NAME="$2"; shift 2 ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(pwd)"
TESSTRAIN_DIR="${PROJECT_DIR}/tesstrain"
GT_DIR="${TESSTRAIN_DIR}/data/${MODEL_NAME}-ground-truth"
FINAL_MODEL_DIR="${PROJECT_DIR}/models/${MODEL_NAME}/tessdata"
LOGS_DIR="${PROJECT_DIR}/logs"

# Locate tessdata_best (where eng.traineddata lives for fine-tuning)
TESSDATA_BEST=""
for candidate in \
    "/usr/share/tesseract-ocr/5/tessdata_best" \
    "/usr/share/tesseract-ocr/4.00/tessdata_best" \
    "/usr/local/share/tessdata" \
    "${HOME}/tessdata_best"; do
    if [ -f "${candidate}/${BASE_MODEL}.traineddata" ]; then
        TESSDATA_BEST="$candidate"
        break
    fi
done

# ─── Pre-flight checks ────────────────────────────────────────────────────────
log_section "Pre-flight Checks"

[ -d "$TESSTRAIN_DIR" ] || log_error \
    "tesstrain submodule not found at ${TESSTRAIN_DIR}.\n  Run: git submodule update --init"

[ -f "${TESSTRAIN_DIR}/Makefile" ] || log_error \
    "tesstrain Makefile missing. Submodule may be empty.\n  Run: git submodule update --init"

[ -d "$GT_DIR" ] || log_error \
    "Ground-truth directory not found: ${GT_DIR}\n  Run: python scripts/04_prepare_training_data.py"

GT_COUNT=$(find "$GT_DIR" -name "*.tif" | wc -l)
[ "$GT_COUNT" -gt 0 ] || log_error "No .tif files found in ${GT_DIR}"

[ -n "$TESSDATA_BEST" ] || log_error \
    "Could not find ${BASE_MODEL}.traineddata in any tessdata_best location.\n  Run: bash scripts/install_dependencies.sh"

log_info "Model name:      ${MODEL_NAME}"
log_info "Base model:      ${BASE_MODEL}"
log_info "Max iterations:  ${MAX_ITER}"
log_info "Learning rate:   ${LR}"
log_info "Training images: ${GT_COUNT}"
log_info "tessdata_best:   ${TESSDATA_BEST}"
log_info "Ground-truth:    ${GT_DIR}"

if [ "$GT_COUNT" -lt 50 ]; then
    log_warn "Only ${GT_COUNT} images. Recommend ≥100 for meaningful fine-tuning."
fi

mkdir -p "$FINAL_MODEL_DIR" "$LOGS_DIR"

LOG_FILE="${LOGS_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
log_info "Training log: ${LOG_FILE}"

# ─── Run tesstrain ────────────────────────────────────────────────────────────
log_section "Running tesstrain"

# Activate virtual environment if it exists
VENV_DIR="${PROJECT_DIR}/.venv"
if [ -d "${VENV_DIR}" ]; then
    log_info "Activating virtual environment..."
    source "${VENV_DIR}/bin/activate"
fi

log_info "Invoking tesstrain Makefile..."
log_info "Monitor progress: tail -f ${LOG_FILE} | grep -E '(BEST|CER|error)'"

# Use gmake on macOS (Homebrew), fallback to make
MAKE_CMD="make"
if command -v gmake &> /dev/null; then
    MAKE_CMD="gmake"
fi

"${MAKE_CMD}" -C "$TESSTRAIN_DIR" training \
    MODEL_NAME="$MODEL_NAME" \
    START_MODEL="$BASE_MODEL" \
    TESSDATA="$TESSDATA_BEST" \
    GROUND_TRUTH_DIR="$GT_DIR" \
    MAX_ITERATIONS="$MAX_ITER" \
    LEARNING_RATE="$LR" \
    PSM=6 \
    2>&1 | tee "$LOG_FILE"

log_success "tesstrain finished"

# ─── Copy final model to project models/ ──────────────────────────────────────
log_section "Installing Model"

TESSTRAIN_OUTPUT="${TESSTRAIN_DIR}/data/${MODEL_NAME}.traineddata"

if [ ! -f "$TESSTRAIN_OUTPUT" ]; then
    # tesstrain may place it under data/<MODEL_NAME>/
    TESSTRAIN_OUTPUT="${TESSTRAIN_DIR}/data/${MODEL_NAME}/${MODEL_NAME}.traineddata"
fi

[ -f "$TESSTRAIN_OUTPUT" ] || log_error \
    "Expected output model not found after training.\n  Looked at: ${TESSTRAIN_OUTPUT}"

FINAL_MODEL="${FINAL_MODEL_DIR}/${MODEL_NAME}.traineddata"
cp "$TESSTRAIN_OUTPUT" "$FINAL_MODEL"
log_success "Model installed: ${FINAL_MODEL}"

# ─── Quick smoke test ─────────────────────────────────────────────────────────
log_section "Quick Smoke Test"

SAMPLE_TIF=$(find "$GT_DIR" -name "*.tif" | head -1)
if [ -n "$SAMPLE_TIF" ]; then
    log_info "Testing on: $(basename ${SAMPLE_TIF})"
    tesseract "$SAMPLE_TIF" stdout \
        --tessdata-dir "$FINAL_MODEL_DIR" \
        -l "$MODEL_NAME" \
        --psm 6 2>/dev/null || true
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo -e "  ${GREEN}Training complete!${NC}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Model: ${FINAL_MODEL}"
echo "  Log:   ${LOG_FILE}"
echo ""
echo "  Next steps:"
echo "    Plot curves: python scripts/plot_training_curves.py --log ${LOG_FILE}"
echo "    Evaluate:    python scripts/06_evaluate.py"
echo "    Inference:   python scripts/07_inference.py --input raw_images/new/"
echo ""
echo "  To install system-wide:"
echo "    sudo cp ${FINAL_MODEL} ~/tessdata_best/"
echo ""
