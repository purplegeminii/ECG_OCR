# =============================================================================
# Makefile — ECG OCR Pipeline
# Convenient shortcuts for common pipeline operations
# =============================================================================

.PHONY: help setup preprocess annotate augment prepare train evaluate infer \
        test clean full-pipeline

PYTHON    := python3
SCRIPTS   := scripts/
CONFIG    := config/config.yaml

# ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ECG Postpaid Meter OCR Pipeline"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install all dependencies"
	@echo ""
	@echo "Pipeline (run in order):"
	@echo "  make preprocess     Clean and prepare raw images"
	@echo "  make annotate       Launch annotation tool"
	@echo "  make augment        Augment training dataset"
	@echo "  make prepare        Prepare tesstrain-ready files"
	@echo "  make train          Fine-tune Tesseract model"
	@echo "  make evaluate       Evaluate model accuracy"
	@echo "  make infer          Run inference on new images"
	@echo ""
	@echo "Utilities:"
	@echo "  make test           Run unit tests"
	@echo "  make validate-gt    Validate ground truth files"
	@echo "  make plot-curves    Plot training CER curves"
	@echo "  make correct        Run iterative correction loop"
	@echo "  make stats          Show dataset statistics"
	@echo "  make clean          Remove generated files (keep raw images)"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make full-pipeline  Run preprocess→augment→prepare→train→evaluate"
	@echo ""

# ─── Setup ───────────────────────────────────────────────────────────────────
setup:
	bash scripts/install_dependencies.sh

# ─── Step 1: Preprocess ──────────────────────────────────────────────────────
preprocess:
	$(PYTHON) $(SCRIPTS)01_preprocess.py \
		--input raw_images/ \
		--output preprocessed/ \
		--config $(CONFIG)

preprocess-single:
	@read -p "Image path: " path; \
	$(PYTHON) $(SCRIPTS)01_preprocess.py \
		--single "$$path" \
		--output preprocessed/ \
		--config $(CONFIG)

preprocess-debug:
	$(PYTHON) $(SCRIPTS)01_preprocess.py \
		--input raw_images/ \
		--output preprocessed/ \
		--debug

# ─── Step 2: Annotate ────────────────────────────────────────────────────────
annotate:
	$(PYTHON) $(SCRIPTS)02_annotate.py --manual \
		--images preprocessed/ \
		--output ground_truth/

annotate-studio:
	$(PYTHON) $(SCRIPTS)02_annotate.py --launch \
		--images preprocessed/

validate-gt:
	$(PYTHON) $(SCRIPTS)02_annotate.py --validate \
		--gt ground_truth/ \
		--images preprocessed/

# ─── Step 3: Augment ─────────────────────────────────────────────────────────
augment:
	$(PYTHON) $(SCRIPTS)03_augment.py \
		--input preprocessed/ \
		--output augmented/ \
		--gt ground_truth/ \
		--out-gt ground_truth/ \
		--factor 5 \
		--config $(CONFIG)

augment-preview:
	$(PYTHON) $(SCRIPTS)03_augment.py \
		--input preprocessed/ \
		--preview

# ─── Step 4: Prepare training data ───────────────────────────────────────────
prepare:
	$(PYTHON) $(SCRIPTS)04_prepare_training_data.py \
		--source augmented/ \
		--gt ground_truth/ \
		--tesstrain-dir tesstrain/ \
		--config $(CONFIG)

stats:
	$(PYTHON) $(SCRIPTS)04_prepare_training_data.py \
		--source augmented/ \
		--gt ground_truth/ \
		--stats-only

# ─── Step 5: Train ───────────────────────────────────────────────────────────
train:
	bash $(SCRIPTS)05_run_training.sh

train-resume:
	bash $(SCRIPTS)05_run_training.sh --resume

# ─── Step 6: Evaluate ────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) $(SCRIPTS)06_evaluate.py \
		--test-dir eval_data/ \
		--gt-dir ground_truth/ \
		--output results/ \
		--config $(CONFIG)

evaluate-compare:
	$(PYTHON) $(SCRIPTS)06_evaluate.py \
		--test-dir eval_data/ \
		--gt-dir ground_truth/ \
		--output results/ \
		--compare

# ─── Step 7: Inference ───────────────────────────────────────────────────────
infer:
	$(PYTHON) $(SCRIPTS)07_inference.py \
		--input raw_images/ \
		--output results/ \
		--format csv

infer-json:
	$(PYTHON) $(SCRIPTS)07_inference.py \
		--input raw_images/ \
		--output results/ \
		--format json

# ─── Iterative correction ────────────────────────────────────────────────────
correct:
	$(PYTHON) $(SCRIPTS)08_iterative_correction.py \
		--eval-dir eval_data/ \
		--gt-dir ground_truth/ \
		--rounds 3

find-errors:
	$(PYTHON) $(SCRIPTS)08_iterative_correction.py \
		--find-errors \
		--threshold 0.10

# ─── Plots ───────────────────────────────────────────────────────────────────
plot-curves:
	@latest_log=$$(ls -t logs/training_*.log 2>/dev/null | head -1); \
	if [ -n "$$latest_log" ]; then \
		$(PYTHON) $(SCRIPTS)plot_training_curves.py --log "$$latest_log" --output results/; \
	else \
		echo "No training log found in logs/"; \
	fi

# ─── Tests ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=scripts --cov-report=html --cov-report=term

# ─── Full pipeline ────────────────────────────────────────────────────────────
full-pipeline: preprocess augment prepare train evaluate
	@echo ""
	@echo "Full pipeline complete!"
	@echo "Results saved to: results/"

# ─── Clean ───────────────────────────────────────────────────────────────────
clean-generated:
	rm -rf preprocessed/* augmented/* tesstrain/data/
	@echo "Cleaned generated files (raw_images and ground_truth preserved)"

clean-models:
	rm -rf models/*/checkpoints/* models/*/extracted/*
	@echo "Cleaned model checkpoints (final models preserved)"

clean-results:
	rm -rf results/*
	@echo "Cleaned results"

clean: clean-generated clean-results
	@echo "Clean complete"
