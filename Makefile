# =============================================================================
# Makefile — ECG OCR Pipeline
# Convenient shortcuts for common pipeline operations
# =============================================================================

.PHONY: help setup \
        preprocess preprocess-single preprocess-debug \
        annotate annotate-studio validate-gt \
        augment augment-preview \
        prepare stats \
        train train-resume \
        evaluate evaluate-compare \
        infer infer-json \
        add-eval add-eval-dir \
        correct find-errors \
        plot-curves \
        test test-cov \
        full-pipeline \
        clean clean-generated clean-models clean-results

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
	@echo "  make correct        Run iterative correction loop (3 rounds + retrain)"
	@echo "  make find-errors    Report high-CER samples without correcting"
	@echo "  make stats          Show dataset statistics"
	@echo "  make add-eval       Add a single image to eval_data/ interactively"
	@echo "  make add-eval-dir   Add a directory of images to eval_data/ interactively"
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
		--compare \
		--config $(CONFIG)

# ─── Step 7: Inference ───────────────────────────────────────────────────────
infer:
	$(PYTHON) $(SCRIPTS)07_inference.py \
		--input raw_images/ \
		--output results/ \
		--format csv \
		--config $(CONFIG)

infer-json:
	$(PYTHON) $(SCRIPTS)07_inference.py \
		--input raw_images/ \
		--output results/ \
		--format json \
		--config $(CONFIG)

# ─── Iterative correction ────────────────────────────────────────────────────
correct:
	$(PYTHON) $(SCRIPTS)08_iterative_correction.py \
		--eval-dir eval_data/ \
		--gt-dir ground_truth/ \
		--corrections corrections/ \
		--rounds 3 \
		--retrain \
		--config $(CONFIG)

find-errors:
	$(PYTHON) $(SCRIPTS)08_iterative_correction.py \
		--find-errors \
		--eval-dir eval_data/ \
		--gt-dir ground_truth/ \
		--threshold 0.10 \
		--config $(CONFIG)

# ─── Add to eval data ─────────────────────────────────────────────────────────
add-eval:
	@read -p "Image path: " path; \
	$(PYTHON) $(SCRIPTS)add_to_eval_data.py \
		--image "$$path" \
		--interactive \
		--config $(CONFIG)

add-eval-dir:
	@read -p "Directory: " dir; \
	$(PYTHON) $(SCRIPTS)add_to_eval_data.py \
		--input "$$dir" \
		--interactive \
		--config $(CONFIG)

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
