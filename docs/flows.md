## Runtime Sequence — Training Pipeline
Execution order when training the model.

```mermaid
sequenceDiagram

    participant User
    participant Pre as 01_preprocess.py
    participant Ann as 02_annotate.py
    participant Aug as 03_augment.py
    participant Prep as 04_prepare_training_data.py
    participant Train as tesstrain
    participant Model as models/

    User->>Pre: Preprocess images
    Pre->>Ann: Send cleaned images
    Ann->>User: Manual annotation
    User->>Aug: Labeled data
    Aug->>Prep: Augmented dataset
    Prep->>Train: Training pairs
    Train->>Model: Trained .traineddata
    Model->>User: Model ready
```

## Runtime Sequence — Batch Inference
Execution order for batch processing.

```mermaid
sequenceDiagram

    participant User
    participant Inf as 07_inference.py
    participant Model
    participant Results as results/
    participant Eval as 06_evaluate.py
    participant IterCorr as 08_iterative_correction.py
    participant Corrections as corrections/

    User->>Inf: Run batch inference
    Inf->>Model: Load ecg_meter.traineddata
    Model->>Inf: Model loaded
    Inf->>Results: Write predictions
    Results->>Eval: Run evaluation
    Eval->>User: Reports & plots
    
    alt Errors Found
        User->>IterCorr: Analyze errors
        IterCorr->>Results: Find worst performers
        IterCorr->>User: Present for review
        User->>Corrections: Save corrected labels
        Corrections->>Model: Incremental retrain
    end
```

## Runtime Sequence — Production API
Execution order for real-time API requests.

```mermaid
sequenceDiagram

    participant Client
    participant API as Flask App
    participant OCR as ocr.py
    participant Model

    Client->>API: POST /read-meter (image)
    API->>OCR: preprocess_and_ocr(image)
    OCR->>Model: pytesseract.image_to_string()
    Model->>OCR: Raw text
    OCR->>API: Cleaned result
    API->>Client: JSON response
```