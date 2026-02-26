## OCR & Image Processing Pipeline
What happens to each ECG image.

```mermaid
flowchart TD

    A[Raw ECG Image] --> B[Grayscale]
    B --> C[ROI Extraction]
    C --> D[Perspective Correction]
    D --> E[Deskew]
    E --> F[Resize to 300+ DPI]
    
    F --> G[Noise Removal]
    G --> H[Adaptive Thresholding]

    H --> I[Save as TIFF]
    I --> J[Tesseract OCR]

    J --> K[Raw Text]
    K --> L[Post-processing / Cleanup]
    L --> M[Final Output]
```