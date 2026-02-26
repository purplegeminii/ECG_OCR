## Training Data Pipeline
How your dataset is built.

```mermaid
flowchart TD

    A[raw_images/] --> B[Preprocess]
    B --> C[preprocessed/]

    C --> D[Manual Annotation]
    D --> E[ground_truth/]

    E --> F[Augmentation]
    F --> G[augmented/]

    G --> H[Prepare Training Data]
    H --> I[tesstrain Format]

    I --> J[Model Training]
    J --> K[models/]
```