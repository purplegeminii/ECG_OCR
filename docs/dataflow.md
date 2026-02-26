## Data Flow Diagram
Where files go and how they transform.

```mermaid
flowchart LR

    RAW[raw_images/]
    PREP[preprocessed/]
    GT[ground_truth/]
    AUG[augmented/]
    MOD[models/]
    RES[results/]
    LOG[logs/]

    RAW --> PREP
    PREP --> GT
    GT --> AUG
    AUG --> MOD
    MOD --> RES

    PREP --> LOG
    GT --> LOG
    AUG --> LOG
    MOD --> LOG
    RES --> LOG
```