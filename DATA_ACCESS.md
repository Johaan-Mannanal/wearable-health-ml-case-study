# Data Access

## Summary

**This project does not use any real human, patient, or wearable-device data.**
All datasets used for training and evaluation are **synthetically generated in code**
using NumPy random distributions. There is therefore no private dataset to request,
license, or download, you can reproduce every dataset locally by running the code.

## Why there is no real dataset

The project began as an exploration of whether machine-learning models could separate
labeled patterns in wearable-style cardiovascular signals (heart rate, heart-rate
variability, and derived metrics). Rather than collecting real Apple Watch / HealthKit
recordings, which would raise consent, privacy, and licensing obligations, the
authors generated **synthetic samples** whose distributions were hand-designed to
loosely resemble published physiological ranges.

This has two important consequences that are stated plainly throughout the repo:

1. **The reported metrics measure separability of the synthetic distributions, not
   real-world predictive performance.** Because the class distributions were designed
   to differ, high accuracy is expected and does **not** demonstrate that the models
   would work on real people.
2. **No participant/subject structure exists.** Each row is an independent random draw,
   so there is no notion of "the same participant appearing in train and test," and the
   results cannot speak to generalization across individuals.

## How to (re)generate the data

The data is produced deterministically (fixed seed `42`) by the functions in
[`src/data_processing.py`](src/data_processing.py):

```python
from src.data_processing import (
    generate_rhythm_data,
    generate_health_risk_data,
    generate_hrv_pattern_data,
    generate_cardio_fitness_data,
)

df = generate_rhythm_data(n_samples=5000, random_state=42)
```

See [`data/README.md`](data/README.md) for the expected schema of each dataset and the
HealthKit fields the synthetic features are modeled after.

## If you want to extend this with real data

Using real wearable data would require, at minimum:

- Informed consent from every participant and an appropriate ethics/IRB review.
- A lawful basis for processing health data (e.g., HIPAA in the US, GDPR special-category
  data in the EU).
- De-identification and secure storage; **never** commit raw health records to git.
- **Group-aware / participant-level splitting** (e.g., `GroupKFold`) so no individual's
  data appears in both training and testing.

None of the above has been done here, because no real data is involved. See
[`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for what would need to change before this could
become publication-quality research.
