# Data Directory

This directory intentionally contains **no data files**. Every dataset in this project is
**synthetic** and generated on demand by [`src/data_processing.py`](../src/data_processing.py)
with a fixed random seed, so nothing needs to be downloaded or stored here.

See [`../DATA_ACCESS.md`](../DATA_ACCESS.md) for why there is no real dataset.

## Expected schema of each synthetic dataset

### 1. Heart-rhythm dataset — `generate_rhythm_data()`
Binary classification: `label` 0 = Normal, 1 = Irregular.

| Column | Type | Approx. range | Modeled after (HealthKit) |
|--------|------|---------------|---------------------------|
| `mean_heart_rate` | float | 40–120 bpm | `HKQuantityTypeIdentifierHeartRate` |
| `std_heart_rate`  | float | 1–20 bpm    | derived from `HeartRateVariabilitySDNN` |
| `pnn50`           | float | 0–0.5       | derived from `HeartRateVariabilityRMSSD` |
| `label`           | int   | {0, 1}      | synthetic ground truth |

### 2. Health-risk dataset — `generate_health_risk_data()`
Binary classification: `risk_level` 0 = Low, 1 = High.

| Column | Type | Approx. range | Notes |
|--------|------|---------------|-------|
| `average_heart_rate` | float | 40–120 bpm | continuous HR |
| `hrv_mean` | float | 10–100 ms | RMSSD-style HRV |
| `respiratory_rate` | float | 8–25 /min | sleep-derived |
| `activity_level` | float | 0–1000 | steps/movement proxy |
| `sleep_quality` | float | 0–1 | normalized score |
| `stress_indicator` | float | 0–1 | normalized score |
| `hr_hrv_ratio` | float | derived | `average_heart_rate / (hrv_mean + 1)` |
| `recovery_score` | float | derived | `sleep_quality * hrv_mean / 50` |
| `risk_level` | int | {0, 1} | synthetic ground truth |

### 3. HRV-pattern dataset — `generate_hrv_pattern_data()`
4-class classification: `label` ∈ {normal, afib, bradycardia, tachycardia}.
13 engineered features summarizing a short RR-interval sequence:
`mean_rr, std_rr, min_rr, max_rr, q25_rr, q75_rr, mean_diff_rr, std_diff_rr, rmssd,
pnn50, low_freq_power, mid_freq_power, high_freq_power`.

### 4. Cardiovascular-fitness dataset — `generate_cardio_fitness_data()`
Regression with three continuous targets (`fitness_level`, `vo2max`, `cardiovascular_age`).

| Column | Type | Notes |
|--------|------|-------|
| `age` | float | years |
| `resting_hr` | float | bpm |
| `hrr_1min` | float | 1-minute heart-rate recovery |
| `rmssd` | float | HRV |
| `fitness_level` / `vo2max` / `cardiovascular_age` | float | synthetic targets |

## Important
These schemas describe **synthetic** features designed to *resemble* the named HealthKit
quantities. They are not recordings from any device or person, and the mapping to real
HealthKit identifiers is aspirational documentation only.
