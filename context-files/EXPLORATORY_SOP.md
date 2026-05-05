# Exploratory Analysis SOP
*Guidelines for Independent WiFi CSI Human Activity Detection Experiments*

## Purpose
This document defines the lightweight, flexible standard operating procedure for our independent exploratory analysis of the WiMANS dataset. Unlike strict, production-level pipelines, this SOP is designed to encourage rapid iteration and discovery while maintaining sufficient traceability for mentoring and review.

## 1. Naming Conventions
To keep runs organized without excessive overhead, all automated experiment runs must be saved under a unique identifier.

**Format:** `YYYYMMDD_HHMM_[model/method]_[key_variables]`
**Example:** `20260420_1640_baseline_cnn_sanitized`

## 2. Folder Structure
Automated scripts must isolate their outputs to prevent overwriting and allow easy comparison.

```text
/
├── outputs/
│   └── <run_id>/           # Logs, plots, and run-level reports
├── saved_models/
│   └── <run_id>/           # Model checkpoints and weights
```

## 3. Documentation Strategy
Documentation is split into two levels to balance speed and traceability:

### Macro Documentation (Project Level)
*   **`proposed_experiments.md`**: The master list of ideas and their current statuses. Used to plan ahead and discuss progress with mentors.
*   **`decision_log.md`**: A chronological diary. Before making a major pivot or starting a new phase, log *what* you are trying, *why*, and the eventual *outcome*.

### Micro Documentation (Run Level)
*   **`config.json`**: Saved automatically in `outputs/<run_id>/` to capture the exact hyperparameter values and seeds used for that run.
*   **`report.md`**: Saved automatically in `outputs/<run_id>/`. A brief summary of the final metrics (accuracy, loss) and a confusion matrix or relevant plots.

## 4. Agility over Perfection
*   Treat failed experiments as valuable data. Log why an approach failed in the `decision_log.md` and move on.
*   Code can be "hacky" during initial exploration. Once an approach proves valuable, refactor it into clean, reusable modules.
