# endowher-mining

## Motivation
This university data mining project studies predictive patterns in two related but distinct gynecological conditions: polycystic ovary syndrome (PCOS) and endometriosis. The goal is to build clean, reproducible starter pipelines that can be extended for rigorous comparative analysis.

## Research Questions
1. Which features best predict PCOS in the Kaggle clinical PCOS dataset?
2. Which features best predict endometriosis in a structured endometriosis dataset?
3. Which shared clinical feature themes appear across both models at interpretation time?

## Datasets
- **PCOS dataset:** Kaggle clinical PCOS dataset (place in `data/raw/pcos/`)
- **Endometriosis dataset:** Structured endometriosis dataset (place in `data/raw/endometriosis/`)

> Scientific constraint: datasets come from different cohorts and are modeled separately. No row-level merge is performed.

## Methods
Each condition has its own train/test split and modeling pipeline:
- Logistic Regression (with scaling)
- Random Forest
- Optional XGBoost (if dependency available)

## Preprocessing
- Missing value handling (numeric + categorical placeholders)
- Categorical one-hot encoding
- Feature scaling for Logistic Regression
- Dataset-specific target column confirmation (manual TODO markers included)

## Evaluation Metrics
For each model:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

Outputs are saved to:
- `reports/figures/` (plots)
- `reports/tables/` (metrics/importance tables)

## Repository Structure
```
endowher-mining/
├── data/
│   ├── raw/
│   │   ├── pcos/
│   │   └── endometriosis/
│   ├── processed/
│   └── metadata/
├── docs/
│   └── projectplan.md
├── notebooks/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
└── src/
    ├── data/
    ├── explain/
    ├── models/
    ├── utils/
    └── run_workflow.py
```

## How to Run
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put source datasets under:
   - `data/raw/pcos/`
   - `data/raw/endometriosis/`
4. Update target column names and input filenames in `src/run_workflow.py` and `src/utils/config.py`.
5. Run the starter pipeline:
   ```bash
   python -m src.run_workflow
   ```

## Future Work
- Hyperparameter optimization and cross-validation
- Calibration analysis and threshold optimization
- Robust feature stability checks
- Fairness and subgroup performance analysis
- Statistical comparison of ROC curves across models
