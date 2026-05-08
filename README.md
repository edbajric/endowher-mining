# Endowher-Mining

## Short Description
A comparative data mining project on PCOS and endometriosis using separate machine learning pipelines. The goal is to predict each condition independently and then compare shared feature patterns.

## Motivation
This university data mining project studies predictive patterns in two related but distinct gynecological conditions: polycystic ovary syndrome (PCOS) and endometriosis. The goal is to build clean, reproducible starter pipelines that can be extended for rigorous comparative analysis.

## Research Questions
1. Which features best predict PCOS in the Kaggle clinical PCOS dataset?
2. Which features best predict endometriosis in a structured endometriosis dataset?
3. Which shared clinical feature themes appear across both models at interpretation time?
4. Can shared-feature analysis support future women’s health screening tools?

## Datasets
- **PCOS dataset:** Kaggle clinical PCOS dataset (place in `data/raw/pcos/`)
- **Endometriosis dataset:** Structured endometriosis dataset (place in `data/raw/endometriosis/`)

> Scientific constraint: datasets come from different cohorts and are modeled separately. No row-level merge is performed.

## Methods
Each condition has its own train/test split and modeling pipeline:
- Logistic Regression (with scaling)
- Decision Tree
- Random Forest
- Optional XGBoost or Gradient Boosting

## Preprocessing
- Separate preprocessing pipeline for each dataset
- Missing value handling (numeric + categorical)
- Categorical one-hot encoding
- Feature scaling for Logistic Regression
- Train/test split before fit-based preprocessing
- Dataset-specific target column confirmation
- No row-level merge between datasets

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

## Shared-Feature Analysis
After training both models separately, the project compares feature importance across the two pipelines. Shared features are grouped into broader themes such as cycle irregularity, hormonal indicators, pain symptoms, fertility-related factors, and metabolic factors.

## Repository Structure
```text
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

## Installation on macOS and Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name endowher-mining --display-name "endowher-mining"
```

## How to Run
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Put source datasets under:
   - `data/raw/pcos/`
   - `data/raw/endometriosis/`
3. Update target column names and input filenames in `src/run_workflow.py` and `src/utils/config.py`.
4. Run the starter pipeline:
   ```bash
   python -m src.run_workflow
   ```

## Expected Results
- PCOS: baseline around 0.85 accuracy, strong result around 0.90
- Endometriosis: baseline around 0.65 accuracy, strong result around 0.75

## Limitations
- Different cohorts and partially different feature spaces
- No direct patient-level merge
- Public datasets only
- Not a clinical diagnostic tool

## Future Work
- Hyperparameter optimization and cross-validation
- Calibration analysis and threshold optimization
- Robust feature stability checks
- Fairness and subgroup performance analysis
- Statistical comparison of ROC curves across models