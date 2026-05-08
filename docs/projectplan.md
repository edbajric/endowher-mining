# Project Plan: Comparative Mining for PCOS and Endometriosis

## Objective
Develop two separate, reproducible binary classification pipelines:
1. PCOS prediction from a Kaggle clinical dataset
2. Endometriosis prediction from a structured endometriosis dataset

The datasets represent different cohorts and **must not** be merged row-wise. Comparison occurs only after model training through shared-feature interpretation.

## Research Questions
- Which clinical variables are most predictive in each condition-specific cohort?
- How do model performance and feature importance differ between PCOS and endometriosis pipelines?
- Which shared clinical themes emerge when comparing top features across both models?

## Planned Workflow
1. Ingest each dataset independently.
2. Validate schema and manually confirm target columns.
3. Preprocess with imputation, encoding, and scaling where appropriate.
4. Train baseline Logistic Regression and Random Forest models.
5. Evaluate with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
6. Save figures/tables for reproducibility.
7. Compare shared-feature importance mappings at interpretation stage.

## Deliverables
- Reproducible source modules under `src/`
- Stored metrics tables in `reports/tables/`
- Stored visualizations in `reports/figures/`
- Starter workflow script for end-to-end execution

## Risks and Mitigations
- **Unknown target column names:** enforce explicit confirmation in config/runner.
- **Schema mismatch across cohorts:** keep pipelines independent and compare only normalized feature labels.
- **Missingness / class imbalance:** include imputation placeholders and class balancing options.
