# MISCADA ASML Classification Summative

**Student Z-code:** Z0206561

## Overview

This repository contains the reproducible R code for the Heart Failure classification task. The objective is to predict whether a patient will suffer a fatal myocardial infarction (`fatal_mi`) using clinical features.

## How to Run

1. Ensure **R 4.4.2 or later** is installed.
2. Clone or download this repository.
3. Open a terminal in the repository directory.
4. Run:

```bash
Rscript report.R
```

The script will automatically install any missing packages from CRAN on first run.

## Output

The script generates the following PDF figures in the working directory:

| File | Contents |
|------|----------|
| `EDA_plots.pdf` | Boxplots of key clinical variables |
| `Correlation_Heatmap.pdf` | Correlation matrix of continuous predictors |
| `Model_Comparison_CV.pdf` | Cross-validation AUC comparison |
| `Variable_Importance.pdf` | Feature importance from the final model |
| `ROC_Curve.pdf` | ROC curve on the test set |
| `Calibration_Plot.pdf` | Probability calibration assessment |
| `Threshold_Sweep.pdf` | Sensitivity vs Specificity trade-off |
| `Precision_Recall_Tradeoff.pdf` | Precision-Recall trade-off |

## Files

- `report.R` — Master script (all analysis, modelling, and figure generation)
- `heart_failure.csv` — Dataset (provided by the course)
