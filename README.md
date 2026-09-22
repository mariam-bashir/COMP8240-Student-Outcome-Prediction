# COMP8240 Student Outcome Prediction

## Replicating and Extending Machine-Learning Models for Student Outcome Prediction

This project reproduces and extends the conventional machine-learning experiments reported by Farhood et al. (2024), *Evaluating and Enhancing Artificial Intelligence Models for Predicting Student Learning Outcomes*.

The project focuses on seven conventional machine-learning models:

- Linear Regression
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbours (KNN)
- XGBoost

Each model is evaluated **with and without Lasso-based feature selection** using repeated **80/20 holdout evaluation** and repeated **five-fold cross-validation**.

The project contains three main stages:

1. Reproduce the Farhood et al. conventional machine-learning experiments on the Mathematics and xAPI datasets.
2. Apply the same modelling approach to an independent higher-education dataset from UCI.
3. Apply the same approach to a newly constructed student-performance dataset created by harmonising two published datasets.

A **GPA 3.5 sensitivity analysis** is also included for the constructed dataset to examine whether the main conclusions remain stable under an alternative outcome threshold.

Accuracy is retained as the primary evaluation metric to maintain direct comparability with the original study.

---

## Research Question

> To what extent can the conventional machine-learning results reported by Farhood et al. be reproduced, and do the relative model-performance patterns and model-dependent effects of Lasso feature selection generalise to independent higher-education data and a newly constructed student-performance dataset?

---

## Project Structure

```text
COMP8240-Student-Outcome-Prediction/
├── data/
│   ├── processed/
│   │   ├── constructed_dataset/
│   │   │   ├── constructed_students.csv
│   │   │   └── sensitivity_analysis/
│   │   │       └── constructed_students_gpa35.csv
│   │   └── existing_dataset/
│   │       └── completed_students.csv
│   └── raw/
│       ├── constructed_dataset/
│       │   ├── ResearchInformation3.csv
│       │   └── Students_Performance_data_set.xlsx
│       ├── existing_dataset/
│       │   └── data.csv
│       └── reproduction/
│           ├── student-mat.csv
│           └── xAPI-Edu-Data.csv
├── external/
│   └── farhood_original/
│       └── Comparative-Study-DL-ML/
│           ├── Deep Learning Model-Hold Out.py
│           ├── LICENSE
│           ├── Machine Learnings-Holdout.py
│           ├── Machine Learnings-k-fold Cross-Validation.py
│           ├── README.md
│           └── student-dataset.csv
├── figures/
│   ├── constructed_dataset/
│   │   ├── holdout/
│   │   ├── kfold/
│   │   └── sensitivity_analysis/
│   │       └── gpa35/
│   ├── existing_dataset/
│   │   ├── holdout/
│   │   └── kfold/
│   └── reproduction/
│       ├── mathematics/
│       │   ├── holdout/
│       │   └── kfold/
│       └── xapi/
│           ├── holdout/
│           └── kfold/
├── notebooks/
│   ├── comparative_analysis/
│   │   └── 03_comparative_analysis.ipynb
│   ├── constructed_dataset/
│   │   └── 01_harmonisation.ipynb
│   └── existing_dataset/
│       └── 01_exploration.ipynb
├── results/
│   ├── comparative_analysis/
│   │   ├── best_model_summary.csv
│   │   ├── consistency_summary.csv
│   │   ├── final_model_comparison.csv
│   │   ├── gpa35_sensitivity_comparison.csv
│   │   ├── holdout_vs_cv_consistency.csv
│   │   ├── lasso_direction_summary.csv
│   │   ├── lasso_effect_comparison.csv
│   │   ├── mean_lasso_effect.csv
│   │   ├── model_rankings.csv
│   │   ├── reproducibility_comparison.csv
│   │   └── reproducibility_summary.csv
│   ├── constructed_dataset/
│   │   ├── holdout_output.txt
│   │   ├── kfold_output.txt
│   │   └── sensitivity_analysis/
│   │       ├── gpa35_holdout_output.txt
│   │       └── gpa35_kfold_output.txt
│   ├── existing_dataset/
│   │   ├── holdout_output.txt
│   │   └── kfold_output.txt
│   └── reproduction/
│       ├── mathematics/
│       │   ├── holdout_output.txt
│       │   └── kfold_output.txt
│       ├── reproduction_summary.txt
│       └── xapi/
│           ├── holdout_output.txt
│           └── kfold_output.txt
├── src/
│   ├── constructed_dataset/
│   │   ├── construct_dataset.py
│   │   ├── constructed_holdout.py
│   │   ├── constructed_kfold.py
│   │   └── sensitivity_analysis/
│   │       ├── create_gpa35_dataset.py
│   │       ├── gpa35_holdout.py
│   │       └── gpa35_kfold.py
│   ├── existing_dataset/
│   │   ├── existing_holdout.py
│   │   └── existing_kfold.py
│   └── reproduction/
│       └── xapi/
│           ├── xapi_holdout.py
│           └── xapi_kfold.py
├── README.md
└── requirements.txt
```

---

## Datasets

### 1. Mathematics dataset

The Mathematics dataset is used for direct reproduction of the conventional machine-learning experiments from Farhood et al.

The project preserves the original modelling logic as closely as possible. Compatibility-only changes were made where required for the current Python environment.

### 2. xAPI-Edu-Data

The xAPI dataset is also used for reproduction.

The public Farhood repository does not provide the corresponding conventional machine-learning xAPI script, so this project reconstructs the preprocessing and evaluation procedure from the published methodology while preserving the same seven models, repeated evaluation design, and Lasso comparison.

### 3. UCI Predict Students' Dropout and Academic Success

The independent UCI higher-education dataset is used to test whether the model-performance patterns generalise beyond the original datasets.

Only completed-student outcomes are retained:

- Graduate → Success
- Dropout → Failure
- Enrolled → Excluded

After preprocessing, the modelling dataset contains **3,630 completed students**.

### 4. Constructed student-performance dataset

A new dataset is created by harmonising two independently published student-performance datasets.

The main outcome is:

- **Higher Academic Performance:** GPA ≥ 3.0
- **Lower Academic Performance:** GPA < 3.0

The final constructed dataset contains **1,596 records**.

The GPA variables are used only to construct the outcome and are not used as predictors.

`source_dataset` is retained for traceability during dataset construction but is excluded from model predictors.

### 5. GPA 3.5 sensitivity dataset

A second version of the constructed dataset is created using:

- **Higher Academic Performance:** GPA ≥ 3.5
- **Lower Academic Performance:** GPA < 3.5

This sensitivity analysis checks whether the main constructed-dataset conclusions depend strongly on the GPA 3.0 threshold.

---

## Methodology

The project follows the conventional machine-learning evaluation approach used in the selected study.

For each main dataset:

- Seven conventional machine-learning models are evaluated.
- Models are tested with and without Lasso-based feature selection.
- Repeated 80/20 holdout evaluation is performed for 100 iterations.
- Repeated five-fold cross-validation is performed for 100 iterations.
- Accuracy is used as the primary performance metric.
- Repeated accuracy distributions are visualised using boxplots.
- Model rankings, Lasso effects, and holdout-versus-cross-validation consistency are compared across datasets.

For reproduction fidelity, the original Farhood preprocessing and modelling order is preserved where possible.

One important methodological limitation is that the original implementation standardises the full feature matrix before splitting the data. This behaviour is retained in this project to preserve comparability with the original study.

---

## Reproduction Results

### Mathematics

Mean accuracy (%):

| Model | Holdout | Holdout + Lasso | 5-Fold CV | 5-Fold CV + Lasso |
|---|---:|---:|---:|---:|
| Linear Regression | 87.80 | 89.46 | 87.75 | 89.28 |
| Logistic Regression | 89.95 | 91.56 | 89.94 | 91.36 |
| SVM | 87.30 | 89.85 | 87.59 | 89.96 |
| Decision Tree | 89.62 | 89.65 | 89.94 | 89.55 |
| Random Forest | 93.58 | 92.86 | 93.83 | 92.96 |
| KNN | 73.05 | 82.42 | 73.44 | 83.36 |
| XGBoost | 93.49 | 92.10 | 93.52 | 91.93 |

The reproduced Mathematics results are extremely close to the published values.

### xAPI

Mean accuracy (%):

| Model | Holdout | Holdout + Lasso | 5-Fold CV | 5-Fold CV + Lasso |
|---|---:|---:|---:|---:|
| Linear Regression | 90.93 | 91.49 | 91.11 | 91.47 |
| Logistic Regression | 91.04 | 91.70 | 91.20 | 91.66 |
| SVM | 91.25 | 91.03 | 91.36 | 91.35 |
| Decision Tree | 88.01 | 88.30 | 87.95 | 88.07 |
| Random Forest | 92.36 | 91.45 | 92.44 | 91.27 |
| KNN | 88.41 | 90.71 | 88.37 | 91.20 |
| XGBoost | 91.66 | 90.72 | 91.80 | 90.21 |

The xAPI reconstruction shows the same broad model-performance patterns as the published study, although the differences are larger than for Mathematics because the original repository does not provide the corresponding xAPI conventional machine-learning implementation.

---

## Independent UCI Results

Mean accuracy (%):

| Model | Holdout | Holdout + Lasso | 5-Fold CV | 5-Fold CV + Lasso |
|---|---:|---:|---:|---:|
| Linear Regression | 90.06 | 90.83 | 90.10 | 90.85 |
| Logistic Regression | 90.26 | 91.11 | 90.28 | 91.09 |
| SVM | 87.86 | 90.47 | 87.81 | 90.52 |
| Decision Tree | 85.86 | 86.13 | 85.76 | 86.16 |
| Random Forest | 90.61 | 90.46 | 90.71 | 90.49 |
| KNN | 75.62 | 85.70 | 75.63 | 85.86 |
| XGBoost | 90.69 | 90.50 | 90.71 | 90.46 |

The UCI results show that strong model performance generalises to an independent higher-education dataset, but the exact ranking of algorithms remains dataset-dependent.

---

## Constructed Dataset Results

### GPA 3.0 main analysis

Mean accuracy (%):

| Model | Holdout | Holdout + Lasso | 5-Fold CV | 5-Fold CV + Lasso |
|---|---:|---:|---:|---:|
| Linear Regression | 87.19 | 87.33 | 87.22 | 86.81 |
| Logistic Regression | 87.49 | 87.08 | 87.60 | 86.29 |
| SVM | 87.23 | 87.60 | 87.33 | 87.39 |
| Decision Tree | 86.55 | 86.24 | 86.44 | 86.02 |
| Random Forest | 88.12 | 87.31 | 88.23 | 87.19 |
| KNN | 83.19 | 86.54 | 83.26 | 87.67 |
| XGBoost | 89.05 | 87.97 | 88.82 | 88.06 |

XGBoost achieves the highest accuracy without Lasso in both evaluation methods.

KNN shows the strongest positive response to Lasso feature selection on the constructed dataset.

### GPA 3.5 sensitivity analysis

Mean accuracy (%):

| Model | Holdout | Holdout + Lasso | 5-Fold CV | 5-Fold CV + Lasso |
|---|---:|---:|---:|---:|
| Linear Regression | 79.79 | 79.60 | 79.76 | 79.57 |
| Logistic Regression | 80.35 | 80.22 | 80.29 | 80.10 |
| SVM | 82.05 | 82.55 | 81.88 | 82.26 |
| Decision Tree | 84.53 | 84.30 | 84.78 | 84.30 |
| Random Forest | 85.70 | 86.27 | 85.85 | 86.00 |
| KNN | 77.12 | 81.02 | 77.24 | 81.86 |
| XGBoost | 88.75 | 88.49 | 88.70 | 88.18 |

Changing the GPA threshold changes absolute accuracy values but does not materially alter the main constructed-dataset conclusion.

XGBoost remains strongest overall, while KNN continues to show a substantial positive Lasso effect.

---

## Main Comparative Findings

The experiments provide **partial evidence of generalisation** beyond the original Farhood datasets.

### Model performance

- Random Forest is strongest on the Mathematics and xAPI datasets without Lasso.
- On the UCI dataset, Random Forest and XGBoost are among the strongest models without Lasso.
- Logistic Regression performs highest on the UCI dataset after Lasso feature selection.
- On the constructed dataset, XGBoost achieves the highest accuracy across the main evaluation settings.

### Effect of Lasso

The effect of Lasso is strongly model-dependent rather than universally beneficial.

- KNN consistently benefits from Lasso across the datasets.
- Random Forest and XGBoost generally show lower accuracy after Lasso.
- Logistic Regression and SVM show more dataset-dependent effects.

### Holdout versus cross-validation

Holdout and five-fold cross-validation results are generally close.

This supports the stability of the main comparative conclusions across the two evaluation approaches.

### Sensitivity analysis

Changing the constructed-data threshold from GPA 3.0 to GPA 3.5 changes the absolute accuracy values but does not materially change the principal model-performance pattern.

---

## Reproducibility Notes

### Mathematics

The Mathematics experiments use the publicly available Farhood implementation with compatibility-only changes required for the current software environment.

### xAPI

The xAPI conventional machine-learning experiment is a paper-based reconstruction because the public Farhood repository does not contain the corresponding xAPI conventional machine-learning script.

### Compatibility changes

Compatibility changes made to the original implementation include:

- Updated pandas syntax.
- Updated Matplotlib boxplot argument syntax.
- Saving figures before displaying them.
- Reconstructing the Mathematics `student-dataset.csv` from the official UCI Mathematics dataset and mapping `G3` to `Grade`.

The modelling and evaluation logic was otherwise kept aligned with the original implementation.

---

## Final Implementation Scope

The final implementation focuses on:

- Reproduction of the original conventional machine-learning results.
- Repeated accuracy evaluation.
- Holdout and five-fold cross-validation.
- With-Lasso and without-Lasso comparisons.
- Model rankings.
- Lasso effect comparisons.
- Holdout-versus-cross-validation consistency.
- Independent UCI generalisation.
- Constructed-dataset generalisation.
- GPA 3.5 sensitivity analysis.
- Published-versus-reproduced result comparison.

Accuracy remains the primary metric throughout the completed implementation.

The initial proposal also discussed supplementary precision, recall, F1-score and ANOVA analyses. These supplementary analyses were not added to the final implementation. The final conclusions therefore rely on the completed repeated-accuracy experiments and comparative analyses documented in this repository.

---

## Environment Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the following commands from the project root directory.

---

## Running the Project

### 1. Construct the harmonised dataset

```bash
python src/constructed_dataset/construct_dataset.py
```

The corresponding harmonisation notebook is:

```text
notebooks/constructed_dataset/01_harmonisation.ipynb
```

### 2. Run the independent UCI experiments

Holdout:

```bash
python src/existing_dataset/existing_holdout.py \
> results/existing_dataset/holdout_output.txt
```

Five-fold cross-validation:

```bash
python src/existing_dataset/existing_kfold.py \
> results/existing_dataset/kfold_output.txt
```

### 3. Run the constructed-dataset experiments

Holdout:

```bash
python src/constructed_dataset/constructed_holdout.py \
> results/constructed_dataset/holdout_output.txt
```

Five-fold cross-validation:

```bash
python src/constructed_dataset/constructed_kfold.py \
> results/constructed_dataset/kfold_output.txt
```

### 4. Run the GPA 3.5 sensitivity analysis

Create the sensitivity dataset:

```bash
python src/constructed_dataset/sensitivity_analysis/create_gpa35_dataset.py
```

Holdout:

```bash
python src/constructed_dataset/sensitivity_analysis/gpa35_holdout.py \
> results/constructed_dataset/sensitivity_analysis/gpa35_holdout_output.txt
```

Five-fold cross-validation:

```bash
python src/constructed_dataset/sensitivity_analysis/gpa35_kfold.py \
> results/constructed_dataset/sensitivity_analysis/gpa35_kfold_output.txt
```

### 5. Run the xAPI reproduction

Holdout:

```bash
python src/reproduction/xapi/xapi_holdout.py \
> results/reproduction/xapi/holdout_output.txt
```

Five-fold cross-validation:

```bash
python src/reproduction/xapi/xapi_kfold.py \
> results/reproduction/xapi/kfold_output.txt
```

### 6. Run the comparative analysis

Open and run:

```text
notebooks/comparative_analysis/03_comparative_analysis.ipynb
```

The comparative notebook generates the final CSV summaries under:

```text
results/comparative_analysis/
```

---

## Key Output Files

The main final outputs are:

```text
results/comparative_analysis/final_model_comparison.csv
results/comparative_analysis/model_rankings.csv
results/comparative_analysis/lasso_effect_comparison.csv
results/comparative_analysis/lasso_direction_summary.csv
results/comparative_analysis/mean_lasso_effect.csv
results/comparative_analysis/best_model_summary.csv
results/comparative_analysis/holdout_vs_cv_consistency.csv
results/comparative_analysis/consistency_summary.csv
results/comparative_analysis/gpa35_sensitivity_comparison.csv
results/comparative_analysis/reproducibility_comparison.csv
results/comparative_analysis/reproducibility_summary.csv
```

---

## Limitations

Important limitations include:

- The original implementation standardises the complete feature matrix before splitting data, which can introduce information leakage. This behaviour is retained for reproduction fidelity.
- The xAPI experiment is reconstructed from the published methodology because the corresponding conventional machine-learning source code is not available in the public Farhood repository.
- The constructed dataset combines variables from independently published datasets, so harmonisation necessarily reduces the feature set to concepts that can be mapped consistently.
- The GPA 3.0 threshold is an operational definition of higher versus lower academic performance, not an institutional pass/fail threshold.
- The GPA 3.5 analysis is used only as a sensitivity check.
- Supplementary precision, recall, F1-score and ANOVA analyses discussed in the initial proposal were not included in the final implementation.

---

## Reference

Farhood, H. et al. (2024). *Evaluating and Enhancing Artificial Intelligence Models for Predicting Student Learning Outcomes*. **Informatics, 11**(3), 46.

DOI: https://doi.org/10.3390/informatics11030046

Original implementation:

https://github.com/aideveloper63/Comparative-Study-DL-ML

---

## Project Status

Core experimental work is complete:

- Mathematics reproduction completed
- xAPI reconstruction completed
- UCI independent-dataset extension completed
- Constructed dataset completed
- GPA 3.5 sensitivity analysis completed
- Comparative model analysis completed
- Reproducibility comparison completed
- Lasso effect comparison completed
- Final repository organisation completed

The repository is ready for final report preparation and submission review.
