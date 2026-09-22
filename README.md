Replicating and Extending Machine-Learning Models for Student Outcome Prediction

Overview

This project reproduces and extends the conventional machine-learning experiments reported by Farhood et al. (2024), Evaluating and Enhancing Artificial Intelligence Models for Predicting Student Learning Outcomes.

The project evaluates seven models: Linear Regression, Logistic Regression, SVM, Decision Tree, Random Forest, KNN, and XGBoost. Each model is tested with and without Lasso-based feature selection using repeated 80/20 holdout evaluation and repeated five-fold cross-validation.

The project has three main stages:

Reproduce the Farhood et al. experiments on the Mathematics and xAPI student datasets.

Apply the same modelling approach to the UCI Predict Students' Dropout and Academic Success dataset.

Apply the same approach to a newly constructed student-performance dataset created by harmonising two published datasets.

A GPA 3.5 sensitivity analysis is also included for the constructed dataset. Accuracy is retained as the primary evaluation metric to maintain comparability with the original study.

Research Question

To what extent can the conventional machine-learning results reported by Farhood et al. be reproduced, and do the relative model-performance patterns and model-dependent effects of Lasso feature selection generalise to independent higher-education data and a newly constructed student-performance dataset?

Project Structure

COMP8240_Project/
├── data/
│   ├── raw/
│   │   ├── constructed_dataset/
│   │   ├── existing_dataset/
│   │   └── reproduction/
│   └── processed/
│       ├── constructed_dataset/
│       │   └── sensitivity_analysis/
│       └── existing_dataset/
├── external/
│   └── farhood_original/
│       └── Comparative-Study-DL-ML/
├── figures/
│   ├── constructed_dataset/
│   ├── existing_dataset/
│   └── reproduction/
├── notebooks/
│   ├── comparative_analysis/
│   │   └── 03_comparative_analysis.ipynb
│   ├── constructed_dataset/
│   │   └── 01_harmonisation.ipynb
│   └── existing_dataset/
│       └── 01_exploration.ipynb
├── results/
│   ├── comparative_analysis/
│   ├── constructed_dataset/
│   ├── existing_dataset/
│   └── reproduction/
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
├── requirements.txt
└── README.md

Datasets

Mathematics dataset

Used for direct reproduction of the conventional machine-learning experiments from Farhood et al. The original modelling logic is preserved as closely as possible, with compatibility-only changes where required by the current Python environment.

xAPI-Edu-Data

Used for reproduction. The public Farhood repository did not provide a conventional machine-learning script for xAPI, so this project reconstructs the preprocessing and evaluation procedure from the published methodology while preserving the same seven models, repeated evaluation design, and Lasso comparison.

UCI Predict Students' Dropout and Academic Success

Used as an independent higher-education dataset to test generalisation beyond the original datasets.

Only completed-student outcomes are retained:

Graduate -> Success

Dropout -> Failure

Enrolled -> excluded

After preprocessing, the modelling dataset contains 3,630 completed students.

Constructed student-performance dataset

A new dataset is created by harmonising two independently published student-performance datasets.

Main outcome:

Higher Academic Performance: GPA >= 3.0

Lower Academic Performance: GPA < 3.0

The final constructed dataset contains 1,596 records. GPA is used only to construct the outcome and is not used as a predictor. source_dataset is retained for traceability during construction but excluded from modelling.

GPA 3.5 sensitivity analysis

An alternative threshold is evaluated:

Higher Academic Performance: GPA >= 3.5

Lower Academic Performance: GPA < 3.5

This checks whether the main constructed-dataset conclusions depend strongly on the GPA 3.0 threshold.

Methodology

For each dataset:

seven models are evaluated;

models are tested with and without Lasso-based feature selection;

repeated 80/20 holdout evaluation is performed for 100 iterations;

repeated five-fold cross-validation is performed for 100 iterations;

accuracy is the primary performance metric;

repeated accuracy distributions are visualised using boxplots.

The original Farhood implementation standardises the full feature matrix before the holdout or cross-validation split. This behaviour is preserved for reproduction fidelity and documented as a methodological limitation.

Reproduction Results

Mathematics

Model

Holdout

Holdout + Lasso

5-Fold CV

5-Fold CV + Lasso

Linear Regression

87.80

89.46

87.75

89.28

Logistic Regression

89.95

91.56

89.94

91.36

SVM

87.30

89.85

87.59

89.96

Decision Tree

89.62

89.65

89.94

89.55

Random Forest

93.58

92.86

93.83

92.96

KNN

73.05

82.42

73.44

83.36

XGBoost

93.49

92.10

93.52

91.93

The reproduced Mathematics results are extremely close to the published values.

xAPI

Model

Holdout

Holdout + Lasso

5-Fold CV

5-Fold CV + Lasso

Linear Regression

90.93

91.49

91.11

91.47

Logistic Regression

91.04

91.70

91.20

91.66

SVM

91.25

91.03

91.36

91.35

Decision Tree

88.01

88.30

87.95

88.07

Random Forest

92.36

91.45

92.44

91.27

KNN

88.41

90.71

88.37

91.20

XGBoost

91.66

90.72

91.80

90.21

The xAPI reconstruction shows the same broad model-performance patterns as the published study, although differences are larger than for Mathematics because the original repository did not provide the corresponding xAPI machine-learning implementation.

Independent UCI Results

Model

Holdout

Holdout + Lasso

5-Fold CV

5-Fold CV + Lasso

Linear Regression

90.06

90.83

90.10

90.85

Logistic Regression

90.26

91.11

90.28

91.09

SVM

87.86

90.47

87.81

90.52

Decision Tree

85.86

86.13

85.76

86.16

Random Forest

90.61

90.46

90.71

90.49

KNN

75.62

85.70

75.63

85.86

XGBoost

90.69

90.50

90.71

90.46

Constructed Dataset Results

GPA 3.0 main analysis

Model

Holdout

Holdout + Lasso

5-Fold CV

5-Fold CV + Lasso

Linear Regression

87.19

87.33

87.22

86.81

Logistic Regression

87.49

87.08

87.60

86.29

SVM

87.23

87.60

87.33

87.39

Decision Tree

86.55

86.24

86.44

86.02

Random Forest

88.12

87.31

88.23

87.19

KNN

83.19

86.54

83.26

87.67

XGBoost

89.05

87.97

88.82

88.06

XGBoost has the highest accuracy without Lasso in both evaluation methods. KNN shows the strongest positive response to Lasso feature selection.

GPA 3.5 sensitivity analysis

Model

Holdout

Holdout + Lasso

5-Fold CV

5-Fold CV + Lasso

Linear Regression

79.79

79.60

79.76

79.57

Logistic Regression

80.35

80.22

80.29

80.10

SVM

82.05

82.55

81.88

82.26

Decision Tree

84.53

84.30

84.78

84.30

Random Forest

85.70

86.27

85.85

86.00

KNN

77.12

81.02

77.24

81.86

XGBoost

88.75

88.49

88.70

88.18

Changing the GPA threshold changes absolute accuracy values but does not materially change the main constructed-dataset conclusion: XGBoost remains strongest overall and KNN continues to show a strong positive Lasso effect.

Main Comparative Findings

The experiments provide partial evidence that the findings of Farhood et al. generalise beyond the original datasets.

Random Forest is strongest on the original Mathematics and xAPI datasets without Lasso. On the independent UCI dataset, Random Forest and XGBoost are among the strongest models without Lasso, while Logistic Regression performs best after Lasso feature selection. On the constructed dataset, XGBoost performs best across the main evaluation settings.

The influence of Lasso is model-dependent rather than universally beneficial. KNN consistently benefits from Lasso across the datasets, while Random Forest and XGBoost generally show lower accuracy after Lasso feature selection. Logistic Regression and SVM show more dataset-dependent effects.

Holdout and five-fold cross-validation results are generally close, supporting the stability of the main comparative conclusions across the two evaluation approaches.

Reproducibility Notes

The Mathematics experiments use the publicly available Farhood implementation with compatibility-only changes required for the current software environment.

The xAPI conventional machine-learning experiment is a paper-based reconstruction because the public repository does not contain the corresponding xAPI machine-learning script.

The project intentionally preserves the original preprocessing and evaluation ordering where possible to prioritise reproduction fidelity. Some methodological choices, including scaling before data splitting, would normally be changed in a new predictive modelling study but are retained here so that comparisons with the original work remain meaningful.

Final Implementation Scope

The final implementation uses accuracy as the primary evaluation measure and focuses on repeated model performance, boxplots, model rankings, holdout-versus-cross-validation consistency, Lasso effects, reproducibility, and sensitivity analysis.

The initial proposal also discussed supplementary precision, recall, F1-score and ANOVA analyses. These supplementary analyses were not added to the final implementation. The final conclusions are therefore based on the completed repeated-accuracy experiments and comparative analyses described above.

Environment Setup

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Run commands from the project root directory.

Running the Project

Constructed dataset

python src/constructed_dataset/construct_dataset.py
python src/constructed_dataset/constructed_holdout.py > results/constructed_dataset/holdout_output.txt
python src/constructed_dataset/constructed_kfold.py > results/constructed_dataset/kfold_output.txt

UCI independent dataset

python src/existing_dataset/existing_holdout.py > results/existing_dataset/holdout_output.txt
python src/existing_dataset/existing_kfold.py > results/existing_dataset/kfold_output.txt

GPA 3.5 sensitivity analysis

python src/constructed_dataset/sensitivity_analysis/create_gpa35_dataset.py
python src/constructed_dataset/sensitivity_analysis/gpa35_holdout.py > results/constructed_dataset/sensitivity_analysis/gpa35_holdout_output.txt
python src/constructed_dataset/sensitivity_analysis/gpa35_kfold.py > results/constructed_dataset/sensitivity_analysis/gpa35_kfold_output.txt

xAPI reproduction

python src/reproduction/xapi/xapi_holdout.py > results/reproduction/xapi/holdout_output.txt
python src/reproduction/xapi/xapi_kfold.py > results/reproduction/xapi/kfold_output.txt

Comparative analysis

Open and run:

notebooks/comparative_analysis/03_comparative_analysis.ipynb

The notebook creates final comparative CSV files under:

results/comparative_analysis/

Key Output Files

results/comparative_analysis/final_model_comparison.csv
results/comparative_analysis/model_rankings.csv
results/comparative_analysis/lasso_effect_comparison.csv
results/comparative_analysis/mean_lasso_effect.csv
results/comparative_analysis/best_model_summary.csv
results/comparative_analysis/holdout_vs_cv_consistency.csv
results/comparative_analysis/consistency_summary.csv
results/comparative_analysis/gpa35_sensitivity_comparison.csv
results/comparative_analysis/reproducibility_comparison.csv
results/comparative_analysis/reproducibility_summary.csv

Reference

Farhood, H. et al. (2024). Evaluating and Enhancing Artificial Intelligence Models for Predicting Student Learning Outcomes. Informatics, 11(3), 46. https://doi.org/10.3390/informatics11030046

Status

Core experimental work is complete. Remaining work is final documentation, report preparation, repository cleanup, and submission checks.
