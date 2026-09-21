from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "constructed_dataset"
    / "constructed_students.csv"
)

RAHMAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "constructed_dataset"
    / "Students_Performance_data_set.xlsx"
)

HASAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "constructed_dataset"
    / "ResearchInformation3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "constructed_dataset"
    / "sensitivity_analysis"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "constructed_students_gpa35.csv"
)


# =========================================================
# 1. Load main harmonised dataset
# =========================================================

constructed_df = pd.read_csv(MAIN_DATA_PATH)

assert constructed_df.shape == (1596, 9)


# =========================================================
# 2. Load original datasets
# =========================================================

rahman = pd.read_excel(RAHMAN_PATH)
hasan = pd.read_csv(HASAN_PATH)


# Clean Hasan text columns exactly as in main construction
text_cols = hasan.select_dtypes(
    include=["object", "string"]
).columns

for col in text_cols:
    hasan[col] = hasan[col].str.strip()


# =========================================================
# 3. Apply same population filtering as main analysis
# =========================================================

rahman_valid = rahman[
    (rahman["What was your previous SGPA?"] > 0)
    &
    (rahman["What is your current CGPA?"] > 0)
].copy()

hasan_cse = hasan[
    hasan["Department"]
    == "Computer Science and Engineering"
].copy()


assert len(rahman_valid) == 1153
assert len(hasan_cse) == 443


# =========================================================
# 4. Recover GPA values in same combined row order
# =========================================================

current_gpa = pd.concat(
    [
        rahman_valid[
            "What is your current CGPA?"
        ].reset_index(drop=True),

        hasan_cse[
            "Overall"
        ].reset_index(drop=True),
    ],
    ignore_index=True,
)

assert len(current_gpa) == 1596


# =========================================================
# 5. Create sensitivity outcome at GPA 3.5
# =========================================================

sensitivity_df = constructed_df.copy()

sensitivity_df["outcome"] = np.where(
    current_gpa >= 3.5,
    "Higher",
    "Lower",
)


# =========================================================
# 6. Validate
# =========================================================

assert sensitivity_df.shape == (1596, 9)

assert sensitivity_df[
    "outcome"
].value_counts().to_dict() == {
    "Lower": 999,
    "Higher": 597,
}

assert sensitivity_df[
    "source_dataset"
].value_counts().to_dict() == {
    "Rahman": 1153,
    "Hasan": 443,
}

assert sensitivity_df.isnull().sum().sum() == 0


print("GPA 3.5 sensitivity dataset")
print("Shape:", sensitivity_df.shape)

print("\nOutcome distribution:")
print(
    sensitivity_df[
        "outcome"
    ].value_counts()
)

print("\nOutcome percentages:")
print(
    sensitivity_df[
        "outcome"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# =========================================================
# 7. Save separately
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sensitivity_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nSaved:")
print(OUTPUT_PATH)
