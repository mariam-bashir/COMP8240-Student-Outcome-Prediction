from pathlib import Path
import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
)

OUTPUT_PATH = OUTPUT_DIR / "constructed_students.csv"


# =========================================================
# 2. Load raw datasets
# =========================================================

rahman = pd.read_excel(RAHMAN_PATH)
hasan = pd.read_csv(HASAN_PATH)

print("Rahman raw shape:", rahman.shape)
print("Hasan raw shape :", hasan.shape)


# =========================================================
# 3. Clean and filter datasets
# =========================================================

rahman_clean = rahman.copy()
hasan_clean = hasan.copy()


# Clean whitespace in Hasan text columns
text_cols = hasan_clean.select_dtypes(
    include=["object", "string"]
).columns

for col in text_cols:
    hasan_clean[col] = hasan_clean[col].str.strip()


# Keep only Computer Science and Engineering students
# from the Hasan dataset
hasan_cse = hasan_clean[
    hasan_clean["Department"]
    == "Computer Science and Engineering"
].copy()


# Remove Rahman records with unusable current or
# previous academic-performance values
rahman_valid = rahman_clean[
    (rahman_clean["What was your previous SGPA?"] > 0)
    &
    (rahman_clean["What is your current CGPA?"] > 0)
].copy()


print("\nAfter cleaning/filtering:")
print("Rahman valid rows:", len(rahman_valid))
print("Hasan CSE rows   :", len(hasan_cse))


# =========================================================
# 4. Common harmonisation functions
# =========================================================

def map_study_stage(value):
    """
    Convert semester number into a broader study stage.
    """
    value = int(value)

    if value <= 4:
        return "Early"
    elif value <= 8:
        return "Middle"
    else:
        return "Late"


def map_study_preparation(hours):
    """
    Convert Rahman daily study hours into the same
    preparation categories used by Hasan.
    """
    if hours <= 1:
        return "0-1 Hour"
    elif hours <= 3:
        return "2-3 Hours"
    else:
        return "More than 3 Hours"


def map_attendance(value):
    """
    Convert numeric attendance percentage into the
    attendance bands used by Hasan.
    """
    if value < 40:
        return "Below 40%"
    elif value < 60:
        return "40%-59%"
    elif value < 80:
        return "60%-79%"
    else:
        return "80%-100%"


def map_family_income(value):
    """
    Convert Rahman numeric monthly family income into
    the same four income categories used by Hasan.
    """
    if value < 15000:
        return "Low"
    elif value < 30000:
        return "Lower middle"
    elif value <= 50000:
        return "Upper middle"
    else:
        return "High"


# =========================================================
# 5. Harmonise Rahman dataset
# =========================================================

rahman_h = pd.DataFrame()


# Gender
rahman_h["gender"] = (
    rahman_valid["Gender"]
    .str.strip()
)


# Study stage
rahman_h["study_stage"] = (
    rahman_valid["Current Semester"]
    .apply(map_study_stage)
)


# Study preparation
rahman_h["study_preparation"] = (
    rahman_valid["How many hour do you study daily?"]
    .apply(map_study_preparation)
)


# Attendance
# One Rahman record contains "94-98".
# Use the midpoint 96 before applying attendance bands.
rahman_attendance = (
    rahman_valid["Average attendance on class"]
    .replace("94-98", 96)
)

rahman_attendance = pd.to_numeric(
    rahman_attendance,
    errors="coerce"
)

if rahman_attendance.isna().any():
    raise ValueError(
        "Rahman attendance contains values that "
        "could not be converted to numeric."
    )

rahman_h["attendance"] = (
    rahman_attendance
    .apply(map_attendance)
)


# Extracurricular participation
rahman_h["extracurricular_participation"] = (
    rahman_valid[
        "Are you engaged with any co-curriculum activities?"
    ]
    .str.strip()
)


# Family income
rahman_h["family_income"] = (
    rahman_valid["What is your monthly family income?"]
    .apply(map_family_income)
)


# Previous academic performance
rahman_h["previous_academic_performance"] = (
    rahman_valid["What was your previous SGPA?"]
)


# Outcome
# Current CGPA is used only to construct the target.
# It is not retained as a predictor.
rahman_h["outcome"] = (
    rahman_valid["What is your current CGPA?"]
    .apply(
        lambda x: "Higher" if x >= 3.0 else "Lower"
    )
)


# Dataset source for traceability only
rahman_h["source_dataset"] = "Rahman"


print("\nRahman harmonised:")
print("Shape:", rahman_h.shape)
print(
    "Missing values:",
    rahman_h.isnull().sum().sum()
)
print(rahman_h["outcome"].value_counts())


# =========================================================
# 6. Harmonise Hasan dataset
# =========================================================

hasan_h = pd.DataFrame()


# Gender
hasan_h["gender"] = (
    hasan_cse["Gender"]
    .str.strip()
)


# Convert values such as "2nd", "3rd", "10th"
# into numeric semester values
hasan_semester = (
    hasan_cse["Semester"]
    .str.extract(r"(\d+)")[0]
    .astype(int)
)


# Study stage
hasan_h["study_stage"] = (
    hasan_semester
    .apply(map_study_stage)
)


# Study preparation
hasan_h["study_preparation"] = (
    hasan_cse["Preparation"]
    .str.strip()
)


# Attendance
hasan_h["attendance"] = (
    hasan_cse["Attendance"]
    .str.strip()
)


# Extracurricular participation
hasan_h["extracurricular_participation"] = (
    hasan_cse["Extra"]
    .str.strip()
)


# Family income
income_map = {
    "Low (Below 15,000)": "Low",
    "Lower middle (15,000-30,000)": "Lower middle",
    "Upper middle (30,000-50,000)": "Upper middle",
    "High (Above 50,000)": "High"
}

hasan_h["family_income"] = (
    hasan_cse["Income"]
    .map(income_map)
)

if hasan_h["family_income"].isna().any():
    raise ValueError(
        "Some Hasan income values were not mapped."
    )


# Previous academic performance
hasan_h["previous_academic_performance"] = (
    hasan_cse["Last"]
)


# Outcome
# Overall GPA is used only to construct the target.
# It is not retained as a predictor.
hasan_h["outcome"] = (
    hasan_cse["Overall"]
    .apply(
        lambda x: "Higher" if x >= 3.0 else "Lower"
    )
)


# Dataset source for traceability only
hasan_h["source_dataset"] = "Hasan"


print("\nHasan harmonised:")
print("Shape:", hasan_h.shape)
print(
    "Missing values:",
    hasan_h.isnull().sum().sum()
)
print(hasan_h["outcome"].value_counts())


# =========================================================
# 7. Put both datasets into identical column order
# =========================================================

common_columns = [
    "gender",
    "study_stage",
    "study_preparation",
    "attendance",
    "extracurricular_participation",
    "family_income",
    "previous_academic_performance",
    "outcome",
    "source_dataset"
]

rahman_h = rahman_h[common_columns]
hasan_h = hasan_h[common_columns]


# =========================================================
# 8. Combine datasets
# =========================================================

constructed_df = pd.concat(
    [rahman_h, hasan_h],
    ignore_index=True
)


# =========================================================
# 9. Validate final constructed dataset
# =========================================================

print("\nFinal constructed dataset:")
print("Shape:", constructed_df.shape)
print(
    "Missing values:",
    constructed_df.isnull().sum().sum()
)

print("\nSource distribution:")
print(
    constructed_df[
        "source_dataset"
    ].value_counts()
)

print("\nOutcome distribution:")
print(
    constructed_df[
        "outcome"
    ].value_counts()
)


# Expected dataset dimensions
assert constructed_df.shape == (1596, 9)

# No missing values
assert constructed_df.isnull().sum().sum() == 0

# Expected source contribution
assert (
    constructed_df["source_dataset"]
    .value_counts()
    .to_dict()
    ==
    {
        "Rahman": 1153,
        "Hasan": 443
    }
)

# Expected outcome distribution
assert (
    constructed_df["outcome"]
    .value_counts()
    .to_dict()
    ==
    {
        "Higher": 1132,
        "Lower": 464
    }
)

# Validate expected categories
assert set(
    constructed_df["gender"].unique()
) == {
    "Male",
    "Female"
}

assert set(
    constructed_df["study_stage"].unique()
) == {
    "Early",
    "Middle",
    "Late"
}

assert set(
    constructed_df[
        "study_preparation"
    ].unique()
) == {
    "0-1 Hour",
    "2-3 Hours",
    "More than 3 Hours"
}

assert set(
    constructed_df["attendance"].unique()
) == {
    "Below 40%",
    "40%-59%",
    "60%-79%",
    "80%-100%"
}

assert set(
    constructed_df[
        "extracurricular_participation"
    ].unique()
) == {
    "Yes",
    "No"
}

assert set(
    constructed_df["family_income"].unique()
) == {
    "Low",
    "Lower middle",
    "Upper middle",
    "High"
}

assert set(
    constructed_df["outcome"].unique()
) == {
    "Higher",
    "Lower"
}

# Previous academic performance should remain
# on the 0-4 GPA scale.
assert (
    constructed_df[
        "previous_academic_performance"
    ]
    .between(0, 4)
    .all()
)

print("\nAll validation checks passed.")


# =========================================================
# 10. Save final constructed dataset
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

constructed_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDataset saved successfully:")
print(OUTPUT_PATH)
print("Saved shape:", constructed_df.shape)
