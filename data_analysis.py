"""
data_analysis.py
----------------
Exploratory data analysis for the placement dataset.

Driven by the canonical feature lists in feature_engineering.py rather than
hardcoded column names, so a future dataset change updates this analysis by
editing one module instead of every section here.
"""
import pandas as pd

from feature_engineering import (
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_raw_dataset,
)

# ============================================================
# LOAD DATASET
# ============================================================

file_path = "data/raw/student_placement.csv"

df = load_raw_dataset(file_path)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(f"- {column}")

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ============================================================
# CATEGORICAL DISTRIBUTIONS
# ============================================================

for column in RAW_CATEGORICAL_FEATURES + [TARGET_COLUMN]:
    print(f"\n========== {column.replace('_', ' ').upper()} ==========")
    print(df[column].value_counts())


# ============================================================
# PLACEMENT PERCENTAGE
# ============================================================

print("\n========== PLACEMENT PERCENTAGE ==========")

placement_percentage = (
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(placement_percentage)


# ============================================================
# PLACEMENT RATE BY CATEGORICAL FEATURE
# ============================================================

for column in RAW_CATEGORICAL_FEATURES:
    print(f"\n========== PLACEMENT RATE BY {column.replace('_', ' ').upper()} ==========")
    print(
        pd.crosstab(df[column], df[TARGET_COLUMN], normalize="index")
        .mul(100)
        .round(2)
    )


# ============================================================
# NUMERICAL FEATURE SUMMARIES BY PLACEMENT
# ============================================================

print("\n========== FEATURE MEANS BY PLACEMENT STATUS ==========")
print(df.groupby(TARGET_COLUMN)[RAW_NUMERICAL_FEATURES].mean().round(2).T)

print("\n========== FEATURE MEDIANS BY PLACEMENT STATUS ==========")
print(df.groupby(TARGET_COLUMN)[RAW_NUMERICAL_FEATURES].median().round(2).T)


# ============================================================
# PLACEMENT RATE BY DISCRETE COUNT FEATURES
# ============================================================

DISCRETE_FEATURES = ["internships", "projects", "workshops_certifications"]

for column in DISCRETE_FEATURES:
    print(f"\n========== PLACEMENT RATE BY {column.replace('_', ' ').upper()} ==========")
    print(
        pd.crosstab(df[column], df[TARGET_COLUMN], normalize="index")
        .mul(100)
        .round(2)
    )


# ============================================================
# PLACEMENT RATE BY BINNED CONTINUOUS FEATURES
# ============================================================

CONTINUOUS_FEATURES = [
    "cgpa", "ssc_marks", "hsc_marks", "aptitude_test_score", "soft_skills_rating",
]

for column in CONTINUOUS_FEATURES:
    # Quantile bins keep group sizes comparable; duplicates="drop" guards
    # columns whose range is narrow enough to produce tied bin edges.
    binned = pd.qcut(df[column], q=5, duplicates="drop")
    print(f"\n========== PLACEMENT RATE BY {column.replace('_', ' ').upper()} ==========")
    print(
        pd.crosstab(binned, df[TARGET_COLUMN], normalize="index")
        .mul(100)
        .round(2)
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION WITH PLACEMENT ==========")

# Convert target to numerical for correlation
df["placement_binary"] = (
    df[TARGET_COLUMN].str.strip().str.lower() == "placed"
).astype(int)

correlations = (
    df[RAW_NUMERICAL_FEATURES + ["placement_binary"]]
    .corr()["placement_binary"]
    .drop("placement_binary")
    .sort_values(ascending=False)
    .round(4)
)

print(correlations)

print("\n========== ANALYSIS COMPLETE ==========")
