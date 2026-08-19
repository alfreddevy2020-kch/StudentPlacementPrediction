import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE

from feature_engineering import engineer_features, fit_normalization_stats, RAW_CATEGORICAL_FEATURES


# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "data/raw/student_placement.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Remove duplicate rows if any exist
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"\nRemoved {duplicate_count} duplicate rows.")
else:
    print("\nNo duplicate rows found.")


# ============================================================
# 3. TARGET CREATION
# ============================================================

df["placement_target"] = df["placement_status"].map({
    "Not Placed": 0,
    "Placed": 1
})


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["placement_target"].value_counts()

print(target_counts)

print("\nTarget percentage:")

target_percentage = (
    df["placement_target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(target_percentage)


# ============================================================
# 4. REMOVE IDENTIFIER
# ============================================================

# student_id is an identifier and contains no meaningful
# predictive information.

df = df.drop(columns=["student_id"])


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# Freeze github_repos_max / linkedin_connections_max from the FULL training
# dataset BEFORE engineering features, so every inference-time caller
# (dashboard, API, simulator) reuses this exact constant instead of
# recomputing its own batch-local max. See feature_engineering.py.
norm_stats = fit_normalization_stats(df)
print(f"Persisted normalization stats -> {norm_stats}")

df = engineer_features(df)


print("Feature engineering completed.")


# ============================================================
# 6. REMOVE TARGET LEAKAGE
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE PREVENTION")
print("=" * 60)

# salary_package_lpa is known only after placement.
# Therefore it cannot be used to predict placement.

df = df.drop(columns=["salary_package_lpa"])

# placement_status is the original text target.
# placement_target is the numerical target.

df = df.drop(columns=["placement_status"])


print("Removed:")
print("- student_id")
print("- salary_package_lpa")
print("- placement_status")

print("\nTarget column:")
print("- placement_target")


# ============================================================
# 7. SEPARATE X AND y
# ============================================================

X = df.drop(columns=["placement_target"])

y = df["placement_target"]


# ============================================================
# 8. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = RAW_CATEGORICAL_FEATURES

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\n" + "=" * 60)
print("FEATURE TYPES")
print("=" * 60)

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting distribution:")
print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 10. LEAKAGE-SAFE PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)


# Numerical preprocessing:
# 1. Fill missing values using median
# 2. Standardize numerical features

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# 1. Fill missing values using most frequent value
# 2. One-hot encode categories

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# IMPORTANT:
# fit_transform is used ONLY on training data.
# Test data is transformed using the already-fitted
# preprocessing pipeline.

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("Preprocessor fitted on training data only.")
print("Training data transformed.")
print("Testing data transformed.")


# ============================================================
# 11. CHECK PROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("PROCESSED DATA")
print("=" * 60)

print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)


# ============================================================
# 12. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 60)

placed_percentage = (
    y_train.mean() * 100
)

not_placed_percentage = (
    (1 - y_train.mean()) * 100
)

print(
    f"Placed: {placed_percentage:.2f}%"
)

print(
    f"Not Placed: {not_placed_percentage:.2f}%"
)


difference = abs(
    placed_percentage -
    not_placed_percentage
)

print(
    f"Difference between classes: {difference:.2f} percentage points"
)


if difference < 10:
    print(
        "\nConclusion: The dataset is relatively balanced."
    )
else:
    print(
        "\nConclusion: The dataset has noticeable class imbalance."
    )


# ============================================================
# 13. SMOTE
# ============================================================

print("\n" + "=" * 60)
print("SMOTE EXPERIMENT")
print("=" * 60)

print(
    "SMOTE is applied ONLY to the training data."
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)


print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# ============================================================
# 14. CLASS WEIGHTING INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("CLASS WEIGHTING")
print("=" * 60)

class_counts = y_train.value_counts()

total_samples = len(y_train)
number_of_classes = len(class_counts)

class_weights = {}

for class_value, count in class_counts.items():

    class_weights[class_value] = (
        total_samples /
        (number_of_classes * count)
    )

print(
    "Recommended class weights:"
)

print(class_weights)


# ============================================================
# 15. SAVE PREPROCESSOR & PROCESSED CSV FILES
# ============================================================

# Extract cleaned feature names
feature_names = [
    col.replace("numerical__", "").replace("categorical__", "")
    for col in preprocessor.get_feature_names_out()
]

# Convert processed arrays to DataFrames
X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
).reset_index(drop=True)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
).reset_index(drop=True)

train_processed = X_train_df.copy()
train_processed["placement_status"] = y_train.reset_index(drop=True)

test_processed = X_test_df.copy()
test_processed["placement_status"] = y_test.reset_index(drop=True)

# 1. Save processed CSVs
os.makedirs("data/processed", exist_ok=True)
train_processed.to_csv("data/processed/train_processed.csv", index=False)
test_processed.to_csv("data/processed/test_processed.csv", index=False)

# 2. Save class weights
class_weights_df = pd.DataFrame(
    {
        "class": list(class_weights.keys()),
        "weight": list(class_weights.values()),
    }
)
class_weights_df.to_csv("data/processed/class_weights.csv", index=False)

# 3. Save preprocessor
os.makedirs("part2/models", exist_ok=True)
joblib.dump(preprocessor, "part2/models/preprocessor.joblib")

print("\n" + "=" * 60)
print("PREPROCESSOR & CSV DATASETS SAVED")
print("=" * 60)
print("data/processed/train_processed.csv")
print("data/processed/test_processed.csv")
print("data/processed/class_weights.csv")
print("part2/models/preprocessor.joblib")


# ============================================================
# 16. TREND FEATURE DOCUMENTATION
# ============================================================

print("\n" + "=" * 60)
print("TREND FEATURE ANALYSIS")
print("=" * 60)

print(
    """
Semester-level trend features were considered,
including:

- CGPA progression
- Attendance progression
- Semester-wise performance change
- Rate of change in academic performance

However, the available dataset contains only
aggregate CGPA and attendance values.

Therefore, genuine semester-level trend features
cannot be calculated without introducing fabricated
data.

The project instead uses aggregate and derived
features such as:

- Academic score
- Adjusted academic score
- Experience score
- Skill score
- Interview readiness
- Placement readiness
- Study/sleep ratio
"""
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Original rows:", len(df) + duplicate_count)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
print("Processed feature columns:", len(feature_names))

print("\nGenerated files:")
print("- data/processed/train_processed.csv")
print("- data/processed/test_processed.csv")
print("- data/processed/class_weights.csv")
print("- part2/models/preprocessor.joblib")

print("\nNo model training performed.")