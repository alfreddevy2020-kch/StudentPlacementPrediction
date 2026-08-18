import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

df["placement_target"] = df["placement_status"].map({"Not Placed": 0, "Placed": 1})


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["placement_target"].value_counts()

print(target_counts)

print("\nTarget percentage:")

target_percentage = df["placement_target"].value_counts(normalize=True).mul(100).round(2)

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


# ------------------------------------------------------------
# Experience
# ------------------------------------------------------------

df["experience_score"] = (
    df["internships_count"] + df["projects_count"] + df["hackathons_participated"]
)

df["internship_project_score"] = df["internships_count"] * df["projects_count"]


# ------------------------------------------------------------
# Overall skills
# ------------------------------------------------------------

skill_columns = [
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "mock_interview_score",
]

df["overall_skill_score"] = df[skill_columns].mean(axis=1)


# ------------------------------------------------------------
# Technical skills
# ------------------------------------------------------------

df["technical_skill_score"] = (df["coding_skill_score"] + df["logical_reasoning_score"]) / 2


# ------------------------------------------------------------
# Soft skills
# ------------------------------------------------------------

df["soft_skill_score"] = (df["communication_skill_score"] + df["mock_interview_score"]) / 2


# ------------------------------------------------------------
# Academic features
# ------------------------------------------------------------

df["cgpa_percentage"] = df["cgpa"] * 10

df["academic_score"] = (df["cgpa_percentage"] + df["attendance_percentage"]) / 2

df["backlog_penalty"] = df["backlogs"] * 10

df["adjusted_academic_score"] = df["academic_score"] - df["backlog_penalty"]


# ------------------------------------------------------------
# Professional presence
# ------------------------------------------------------------

github_max = df["github_repos"].max()
linkedin_max = df["linkedin_connections"].max()

# Avoid division by zero
if github_max > 0:
    df["github_normalized"] = df["github_repos"] / github_max
else:
    df["github_normalized"] = 0

if linkedin_max > 0:
    df["linkedin_normalized"] = df["linkedin_connections"] / linkedin_max
else:
    df["linkedin_normalized"] = 0

df["professional_presence_score"] = (df["github_normalized"] + df["linkedin_normalized"]) / 2


# ------------------------------------------------------------
# Achievement
# ------------------------------------------------------------

df["achievement_score"] = (
    df["certifications_count"] + df["hackathons_participated"] + (df["extracurricular_score"] / 20)
)


# ------------------------------------------------------------
# Leadership
# ------------------------------------------------------------

df["leadership_profile_score"] = (df["leadership_score"] + df["extracurricular_score"]) / 2


# ------------------------------------------------------------
# Volunteer experience
# ------------------------------------------------------------

df["volunteer_binary"] = df["volunteer_experience"].map({"No": 0, "Yes": 1})


# ------------------------------------------------------------
# Study / sleep relationship
# ------------------------------------------------------------

df["study_sleep_ratio"] = df["study_hours_per_day"] / (df["sleep_hours"] + 0.1)


# ------------------------------------------------------------
# Interview readiness
# ------------------------------------------------------------

df["interview_readiness_score"] = (
    df["communication_skill_score"]
    + df["aptitude_score"]
    + df["logical_reasoning_score"]
    + df["mock_interview_score"]
) / 4


# ------------------------------------------------------------
# Placement readiness
# ------------------------------------------------------------

df["placement_readiness_score"] = (
    df["overall_skill_score"]
    + df["academic_score"]
    + (df["experience_score"] * 5)
    + df["leadership_score"]
) / 4


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

categorical_features = ["gender", "branch", "college_tier", "volunteer_experience"]

numerical_features = [column for column in X.columns if column not in categorical_features]


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
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTesting distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


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
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# Categorical preprocessing:
# 1. Fill missing values using most frequent value
# 2. One-hot encode categories

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
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

placed_percentage = y_train.mean() * 100

not_placed_percentage = (1 - y_train.mean()) * 100

print(f"Placed: {placed_percentage:.2f}%")

print(f"Not Placed: {not_placed_percentage:.2f}%")


difference = abs(placed_percentage - not_placed_percentage)

print(f"Difference between classes: {difference:.2f} percentage points")


if difference < 10:
    print("\nConclusion: The dataset is relatively balanced.")
else:
    print("\nConclusion: The dataset has noticeable class imbalance.")


# ============================================================
# 13. SMOTE
# ============================================================

print("\n" + "=" * 60)
print("SMOTE EXPERIMENT")
print("=" * 60)

print("SMOTE is applied ONLY to the training data.")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)


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
    class_weights[class_value] = total_samples / (number_of_classes * count)

print("Recommended class weights:")

print(class_weights)


# ============================================================
# 15. SAVE PREPROCESSOR
# ============================================================

PREPROCESSOR_FILE = "preprocessor.pkl"

joblib.dump(preprocessor, PREPROCESSOR_FILE)

print("\n" + "=" * 60)
print("PREPROCESSOR SAVED")
print("=" * 60)

print(PREPROCESSOR_FILE)


# ============================================================
# 16. SAVE PROCESSED DATA
# ============================================================

np.save("X_train_processed.npy", X_train_processed)

np.save("X_test_processed.npy", X_test_processed)

np.save("y_train.npy", y_train.to_numpy())

np.save("y_test.npy", y_test.to_numpy())


# SMOTE data is saved separately because it is
# an experimental balanced training dataset.

np.save("X_train_smote.npy", X_train_smote)

np.save("y_train_smote.npy", y_train_smote.to_numpy())


print("\nProcessed datasets saved.")


# ============================================================
# 17. TREND FEATURE DOCUMENTATION
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
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Original rows:", 100000)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nGenerated files:")

print("preprocessor.pkl")
print("X_train_processed.npy")
print("X_test_processed.npy")
print("y_train.npy")
print("y_test.npy")
print("X_train_smote.npy")
print("y_train_smote.npy")

print("\nNo model training performed.")
