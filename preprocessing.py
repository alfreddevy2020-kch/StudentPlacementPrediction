import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv("data/raw/student_placement.csv")

print("Original dataset shape:", df.shape)


# ============================================================
# 2. REMOVE IRRELEVANT / LEAKAGE COLUMNS
# ============================================================

# student_id:
# - Only identifies the student
# - Does not provide useful information for prediction
#
# salary_package_lpa:
# - This is known AFTER placement
# - Using it would cause target leakage

df = df.drop(
    columns=["student_id", "salary_package_lpa"]
)

print("Shape after removing irrelevant columns:", df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["placement_status"])

y = df["placement_status"]

print("\n========== FEATURES AND TARGET ==========")

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")

print(y.value_counts())

print("\nTarget percentages:")

print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = [
    "gender",
    "extracurricular_activities"
]

numerical_features = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "cgpa",
    "entrance_exam_score",
    "technical_skill_score",
    "soft_skill_score",
    "internship_count",
    "live_projects",
    "work_experience_months",
    "certifications",
    "attendance_percentage",
    "backlogs"
]


print("\n========== FEATURE TYPES ==========")

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 7. CREATE PREPROCESSING PIPELINE
# ============================================================

print("\n========== CREATING PREPROCESSING PIPELINE ==========")

preprocessor = ColumnTransformer(
    transformers=[
        
        # Numerical features
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),

        # Categorical features
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 8. FIT ONLY ON TRAINING DATA
# ============================================================

print("\n========== FITTING PREPROCESSOR ==========")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ============================================================
# 9. GET PROCESSED FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print("\n========== PROCESSED FEATURES ==========")

for feature in feature_names:
    print(feature)


# ============================================================
# 10. CONVERT PROCESSED DATA TO DATAFRAMES
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# 11. RESET INDEX
# ============================================================

X_train_processed = X_train_processed.reset_index(drop=True)
X_test_processed = X_test_processed.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# ============================================================
# 12. CHECK CLASS IMBALANCE
# ============================================================

print("\n========== CLASS IMBALANCE ==========")

class_counts = y_train.value_counts()

print(class_counts)

print("\nClass percentages:")

print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 13. CALCULATE CLASS WEIGHTS
# ============================================================

# We will use these weights during model training.
#
# The minority class (1 = placed) receives a higher weight.
# This prevents the model from simply favoring class 0.

total_samples = len(y_train)
number_of_classes = y_train.nunique()

class_weights = {}

for class_value, count in class_counts.items():

    class_weights[class_value] = (
        total_samples /
        (number_of_classes * count)
    )


print("\n========== CLASS WEIGHTS ==========")

for class_value, weight in class_weights.items():

    print(
        f"Class {class_value}: "
        f"{weight:.3f}"
    )


# ============================================================
# 14. CREATE PROCESSED DATA FOLDER
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ============================================================
# 15. SAVE PROCESSED TRAINING DATA
# ============================================================

train_processed = X_train_processed.copy()

train_processed["placement_status"] = y_train

train_processed.to_csv(
    "data/processed/train_processed.csv",
    index=False
)


# ============================================================
# 16. SAVE PROCESSED TESTING DATA
# ============================================================

test_processed = X_test_processed.copy()

test_processed["placement_status"] = y_test

test_processed.to_csv(
    "data/processed/test_processed.csv",
    index=False
)


# ============================================================
# 17. SAVE CLASS WEIGHTS
# ============================================================

class_weights_df = pd.DataFrame(
    {
        "class": list(class_weights.keys()),
        "weight": list(class_weights.values())
    }
)

class_weights_df.to_csv(
    "data/processed/class_weights.csv",
    index=False
)


# ============================================================
# 18. FINAL INFORMATION
# ============================================================

print("\n========== PREPROCESSING COMPLETE ==========")

print(
    "\nTraining data saved to:"
)

print(
    "data/processed/train_processed.csv"
)

print(
    "\nTesting data saved to:"
)

print(
    "data/processed/test_processed.csv"
)

print(
    "\nClass weights saved to:"
)

print(
    "data/processed/class_weights.csv"
)

print("\nFinal training shape:", X_train_processed.shape)
print("Final testing shape:", X_test_processed.shape)

print("\n============================================")