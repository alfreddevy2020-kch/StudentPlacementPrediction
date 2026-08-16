import pandas as pd

# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("data/raw/student_placement.csv")


# ==============================
# 2. Remove irrelevant columns
# ==============================

# student_id is only an identifier.
# salary_package_lpa is removed because it can cause data leakage.
df = df.drop(columns=["student_id", "salary_package_lpa"])


# ==============================
# 3. Dataset shape
# ==============================

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# ==============================
# 4. Column information
# ==============================

print("\n========== COLUMN INFORMATION ==========")
df.info()


# ==============================
# 5. Missing values
# ==============================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==============================
# 6. Duplicate rows
# ==============================

print("\n========== DUPLICATES ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ==============================
# 7. Unique values
# ==============================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])


# ==============================
# 8. Target distribution
# ==============================

print("\n========== PLACEMENT STATUS ==========")

print(df["placement_status"].value_counts())

print("\nPlacement percentages:")

print(
    df["placement_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==============================
# 9. Statistical summary
# ==============================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ==============================
# 10. Average features by placement
# ==============================

print("\n========== AVERAGE FEATURES BY PLACEMENT STATUS ==========")

numeric_features = [
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

average_by_placement = df.groupby("placement_status")[numeric_features].mean()

print(average_by_placement.round(2))


# ==============================
# 11. Minimum values by placement
# ==============================

print("\n========== MINIMUM VALUES BY PLACEMENT STATUS ==========")

minimum_by_placement = df.groupby("placement_status")[numeric_features].min()

print(minimum_by_placement.T)


# ==============================
# 12. Maximum values by placement
# ==============================

print("\n========== MAXIMUM VALUES BY PLACEMENT STATUS ==========")

maximum_by_placement = df.groupby("placement_status")[numeric_features].max()

print(maximum_by_placement.T)


# ==============================
# 13. Median values by placement
# ==============================

print("\n========== MEDIAN VALUES BY PLACEMENT STATUS ==========")

median_by_placement = df.groupby("placement_status")[numeric_features].median()

print(median_by_placement.round(2).T)


# ==============================
# 14. Categorical feature distribution
# ==============================

print("\n========== GENDER DISTRIBUTION ==========")
print(df["gender"].value_counts())

print("\n========== GENDER VS PLACEMENT ==========")
print(
    pd.crosstab(
        df["gender"],
        df["placement_status"],
        normalize="index"
    ).mul(100).round(2)
)


print("\n========== EXTRACURRICULAR ACTIVITIES ==========")
print(df["extracurricular_activities"].value_counts())

print("\n========== EXTRACURRICULAR ACTIVITIES VS PLACEMENT ==========")
print(
    pd.crosstab(
        df["extracurricular_activities"],
        df["placement_status"],
        normalize="index"
    ).mul(100).round(2)
)


# ==============================
# 15. Correlation with placement
# ==============================

print("\n========== CORRELATION WITH PLACEMENT ==========")

# Convert categorical columns temporarily for correlation analysis
correlation_df = df.copy()

correlation_df["gender"] = correlation_df["gender"].map({
    "Male": 1,
    "Female": 0
})

correlation_df["extracurricular_activities"] = (
    correlation_df["extracurricular_activities"].map({
        "Yes": 1,
        "No": 0
    })
)

correlation = (
    correlation_df
    .corr(numeric_only=True)["placement_status"]
    .sort_values(ascending=False)
)

print(correlation.round(3))


print("\n========== DATA ANALYSIS COMPLETE ==========")