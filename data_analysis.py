import pandas as pd

# ============================================================
# LOAD DATASET
# ============================================================

file_path = "data/raw/student_placement.csv"

df = pd.read_csv(file_path)


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
    print(column)

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

print("\n========== GENDER ==========")
print(df["gender"].value_counts())

print("\n========== BRANCH ==========")
print(df["branch"].value_counts())

print("\n========== COLLEGE TIER ==========")
print(df["college_tier"].value_counts())

print("\n========== VOLUNTEER EXPERIENCE ==========")
print(df["volunteer_experience"].value_counts())

print("\n========== PLACEMENT STATUS ==========")
print(df["placement_status"].value_counts())


# ============================================================
# PLACEMENT PERCENTAGE
# ============================================================

print("\n========== PLACEMENT PERCENTAGE ==========")

placement_percentage = df["placement_status"].value_counts(normalize=True).mul(100).round(2)

print(placement_percentage)


# ============================================================
# PLACEMENT RATE BY GENDER
# ============================================================

print("\n========== PLACEMENT RATE BY GENDER ==========")

gender_placement = (
    pd.crosstab(df["gender"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(gender_placement)


# ============================================================
# PLACEMENT RATE BY BRANCH
# ============================================================

print("\n========== PLACEMENT RATE BY BRANCH ==========")

branch_placement = (
    pd.crosstab(df["branch"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(branch_placement)


# ============================================================
# PLACEMENT RATE BY COLLEGE TIER
# ============================================================

print("\n========== PLACEMENT RATE BY COLLEGE TIER ==========")

tier_placement = (
    pd.crosstab(df["college_tier"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(tier_placement)


# ============================================================
# PLACEMENT RATE BY VOLUNTEER EXPERIENCE
# ============================================================

print("\n========== PLACEMENT RATE BY VOLUNTEER EXPERIENCE ==========")

volunteer_placement = (
    pd.crosstab(df["volunteer_experience"], df["placement_status"], normalize="index")
    .mul(100)
    .round(2)
)

print(volunteer_placement)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "age",
    "cgpa",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs",
    "extracurricular_score",
    "leadership_score",
    "sleep_hours",
    "study_hours_per_day",
]


# ============================================================
# MEAN OF NUMERICAL FEATURES BY PLACEMENT
# ============================================================

print("\n========== FEATURE MEANS BY PLACEMENT STATUS ==========")

mean_by_placement = df.groupby("placement_status")[numerical_features].mean().round(2)

print(mean_by_placement.T)


# ============================================================
# MEDIAN OF NUMERICAL FEATURES BY PLACEMENT
# ============================================================

print("\n========== FEATURE MEDIANS BY PLACEMENT STATUS ==========")

median_by_placement = df.groupby("placement_status")[numerical_features].median().round(2)

print(median_by_placement.T)


# ============================================================
# PLACEMENT RATE BY INTERNSHIP COUNT
# ============================================================

print("\n========== PLACEMENT RATE BY INTERNSHIP COUNT ==========")

internship_placement = (
    pd.crosstab(df["internships_count"], df["placement_status"], normalize="index")
    .mul(100)
    .round(2)
)

print(internship_placement)


# ============================================================
# PLACEMENT RATE BY PROJECT COUNT
# ============================================================

print("\n========== PLACEMENT RATE BY PROJECT COUNT ==========")

project_placement = (
    pd.crosstab(df["projects_count"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(project_placement)


# ============================================================
# PLACEMENT RATE BY CERTIFICATION COUNT
# ============================================================

print("\n========== PLACEMENT RATE BY CERTIFICATION COUNT ==========")

certification_placement = (
    pd.crosstab(df["certifications_count"], df["placement_status"], normalize="index")
    .mul(100)
    .round(2)
)

print(certification_placement)


# ============================================================
# PLACEMENT RATE BY BACKLOGS
# ============================================================

print("\n========== PLACEMENT RATE BY BACKLOGS ==========")

backlog_placement = (
    pd.crosstab(df["backlogs"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(backlog_placement)


# ============================================================
# CGPA GROUPS
# ============================================================

print("\n========== PLACEMENT RATE BY CGPA GROUP ==========")

df["cgpa_group"] = pd.cut(
    df["cgpa"],
    bins=[0, 6, 7, 8, 9, 10],
    labels=["<=6", "6-7", "7-8", "8-9", "9-10"],
    include_lowest=True,
)

cgpa_placement = (
    pd.crosstab(df["cgpa_group"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(cgpa_placement)


# ============================================================
# CODING SKILL GROUPS
# ============================================================

print("\n========== PLACEMENT RATE BY CODING SKILL ==========")

df["coding_group"] = pd.cut(
    df["coding_skill_score"],
    bins=[0, 40, 60, 70, 80, 90, 100],
    labels=["<=40", "40-60", "60-70", "70-80", "80-90", "90-100"],
    include_lowest=True,
)

coding_placement = (
    pd.crosstab(df["coding_group"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(coding_placement)


# ============================================================
# ATTENDANCE GROUPS
# ============================================================

print("\n========== PLACEMENT RATE BY ATTENDANCE ==========")

df["attendance_group"] = pd.cut(
    df["attendance_percentage"],
    bins=[0, 60, 70, 80, 90, 100],
    labels=["<=60", "60-70", "70-80", "80-90", "90-100"],
    include_lowest=True,
)

attendance_placement = (
    pd.crosstab(df["attendance_group"], df["placement_status"], normalize="index").mul(100).round(2)
)

print(attendance_placement)


# ============================================================
# STUDY HOURS
# ============================================================

print("\n========== PLACEMENT RATE BY STUDY HOURS ==========")

df["study_hours_group"] = pd.cut(
    df["study_hours_per_day"],
    bins=[0, 2, 4, 6, 8, 24],
    labels=["<=2", "2-4", "4-6", "6-8", "8+"],
    include_lowest=True,
)

study_placement = (
    pd.crosstab(df["study_hours_group"], df["placement_status"], normalize="index")
    .mul(100)
    .round(2)
)

print(study_placement)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION WITH PLACEMENT ==========")

# Convert target to numerical
df["placement_binary"] = df["placement_status"].map({"Not Placed": 0, "Placed": 1})

correlation_data = df[numerical_features + ["placement_binary"]]

correlations = (
    correlation_data.corr()["placement_binary"]
    .drop("placement_binary")
    .sort_values(ascending=False)
)

print(correlations.round(4))


# ============================================================
# ABSOLUTE CORRELATION RANKING
# ============================================================

print("\n========== FEATURE IMPORTANCE BY ABSOLUTE CORRELATION ==========")

absolute_correlations = correlations.abs().sort_values(ascending=False)

print(absolute_correlations.round(4))


# ============================================================
# SALARY ANALYSIS
# ============================================================

print("\n========== SALARY ANALYSIS - PLACED STUDENTS ==========")

placed_students = df[df["placement_status"] == "Placed"]

print(placed_students["salary_package_lpa"].describe())


# ============================================================
# SALARY BY BRANCH
# ============================================================

print("\n========== AVERAGE SALARY BY BRANCH ==========")

salary_by_branch = (
    placed_students.groupby("branch")["salary_package_lpa"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print(salary_by_branch)


# ============================================================
# SALARY BY COLLEGE TIER
# ============================================================

print("\n========== AVERAGE SALARY BY COLLEGE TIER ==========")

salary_by_tier = (
    placed_students.groupby("college_tier")["salary_package_lpa"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print(salary_by_tier)


# ============================================================
# OUTLIER CHECK
# ============================================================

print("\n========== OUTLIER CHECK ==========")

for column in numerical_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"{column}: {len(outliers)} outliers")


# ============================================================
# CLEANUP TEMPORARY COLUMNS
# ============================================================

df.drop(
    columns=[
        "cgpa_group",
        "coding_group",
        "attendance_group",
        "study_hours_group",
        "placement_binary",
    ],
    inplace=True,
)


print("\n========== EDA COMPLETE ==========")
