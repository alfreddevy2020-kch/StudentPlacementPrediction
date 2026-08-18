import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/raw/student_placement.csv"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("VISUALIZATION DATASET")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# TARGET
# ============================================================

df["placement_binary"] = (
    df["placement_status"].str.strip().str.lower() == "placed"
).astype(int)

# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Created: {filename}")


# ============================================================
# 1. PLACEMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="placement_status"
)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

save_plot("placement_distribution.png")


# ============================================================
# 2. PLACEMENT PERCENTAGE
# ============================================================

placement_percentage = (
    df["placement_status"]
    .value_counts(normalize=True)
    .mul(100)
)

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=placement_percentage.index,
    y=placement_percentage.values
)

for i, value in enumerate(placement_percentage.values):
    ax.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.title("Placement Percentage")
plt.xlabel("Placement Status")
plt.ylabel("Percentage (%)")
plt.ylim(0, 65)

save_plot("placement_percentage.png")


# ============================================================
# 3. PLACEMENT VS INTERNSHIPS
# ============================================================

internship_rate = (
    df.groupby("internships_count")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=internship_rate.index,
    y=internship_rate.values
)

plt.title("Placement Rate by Internship Count")
plt.xlabel("Number of Internships")
plt.ylabel("Placement Rate (%)")

save_plot("internships_vs_placement.png")


# ============================================================
# 4. PLACEMENT VS PROJECTS
# ============================================================

project_rate = (
    df.groupby("projects_count")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=project_rate.index,
    y=project_rate.values
)

plt.title("Placement Rate by Project Count")
plt.xlabel("Number of Projects")
plt.ylabel("Placement Rate (%)")

save_plot("projects_vs_placement.png")


# ============================================================
# 5. PLACEMENT VS BACKLOGS
# ============================================================

backlog_rate = (
    df.groupby("backlogs")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=backlog_rate.index,
    y=backlog_rate.values
)

plt.title("Placement Rate by Number of Backlogs")
plt.xlabel("Number of Backlogs")
plt.ylabel("Placement Rate (%)")

save_plot("backlogs_vs_placement.png")


# ============================================================
# 6. PLACEMENT VS CGPA
# ============================================================

df["cgpa_group"] = pd.cut(
    df["cgpa"],
    bins=[0, 6, 7, 8, 9, 10],
    labels=["<=6", "6-7", "7-8", "8-9", "9-10"],
    include_lowest=True
)

cgpa_rate = (
    df.groupby("cgpa_group", observed=False)["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=cgpa_rate.index.astype(str),
    y=cgpa_rate.values
)

plt.title("Placement Rate by CGPA Group")
plt.xlabel("CGPA Group")
plt.ylabel("Placement Rate (%)")

save_plot("cgpa_vs_placement.png")


# ============================================================
# 7. CODING SKILL VS PLACEMENT
# ============================================================

df["coding_group"] = pd.cut(
    df["coding_skill_score"],
    bins=[0, 40, 60, 70, 80, 90, 100],
    labels=[
        "<=40",
        "40-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100"
    ],
    include_lowest=True
)

coding_rate = (
    df.groupby("coding_group", observed=False)["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=coding_rate.index.astype(str),
    y=coding_rate.values
)

plt.title("Placement Rate by Coding Skill")
plt.xlabel("Coding Skill Score")
plt.ylabel("Placement Rate (%)")

save_plot("coding_skill_vs_placement.png")


# ============================================================
# 8. ATTENDANCE VS PLACEMENT
# ============================================================

df["attendance_group"] = pd.cut(
    df["attendance_percentage"],
    bins=[0, 60, 70, 80, 90, 100],
    labels=[
        "<=60",
        "60-70",
        "70-80",
        "80-90",
        "90-100"
    ],
    include_lowest=True
)

attendance_rate = (
    df.groupby("attendance_group", observed=False)["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=attendance_rate.index.astype(str),
    y=attendance_rate.values
)

plt.title("Placement Rate by Attendance")
plt.xlabel("Attendance Percentage")
plt.ylabel("Placement Rate (%)")

save_plot("attendance_vs_placement.png")


# ============================================================
# 9. STUDY HOURS VS PLACEMENT
# ============================================================

df["study_hours_group"] = pd.cut(
    df["study_hours_per_day"],
    bins=[-1, 2, 4, 6, 8, float("inf")],
    labels=[
        "<=2",
        "2-4",
        "4-6",
        "6-8",
        "8+"
    ]
)

study_rate = (
    df.groupby("study_hours_group", observed=False)["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=study_rate.index.astype(str),
    y=study_rate.values
)

plt.title("Placement Rate by Study Hours per Day")
plt.xlabel("Study Hours per Day")
plt.ylabel("Placement Rate (%)")

save_plot("study_hours_vs_placement.png")


# ============================================================
# 10. PLACEMENT RATE BY BRANCH
# ============================================================

branch_rate = (
    df.groupby("branch")["placement_binary"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=branch_rate.index,
    y=branch_rate.values
)

plt.title("Placement Rate by Branch")
plt.xlabel("Branch")
plt.ylabel("Placement Rate (%)")

save_plot("branch_vs_placement.png")


# ============================================================
# 11. PLACEMENT RATE BY COLLEGE TIER
# ============================================================

tier_rate = (
    df.groupby("college_tier")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=tier_rate.index,
    y=tier_rate.values
)

plt.title("Placement Rate by College Tier")
plt.xlabel("College Tier")
plt.ylabel("Placement Rate (%)")

save_plot("college_tier_vs_placement.png")


# ============================================================
# 12. PLACEMENT RATE BY GENDER
# ============================================================

gender_rate = (
    df.groupby("gender")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=gender_rate.index,
    y=gender_rate.values
)

plt.title("Placement Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Placement Rate (%)")

save_plot("gender_vs_placement.png")


# ============================================================
# 13. PLACEMENT RATE BY VOLUNTEER EXPERIENCE
# ============================================================

volunteer_rate = (
    df.groupby("volunteer_experience")["placement_binary"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=volunteer_rate.index,
    y=volunteer_rate.values
)

plt.title("Placement Rate by Volunteer Experience")
plt.xlabel("Volunteer Experience")
plt.ylabel("Placement Rate (%)")

save_plot("volunteer_vs_placement.png")


# ============================================================
# 14. FEATURE CORRELATION WITH PLACEMENT
# ============================================================

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

correlations = (
    df[numeric_columns]
    .corr()["placement_binary"]
    .drop("placement_binary")
    .sort_values()
)

plt.figure(figsize=(10, 8))

correlations.plot(kind="barh")

plt.title("Feature Correlation with Placement")
plt.xlabel("Correlation")
plt.ylabel("Feature")

save_plot("placement_correlation.png")


# ============================================================
# 15. CORRELATION HEATMAP
# ============================================================

selected_features = [
    "cgpa",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs",
    "extracurricular_score",
    "leadership_score",
    "sleep_hours",
    "study_hours_per_day",
    "placement_binary"
]

correlation_matrix = df[selected_features].corr()

plt.figure(figsize=(14, 11))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Placement Features")

save_plot("correlation_heatmap.png")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print(f"All visualizations saved in: {OUTPUT_DIR}/")
print()