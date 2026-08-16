import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("data/raw/student_placement.csv")


# ==============================
# 2. Remove irrelevant columns
# ==============================

# student_id is only an identifier.
# salary_package_lpa should not be used for placement analysis
# because it is related to the outcome.
df = df.drop(columns=["student_id", "salary_package_lpa"])


# ==============================
# 3. Create output folder
# ==============================

import os

os.makedirs("visualizations", exist_ok=True)


# ==============================
# 4. Set plotting style
# ==============================

sns.set_theme(style="whitegrid")


# ============================================================
# 5. PLACEMENT STATUS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

placement_counts = df["placement_status"].value_counts()

plt.bar(
    ["Not Placed", "Placed"],
    [
        placement_counts.get(0, 0),
        placement_counts.get(1, 0)
    ]
)

plt.title("Student Placement Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    "visualizations/placement_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. PLACEMENT STATUS PIE CHART
# ============================================================

plt.figure(figsize=(7, 7))

plt.pie(
    placement_counts,
    labels=["Not Placed", "Placed"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage of Students Placed")

plt.tight_layout()

plt.savefig(
    "visualizations/placement_percentage.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. CGPA VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="placement_status",
    y="cgpa"
)

plt.title("CGPA Distribution by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("CGPA")

plt.tight_layout()

plt.savefig(
    "visualizations/cgpa_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. TECHNICAL SKILL VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="placement_status",
    y="technical_skill_score"
)

plt.title("Technical Skill Score by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Technical Skill Score")

plt.tight_layout()

plt.savefig(
    "visualizations/technical_skill_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. SOFT SKILL VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="placement_status",
    y="soft_skill_score"
)

plt.title("Soft Skill Score by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Soft Skill Score")

plt.tight_layout()

plt.savefig(
    "visualizations/soft_skill_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. BACKLOGS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="backlogs",
    hue="placement_status"
)

plt.title("Backlogs and Placement Status")
plt.xlabel("Number of Backlogs")
plt.ylabel("Number of Students")

plt.legend(
    title="Placement Status",
    labels=["Not Placed", "Placed"]
)

plt.tight_layout()

plt.savefig(
    "visualizations/backlogs_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. INTERNSHIP COUNT VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="internship_count",
    hue="placement_status"
)

plt.title("Internship Count and Placement Status")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")

plt.legend(
    title="Placement Status",
    labels=["Not Placed", "Placed"]
)

plt.tight_layout()

plt.savefig(
    "visualizations/internships_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. LIVE PROJECTS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="live_projects",
    hue="placement_status"
)

plt.title("Live Projects and Placement Status")
plt.xlabel("Number of Live Projects")
plt.ylabel("Number of Students")

plt.legend(
    title="Placement Status",
    labels=["Not Placed", "Placed"]
)

plt.tight_layout()

plt.savefig(
    "visualizations/projects_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. WORK EXPERIENCE VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="placement_status",
    y="work_experience_months"
)

plt.title("Work Experience by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Work Experience (Months)")

plt.tight_layout()

plt.savefig(
    "visualizations/work_experience_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. ATTENDANCE VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="placement_status",
    y="attendance_percentage"
)

plt.title("Attendance Percentage by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Attendance Percentage")

plt.tight_layout()

plt.savefig(
    "visualizations/attendance_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. GENDER VS PLACEMENT
# ============================================================

gender_placement = pd.crosstab(
    df["gender"],
    df["placement_status"]
)

gender_placement.plot(
    kind="bar",
    figsize=(8, 6)
)

plt.title("Gender and Placement Status")
plt.xlabel("Gender")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.legend(
    ["Not Placed", "Placed"],
    title="Placement Status"
)

plt.tight_layout()

plt.savefig(
    "visualizations/gender_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. EXTRACURRICULAR ACTIVITIES VS PLACEMENT
# ============================================================

extra_placement = pd.crosstab(
    df["extracurricular_activities"],
    df["placement_status"]
)

extra_placement.plot(
    kind="bar",
    figsize=(8, 6)
)

plt.title("Extracurricular Activities and Placement Status")
plt.xlabel("Extracurricular Activities")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.legend(
    ["Not Placed", "Placed"],
    title="Placement Status"
)

plt.tight_layout()

plt.savefig(
    "visualizations/extracurricular_vs_placement.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. CORRELATION HEATMAP
# ============================================================

correlation_df = df.copy()

# Convert categorical values to numerical values
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


plt.figure(figsize=(14, 10))

correlation_matrix = correlation_df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. CORRELATION WITH PLACEMENT
# ============================================================

placement_correlation = (
    correlation_df
    .corr(numeric_only=True)["placement_status"]
    .drop("placement_status")
    .sort_values()
)

plt.figure(figsize=(10, 7))

placement_correlation.plot(
    kind="barh"
)

plt.title("Feature Correlation with Placement Status")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "visualizations/placement_correlation.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("VISUALIZATION COMPLETE")
print("==========================================")
print("All graphs have been saved in:")
print("visualizations/")
print("==========================================")