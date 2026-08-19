"""
visualization.py
----------------
Exploratory visualisations for the placement dataset.

Driven by the canonical feature lists in feature_engineering.py rather than
hardcoded column names, so a future dataset change updates the plots by
editing one module instead of every section here.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feature_engineering import (
    load_raw_dataset,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/raw/student_placement.csv"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Count-style features get a bar per discrete value; continuous ones are
# binned into quantiles first.
DISCRETE_FEATURES = ["internships", "projects", "workshops_certifications"]
CONTINUOUS_FEATURES = [
    "cgpa", "ssc_marks", "hsc_marks", "aptitude_test_score", "soft_skills_rating",
]

# ============================================================
# LOAD DATA
# ============================================================

df = load_raw_dataset(DATA_FILE)

print("=" * 60)
print("VISUALIZATION DATASET")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# TARGET
# ============================================================

df["placement_binary"] = (
    df[TARGET_COLUMN].str.strip().str.lower() == "placed"
).astype(int)

# ============================================================
# HELPER FUNCTIONS
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


def plot_rate_by_group(series, title, xlabel, filename, rotate=False):
    """Bar chart of placement rate (%) for each value of `series`."""
    rate = df.groupby(series, observed=True)["placement_binary"].mean().mul(100)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=rate.index.astype(str), y=rate.values)

    for i, value in enumerate(rate.values):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Placement Rate (%)")
    plt.ylim(0, min(105, max(rate.values) + 12))
    if rotate:
        plt.xticks(rotation=30, ha="right")

    save_plot(filename)


# ============================================================
# 1. PLACEMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x=TARGET_COLUMN)
plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
save_plot("placement_distribution.png")


# ============================================================
# 2. PLACEMENT PERCENTAGE
# ============================================================

placement_percentage = (
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
)

plt.figure(figsize=(8, 6))
ax = sns.barplot(
    x=placement_percentage.index.astype(str),
    y=placement_percentage.values,
)

for i, value in enumerate(placement_percentage.values):
    ax.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.title("Placement Percentage")
plt.xlabel("Placement Status")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
save_plot("placement_percentage.png")


# ============================================================
# 3. PLACEMENT RATE BY DISCRETE COUNT FEATURES
# ============================================================

for column in DISCRETE_FEATURES:
    plot_rate_by_group(
        df[column],
        title=f"Placement Rate by {column.replace('_', ' ').title()}",
        xlabel=column.replace("_", " ").title(),
        filename=f"{column}_vs_placement.png",
    )


# ============================================================
# 4. PLACEMENT RATE BY BINNED CONTINUOUS FEATURES
# ============================================================

for column in CONTINUOUS_FEATURES:
    # Quantile bins keep group sizes comparable across differently-shaped
    # distributions; duplicates="drop" guards narrow-range columns.
    binned = pd.qcut(df[column], q=5, duplicates="drop")
    plot_rate_by_group(
        binned,
        title=f"Placement Rate by {column.replace('_', ' ').title()}",
        xlabel=f"{column.replace('_', ' ').title()} (quintile)",
        filename=f"{column}_vs_placement.png",
        rotate=True,
    )


# ============================================================
# 5. PLACEMENT RATE BY CATEGORICAL FEATURES
# ============================================================

for column in RAW_CATEGORICAL_FEATURES:
    plot_rate_by_group(
        df[column],
        title=f"Placement Rate by {column.replace('_', ' ').title()}",
        xlabel=column.replace("_", " ").title(),
        filename=f"{column}_vs_placement.png",
    )


# ============================================================
# 6. FEATURE CORRELATION WITH PLACEMENT
# ============================================================

correlations = (
    df[RAW_NUMERICAL_FEATURES + ["placement_binary"]]
    .corr()["placement_binary"]
    .drop("placement_binary")
    .sort_values()
)

plt.figure(figsize=(9, 6))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title("Feature Correlation with Placement")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.axvline(0, color="black", linewidth=0.8)
save_plot("placement_correlation.png")


# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

correlation_matrix = df[RAW_NUMERICAL_FEATURES + ["placement_binary"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 8},
)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
save_plot("correlation_heatmap.png")


print("\n" + "=" * 60)
print(f"All visualisations written to {OUTPUT_DIR}/")
print("=" * 60)
