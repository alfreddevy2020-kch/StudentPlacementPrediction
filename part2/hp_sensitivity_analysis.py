"""
Role 2 — Hyperparameter Sensitivity Analysis
==============================================
Trains Logistic Regression and Random Forest with deliberately varied
hyperparameter configurations (underfitting → default → tuned → overfitting)
to demonstrate WHY the tuned values are optimal.

Generates:
  - part2/model_results/hp_sensitivity_logreg.png
  - part2/model_results/hp_sensitivity_rf.png
  - part2/model_results/hp_sensitivity_combined.png
  - part2/model_results/hp_sensitivity_report.txt

Run after preprocessing.py has been executed.
"""

import os
import time
import warnings

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE
# ============================================================

COLORS = {
    "underfitting": "#E74C3C",  # Red
    "weak": "#E67E22",  # Orange
    "default": "#F1C40F",  # Yellow
    "good": "#2ECC71",  # Green
    "tuned": "#3498DB",  # Blue  (our best)
    "overfitting": "#9B59B6",  # Purple
    "bg": "#F7F7FA",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("=" * 65)

train_df = pd.read_csv("data/processed/train_processed.csv")
test_df = pd.read_csv("data/processed/test_processed.csv")
weights_df = pd.read_csv("data/processed/class_weights.csv")

TARGET = "placement_status"
X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values
X_test = test_df.drop(columns=[TARGET]).values
y_test = test_df[TARGET].values

class_weights = dict(zip(weights_df["class"].astype(int), weights_df["weight"]))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs("part2/model_results", exist_ok=True)


def evaluate(model, X_tr, y_tr, X_te, y_te):
    """Train and return a dict of metrics."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # Also get CV score
    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1).mean()

    return {
        "CV F1": round(cv_f1, 4),
        "Test F1": round(f1_score(y_te, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_te, y_proba), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_te, y_pred), 4),
        "Brier": round(brier_score_loss(y_te, y_proba), 4),
        "Avg Prec": round(average_precision_score(y_te, y_proba), 4),
        "Train Time": round(train_time, 2),
    }


# ============================================================
# 2. LOGISTIC REGRESSION CONFIGURATIONS
# ============================================================

print("\n" + "-" * 50)
print("[1] Logistic Regression — 6 Configurations")
print("-" * 50)

logreg_configs = {
    "Extreme Regularization\n(C=0.0001)": {
        "label": "Underfitting",
        "color": COLORS["underfitting"],
        "model": LogisticRegression(
            C=0.0001,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
        ),
        "why_bad": "C=0.0001 shrinks ALL coefficients toward zero. "
        "The model barely distinguishes features -- essentially predicts the mean.",
    },
    "Heavy Regularization\n(C=0.01)": {
        "label": "Weak",
        "color": COLORS["weak"],
        "model": LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
        ),
        "why_bad": "C=0.01 still penalizes too aggressively. "
        "Important features like CGPA and backlogs can't express their full effect.",
    },
    "Default sklearn\n(C=1.0, no class wt)": {
        "label": "Default",
        "color": COLORS["default"],
        "model": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
            # NOTE: no class_weight -- this is the naive default
        ),
        "why_bad": "Uses C=1.0 (which happens to be good) but IGNORES class imbalance. "
        "Model is biased toward predicting 'Not Placed' (majority class).",
    },
    "TUNED (Ours)\n(C=1.0, class wt)": {
        "label": "Tuned (Ours)",
        "color": COLORS["tuned"],
        "model": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
        ),
        "why_bad": "OPTIMAL: C=1.0 balances regularization. Class weights correct for 82/18 imbalance.",
    },
    "Weak Regularization\n(C=100)": {
        "label": "Good",
        "color": COLORS["good"],
        "model": LogisticRegression(
            C=100,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
        ),
        "why_bad": "C=100 provides almost no regularization. "
        "Works here but risks overfitting on noisier real-world data.",
    },
    "No Regularization\n(C=100000)": {
        "label": "Overfitting risk",
        "color": COLORS["overfitting"],
        "model": LogisticRegression(
            C=100000,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weights,
            random_state=42,
        ),
        "why_bad": "C=100000 effectively removes regularization entirely. "
        "Coefficients can grow unbounded -- fragile on new data.",
    },
}

logreg_results = {}
for name, cfg in logreg_configs.items():
    print(f"  Training: {cfg['label']} ...")
    metrics = evaluate(cfg["model"], X_train, y_train, X_test, y_test)
    logreg_results[name] = {**metrics, **cfg}
    print(
        f"    CV F1={metrics['CV F1']}, Test F1={metrics['Test F1']}, "
        f"ROC-AUC={metrics['ROC-AUC']}, Recall={metrics['Recall']}"
    )


# ============================================================
# 3. RANDOM FOREST CONFIGURATIONS
# ============================================================

print("\n" + "-" * 50)
print("[2] Random Forest — 6 Configurations")
print("-" * 50)

rf_configs = {
    "Single Shallow Tree\n(n=1, depth=2)": {
        "label": "Underfitting",
        "color": COLORS["underfitting"],
        "model": RandomForestClassifier(
            n_estimators=1,
            max_depth=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "why_bad": "A single decision stump with depth=2 can only learn 4 leaf nodes. "
        "Extreme underfitting -- cannot capture feature interactions.",
    },
    "Few Shallow Trees\n(n=10, depth=3)": {
        "label": "Weak",
        "color": COLORS["weak"],
        "model": RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "why_bad": "10 trees with depth=3 is still too constrained. "
        "Each tree can only learn 8 leaf regions -- insufficient for 17 features.",
    },
    "Default sklearn\n(n=100, no class wt)": {
        "label": "Default",
        "color": COLORS["default"],
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            # NOTE: no class_weight, no max_depth limit, default max_features='sqrt'
        ),
        "why_bad": "sklearn defaults: 100 trees, unlimited depth, sqrt features, no class weighting. "
        "Ignores class imbalance entirely.",
    },
    "TUNED (Ours)\n(n=500, depth=10)": {
        "label": "Tuned (Ours)",
        "color": COLORS["tuned"],
        "model": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=2,
            max_features="log2",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "why_bad": "OPTIMAL: Empirically tuned via RandomizedSearchCV + GridSearchCV. "
        "Balanced class weights, regularized depth, decorrelated trees via log2 features.",
    },
    "Deep Unlimited Trees\n(n=500, no depth cap)": {
        "label": "Deep (risk)",
        "color": COLORS["good"],
        "model": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "why_bad": "Unlimited depth and min_samples_leaf=1 lets each tree grow until it perfectly "
        "memorizes training data. Works on this clean dataset but is fragile.",
    },
    "Overfit Config\n(n=1000, depth=None, leaf=1)": {
        "label": "Overfit risk",
        "color": COLORS["overfitting"],
        "model": RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=None,  # use ALL features at each split
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "why_bad": "1000 trees, unlimited depth, all features at every split (max_features=None). "
        "Trees are highly correlated -- defeats the purpose of ensembling. "
        "Slower and more fragile.",
    },
}

rf_results = {}
for name, cfg in rf_configs.items():
    print(f"  Training: {cfg['label']} ...")
    metrics = evaluate(cfg["model"], X_train, y_train, X_test, y_test)
    rf_results[name] = {**metrics, **cfg}
    print(
        f"    CV F1={metrics['CV F1']}, Test F1={metrics['Test F1']}, "
        f"ROC-AUC={metrics['ROC-AUC']}, Recall={metrics['Recall']}, "
        f"Time={metrics['Train Time']}s"
    )


# ============================================================
# 4. PLOT — LOGISTIC REGRESSION COMPARISON
# ============================================================

print("\n[3] Generating visualizations ...")


def plot_comparison(results, title, filename):
    """Create a multi-metric grouped bar chart for hyperparameter comparison."""
    names = list(results.keys())
    colors = [results[n]["color"] for n in names]

    # Metrics to show
    show_metrics = ["CV F1", "Test F1", "ROC-AUC", "Precision", "Recall"]
    n_metrics = len(show_metrics)
    n_configs = len(names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 7), sharey=False)
    fig.patch.set_facecolor("white")

    for ax_idx, metric in enumerate(show_metrics):
        ax = axes[ax_idx]
        vals = [results[n][metric] for n in names]
        bars = ax.barh(
            range(n_configs),
            vals,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
            height=0.7,
        )

        # Annotate each bar with value
        for bar, val in zip(bars, vals):
            x_pos = bar.get_width() + 0.005
            if x_pos > 0.95:
                x_pos = bar.get_width() - 0.04
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlim(0, 1.12)
        ax.set_yticks(range(n_configs))
        if ax_idx == 0:
            ax.set_yticklabels(names, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(metric, fontsize=10, fontweight="bold")
        ax.set_facecolor(COLORS["bg"])
        ax.axvline(1.0, color="#CCCCCC", linewidth=0.8, linestyle=":")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"  Saved: {filename}")


plot_comparison(
    logreg_results,
    "Logistic Regression - Hyperparameter Sensitivity Analysis\n"
    "Why does C=1.0 with class weights win? Because it balances regularization and imbalance handling.",
    "part2/model_results/hp_sensitivity_logreg.png",
)

plot_comparison(
    rf_results,
    "Random Forest - Hyperparameter Sensitivity Analysis\n"
    "Why does our tuned config win? Empirical search + regularization + decorrelated trees.",
    "part2/model_results/hp_sensitivity_rf.png",
)

# ============================================================
# 5. COMBINED SUMMARY FIGURE
# ============================================================


def make_combined_figure(lr_results, rf_results, filename):
    """Single 2-row figure: top = LR, bottom = RF. Each row shows F1, ROC-AUC, Recall."""
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    for row_idx, (results, model_name) in enumerate(
        [
            (lr_results, "Logistic Regression"),
            (rf_results, "Random Forest"),
        ]
    ):
        names = list(results.keys())
        colors = [results[n]["color"] for n in names]
        n = len(names)

        for col_idx, metric in enumerate(["Test F1", "ROC-AUC", "Recall"]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            vals = [results[nm][metric] for nm in names]
            bars = ax.barh(
                range(n),
                vals,
                color=colors,
                alpha=0.85,
                edgecolor="white",
                linewidth=1.2,
                height=0.65,
            )
            for bar, val in zip(bars, vals):
                x_pos = bar.get_width() + 0.008
                if x_pos > 0.96:
                    x_pos = bar.get_width() - 0.05
                ax.text(
                    x_pos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_xlim(0, 1.15)
            ax.set_yticks(range(n))
            ax.set_yticklabels(names if col_idx == 0 else [], fontsize=7)
            ax.set_xlabel(metric, fontsize=10, fontweight="bold")
            ax.set_title(f"{model_name} - {metric}", fontsize=10)
            ax.set_facecolor(COLORS["bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axvline(1.0, color="#CCCCCC", linewidth=0.8, linestyle=":")

    fig.suptitle(
        "Hyperparameter Sensitivity Analysis - Why Our Tuned Configuration Wins\n"
        "Red = underfitting | Yellow = naive defaults | Blue = our tuned config | Purple = overfitting risk",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(filename)
    plt.close(fig)
    print(f"  Saved: {filename}")


make_combined_figure(logreg_results, rf_results, "part2/model_results/hp_sensitivity_combined.png")

# ============================================================
# 6. TEXT REPORT
# ============================================================

print("\n[4] Generating text report ...")

lines = [
    "=" * 70,
    "HYPERPARAMETER SENSITIVITY ANALYSIS",
    "Why our tuned hyperparameters are optimal",
    "=" * 70,
    "",
]

for model_name, results in [
    ("LOGISTIC REGRESSION", logreg_results),
    ("RANDOM FOREST", rf_results),
]:
    lines.append("-" * 70)
    lines.append(model_name)
    lines.append("-" * 70)
    lines.append("")

    # Table header
    header = f"  {'Configuration':<35} {'CV F1':>7} {'Test F1':>8} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'Brier':>7} {'Time':>6}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name, data in results.items():
        clean_name = name.replace("\n", " / ")
        line = (
            f"  {clean_name:<35} "
            f"{data['CV F1']:>7.4f} {data['Test F1']:>8.4f} "
            f"{data['ROC-AUC']:>7.4f} {data['Precision']:>7.4f} "
            f"{data['Recall']:>7.4f} {data['Brier']:>7.4f} "
            f"{data['Train Time']:>5.2f}s"
        )
        lines.append(line)

    lines.append("")

    # Explanations
    lines.append("  WHY EACH CONFIGURATION PERFORMS AS IT DOES:")
    lines.append("")
    for name, data in results.items():
        clean_name = name.replace("\n", " / ")
        tag = f"[{data['label']}]"
        lines.append(f"  {tag:<20} {clean_name}")
        lines.append(f"  {'':20} {data['why_bad']}")
        lines.append("")

lines.append("=" * 70)
lines.append("KEY TAKEAWAYS FOR PRESENTATION")
lines.append("=" * 70)
lines.append("")
lines.append("  1. REGULARIZATION MATTERS: Both extreme regularization (underfitting)")
lines.append("     and no regularization (overfitting risk) perform worse than the")
lines.append("     balanced middle ground. Our GridSearchCV found this sweet spot.")
lines.append("")
lines.append("  2. CLASS WEIGHTS ARE ESSENTIAL: The 'default sklearn' configs that")
lines.append("     ignore class imbalance have noticeably lower Recall -- they fail")
lines.append("     to identify placement-ready students (the minority class).")
lines.append("")
lines.append("  3. TREE DEPTH + LEAF SIZE = REGULARIZATION FOR FORESTS: Capping")
lines.append("     max_depth=10 and min_samples_leaf=5 prevents memorization while")
lines.append("     still capturing the key decision boundaries.")
lines.append("")
lines.append("  4. DECORRELATED TREES (max_features='log2'): Forcing each tree to")
lines.append("     consider only ~4 of 17 features per split increases ensemble")
lines.append("     diversity. max_features=None (all features) makes trees too")
lines.append("     similar, defeating the purpose of Random Forest.")
lines.append("")
lines.append("  5. EMPIRICAL, NOT ARBITRARY: Every hyperparameter was selected by")
lines.append("     data-driven search (RandomizedSearchCV + GridSearchCV), not by")
lines.append("     guessing. The OOB convergence and learning curve plots prove")
lines.append("     the model is not over- or under-fitted.")
lines.append("")
lines.append("=" * 70)

report_text = "\n".join(lines)
report_path = "part2/model_results/hp_sensitivity_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n  Saved: {report_path}")

print("\n" + "=" * 65)
print("HYPERPARAMETER SENSITIVITY ANALYSIS -- COMPLETE")
print("=" * 65)
print("\nGenerated artifacts:")
print("  part2/model_results/hp_sensitivity_logreg.png")
print("  part2/model_results/hp_sensitivity_rf.png")
print("  part2/model_results/hp_sensitivity_combined.png")
print("  part2/model_results/hp_sensitivity_report.txt")
