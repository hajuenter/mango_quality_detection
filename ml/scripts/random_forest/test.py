import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from datetime import datetime
import joblib
import sys
from collections import Counter
import pandas as pd

from features import extract_features

TEST_DIR = "ml/dataset/mango_dataset_ml_split/test"
MODEL_PATH = "ml/models/random_forest/random_forest_mango.pkl"
RESULTS_DIR = "ml/results/random_forest/test"

THRESHOLD_ROTTEN = 0.65

if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = os.path.join(RESULTS_DIR, "test_log.txt")
sys.stdout = open(log_path, "w")

print("=" * 60)
print("EVALUASI MODEL RANDOM FOREST - TEST SET")
print("=" * 60)

print("\n[1/5] Loading model...")
rf_best = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")
print(f"Model type: {type(rf_best).__name__}")
print(f"Number of estimators: {rf_best.n_estimators}")
print(f"Max depth: {rf_best.max_depth}")

print("\n[2/5] Loading test dataset...")
X_test, y_test = [], []
classes = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)

for label in classes:
    folder_path = os.path.join(TEST_DIR, label)
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"  - {label}: {len(files)} images")
    for f in files:
        X_test.append(extract_features(f))
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"\nTest set loaded: {len(X_test)} samples, {X_test.shape[1]} features")
test_dist = Counter(y_test)
print(f"Distribution: {dict(test_dist)}")


print("\n[3/5] Predicting...")

y_pred_default = rf_best.predict(X_test)

y_pred_proba = rf_best.predict_proba(X_test)
rotten_idx = list(rf_best.classes_).index("mango_rotten")
y_pred_threshold = np.array(
    [
        "mango_rotten" if proba[rotten_idx] >= THRESHOLD_ROTTEN else "mango_healthy"
        for proba in y_pred_proba
    ]
)

print(f"Predictions completed")
print(f"  - Default method (threshold 0.5)")
print(f"  - Threshold method (threshold {THRESHOLD_ROTTEN})")


print("\n[4/5] Computing metrics...")


def compute_metrics(y_true, y_pred, method_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n{method_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


metrics_default = compute_metrics(y_test, y_pred_default, "Default Method (0.5)")
metrics_threshold = compute_metrics(
    y_test, y_pred_threshold, f"Threshold Method ({THRESHOLD_ROTTEN})"
)


results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": MODEL_PATH,
    "test_samples": len(y_test),
    "feature_dimension": int(X_test.shape[1]),
    "classes": classes,
    "distribution": dict(test_dist),
    "methods": {
        "default": {
            "threshold": 0.5,
            "metrics": metrics_default,
            "per_class": {},
        },
        "threshold_tuned": {
            "threshold": float(THRESHOLD_ROTTEN),
            "metrics": metrics_threshold,
            "per_class": {},
        },
    },
}


for method_key, report in [
    ("default", metrics_default["report"]),
    ("threshold_tuned", metrics_threshold["report"]),
]:
    for cls in classes:
        if cls in report:
            results["methods"][method_key]["per_class"][cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1_score": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

json_path = os.path.join(RESULTS_DIR, "test_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\n[5/5] Results saved to: {json_path}")

print("\nGenerating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm_default = metrics_default["confusion_matrix"]
cm_threshold = metrics_threshold["confusion_matrix"]

for ax, cm, title, method in zip(
    axes,
    [cm_default, cm_threshold],
    [
        f"Default Method (Threshold 0.5)\nAcc: {metrics_default['accuracy']:.4f}",
        f"Threshold Tuned ({THRESHOLD_ROTTEN})\nAcc: {metrics_threshold['accuracy']:.4f}",
    ],
    ["default", "threshold"],
):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_comparison.png"), dpi=300)
print("Confusion matrix comparison saved")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_names = ["Precision", "Recall", "F1-Score"]
metric_keys = ["precision", "recall", "f1_score"]

for ax, metric_name, metric_key in zip(axes, metrics_names, metric_keys):
    default_values = [
        results["methods"]["default"]["per_class"][cls][metric_key] for cls in classes
    ]
    threshold_values = [
        results["methods"]["threshold_tuned"]["per_class"][cls][metric_key]
        for cls in classes
    ]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, default_values, width, label="Default (0.5)", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        threshold_values,
        width,
        label=f"Threshold ({THRESHOLD_ROTTEN})",
        alpha=0.8,
    )

    ax.set_xlabel("Classes", fontweight="bold")
    ax.set_ylabel(metric_name, fontweight="bold")
    ax.set_title(f"{metric_name} Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_class_metrics_comparison.png"), dpi=300)
print("Per-class metrics comparison saved")


fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
default_values = [
    metrics_default["accuracy"],
    metrics_default["precision"],
    metrics_default["recall"],
    metrics_default["f1_score"],
]
threshold_values = [
    metrics_threshold["accuracy"],
    metrics_threshold["precision"],
    metrics_threshold["recall"],
    metrics_threshold["f1_score"],
]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    default_values,
    width,
    label="Default (0.5)",
    alpha=0.8,
    color="#3498db",
)
bars2 = ax.bar(
    x + width / 2,
    threshold_values,
    width,
    label=f"Threshold ({THRESHOLD_ROTTEN})",
    alpha=0.8,
    color="#e74c3c",
)

ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Overall Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis="y", alpha=0.3)


for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overall_metrics_comparison.png"), dpi=300)
print("Overall metrics comparison saved")


if len(classes) == 2:
    fig, ax = plt.subplots(figsize=(10, 8))

    y_test_binary = (y_test == "mango_rotten").astype(int)
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba[:, rotten_idx])
    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    default_idx = np.argmin(np.abs(thresholds - 0.5))
    threshold_idx = np.argmin(np.abs(thresholds - THRESHOLD_ROTTEN))

    ax.plot(
        fpr[default_idx], tpr[default_idx], "go", markersize=10, label=f"Default (0.5)"
    )
    ax.plot(
        fpr[threshold_idx],
        tpr[threshold_idx],
        "ro",
        markersize=10,
        label=f"Tuned ({THRESHOLD_ROTTEN})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curve - Mango Rotten Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
    print("ROC curve saved")

plt.close("all")

print("\n[6/6] Exporting analysis results with tree voting to Excel...")

feature_importances = rf_best.feature_importances_
feature_names = [f"feature_{i+1}" for i in range(len(feature_importances))]

df_features = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)

per_class_data = []
for cls in classes:
    per_class_data.append(
        {
            "Class": cls,
            "Precision": results["methods"]["threshold_tuned"]["per_class"][cls][
                "precision"
            ],
            "Recall": results["methods"]["threshold_tuned"]["per_class"][cls]["recall"],
            "F1-Score": results["methods"]["threshold_tuned"]["per_class"][cls][
                "f1_score"
            ],
            "Support": results["methods"]["threshold_tuned"]["per_class"][cls][
                "support"
            ],
        }
    )
df_per_class = pd.DataFrame(per_class_data)

df_predictions = pd.DataFrame(
    {
        "True Label": y_test,
        "Predicted Label": y_pred_threshold,
        "Proba Rotten": y_pred_proba[:, rotten_idx],
    }
)

print("Mengambil voting dari setiap pohon dalam Random Forest...")

all_tree_preds = np.array([tree.predict(X_test) for tree in rf_best.estimators_]).T

if np.issubdtype(all_tree_preds.dtype, np.number):
    label_map = {i: cls for i, cls in enumerate(rf_best.classes_)}
    all_tree_preds = np.vectorize(label_map.get)(all_tree_preds)

voting_data = []
for i, (true_label, final_pred, probs) in enumerate(
    zip(y_test, y_pred_threshold, y_pred_proba)
):
    votes = all_tree_preds[i]
    count_healthy = np.sum(votes == "mango_healthy")
    count_rotten = np.sum(votes == "mango_rotten")
    total_trees = len(votes)
    percent_rotten = count_rotten / total_trees
    percent_healthy = count_healthy / total_trees

    voting_data.append(
        {
            "Sample Index": i + 1,
            "True Label": true_label,
            "Predicted Label": final_pred,
            "Proba Rotten": probs[rotten_idx],
            "Votes Healthy": int(count_healthy),
            "Votes Rotten": int(count_rotten),
            "Total Trees": total_trees,
            "% Healthy": round(percent_healthy * 100, 2),
            "% Rotten": round(percent_rotten * 100, 2),
        }
    )

df_voting = pd.DataFrame(voting_data)

# --- Simpan semua ke Excel ---
excel_path = os.path.join(RESULTS_DIR, "random_forest_analysis.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_features.to_excel(writer, index=False, sheet_name="Feature Importance")
    df_per_class.to_excel(writer, index=False, sheet_name="Per Class Metrics")
    df_predictions.to_excel(writer, index=False, sheet_name="Predictions")
    df_voting.to_excel(writer, index=False, sheet_name="Voting Detail")

print(f"Excel hasil analisis lengkap disimpan ke: {excel_path}")


print("\n" + "=" * 60)
print("SUMMARY - BEST METHOD RECOMMENDATION")
print("=" * 60)


if metrics_threshold["accuracy"] > metrics_default["accuracy"]:
    print(f"RECOMMENDATION: Use Threshold Method ({THRESHOLD_ROTTEN})")
    print(
        f"Accuracy improvement: {metrics_threshold['accuracy'] - metrics_default['accuracy']:+.4f}"
    )
else:
    print(f"RECOMMENDATION: Use Default Method (0.5)")
    print(
        f"Accuracy difference: {metrics_default['accuracy'] - metrics_threshold['accuracy']:+.4f}"
    )

print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT - THRESHOLD METHOD")
print("=" * 60)
print(
    classification_report(
        y_test, y_pred_threshold, target_names=classes, zero_division=0
    )
)

print("\n" + "=" * 60)
print("Test evaluation completed successfully!")
print(f"All results saved to: {RESULTS_DIR}")
print("=" * 60)
