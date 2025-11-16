import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.inspection import permutation_importance

MODEL_PATH = "ml/models/knn/knn_mango.pkl"
SCALER_PATH = "ml/models/knn/scaler.pkl"
FEATURE_PATH = "ml/results/knn/features/mango_features_all.xlsx"
RESULTS_DIR = "ml/results/knn/test"

FEATURES = [
    "avg_red",
    "avg_green",
    "avg_blue",
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "saturation_mean",
    "brightness_mean",
    "entropy",
]

CLASS_NAMES = {0: "Healthy", 1: "Rotten"}

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80)
print("ENHANCED KNN MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
print("=" * 80)

print("\n[1/7] Loading model, scaler, and test data...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df_test = pd.read_excel(FEATURE_PATH, sheet_name="test")

X_test = df_test[FEATURES].values
y_test = df_test["label"].values
X_test_scaled = scaler.transform(X_test)

print(f"✓ Model loaded: {type(model).__name__}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {len(FEATURES)}")

print("\n[2/7] Making predictions...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
conf_mat = confusion_matrix(y_test, y_pred)

print(f"✓ Accuracy : {accuracy:.4f}")
print(f"✓ Precision: {precision:.4f}")
print(f"✓ Recall   : {recall:.4f}")
print(f"✓ F1-Score : {f1:.4f}")

print("\n[3/7] Analyzing misclassified samples...")

error_indices = np.where(y_test != y_pred)[0]
correct_indices = np.where(y_test == y_pred)[0]

print(
    f"✓ Misclassified samples: {len(error_indices)} ({len(error_indices)/len(y_test)*100:.2f}%)"
)
print(
    f"✓ Correctly classified : {len(correct_indices)} ({len(correct_indices)/len(y_test)*100:.2f}%)"
)

# Create error analysis dataframe
df_errors = df_test.iloc[error_indices].copy()
df_errors["predicted_label"] = y_pred[error_indices]
df_errors["true_label_name"] = df_errors["label"].map(CLASS_NAMES)
df_errors["predicted_label_name"] = df_errors["predicted_label"].map(CLASS_NAMES)

# Add prediction probabilities
df_errors["prob_healthy"] = y_pred_proba[error_indices, 0]
df_errors["prob_rotten"] = y_pred_proba[error_indices, 1]
df_errors["confidence"] = np.max(y_pred_proba[error_indices], axis=1)

# Error type breakdown (SEBELUM reorder columns)
false_positives = len(
    df_errors[(df_errors["label"] == 0) & (df_errors["predicted_label"] == 1)]
)
false_negatives = len(
    df_errors[(df_errors["label"] == 1) & (df_errors["predicted_label"] == 0)]
)

# Reorder columns for better readability
cols_order = [
    "image_name",
    "image_path",
    "label",  # Keep original label
    "true_label_name",
    "predicted_label",  # Keep predicted label
    "predicted_label_name",
    "prob_healthy",
    "prob_rotten",
    "confidence",
] + FEATURES

df_errors = df_errors[cols_order]

# Save misclassified images info
error_excel_path = os.path.join(RESULTS_DIR, f"misclassified_images_{timestamp}.xlsx")
df_errors.to_excel(error_excel_path, index=False)
print(f"✓ Error analysis saved: {error_excel_path}")

print(f"\n  Error Breakdown:")
print(f"  - False Positives (Healthy → Rotten): {false_positives}")
print(f"  - False Negatives (Rotten → Healthy): {false_negatives}")

print("\n[4/7] Computing feature importance...")

perm_importance = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

feature_importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std,
    }
).sort_values("importance_mean", ascending=False)

print("\n  Feature Importance Ranking:")
for idx, row in feature_importance_df.iterrows():
    print(
        f"  {row['feature']:20s}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})"
    )

# Save feature importance
feat_imp_path = os.path.join(RESULTS_DIR, f"feature_importance_{timestamp}.xlsx")
feature_importance_df.to_excel(feat_imp_path, index=False)
print(f"\n✓ Feature importance saved: {feat_imp_path}")

print("\n[5/7] Analyzing different probability thresholds...")

thresholds_to_test = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds_to_test:
    # Apply threshold (predict rotten if prob_rotten >= threshold)
    y_pred_thresh = (y_pred_proba[:, 1] >= thresh).astype(int)

    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, average="weighted")
    rec = recall_score(y_test, y_pred_thresh, average="weighted")
    f1_thresh = f1_score(y_test, y_pred_thresh, average="weighted")

    # Confusion matrix for this threshold
    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()

    threshold_results.append(
        {
            "threshold": thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1_thresh,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
    )

df_thresholds = pd.DataFrame(threshold_results)
print("\n  Threshold Analysis:")
print(df_thresholds.to_string(index=False))

thresh_path = os.path.join(RESULTS_DIR, f"threshold_analysis_{timestamp}.xlsx")
df_thresholds.to_excel(thresh_path, index=False)
print(f"\n✓ Threshold analysis saved: {thresh_path}")

print("\n[6/7] Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Healthy", "Rotten"],
    yticklabels=["Healthy", "Rotten"],
    ax=ax1,
)
ax1.set_title(
    f"Confusion Matrix\nAccuracy: {accuracy:.4f}", fontsize=12, fontweight="bold"
)
ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("True Label")

# 2. Feature Importance
ax2 = fig.add_subplot(gs[0, 1])
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
bars = ax2.barh(
    feature_importance_df["feature"],
    feature_importance_df["importance_mean"],
    color=colors,
)
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance (Permutation)", fontsize=12, fontweight="bold")
ax2.invert_yaxis()
for i, (bar, std) in enumerate(zip(bars, feature_importance_df["importance_std"])):
    ax2.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f" {bar.get_width():.3f}",
        va="center",
        fontsize=9,
    )

# 3. ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
ax3.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve", fontsize=12, fontweight="bold")
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3)

# 4. Prediction Distribution
ax4 = fig.add_subplot(gs[1, 0])
pred_df = pd.DataFrame(
    {"True Label": [CLASS_NAMES[y] for y in y_test], "Correct": y_test == y_pred}
)
sns.countplot(
    data=pred_df, x="True Label", hue="Correct", palette=["#FF6B6B", "#51CF66"], ax=ax4
)
ax4.set_title("Prediction Correctness per Class", fontsize=12, fontweight="bold")
ax4.set_ylabel("Count")
ax4.legend(title="Correct", labels=["False", "True"])

# 5. Probability Distribution (Correct vs Error)
ax5 = fig.add_subplot(gs[1, 1])
correct_conf = np.max(y_pred_proba[correct_indices], axis=1)
error_conf = np.max(y_pred_proba[error_indices], axis=1)
ax5.hist(
    correct_conf, bins=20, alpha=0.7, label="Correct", color="green", edgecolor="black"
)
ax5.hist(
    error_conf,
    bins=20,
    alpha=0.7,
    label="Misclassified",
    color="red",
    edgecolor="black",
)
ax5.set_xlabel("Prediction Confidence")
ax5.set_ylabel("Frequency")
ax5.set_title(
    "Confidence Distribution: Correct vs Misclassified", fontsize=12, fontweight="bold"
)
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Threshold Performance
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(
    df_thresholds["threshold"],
    df_thresholds["accuracy"],
    marker="o",
    label="Accuracy",
    linewidth=2,
)
ax6.plot(
    df_thresholds["threshold"],
    df_thresholds["precision"],
    marker="s",
    label="Precision",
    linewidth=2,
)
ax6.plot(
    df_thresholds["threshold"],
    df_thresholds["recall"],
    marker="^",
    label="Recall",
    linewidth=2,
)
ax6.plot(
    df_thresholds["threshold"],
    df_thresholds["f1_score"],
    marker="d",
    label="F1-Score",
    linewidth=2,
)
ax6.set_xlabel("Threshold")
ax6.set_ylabel("Score")
ax6.set_title("Metrics vs Threshold", fontsize=12, fontweight="bold")
ax6.legend()
ax6.grid(alpha=0.3)
ax6.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")

# 7. Error Analysis - Feature Comparison
ax7 = fig.add_subplot(gs[2, :2])
top_features = feature_importance_df.head(5)["feature"].tolist()
error_feature_means = df_test.iloc[error_indices][top_features].mean()
correct_feature_means = df_test.iloc[correct_indices][top_features].mean()

x = np.arange(len(top_features))
width = 0.35
bars1 = ax7.bar(
    x - width / 2,
    error_feature_means,
    width,
    label="Misclassified",
    color="#FF6B6B",
    alpha=0.8,
)
bars2 = ax7.bar(
    x + width / 2,
    correct_feature_means,
    width,
    label="Correct",
    color="#51CF66",
    alpha=0.8,
)

ax7.set_xlabel("Top 5 Features")
ax7.set_ylabel("Mean Value")
ax7.set_title(
    "Feature Comparison: Misclassified vs Correct Samples",
    fontsize=12,
    fontweight="bold",
)
ax7.set_xticks(x)
ax7.set_xticklabels(top_features, rotation=45, ha="right")
ax7.legend()
ax7.grid(alpha=0.3, axis="y")

# 8. Class-wise Error Rate
ax8 = fig.add_subplot(gs[2, 2])
class_errors = {
    "Healthy": (
        false_positives / np.sum(y_test == 0) * 100 if np.sum(y_test == 0) > 0 else 0
    ),
    "Rotten": (
        false_negatives / np.sum(y_test == 1) * 100 if np.sum(y_test == 1) > 0 else 0
    ),
}
colors_pie = ["#51CF66", "#FF6B6B"]
ax8.bar(
    class_errors.keys(),
    class_errors.values(),
    color=colors_pie,
    alpha=0.8,
    edgecolor="black",
)
ax8.set_ylabel("Error Rate (%)")
ax8.set_title("Error Rate by Class", fontsize=12, fontweight="bold")
ax8.grid(alpha=0.3, axis="y")
for i, (k, v) in enumerate(class_errors.items()):
    ax8.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

plt.suptitle(
    f"KNN Model - Comprehensive Analysis Report\nTimestamp: {timestamp}",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

vis_path = os.path.join(RESULTS_DIR, f"comprehensive_analysis_{timestamp}.png")
plt.savefig(vis_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Comprehensive visualization saved: {vis_path}")

print("\n[7/7] Saving comprehensive report...")

# Create detailed JSON report
report = {
    "timestamp": timestamp,
    "model_info": {
        "type": type(model).__name__,
        "parameters": model.get_params(),
    },
    "dataset_info": {
        "test_size": len(X_test),
        "features": FEATURES,
        "class_distribution": {
            "healthy": int(np.sum(y_test == 0)),
            "rotten": int(np.sum(y_test == 1)),
        },
    },
    "performance_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    },
    "confusion_matrix": {
        "matrix": conf_mat.tolist(),
        "true_negative": int(conf_mat[0, 0]),
        "false_positive": int(conf_mat[0, 1]),
        "false_negative": int(conf_mat[1, 0]),
        "true_positive": int(conf_mat[1, 1]),
    },
    "error_analysis": {
        "total_errors": len(error_indices),
        "error_rate": float(len(error_indices) / len(y_test)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "avg_error_confidence": float(np.mean(error_conf)),
        "avg_correct_confidence": float(np.mean(correct_conf)),
    },
    "feature_importance": feature_importance_df.to_dict(orient="records"),
    "threshold_analysis": df_thresholds.to_dict(orient="records"),
    "classification_report": classification_report(
        y_test, y_pred, target_names=["Healthy", "Rotten"], output_dict=True
    ),
    "files_generated": {
        "misclassified_images": error_excel_path,
        "feature_importance": feat_imp_path,
        "threshold_analysis": thresh_path,
        "visualization": vis_path,
    },
}

report_path = os.path.join(RESULTS_DIR, f"comprehensive_report_{timestamp}.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print(f"✓ Comprehensive report saved: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 80)
print(f"Overall Performance:")
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score : {f1:.4f}")
print(f"   ROC AUC  : {roc_auc:.4f}")

print(f"Error Analysis:")
print(f"   Total Errors     : {len(error_indices)} / {len(y_test)}")
print(f"   False Positives  : {false_positives} (Healthy predicted as Rotten)")
print(f"   False Negatives  : {false_negatives} (Rotten predicted as Healthy)")

print(f"Top 3 Most Important Features:")
for idx, row in feature_importance_df.head(3).iterrows():
    print(f"   {idx+1}. {row['feature']:20s}: {row['importance_mean']:.4f}")

print(f"Best Threshold:")
best_thresh_row = df_thresholds.loc[df_thresholds["f1_score"].idxmax()]
print(f"   Threshold: {best_thresh_row['threshold']:.2f}")
print(f"   F1-Score : {best_thresh_row['f1_score']:.4f}")
print(f"   Accuracy : {best_thresh_row['accuracy']:.4f}")

print(f"All Results Saved In:")
print(f"   {RESULTS_DIR}")
print("\n" + "=" * 80)
print("Enhanced evaluation completed successfully!")
print("=" * 80)
