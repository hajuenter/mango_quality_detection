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
)
from datetime import datetime
import joblib
import sys

from features import extract_features

# ---------- Konfigurasi ----------
TEST_DIR = "ml/dataset/mango_dataset_ml_v2_split/test"
MODEL_PATH = "ml/models/random_forest/random_forest_mango.pkl"
RESULTS_DIR = "ml/results/random_forest/test"
# ---------------------------------

# Hapus folder hasil lama jika ada
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Redirect print ke file log
log_path = os.path.join(RESULTS_DIR, "test_log.txt")
sys.stdout = open(log_path, "w")

print("Loading model...")
rf_best = joblib.load(MODEL_PATH)

print("Loading test dataset...")
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
    for f in files:
        X_test.append(extract_features(f))
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Prediksi
print("Predicting...")
y_pred = rf_best.predict(X_test)

# Hitung metrik
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Siapkan hasil untuk JSON
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": MODEL_PATH,
    "test_samples": len(y_test),
    "classes": classes,
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    },
    "per_class_metrics": {},
    "confusion_matrix": cm.tolist(),
}

# Per-class metrics
for cls in classes:
    if cls in report:
        results["per_class_metrics"][cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1_score": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

# Simpan JSON
json_path = os.path.join(RESULTS_DIR, "test_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Hasil test disimpan ke: {json_path}")

# ========== VISUALISASI ==========

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={"label": "Count"},
)
plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}", fontsize=14, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
print(f"Confusion matrix disimpan")

# 2. Per-Class Metrics Bar Chart
metrics_data = {
    "Precision": [results["per_class_metrics"][cls]["precision"] for cls in classes],
    "Recall": [results["per_class_metrics"][cls]["recall"] for cls in classes],
    "F1-Score": [results["per_class_metrics"][cls]["f1_score"] for cls in classes],
}

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, metrics_data["Precision"], width, label="Precision")
bars2 = ax.bar(x, metrics_data["Recall"], width, label="Recall")
bars3 = ax.bar(x + width, metrics_data["F1-Score"], width, label="F1-Score")

ax.set_xlabel("Classes", fontweight="bold")
ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_class_metrics.png"), dpi=300)
print(f"Per-class metrics disimpan")

# 3. Overall Metrics Bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

overall_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
overall_values = [accuracy, precision, recall, f1]
colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

bars = ax1.bar(
    overall_metrics, overall_values, color=colors, alpha=0.7, edgecolor="black"
)
ax1.set_ylabel("Score", fontweight="bold")
ax1.set_title("Overall Model Performance", fontsize=12, fontweight="bold")
ax1.set_ylim([0, 1.1])
ax1.grid(axis="y", alpha=0.3)

# Tambahkan nilai di atas bar
for bar, val in zip(bars, overall_values):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Support per class
support_values = [results["per_class_metrics"][cls]["support"] for cls in classes]
ax2.bar(classes, support_values, color="#9b59b6", alpha=0.7, edgecolor="black")
ax2.set_xlabel("Classes", fontweight="bold")
ax2.set_ylabel("Number of Samples", fontweight="bold")
ax2.set_title("Test Set Distribution", fontsize=12, fontweight="bold")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overall_metrics.png"), dpi=300)
print(f"Overall metrics disimpan")

plt.close("all")

# Cetak ringkasan ke log
print("\n" + "=" * 50)
print("HASIL EVALUASI TEST SET")
print("=" * 50)
print(f"Akurasi Test: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("=" * 50)
