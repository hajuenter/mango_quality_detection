import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import joblib

from features import extract_features

# ---------- Konfigurasi ----------
DATASET_DIR = "ml/dataset/mango_dataset_ml_v2_split/train"
VAL_DIR = "ml/dataset/mango_dataset_ml_v2_split/val"
MODEL_PATH = "ml/models/random_forest/random_forest_mango.pkl"
RESULTS_DIR = "ml/results/random_forest/train"
N_ESTIMATORS = 100
DEPTHS = [2, 4, 6, 8, 10, None]
# ---------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# -----------------------------
# Fungsi load dataset
# -----------------------------
def load_dataset(folder):
    X, y = [], []
    classes = sorted(
        [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    )
    print(f"📂 Loading from {folder}...")
    for label in classes:
        folder_path = os.path.join(folder, label)
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  - {label}: {len(files)} images")
        for f in files:
            X.append(extract_features(f))
            y.append(label)
    return np.array(X), np.array(y)


# -----------------------------
# Load train & val
# -----------------------------
X_train, y_train = load_dataset(DATASET_DIR)
X_val, y_val = load_dataset(VAL_DIR)

print(
    f"\n📊 Dataset loaded: {len(X_train)} train, {len(X_val)} val, Feature dim: {X_train.shape[1]}"
)

# -----------------------------
# Hyperparameter tuning max_depth
# -----------------------------
best_acc = 0
best_depth = None
train_scores = []
val_scores = []

print("\n🔧 Hyperparameter Tuning (max_depth)...")
for depth in DEPTHS:
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=depth, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    depth_str = str(depth) if depth is not None else "None"
    print(
        f"  max_depth={depth_str:4s} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )
    if val_acc > best_acc:
        best_acc = val_acc
        best_depth = depth

print(f"\n✅ Best max_depth: {best_depth}, Validation Accuracy: {best_acc:.4f}")

# -----------------------------
# Train final model
# -----------------------------
rf_best = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=best_depth, random_state=42, n_jobs=-1
)
rf_best.fit(X_train, y_train)

final_train_acc = accuracy_score(y_train, rf_best.predict(X_train))
final_val_acc = accuracy_score(y_val, rf_best.predict(X_val))
joblib.dump(rf_best, MODEL_PATH)
print(f"\n✅ Model disimpan di {MODEL_PATH}")

# -----------------------------
# Simpan training results JSON
# -----------------------------
training_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "RandomForestClassifier",
    "hyperparameters": {
        "n_estimators": N_ESTIMATORS,
        "best_max_depth": best_depth,
        "random_state": 42,
    },
    "dataset": {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "feature_dimension": int(X_train.shape[1]),
        "classes": sorted(np.unique(y_train).tolist()),
    },
    "training_history": {
        "max_depths": [str(d) if d is not None else "None" for d in DEPTHS],
        "train_accuracy": [float(a) for a in train_scores],
        "val_accuracy": [float(a) for a in val_scores],
    },
    "final_results": {
        "train_accuracy": float(final_train_acc),
        "val_accuracy": float(final_val_acc),
    },
}
json_path = os.path.join(RESULTS_DIR, "training_results.json")
with open(json_path, "w") as f:
    json.dump(training_results, f, indent=4)
print(f"✅ Training results JSON disimpan ke: {json_path}")

# -----------------------------
# Plot Training vs Validation Accuracy
# -----------------------------
plt.figure(figsize=(10, 6))
depth_labels = [str(d) if d is not None else "None" for d in DEPTHS]
x_pos = np.arange(len(depth_labels))
plt.plot(x_pos, train_scores, marker="o", label="Train Accuracy", linewidth=2)
plt.plot(x_pos, val_scores, marker="s", label="Validation Accuracy", linewidth=2)
plt.xlabel("Max Depth", fontweight="bold")
plt.ylabel("Accuracy", fontweight="bold")
plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_validation_curve.png"), dpi=300)
print(f"✅ Training curve disimpan")

# -----------------------------
# Plot Overfitting Analysis
# -----------------------------
plt.figure(figsize=(10, 6))
overfit_gap = np.array(train_scores) - np.array(val_scores)
colors = [
    "green" if gap < 0.1 else "orange" if gap < 0.2 else "red" for gap in overfit_gap
]
plt.bar(depth_labels, overfit_gap, color=colors, alpha=0.7, edgecolor="black")
plt.xlabel("Max Depth", fontweight="bold")
plt.ylabel("Train - Val Accuracy", fontweight="bold")
plt.title("Overfitting Analysis (Lower is Better)", fontsize=14, fontweight="bold")
plt.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="Threshold: 0.1")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overfitting_analysis.png"), dpi=300)
print(f"✅ Overfitting analysis disimpan")

# -----------------------------
# Model Summary Table
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("off")
summary_data = [
    ["Model", "Random Forest Classifier"],
    ["N Estimators", N_ESTIMATORS],
    ["Best Max Depth", best_depth],
    ["Training Samples", len(X_train)],
    ["Validation Samples", len(X_val)],
    ["Final Train Accuracy", f"{final_train_acc:.4f}"],
    ["Final Validation Accuracy", f"{final_val_acc:.4f}"],
    ["Overfitting Gap", f"{final_train_acc - final_val_acc:.4f}"],
]
table = ax.table(
    cellText=summary_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_summary.png"), dpi=300)
print(f"✅ Training summary tabel disimpan")

plt.close("all")
print("\n✅ Training selesai! Semua hasil disimpan di:", RESULTS_DIR)
