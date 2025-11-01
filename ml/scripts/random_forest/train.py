import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import joblib
import shutil
import seaborn as sns

from features import extract_features


DATASET_DIR = "ml/dataset/mango_dataset_ml_split/train"
VAL_DIR = "ml/dataset/mango_dataset_ml_split/val"
TEST_DIR = "ml/dataset/mango_dataset_ml_split/test"

MODEL_DIR = "ml/models/random_forest"
RESULTS_DIR = "ml/results/random_forest/train"

N_ESTIMATORS = 150
DEPTHS = [4, 6, 8, 10, 12]
RANDOM_STATE = 42


if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


log_path = os.path.join(RESULTS_DIR, "train_log.txt")
sys.stdout = open(log_path, "w")

print("Memulai training Random Forest...\n")


def load_dataset(folder):
    X, y = [], []
    classes = sorted(
        [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    )
    print(f"Loading dari {folder}...")
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


X_train, y_train = load_dataset(DATASET_DIR)
X_val, y_val = load_dataset(VAL_DIR)
X_test, y_test = load_dataset(TEST_DIR)

print(
    f"\nDataset loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, Feature dim={X_train.shape[1]}"
)


from collections import Counter

train_dist = Counter(y_train)
val_dist = Counter(y_val)
test_dist = Counter(y_test)

print("\n=== Distribusi Dataset ===")
print(f"Train: {dict(train_dist)}")
print(f"Val: {dict(val_dist)}")
print(f"Test: {dict(test_dist)}")


class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"\nClass Weights: {class_weight_dict}")


best_acc = 0
best_depth = None
train_scores = []
val_scores = []

print("\n=== Hyperparameter Tuning (max_depth) ===")
for depth in DEPTHS:
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        max_samples=0.8,
        class_weight=class_weight_dict,
    )
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    overfitting_gap = train_acc - val_acc
    print(
        f"max_depth={depth:>2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {overfitting_gap:.4f}"
    )
    if val_acc > best_acc:
        best_acc = val_acc
        best_depth = depth

print(f"\nBest max_depth: {best_depth}, Validation Accuracy: {best_acc:.4f}")


model_name = f"random_forest_mango.pkl"
model_path = os.path.join(MODEL_DIR, model_name)

rf_best = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=best_depth,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    max_samples=0.8,
    class_weight=class_weight_dict,
)
rf_best.fit(X_train, y_train)


y_train_pred = rf_best.predict(X_train)
y_val_pred = rf_best.predict(X_val)
y_test_pred = rf_best.predict(X_test)

final_train_acc = accuracy_score(y_train, y_train_pred)
final_val_acc = accuracy_score(y_val, y_val_pred)
final_test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n=== Final Model Performance ===")
print(f"Train Accuracy: {final_train_acc:.4f}")
print(f"Val Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy: {final_test_acc:.4f}")
print(f"Overfitting Gap: {final_train_acc - final_val_acc:.4f}")


print("\n=== Classification Report (Test Set) ===")
print(
    classification_report(y_test, y_test_pred, target_names=sorted(np.unique(y_test)))
)

joblib.dump(rf_best, model_path)
print(f"\nModel disimpan ke: {model_path}")


training_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "RandomForestClassifier (Optimized)",
    "hyperparameters": {
        "n_estimators": N_ESTIMATORS,
        "best_max_depth": best_depth,
        "random_state": RANDOM_STATE,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "max_samples": 0.8,
        "class_weight": {k: float(v) for k, v in class_weight_dict.items()},
    },
    "dataset": {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_dimension": int(X_train.shape[1]),
        "classes": sorted(np.unique(y_train).tolist()),
        "distribution": {
            "train": dict(train_dist),
            "val": dict(val_dist),
            "test": dict(test_dist),
        },
    },
    "training_history": {
        "max_depths": [int(d) for d in DEPTHS],
        "train_accuracy": [float(a) for a in train_scores],
        "val_accuracy": [float(a) for a in val_scores],
        "overfitting_gaps": [float(t - v) for t, v in zip(train_scores, val_scores)],
    },
    "final_results": {
        "train_accuracy": float(final_train_acc),
        "val_accuracy": float(final_val_acc),
        "test_accuracy": float(final_test_acc),
        "overfitting_gap": float(final_train_acc - final_val_acc),
    },
}
json_path = os.path.join(RESULTS_DIR, "training_results.json")
with open(json_path, "w") as f:
    json.dump(training_results, f, indent=4)
print(f"Hasil training JSON disimpan ke: {json_path}")


plt.figure(figsize=(10, 6))
depth_labels = [str(d) for d in DEPTHS]
x_pos = np.arange(len(depth_labels))
plt.plot(x_pos, train_scores, marker="o", label="Train Accuracy", linewidth=2)
plt.plot(x_pos, val_scores, marker="s", label="Validation Accuracy", linewidth=2)
plt.xlabel("Max Depth", fontweight="bold")
plt.ylabel("Accuracy", fontweight="bold")
plt.title(
    "Training vs Validation Accuracy (With Class Weights)",
    fontsize=14,
    fontweight="bold",
)
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_validation_curve.png"), dpi=300)
print(f"Grafik training-validation disimpan")


plt.figure(figsize=(10, 6))
overfit_gap = np.array(train_scores) - np.array(val_scores)
colors = [
    "green" if gap < 0.05 else "orange" if gap < 0.10 else "red" for gap in overfit_gap
]
plt.bar(depth_labels, overfit_gap, color=colors, alpha=0.7, edgecolor="black")
plt.xlabel("Max Depth", fontweight="bold")
plt.ylabel("Train - Val Accuracy", fontweight="bold")
plt.title("Overfitting Analysis (Lower is Better)", fontsize=14, fontweight="bold")
plt.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Good: <0.05")
plt.axhline(
    y=0.10, color="orange", linestyle="--", alpha=0.5, label="Acceptable: <0.10"
)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overfitting_analysis.png"), dpi=300)
print(f"Grafik overfitting disimpan")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_val = confusion_matrix(y_val, y_val_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

for ax, cm, title in zip(axes, [cm_val, cm_test], ["Validation", "Test"]):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=sorted(np.unique(y_test)),
        yticklabels=sorted(np.unique(y_test)),
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
print(f"Confusion matrix disimpan")


fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
summary_data = [
    ["Model", "Random Forest (Optimized)"],
    ["N Estimators", N_ESTIMATORS],
    ["Best Max Depth", best_depth],
    ["Training Samples", len(X_train)],
    ["Validation Samples", len(X_val)],
    ["Test Samples", len(X_test)],
    ["Feature Dimension", X_train.shape[1]],
    ["Train Accuracy", f"{final_train_acc:.4f}"],
    ["Val Accuracy", f"{final_val_acc:.4f}"],
    ["Test Accuracy", f"{final_test_acc:.4f}"],
    ["Overfitting Gap", f"{final_train_acc - final_val_acc:.4f}"],
]
table = ax.table(
    cellText=summary_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_summary.png"), dpi=300)
print(f"Tabel summary disimpan")

plt.close("all")

print("\n Training selesai tanpa error!")
print(f"Semua hasil tersimpan di: {RESULTS_DIR}")
