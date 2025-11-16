import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from collections import Counter

FEATURE_PATH = "ml/results/knn/features/mango_features_all.xlsx"
RESULTS_DIR = "ml/models/knn"
LOG_DIR_PATH = "ml/results/knn"
LOG_DIR = os.path.join(LOG_DIR_PATH, "logs")
VISUAL_DIR = os.path.join(LOG_DIR_PATH, "train")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_path = os.path.join(LOG_DIR, f"train_log_{timestamp}.txt")
log_json_path = os.path.join(LOG_DIR, f"train_log_{timestamp}.json")


def log_write(text):
    """Tulis ke file log tanpa mencetak ke terminal"""
    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


df_train_val = pd.read_excel(FEATURE_PATH, sheet_name="train_val")
df_test = pd.read_excel(FEATURE_PATH, sheet_name="test")

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

X_train_val = df_train_val[FEATURES].values
y_train_val = df_train_val["label"].values
X_test = df_test[FEATURES].values
y_test = df_test["label"].values

train_dist = Counter(y_train_val)
test_dist = Counter(y_test)

log_write("=== Dataset Info ===")
log_write(f"Train/Val samples: {len(X_train_val)} | Test samples: {len(X_test)}")
log_write(f"Train/Val distribution: {dict(train_dist)}")
log_write(f"Test distribution: {dict(test_dist)}\n")

corr_matrix = df_train_val[FEATURES].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix (Training Set)", fontsize=14)
plt.tight_layout()

corr_path = os.path.join(VISUAL_DIR, f"correlation_matrix_{timestamp}.png")
plt.savefig(corr_path, dpi=300)
plt.close()

log_write(f"Correlation matrix disimpan ke: {corr_path}")

scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    "n_neighbors": list(range(3, 21, 2)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

knn = KNeighborsClassifier()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    knn, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
)
grid_search.fit(X_train_val_scaled, y_train_val)

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

log_write(f"Best Parameters: {best_params}")
log_write(f"Best CV Accuracy: {best_cv_score:.4f}")

# === 5. Evaluasi di test set ===
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
cls_report = classification_report(
    y_test, y_pred, target_names=["healthy", "rotten"], output_dict=True
)

log_write("\n=== TEST PERFORMANCE ===")
log_write(f"Accuracy: {test_acc:.4f}")
log_write(f"Confusion Matrix:\n{conf_mat}")
log_write(classification_report(y_test, y_pred, target_names=["healthy", "rotten"]))

model_path = os.path.join(RESULTS_DIR, f"knn_best_model_{timestamp}.pkl")
scaler_path = os.path.join(RESULTS_DIR, f"scaler_{timestamp}.pkl")
joblib.dump(best_knn, model_path)
joblib.dump(scaler, scaler_path)

log_write(f"Model disimpan: {model_path}")
log_write(f"Scaler disimpan: {scaler_path}")

log_data = {
    "timestamp": timestamp,
    "model": "KNeighborsClassifier",
    "best_params": best_params,
    "best_cv_accuracy": best_cv_score,
    "test_accuracy": test_acc,
    "confusion_matrix": conf_mat.tolist(),
    "classification_report": cls_report,
    "dataset_info": {
        "train_val_size": len(X_train_val),
        "test_size": len(X_test),
        "train_val_distribution": {int(k): int(v) for k, v in train_dist.items()},
        "test_distribution": {int(k): int(v) for k, v in test_dist.items()},
    },
    "model_path": model_path,
    "scaler_path": scaler_path,
}

with open(log_json_path, "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=4)

log_write(f"Log tersimpan ke:")
log_write(f"- {log_txt_path}")
log_write(f"- {log_json_path}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Healthy", "Rotten"],
    yticklabels=["Healthy", "Rotten"],
)
plt.title(f"Confusion Matrix (Accuracy: {test_acc:.4f})", fontsize=13)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, f"confusion_matrix_{timestamp}.png"), dpi=300)
plt.close()

pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]

plt.figure(figsize=(7, 5))
sns.countplot(x="Actual", hue="Correct", data=pred_df, palette="Set2")
plt.title("Prediction Correctness per Class")
plt.xlabel("True Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUAL_DIR, f"prediction_correctness_{timestamp}.png"), dpi=300
)
plt.close()

error_mask = pred_df["Correct"] == False
if error_mask.any():
    plt.figure(figsize=(6, 4))
    sns.histplot(X_test_scaled[error_mask][:, 0], bins=20, color="red", alpha=0.7)
    plt.title("Feature[0] Distribution of Misclassified Samples")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VISUAL_DIR, f"error_distribution_{timestamp}.png"), dpi=300
    )
    plt.close()

log_write(f"Visualisasi tersimpan di: {VISUAL_DIR}")
log_write("Training selesai tanpa error!")
