import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys

from features import IMAGE_SIZE, LBP_P, LBP_R

# ---------- Konfigurasi ----------
MODEL_PATH = "ml/models/random_forest/random_forest_mango.pkl"
RESULTS_DIR = "ml/results/random_forest/features"
# ---------------------------------

# Hapus folder lama jika ada
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Redirect print ke file log
log_path = os.path.join(RESULTS_DIR, "feature_importance_log.txt")
sys.stdout = open(log_path, "w")

print("Loading model...")
rf_model = joblib.load(MODEL_PATH)

# Dapatkan feature importance
importances = rf_model.feature_importances_

# Buat nama fitur
n_color_features = 6  # R_mean,G_mean,B_mean,R_std,G_std,B_std
n_texture_features = len(importances) - n_color_features

feature_names = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"] + [
    f"LBP_bin_{i}" for i in range(n_texture_features)
]

# Top N fitur
n_features = len(importances)
top_n = min(20, n_features)

# Urutkan indeks
indices = np.argsort(importances)[::-1]
top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_names = [feature_names[i] for i in top_indices]

# Print Top N
print(f"\nTop {top_n} Most Important Features:")
print("=" * 50)
for i, idx in enumerate(top_indices):
    print(f"{i+1:2d}. {feature_names[idx]:15s} : {importances[idx]:.6f}")

# Simpan ke JSON
importance_data = {
    "model_path": MODEL_PATH,
    "total_features": n_features,
    "color_features": n_color_features,
    "texture_features": n_texture_features,
    "feature_importances": {
        feature_names[i]: float(importances[i]) for i in range(n_features)
    },
    "top_features": [
        {
            "rank": i + 1,
            "feature_name": top_names[i],
            "importance": float(top_importances[i]),
        }
        for i in range(top_n)
    ],
}

json_path = os.path.join(RESULTS_DIR, "feature_importance.json")
with open(json_path, "w") as f:
    json.dump(importance_data, f, indent=4)
print(f"\nFeature importance disimpan ke: {json_path}")

# ========== VISUALISASI ==========

# 1. Top N Feature Importance
plt.figure(figsize=(12, 8))
colors = ["#e74c3c" if "LBP" in name else "#3498db" for name in top_names]
plt.barh(range(top_n), top_importances, color=colors, alpha=0.7, edgecolor="black")
plt.yticks(range(top_n), top_names)
plt.xlabel("Importance", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.title(
    f"Top {top_n} Most Important Features (Blue: Color, Red: Texture)",
    fontsize=14,
    fontweight="bold",
)
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_feature_importance.png"), dpi=300)
print(f"Top feature importance disimpan")

# 2. Color vs Texture Feature Importance
color_importance = importances[:n_color_features].sum()
texture_importance = importances[n_color_features:].sum()

plt.figure(figsize=(10, 6))
categories = ["Color Features (RGB mean + std)", "Texture Features (LBP histogram)"]
values = [color_importance, texture_importance]
colors_pie = ["#3498db", "#e74c3c"]

plt.subplot(1, 2, 1)
plt.pie(
    values,
    labels=categories,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=90,
    explode=(0.05, 0.05),
)
plt.title("Feature Type Contribution", fontweight="bold")

plt.subplot(1, 2, 2)
plt.bar(categories, values, color=colors_pie, alpha=0.7, edgecolor="black")
plt.ylabel("Total Importance", fontweight="bold")
plt.title("Feature Type Importance", fontweight="bold")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_type_comparison.png"), dpi=300)
print(f"Feature type comparison disimpan")

# 3. Individual Color Feature Contribution
plt.figure(figsize=(10, 6))
color_features = feature_names[:n_color_features]
color_values = importances[:n_color_features]
colors_bar = ["#e74c3c", "#2ecc71", "#3498db", "#e74c3c", "#2ecc71", "#3498db"]
plt.bar(color_features, color_values, color=colors_bar, alpha=0.7, edgecolor="black")
plt.xlabel("Color Feature", fontweight="bold")
plt.ylabel("Importance", fontweight="bold")
plt.title("Individual Color Feature Importance", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "color_feature_importance.png"), dpi=300)
print(f"Color feature importance disimpan")

# 4. LBP Histogram Distribution
plt.figure(figsize=(14, 6))
lbp_importances = importances[n_color_features:]
lbp_bins = range(len(lbp_importances))

plt.subplot(1, 2, 1)
plt.plot(lbp_bins, lbp_importances, marker="o", linewidth=2, markersize=4)
plt.xlabel("LBP Bin", fontweight="bold")
plt.ylabel("Importance", fontweight="bold")
plt.title("LBP Histogram Bin Importance", fontweight="bold")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(lbp_importances, bins=20, color="#e74c3c", alpha=0.7, edgecolor="black")
plt.xlabel("Importance Value", fontweight="bold")
plt.ylabel("Frequency", fontweight="bold")
plt.title("Distribution of LBP Feature Importance", fontweight="bold")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lbp_importance_distribution.png"), dpi=300)
print(f"LBP importance distribution disimpan")

plt.close("all")
print("\n" + "=" * 50)
print("Feature importance analysis selesai!")
print(f"Hasil disimpan di: {RESULTS_DIR}")
print("=" * 50)
