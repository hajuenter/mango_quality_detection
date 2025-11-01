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

# Ambil feature importances
importances = rf_model.feature_importances_
n_features = len(importances)

# ---------- Definisikan jumlah fitur ----------
n_color = 12  # RGB + HSV (mean + std)
n_texture = 10  # LBP histogram
n_shape = 3  # edge_density, num_contours, smoothness
n_stat = 4  # entropy, contrast, skewness, kurtosis

# Total 29 fitur
assert n_features == (n_color + n_texture + n_shape + n_stat), (
    f"Model memiliki {n_features} fitur, "
    f"tapi diharapkan {n_color + n_texture + n_shape + n_stat}"
)

# ---------- Nama fitur ----------
color_names = [
    "R_mean",
    "G_mean",
    "B_mean",
    "R_std",
    "G_std",
    "B_std",
    "H_mean",
    "S_mean",
    "V_mean",
    "H_std",
    "S_std",
    "V_std",
]
texture_names = [f"LBP_bin_{i}" for i in range(n_texture)]
shape_names = ["edge_density", "num_contours", "smoothness"]
stat_names = ["entropy", "contrast", "skewness", "kurtosis"]

feature_names = color_names + texture_names + shape_names + stat_names

# ---------- Urutkan fitur ----------
indices = np.argsort(importances)[::-1]
top_n = min(20, n_features)
top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_names = [feature_names[i] for i in top_indices]

# ---------- Print hasil ----------
print(f"\nTop {top_n} Most Important Features:")
print("=" * 50)
for i, idx in enumerate(top_indices):
    print(f"{i+1:2d}. {feature_names[idx]:20s} : {importances[idx]:.6f}")

# ---------- Simpan JSON ----------
importance_data = {
    "model_path": MODEL_PATH,
    "total_features": n_features,
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

# ---------- VISUALISASI ----------

# 🎯 1. Top-N Feature Importance
plt.figure(figsize=(12, 8))
colors = []
for name in top_names:
    if "LBP" in name:
        colors.append("#e74c3c")  # Merah untuk tekstur
    elif any(x in name for x in ["R_", "G_", "B_", "H_", "S_", "V_"]):
        colors.append("#3498db")  # Biru untuk warna
    elif name in shape_names:
        colors.append("#2ecc71")  # Hijau untuk bentuk
    else:
        colors.append("#f1c40f")  # Kuning untuk statistik

plt.barh(range(top_n), top_importances, color=colors, alpha=0.8, edgecolor="black")
plt.yticks(range(top_n), top_names)
plt.xlabel("Importance", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_feature_importance.png"), dpi=300)
print("Top feature importance disimpan")

# 🎨 2. Total kontribusi berdasarkan tipe fitur
groups = {
    "Color": importances[:n_color].sum(),
    "Texture": importances[n_color : n_color + n_texture].sum(),
    "Shape": importances[n_color + n_texture : n_color + n_texture + n_shape].sum(),
    "Statistical": importances[-n_stat:].sum(),
}

plt.figure(figsize=(10, 6))
plt.pie(
    groups.values(),
    labels=groups.keys(),
    autopct="%1.1f%%",
    colors=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    startangle=90,
    explode=(0.05, 0.05, 0.05, 0.05),
)
plt.title("Feature Type Contribution", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_type_contribution.png"), dpi=300)
print("Feature type contribution disimpan")

# 🎯 3. Distribusi Importance per Kelompok
plt.figure(figsize=(12, 6))
plt.bar(
    groups.keys(), groups.values(), color=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"]
)
plt.ylabel("Total Importance", fontweight="bold")
plt.title("Total Importance per Feature Group", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_group_importance.png"), dpi=300)
print("Feature group importance disimpan")

plt.close("all")

print("\n" + "=" * 50)
print("Feature importance analysis selesai!")
print(f"Hasil disimpan di: {RESULTS_DIR}")
print("=" * 50)
