import os
import cv2
import numpy as np
from collections import Counter

# ================== Konfigurasi ==================
DATASET_DIR = "ml/dataset/mango_dataset_ml_split/train"
IMAGE_SIZE = (224, 224)  # ukuran resize gambar
VALID_EXT = (".jpg", ".jpeg", ".png")
# =================================================


def extract_features(image_path):
    """Ekstraksi fitur sederhana: mean dan std warna RGB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    mean_color = img.mean(axis=(0, 1))  # rata-rata warna
    std_color = img.std(axis=(0, 1))  # keragaman warna
    return np.concatenate([mean_color, std_color])


def load_dataset(dataset_dir):
    """Load dataset: X = fitur, y = label"""
    X, y = [], []
    classes = sorted(
        [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    )

    for label in classes:
        folder = os.path.join(dataset_dir, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)]
        for file in files:
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y, classes


# ================== Main ==================
X, y, classes = load_dataset(DATASET_DIR)

print("📂 Total data:", len(X))
print("🧩 Kelas terdeteksi:", classes)

# Cek distribusi kelas
print("\n📊 Distribusi kelas:")
count = Counter(y)
for label, num in count.items():
    percent = (num / len(y)) * 100
    print(f" - {label}: {num} data ({percent:.2f}%)")

print("\n✅ Pengecekan keseimbangan selesai.")
