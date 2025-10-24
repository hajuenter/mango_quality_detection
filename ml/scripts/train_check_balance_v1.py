import os
import cv2
import numpy as np
from collections import Counter

# Path dataset
train_dir = "ml/dataset/mango_dataset_ml_v1_split/train"

# Ambil daftar label (otomatis dari folder)
classes = sorted(os.listdir(train_dir))


def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    mean_color = img.mean(axis=(0, 1))  # rata-rata warna
    std_color = img.std(axis=(0, 1))  # keragaman warna
    return np.concatenate([mean_color, std_color])


# ======== Load Data ========
X = []
y = []

for label in classes:
    folder = os.path.join(train_dir, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("📂 Total data:", len(X))
print("🧩 Kelas terdeteksi:", classes)

# ======== Cek Distribusi Kelas Asli ========
print("\n📊 Distribusi kelas asli:")
print(Counter(y))

# ======== Ubah Label Jadi Binary (Healthy vs Rotten) ========
y_binary = np.array(["Healthy" if c == "Healthy" else "Rotten" for c in y])
print("\n🧩 Distribusi kelas biner (Healthy vs Rotten):")
print(Counter(y_binary))

# ======== Rasio Persentase ========
total = len(y_binary)
count = Counter(y_binary)
for label, num in count.items():
    percent = (num / total) * 100
    print(f" - {label}: {num} data ({percent:.2f}%)")

print("\n✅ Pengecekan keseimbangan selesai.")
