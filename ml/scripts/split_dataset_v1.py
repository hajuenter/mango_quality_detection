import os
import shutil
from sklearn.model_selection import train_test_split

# Path dataset asli
base_dir = "ml/dataset/mango_dataset_ml_v1/image"
classes = ["Anthracnose", "Bacterial Canker", "Scab", "Stem End Rot", "Healthy"]

# Path output dataset split
output_dir = "ml/dataset/mango_dataset_ml_v1_split"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# 🧹 Hapus folder lama jika ada, agar tidak ganda
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Buat ulang folder target
for split_dir in [train_dir, val_dir, test_dir]:
    for label in classes:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)

# Kumpulkan semua path dan label
image_paths = []
labels = []

for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    for img in os.listdir(cls_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(cls_path, img))
            labels.append(cls)

# Hitung total sebelum split
total_before = len(image_paths)
print(f"📂 Total gambar sebelum split: {total_before}")

# Split pertama: train (70%) dan sisanya (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)

# Split kedua: dari sisanya (30%) jadi val (15%) dan test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


# Fungsi salin file ke folder target
def copy_files(file_list, labels_list, dest_dir):
    for img_path, label in zip(file_list, labels_list):
        dest = os.path.join(dest_dir, label, os.path.basename(img_path))
        shutil.copy(img_path, dest)


# Salin file hasil split
copy_files(X_train, y_train, train_dir)
copy_files(X_val, y_val, val_dir)
copy_files(X_test, y_test, test_dir)


# Fungsi hitung jumlah gambar di setiap split
def count_images(split_dir):
    count = 0
    for cls in classes:
        path = os.path.join(split_dir, cls)
        count += len(
            [
                f
                for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
    return count


# Hitung jumlah gambar per split
train_count = count_images(train_dir)
val_count = count_images(val_dir)
test_count = count_images(test_dir)

print("\n📊 Jumlah gambar setelah split:")
print(f"Train: {train_count}")
print(f"Val:   {val_count}")
print(f"Test:  {test_count}")
print(f"Total setelah split: {train_count + val_count + test_count}")

# Cek konsistensi
if total_before == (train_count + val_count + test_count):
    print("✅ Semua gambar terdistribusi dengan benar, tidak ada yang hilang.")
else:
    print("⚠️ Jumlah gambar tidak cocok! Periksa ulang folder atau proses copy.")
