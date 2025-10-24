import os
import shutil
from sklearn.model_selection import train_test_split

# Path dataset asli
base_dir = "ml/dataset/mango_dataset_ml_v1/image"
classes = ["Anthracnose", "Bacterial Canker", "Scab", "Stem End Rot", "Healthy"]

# Path output dataset split
output_dir = "ml/dataset/mango_dataset_ml_v1_split"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Buat folder target
for split_dir in [train_dir, test_dir]:
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

# Split train-test 80/20 dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)


# Fungsi salin file ke folder target
def copy_files(file_list, labels_list, dest_dir):
    for img_path, label in zip(file_list, labels_list):
        dest = os.path.join(dest_dir, label, os.path.basename(img_path))
        shutil.copy(img_path, dest)


# Salin file hasil split
copy_files(X_train, y_train, train_dir)
copy_files(X_test, y_test, test_dir)

print("✅ Dataset split selesai!")
print(f"Train dir: {train_dir}")
print(f"Test dir: {test_dir}")
