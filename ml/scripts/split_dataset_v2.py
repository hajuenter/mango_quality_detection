import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "ml/dataset/mango_dataset_ml_v2"
OUTPUT_DIR = "ml/dataset/mango_dataset_ml_v2_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_files(file_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f in file_list:
        shutil.copy(f, target_dir)


for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    files = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # Split train + temp
    train_files, temp_files = train_test_split(
        files, test_size=(1 - TRAIN_RATIO), random_state=42, shuffle=True
    )

    # Split val + test dari temp
    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        random_state=42,
        shuffle=True,
    )

    copy_files(train_files, os.path.join(OUTPUT_DIR, "train", class_name))
    copy_files(val_files, os.path.join(OUTPUT_DIR, "val", class_name))
    copy_files(test_files, os.path.join(OUTPUT_DIR, "test", class_name))

print("✅ Dataset berhasil di-split menjadi train, val, test!")
