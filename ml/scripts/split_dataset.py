import os
import shutil
from sklearn.model_selection import train_test_split

# 📁 Path dataset
DATASET_DIR = "ml/dataset/mango_dataset_ml"
OUTPUT_DIR = "ml/dataset/mango_dataset_ml_split"

# 🔢 Rasio split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 🧹 Hapus folder output lama jika ada
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_files(file_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f in file_list:
        shutil.copy(f, target_dir)


# Statistik
total_before = 0
train_total, val_total, test_total = 0, 0, 0

print("📂 Memulai proses split dataset...\n")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Ambil semua file gambar di folder kelas
    files = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    total_before += len(files)

    # Split train + temp
    train_files, temp_files = train_test_split(
        files, test_size=(1 - TRAIN_RATIO), random_state=42, shuffle=True
    )

    # Split val + test dari temp
    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42,
        shuffle=True,
    )

    # Copy ke folder masing-masing
    copy_files(train_files, os.path.join(OUTPUT_DIR, "train", class_name))
    copy_files(val_files, os.path.join(OUTPUT_DIR, "val", class_name))
    copy_files(test_files, os.path.join(OUTPUT_DIR, "test", class_name))

    # Hitung per kelas
    train_total += len(train_files)
    val_total += len(val_files)
    test_total += len(test_files)

    print(f"📸 {class_name}:")
    print(
        f"   Total: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
    )

# Ringkasan total
print("\n📊 Ringkasan keseluruhan:")
print(f"Total gambar sebelum split : {total_before}")
print(f"Train total                : {train_total}")
print(f"Val total                  : {val_total}")
print(f"Test total                 : {test_total}")
print(f"Total setelah split        : {train_total + val_total + test_total}")

# Validasi konsistensi
if total_before == (train_total + val_total + test_total):
    print(
        "\n✅ Semua gambar terdistribusi dengan benar, tidak ada yang hilang atau ganda."
    )
else:
    print("\n⚠️ Jumlah gambar tidak cocok! Periksa ulang hasil split di folder output.")
