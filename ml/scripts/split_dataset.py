import os
import shutil
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

DATASET_DIR = "ml/dataset/mango_dataset_ml"
OUTPUT_DIR = "ml/dataset/mango_dataset_ml_split"

LOG_DIR = "ml/results"
os.makedirs(LOG_DIR, exist_ok=True)

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


total_before = 0
train_total, val_total, test_total = 0, 0, 0

summary = {
    "split_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_dir": DATASET_DIR,
    "output_dir": OUTPUT_DIR,
    "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    "classes": {},
}

log_lines = []
log_lines.append("Memulai proses split dataset...\n")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    files = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    total_before += len(files)

    train_files, temp_files = train_test_split(
        files, test_size=(1 - TRAIN_RATIO), random_state=42, shuffle=True
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42,
        shuffle=True,
    )

    copy_files(train_files, os.path.join(OUTPUT_DIR, "train", class_name))
    copy_files(val_files, os.path.join(OUTPUT_DIR, "val", class_name))
    copy_files(test_files, os.path.join(OUTPUT_DIR, "test", class_name))

    train_total += len(train_files)
    val_total += len(val_files)
    test_total += len(test_files)

    # Simpan ke log
    line = f"{class_name}: Total={len(files)}, Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    print("" + line)
    log_lines.append(line)

    # Simpan ke JSON summary
    summary["classes"][class_name] = {
        "total": len(files),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }

final_summary = (
    f"\nTotal sebelum split : {total_before}\n"
    f"Train total         : {train_total}\n"
    f"Val total           : {val_total}\n"
    f"Test total          : {test_total}\n"
    f"Total setelah split : {train_total + val_total + test_total}"
)

print(final_summary)
log_lines.append(final_summary)

# Cek konsistensi
if total_before == (train_total + val_total + test_total):
    print("Semua gambar terdistribusi dengan benar.")
    log_lines.append("Semua gambar terdistribusi dengan benar.")
else:
    print("Jumlah gambar tidak cocok! Periksa ulang!")
    log_lines.append("Jumlah gambar tidak cocok!")

# Simpan LOG .txt
with open(os.path.join(LOG_DIR, "split_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

# Simpan JSON
summary["totals"] = {
    "before": total_before,
    "train": train_total,
    "val": val_total,
    "test": test_total,
    "after": train_total + val_total + test_total,
}

with open(os.path.join(LOG_DIR, "split_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("Log disimpan ke:", os.path.join(LOG_DIR, "split_log.txt"))
print("JSON ringkasan disimpan ke:", os.path.join(LOG_DIR, "split_summary.json"))
