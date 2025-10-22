from ultralytics import YOLO
import os
import shutil


def test_model():
    # Path model hasil training
    model_path = "ml/results/mango_quality_cls/weights/best.pt"
    model = YOLO(model_path)

    # Path dataset validasi
    val_dir = "ml/dataset/mango_dataset_dl_split/val"

    # Jalankan evaluasi
    results = model.val(
        data=val_dir, save=True, project="ml/results", name="mango_quality_test"
    )

    # Path output YOLO (hasil evaluasi otomatis)
    result_dir = results.save_dir

    # Pastikan folder models ada
    os.makedirs("ml/models/test", exist_ok=True)

    # File-file metrik visual yang ingin disalin
    visual_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "labels_correlogram.jpg",
        "results.png",
    ]

    # Salin file ke ml/models jika ada
    for vf in visual_files:
        src = os.path.join(result_dir, vf)
        if os.path.exists(src):
            dst = os.path.join("ml/models/test", vf)
            shutil.copy(src, dst)
            print(f"📊 Disalin: {vf}")

    print("\n✅ Evaluasi selesai!")
    print("📂 Semua metrik visual disimpan di: ml/models/test")


if __name__ == "__main__":
    test_model()
