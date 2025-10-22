from ultralytics import YOLO
import os
import shutil


def train_model():
    # Path dataset
    dataset_path = "ml/dataset/mango_dataset_dl_split"

    # Load pretrained YOLOv8 untuk klasifikasi
    model = YOLO("yolov8s-cls.pt")

    # Jalankan training
    results = model.train(
        data=dataset_path,
        epochs=50,
        imgsz=224,
        batch=16,
        patience=7,  # early stopping 7 epoch tanpa perbaikan
        project="ml/results",
        name="mango_quality_cls",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # Path hasil training YOLO
    result_dir = results.save_dir

    # Pastikan folder models ada untuk simpan grafik
    os.makedirs("ml/models/train", exist_ok=True)

    # File-file visualisasi hasil training
    visual_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
    ]

    for vf in visual_files:
        src = os.path.join(result_dir, vf)
        if os.path.exists(src):
            dst = os.path.join("ml/models/train", vf)
            shutil.copy(src, dst)
            print(f"📊 Saved visualization: {vf}")

    summary_path = os.path.join("ml/models/train", "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Mango Quality Classification - YOLOv8 Results\n")
        f.write("=" * 50 + "\n")
        f.write(
            str(results.metrics)
            if hasattr(results, "metrics")
            else "No metrics available.\n"
        )
    print("🧾 Saved metrics summary: metrics_summary.txt")

    print("\n✅ Training finished successfully!")
    print("📁 Model files: ml/result/mango_quality_cls/weights/")
    print("📈 Visualizations: ml/models/train/")


if __name__ == "__main__":
    train_model()
