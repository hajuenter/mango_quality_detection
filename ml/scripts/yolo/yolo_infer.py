# from ultralytics import YOLO
# import os


# def infer_image():
#     # Load model hasil training
#     model_path = "ml/result/mango_quality_cls/weights/best.pt"
#     model = YOLO(model_path)

#     # Path gambar uji tunggal (ganti sesuai lokasi gambar kamu)
#     image_path = "ml/dataset/mango_dataset_dl_split/val/mango_healthy/xxx.jpg"

#     # Prediksi
#     results = model.predict(
#         source=image_path, save=True, project="ml/result", name="mango_infer"
#     )

#     # Tampilkan hasil
#     for r in results:
#         probs = r.probs
#         predicted_class = r.names[probs.top1]
#         confidence = probs.top1conf.item()
#         print(f"Prediksi: {predicted_class} ({confidence:.2f})")

#     print("\n✅ Inference selesai! Hasil disimpan di: ml/result/mango_infer")


# if __name__ == "__main__":
#     infer_image()
