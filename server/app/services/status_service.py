import cv2
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from app.services.model_service import predict_image
from app.services.detection_service import save_detection_to_firestore
from app.services.season_service import get_active_season
from app.config.config import Config


def capture_and_predict_from_webcam(frame=None):
    """
    Prediksi dari frame webcam yang sudah di-capture atau capture baru.

    Args:
        frame: numpy array (BGR) dari webcam. Jika None, akan capture baru.
    """
    try:
        # 1. Gunakan frame yang diberikan atau capture baru
        if frame is None:
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                raise RuntimeError("Kamera tidak bisa dibuka")

            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Gagal mengambil gambar dari webcam")

        # Validasi gambar
        print(f"🔍 Dimensi gambar dari webcam: {frame.shape}")

        # Pastikan gambar dalam format BGR yang benar
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 2. Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = secure_filename(f"webcam_{timestamp}_{unique_id}.jpg")
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        # 3. Simpan ke folder uploads dengan kualitas tinggi
        cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Validasi file tersimpan
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise RuntimeError("Gagal menyimpan gambar ke folder uploads")

        print(f"✅ Gambar disimpan: {file_path} ({os.path.getsize(file_path)} bytes)")

        # 4. Buat URL publik TANPA request.host_url
        public_url = Config.get_public_url(filename)
        print(f"🌐 Public URL: {public_url}")

        # 5. Jalankan prediksi
        result = predict_image(file_path)

        # 6. Get active season
        active_season = get_active_season()
        season_name = active_season["name"] if active_season else None
        season_status = active_season["status"] if active_season else "none"

        # 7. Simpan hasil ke Firestore
        detection_data = save_detection_to_firestore(
            result=result,
            image_path=public_url,
            season_name=season_name,
            season_status=season_status,
        )

        print(f"✅ Prediksi berhasil: {result['label']} ({result['confidence']:.2%})")

        return {
            "success": True,
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "method": result["method"],
            "image_url": public_url,
            "saved": detection_data,
            "message": "Prediksi dari webcam berhasil dan disimpan ke Firestore",
        }

    except Exception as e:
        import traceback

        print(f"❌ Error di capture_and_predict_from_webcam: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "message": str(e)}
