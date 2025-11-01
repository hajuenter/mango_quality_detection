import cv2
import numpy as np
import os
import sys
import tempfile
import joblib
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_PATH = os.path.join(ML_DIR, "models", "random_forest", "random_forest_mango.pkl")

SCRIPTS_PATH = os.path.join(ML_DIR, "scripts", "random_forest")
sys.path.append(SCRIPTS_PATH)

from features import extract_features

print("📦 Loading Random Forest model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")


def is_mango_present(frame, min_area_ratio=0.03):
    """
    Deteksi keberadaan mangga berdasarkan kombinasi warna dan bentuk.
    Menggunakan area warna khas mangga + filtering bentuk.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rentang warna hijau–kuning khas mangga matang dan mentah
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([35, 255, 255])

    # gabungkan dua mask (mangga hijau dan mangga kuning)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_green, mask_yellow)

    # bersihkan noise kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = frame.shape
    frame_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / frame_area

        # filter: area cukup besar dan bentuk agak lonjong (mangga)
        if area_ratio > min_area_ratio:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / float(bh)
            if 0.5 < aspect_ratio < 2.0:  # bentuk lonjong (tidak pipih seperti daun)
                return True
    return False


def predict_image(image_path):
    """Prediksi kualitas mangga menggunakan Random Forest."""
    features = extract_features(image_path)
    features = np.expand_dims(features, axis=0)
    pred_proba = model.predict_proba(features)[0]
    classes = model.classes_.tolist()

    healthy_idx = classes.index("mango_healthy")
    rotten_idx = classes.index("mango_rotten")
    pred_label = model.predict(features)[0]
    confidence = float(max(pred_proba))

    return {
        "label": pred_label,
        "confidence": confidence,
        "probabilities": {
            "mango_healthy": float(pred_proba[healthy_idx]),
            "mango_rotten": float(pred_proba[rotten_idx]),
        },
    }


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Tidak dapat membuka webcam.")
        return

    print("🎥 Tekan 'SPACE' untuk jepret & prediksi, 'q' untuk keluar.")
    last_prediction = "Tekan spasi untuk mulai prediksi..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # tampilkan hasil prediksi terakhir di layar
        color = (0, 255, 0) if "healthy" in last_prediction else (0, 0, 255)
        if "Tekan" in last_prediction or "No mango" in last_prediction:
            color = (255, 255, 255)

        cv2.putText(
            frame, last_prediction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        cv2.imshow("Mango Quality Detection (Random Forest)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # tombol spasi
            print("📸 Mengambil gambar untuk prediksi...")

            if not is_mango_present(frame):
                last_prediction = "🚫 Tidak ada mangga terdeteksi"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {last_prediction}")
                continue

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()

            cv2.imwrite(tmp_path, frame)
            result = predict_image(tmp_path)

            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"⚠️ Gagal hapus temp file: {e}")

            label = f"{result['label']} ({result['confidence']:.2f})"
            last_prediction = label
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 {label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
