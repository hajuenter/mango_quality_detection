import cv2
import threading
import time
from app.services.trigger_service import trigger_service
from app.services.status_service import capture_and_predict_from_webcam


class VideoService:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.is_running = False

    def _open_camera(self):
        """Buka kamera hanya sekali"""
        if self.cap is None or not self.cap.isOpened():
            # Gunakan CAP_DSHOW agar startup lebih cepat di Windows
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("🎥 Kamera dibuka (CAP_DSHOW aktif)")

    def _close_camera(self):
        """Tutup kamera"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print("🛑 Kamera ditutup")

    def start_capture_thread(self):
        """Mulai thread capture frame (jika belum jalan)"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            thread.start()
            print("✅ Thread capture dimulai (kamera siap)")

    def _capture_loop(self):
        """Loop ambil frame & cek trigger ESP32"""
        self._open_camera()

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                break

            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.latest_frame = frame.copy()

            # Cek trigger ESP32
            if trigger_service.get_and_reset_trigger():
                print("📸 Trigger terdeteksi! Memulai capture dan prediksi...")
                frame_copy = (
                    self.latest_frame.copy() if self.latest_frame is not None else None
                )
                if frame_copy is not None:
                    threading.Thread(
                        target=capture_and_predict_from_webcam,
                        args=(frame_copy,),
                        daemon=True,
                    ).start()

            time.sleep(0.03)  # batasi FPS agar tidak berat (≈30 FPS)

        self._close_camera()
        print("🧵 Thread capture berhenti")

    def generate_frames(self):
        """Generator MJPEG untuk streaming video"""
        self.start_capture_thread()

        try:
            while self.is_running:
                with self.lock:
                    if self.latest_frame is None:
                        continue
                    frame = self.latest_frame.copy()

                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except GeneratorExit:
            print("❌ Stream video berakhir")

        finally:
            self.is_running = False
            self._close_camera()

    def release(self):
        """Manual stop"""
        self.is_running = False
        self._close_camera()


# Singleton instance
video_service = VideoService()
