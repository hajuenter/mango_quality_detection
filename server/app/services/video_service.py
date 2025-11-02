import cv2


class VideoService:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def generate_frames(self):
        """Generator untuk streaming frame webcam"""
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Encode frame ke JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Format multipart MJPEG
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def release(self):
        """Lepaskan kamera"""
        self.cap.release()
