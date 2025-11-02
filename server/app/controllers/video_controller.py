from flask import Response
from app.services.video_service import VideoService

video_service = VideoService()


def get_video_feed_controller():
    """Controller untuk mengembalikan stream kamera"""
    try:
        return Response(
            video_service.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        print(f"[ERROR] Gagal streaming video: {e}")
        return {"error": str(e)}, 500
