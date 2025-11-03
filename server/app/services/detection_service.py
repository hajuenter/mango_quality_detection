from datetime import datetime
from app.config.firebase_db import firebase_db


def save_detection_to_firestore(
    result, image_path, season_name=None, season_status="none"
):
    try:
        now = datetime.now()

        detection_data = {
            "label": result["label"],
            "confidence": result["confidence"],
            "timestamp": now,
            "date": now.strftime("%Y-%m-%d"),
            "month": now.strftime("%Y-%m"),
            "year": now.strftime("%Y"),
            "image_url": image_path,
            "method": result["method"],
            "season_name": season_name,
            "season_status": season_status,
        }

        firebase_db.collection("mango_detections").add(detection_data)

        return detection_data

    except Exception as e:
        raise RuntimeError(f"Gagal menyimpan ke Firestore: {e}")


def get_all_detections():
    try:
        docs = (
            firebase_db.collection("mango_detections")
            .order_by("timestamp", direction="DESCENDING")
            .stream()
        )

        detections = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            detections.append(data)

        return detections

    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari Firestore: {e}")


def get_healthy_detections():
    try:
        docs = (
            firebase_db.collection("mango_detections")
            .where("label", "==", "mango_healthy")
            .order_by("timestamp", direction="DESCENDING")
            .stream()
        )

        detections = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            detections.append(data)

        return detections
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data healthy: {e}")


def get_rotten_detections():
    try:
        docs = (
            firebase_db.collection("mango_detections")
            .where("label", "==", "mango_rotten")
            .order_by("timestamp", direction="DESCENDING")
            .stream()
        )

        detections = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            detections.append(data)

        return detections
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data rotten: {e}")


def get_latest_detections(limit=5):
    try:
        docs = (
            firebase_db.collection("mango_detections")
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
            .stream()
        )

        detections = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            detections.append(data)

        return detections

    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data terbaru: {e}")
