from flask import jsonify
from app.services.detection_service import (
    get_all_detections,
    get_healthy_detections,
    get_latest_detections,
    get_rotten_detections,
)
from app.schemas.detection_schema import GetAllDetectionsSchema


def get_detections_controller():
    try:
        detections = get_all_detections()
        schema = GetAllDetectionsSchema(many=True)
        result = schema.dump(detections)

        return (
            jsonify({"success": True, "count": len(result), "detections": result}),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_healthy_detections_controller():
    try:
        detections = get_healthy_detections()
        schema = GetAllDetectionsSchema(many=True)
        result = schema.dump(detections)

        return (
            jsonify({"success": True, "count": len(result), "detections": result}),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_rotten_detections_controller():
    try:
        detections = get_rotten_detections()
        schema = GetAllDetectionsSchema(many=True)
        result = schema.dump(detections)

        return (
            jsonify({"success": True, "count": len(result), "detections": result}),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_latest_detections_controller():
    try:
        detections = get_latest_detections()
        schema = GetAllDetectionsSchema(many=True)
        result = schema.dump(detections)

        return (
            jsonify({"success": True, "count": len(result), "detections": result}),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
