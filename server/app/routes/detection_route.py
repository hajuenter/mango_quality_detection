from flask import Blueprint
from app.controllers.detection_controller import (
    get_detections_controller,
    get_healthy_detections_controller,
    get_rotten_detections_controller,
    get_latest_detections_controller,
)
from app.middleware.auth_middleware import firebase_required

detection_bp = Blueprint("detection_bp", __name__, url_prefix="/api")

detection_bp.route("/detections", methods=["GET"])(
    firebase_required(get_detections_controller)
)
detection_bp.route("/detections/healthy", methods=["GET"])(
    firebase_required(get_healthy_detections_controller)
)
detection_bp.route("/detections/rotten", methods=["GET"])(
    firebase_required(get_rotten_detections_controller)
)
detection_bp.route("/detections/latest", methods=["GET"])(
    firebase_required(get_latest_detections_controller)
)
