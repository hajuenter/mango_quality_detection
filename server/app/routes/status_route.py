from flask import Blueprint
from app.controllers.status_controller import (
    get_status_controller,
    trigger_capture_controller,
)

status_bp = Blueprint("status_bp", __name__, url_prefix="/api")

status_bp.route("/status", methods=["GET"])(get_status_controller)
status_bp.route("/trigger", methods=["POST"])(trigger_capture_controller)
