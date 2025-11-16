from flask import Blueprint
from app.controllers.season_controller import (
    start_season_controller,
    stop_season_controller,
    get_current_season_controller,
)
from app.middleware.auth_middleware import firebase_required

season_bp = Blueprint("season_bp", __name__, url_prefix="/api/season")

season_bp.route("/start", methods=["POST"])(firebase_required(start_season_controller))
season_bp.route("/stop", methods=["POST"])(firebase_required(stop_season_controller))
season_bp.route("/current", methods=["GET"])(
    firebase_required(get_current_season_controller)
)
