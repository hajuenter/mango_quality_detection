from flask import Blueprint
from app.controllers.video_controller import get_video_feed_controller

video_bp = Blueprint("video_bp", __name__, url_prefix="/api")

video_bp.route("/video_feed", methods=["GET"])(get_video_feed_controller)
