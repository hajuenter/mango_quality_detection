from flask import Blueprint
from app.controllers.file_controller import get_uploaded_file_controller

file_bp = Blueprint("file_bp", __name__, url_prefix="/uploads")


@file_bp.route("/<path:filename>", methods=["GET"])
def serve_uploaded_file(filename):
    return get_uploaded_file_controller(filename)
