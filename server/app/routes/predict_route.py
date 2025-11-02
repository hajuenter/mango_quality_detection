from flask import Blueprint
from app.controllers.predict_controller import predict_controller

predict_bp = Blueprint("predict_bp", __name__, url_prefix="/api")
predict_bp.route("/predict", methods=["POST"])(predict_controller)
