from flask import Flask
from flask_cors import CORS

from app.routes.predict_route import predict_bp
from app.routes.detection_route import detection_bp
from app.routes.video_route import video_bp
from app.routes.file_route import file_bp


def create_app():
    app = Flask(__name__)

    # Aktifkan CORS untuk semua endpoint di bawah /api/*
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(predict_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(file_bp)

    return app
