from flask import request, jsonify
import os
import traceback
from werkzeug.utils import secure_filename
from marshmallow import ValidationError
from app.services.model_service_old import predict_image
from app.services.detection_service import save_detection_to_firestore
from app.schemas.predict_schema import PredictSchema
from app.config.config import Config
from app.services.season_service import get_active_season


def predict_controller():
    """
    Handle request prediksi gambar mangga
    """
    schema = PredictSchema()

    try:
        # Ambil file dari form-data
        file = request.files.get("file")
        schema.load({"file": file})  # Validasi file upload

        # Simpan file di folder uploads
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Buat URL publik menggunakan Config (konsisten)
        public_url = Config.get_public_url(filename)

        # Jalankan prediksi
        result = predict_image(file_path)
        active_season = get_active_season()
        season_name = active_season["name"] if active_season else None
        season_status = active_season["status"] if active_season else "none"

        # Simpan hasil ke Firestore
        detection_data = save_detection_to_firestore(
            result=result,
            image_path=public_url,
            season_name=season_name,
            season_status=season_status,
        )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Prediksi berhasil dan disimpan ke Firestore.",
                    "result": result,
                    "saved": detection_data,
                }
            ),
            200,
        )

    except ValidationError as err:
        return jsonify({"success": False, "errors": err.messages}), 400
    except Exception as e:
        print("❌ Error di predict_controller:", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
