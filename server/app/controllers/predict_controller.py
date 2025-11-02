from flask import request, jsonify
import os
import traceback
from werkzeug.utils import secure_filename
from marshmallow import ValidationError
from app.services.model_service import predict_image
from app.services.detection_service import save_detection_to_firestore
from app.schemas.predict_schema import PredictSchema
from app.config.config import Config


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

        # Buat URL publik agar bisa diakses Flutter
        public_url = f"{request.host_url}uploads/{filename}"

        # Jalankan prediksi
        result = predict_image(file_path)

        # Simpan hasil ke Firestore (gunakan URL publik)
        detection_data = save_detection_to_firestore(
            result=result,
            image_path=public_url,
        )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "✅ Prediksi berhasil dan disimpan ke Firestore.",
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
