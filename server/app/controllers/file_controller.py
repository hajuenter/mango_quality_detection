from flask import jsonify
from app.services.file_service import get_uploaded_file
from app.schemas.file_request_schema import FileRequestSchema


def get_uploaded_file_controller(filename):
    """
    Controller untuk menangani permintaan file upload.
    """
    try:
        # Validasi nama file pakai schema
        schema = FileRequestSchema()
        errors = schema.validate({"filename": filename})
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        # Jika valid, ambil file dari service
        return get_uploaded_file(filename)

    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "message": f"Terjadi kesalahan: {e}"}), 500
