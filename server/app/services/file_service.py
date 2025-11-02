import os
from flask import send_from_directory
from app.config.config import Config


def get_uploaded_file(filename: str):
    """
    Mengambil file dari folder uploads dan mengembalikannya sebagai response Flask.
    """
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}' tidak ditemukan di uploads.")

    return send_from_directory(Config.UPLOAD_FOLDER, filename)
