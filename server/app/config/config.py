import os


class Config:
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    # MODEL_PATH = os.path.join(
    #     BASE_DIR, "ml", "models", "random_forest", "random_forest_mango.pkl"
    # )

    MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "knn", "knn_mango.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "ml", "models", "knn", "scaler.pkl")

    FIREBASE_CRED_PATH = os.path.join(os.path.dirname(__file__), "mango-db.json")

    BASE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    UPLOAD_FOLDER = os.path.join(BASE_DIR_UP, "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    BASE_URL = "http://172.28.48.1:5000"

    @staticmethod
    def get_public_url(filename):
        """Generate public URL untuk file uploads"""
        return f"{Config.BASE_URL}/uploads/{filename}"
