import firebase_admin
from firebase_admin import credentials, firestore
from app.config.config import Config

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(Config.FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
except Exception as e:
    raise RuntimeError(f"Gagal inisialisasi Firebase: {e}")


firebase_db = firestore.client()
