from datetime import datetime
from app.config.firebase_db import firebase_db


def get_active_season():
    """
    Ambil data musim aktif dari Firestore (dokumen global 'active')
    """
    try:
        doc_ref = firebase_db.collection("season").document("active")
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            if data.get("status") == "active" and data.get("name"):
                return {
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "id": data.get("id"),
                }
        return None

    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data musim: {e}")


def is_name_exist(name: str) -> bool:
    """
    Cek apakah nama musim sudah pernah digunakan
    """
    try:
        query = firebase_db.collection("season").where("name", "==", name).get()
        return len(query) > 0
    except Exception as e:
        raise RuntimeError(f"Gagal memeriksa nama musim: {e}")


def start_new_season(name: str):
    try:
        # Cek apakah ada musim aktif
        active = get_active_season()
        if active and active["status"] == "active":
            raise ValueError(
                "Masih ada musim aktif. Nonaktifkan dulu sebelum memulai yang baru."
            )

        # Cek apakah nama sudah digunakan
        if is_name_exist(name):
            raise ValueError(
                f"Nama musim '{name}' sudah pernah digunakan. Gunakan nama lain."
            )

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        season_id = f"{now}_{name.replace(' ', '_')}"

        season_data = {
            "id": season_id,
            "name": name,
            "status": "active",
            "started_at": datetime.now(),
            "ended_at": None,
        }

        # Simpan sebagai dokumen riwayat
        firebase_db.collection("season").document(season_id).set(season_data)

        # Update dokumen global active
        firebase_db.collection("season").document("active").set(
            {"id": season_id, "name": name, "status": "active"}, merge=True
        )

        return {"id": season_id, "name": name, "status": "active"}

    except Exception as e:
        raise RuntimeError(f"Gagal memulai musim: {e}")


def stop_active_season():
    """
    Hentikan musim aktif sekarang:
    - Ubah status musim aktif di riwayat jadi 'inactive'
    - Kosongkan dokumen 'active'
    """
    try:
        active = get_active_season()
        if not active or active["status"] != "active":
            return None

        # Update musim di riwayat
        season_doc = firebase_db.collection("season").document(active["id"])
        season_doc.update({"status": "inactive", "ended_at": datetime.now()})

        # Kosongkan dokumen 'active'
        firebase_db.collection("season").document("active").set(
            {"name": None, "status": "none", "id": None}
        )

        return {"name": None, "status": "none", "id": None}

    except Exception as e:
        raise RuntimeError(f"Gagal menghentikan musim: {e}")
