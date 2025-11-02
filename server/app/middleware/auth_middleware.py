from functools import wraps
from flask import request, jsonify
from firebase_admin import auth


def firebase_required(f):
    """
    Middleware decorator untuk memverifikasi Firebase ID Token.
    Pastikan header Authorization berisi: 'Bearer <token>'
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Authorization header missing"}), 401

        # Format header: "Bearer <token>"
        parts = auth_header.split("Bearer ")
        if len(parts) != 2 or not parts[1].strip():
            return jsonify({"error": "Invalid Authorization header format"}), 401

        id_token = parts[1].strip()

        try:
            # Verifikasi token dengan Firebase Admin SDK
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({"error": f"Invalid or expired token: {str(e)}"}), 401

        return f(*args, **kwargs)

    return wrapper
