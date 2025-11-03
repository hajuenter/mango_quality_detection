from flask import request, jsonify
from marshmallow import ValidationError
from app.services.season_service import (
    get_active_season,
    start_new_season,
    stop_active_season,
)
from app.schemas.season_schema import SeasonStartSchema, SeasonResponseSchema


def start_season_controller():
    schema = SeasonStartSchema()
    try:
        data = schema.load(request.get_json())
        season = start_new_season(data["name"])

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Musim '{data['name']}' berhasil dimulai.",
                    "data": season,
                }
            ),
            200,
        )

    except ValidationError as err:
        return jsonify({"success": False, "errors": err.messages}), 400
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def stop_season_controller():
    try:
        stopped = stop_active_season()

        if not stopped:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Tidak ada musim aktif yang bisa dihentikan.",
                    }
                ),
                400,
            )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Musim aktif telah dihentikan.",
                    "data": stopped,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_current_season_controller():
    try:
        active = get_active_season()
        schema = SeasonResponseSchema()
        result = schema.dump(active or {"name": None, "status": "none"})
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
