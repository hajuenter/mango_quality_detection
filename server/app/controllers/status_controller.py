from flask import jsonify, request
from marshmallow import ValidationError
from app.services.trigger_service import trigger_service
from app.schemas.trigger_schema import TriggerRequestSchema, TriggerResponseSchema


def trigger_capture_controller():
    """
    Endpoint untuk ESP32 mengirim trigger.
    ESP32 POST ke /api/trigger dengan JSON: {"trigger": true}
    """
    request_schema = TriggerRequestSchema()
    response_schema = TriggerResponseSchema()

    try:
        # Validasi input dari ESP32
        data = request.get_json()
        validated_data = request_schema.load(data)

        trigger_value = validated_data.get("trigger", False)

        if trigger_value:
            trigger_service.set_trigger(True)
            response = {
                "success": True,
                "message": "Trigger diterima, webcam akan mengambil foto dan mendeteksi",
            }
        else:
            response = {"success": True, "message": "Trigger false, tidak ada aksi"}

        return jsonify(response_schema.dump(response)), 200

    except ValidationError as err:
        return (
            jsonify(
                {"success": False, "message": "Validasi gagal", "errors": err.messages}
            ),
            400,
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def get_status_controller():
    """
    Manual trigger untuk testing (opsional).
    Endpoint ini tidak mengembalikan hasil prediksi langsung,
    hanya mengaktifkan trigger.
    """
    response_schema = TriggerResponseSchema()

    try:
        # Set trigger
        trigger_service.set_trigger(True)

        response = {
            "success": True,
            "message": "Manual trigger berhasil diset. Webcam akan capture di frame berikutnya.",
        }

        return jsonify(response_schema.dump(response)), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
