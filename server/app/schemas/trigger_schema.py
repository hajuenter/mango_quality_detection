from marshmallow import Schema, fields, validates, ValidationError


class TriggerRequestSchema(Schema):
    """Schema untuk request trigger dari ESP32"""

    trigger = fields.Boolean(required=True)

    @validates("trigger")
    def validate_trigger(self, value):
        if not isinstance(value, bool):
            raise ValidationError("Trigger harus berupa boolean (true/false)")


class TriggerResponseSchema(Schema):
    """Schema untuk response trigger"""

    success = fields.Boolean(required=True)
    message = fields.Str(required=True)
