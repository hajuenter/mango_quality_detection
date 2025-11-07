from marshmallow import Schema, fields


class StatusResponseSchema(Schema):
    success = fields.Boolean(required=True)
    label = fields.Str(required=False)
    confidence = fields.Float(required=False)
    probabilities = fields.Dict(required=False)
    method = fields.Str(required=False)
    image_url = fields.Str(required=False)
    saved = fields.Dict(required=False)
    message = fields.Str(required=True)
