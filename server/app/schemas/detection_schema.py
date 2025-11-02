from marshmallow import Schema, fields


class GetAllDetectionsSchema(Schema):
    id = fields.Str(required=True)
    label = fields.Str(required=True)
    confidence = fields.Float(required=True)
    timestamp = fields.DateTime(required=True)
    date = fields.Str(required=True)
    month = fields.Str(required=True)
    year = fields.Str(required=True)
    image_url = fields.Str(required=True)
    method = fields.Str(required=True)
