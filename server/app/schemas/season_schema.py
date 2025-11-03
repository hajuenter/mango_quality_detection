from marshmallow import Schema, fields, validate


class SeasonStartSchema(Schema):
    """
    Schema untuk memulai musim baru
    """

    name = fields.Str(
        required=True,
        validate=validate.Length(min=3, error="Nama musim minimal 3 karakter."),
        error_messages={"required": "Field 'name' wajib diisi."},
    )


class SeasonResponseSchema(Schema):
    """
    Schema untuk response data musim
    """

    id = fields.Str(allow_none=True)
    name = fields.Str(allow_none=True)
    status = fields.Str(validate=validate.OneOf(["active", "inactive", "none"]))
    started_at = fields.DateTime(allow_none=True)
    ended_at = fields.DateTime(allow_none=True)
