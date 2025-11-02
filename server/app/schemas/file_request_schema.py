from marshmallow import Schema, fields, validates, ValidationError


class FileRequestSchema(Schema):
    filename = fields.Str(required=True)

    @validates("filename")
    def validate_filename(self, value, **kwargs):
        if ".." in value or value.startswith("/"):
            raise ValidationError(
                "Nama file tidak valid atau berpotensi path traversal."
            )
