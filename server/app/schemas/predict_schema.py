from marshmallow import Schema, fields, ValidationError


def validate_file(file):
    """Validasi file upload gambar"""
    if file is None:
        raise ValidationError("File harus disertakan.")
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise ValidationError("File harus berformat .jpg, .jpeg, atau .png.")

    file.seek(0, 2)
    size = file.tell()
    if size > 5 * 1024 * 1024:
        raise ValidationError("Ukuran file tidak boleh lebih dari 5MB.")
    file.seek(0)


class PredictSchema(Schema):
    """Schema validasi input prediksi"""

    file = fields.Raw(required=True, validate=validate_file)
