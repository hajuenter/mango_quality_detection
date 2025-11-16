from imagekitio import ImageKit
from dotenv import load_dotenv
import os

load_dotenv()

imagekit = ImageKit(
    public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
    private_key=os.getenv("IMAGEKIT_PRIVATE_KEY"),
    url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT"),
)


def upload_to_imagekit(file_path, filename):
    try:
        with open(file_path, "rb") as f:
            upload_result = imagekit.upload(file=f, file_name=filename)

        return upload_result.url

    except Exception as e:
        raise RuntimeError(f"Gagal upload ke ImageKit: {str(e)}")
