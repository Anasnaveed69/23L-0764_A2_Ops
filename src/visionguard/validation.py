"""Upload validation shared by the Flask UI and the model API.

Three independent gates, cheapest first:
  1. extension allow-list      -> 415 Unsupported Media Type
  2. size limit                -> 413 Payload Too Large (also enforced by Flask MAX_CONTENT_LENGTH)
  3. full decode with Pillow   -> 422 Unprocessable (looks like an image, cannot be decoded)
The bytes stay in memory; nothing is written to disk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from werkzeug.utils import secure_filename

from .imaging import ImageDecodeError, open_image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}


class UploadError(Exception):
    def __init__(self, status: int, title: str, message: str):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


@dataclass
class CheckedUpload:
    data: bytes
    filename: str      # sanitised, safe to store / display
    sha256: str
    width: int
    height: int
    size_kb: float


def check_upload(file_storage, max_bytes: int) -> CheckedUpload:
    if file_storage is None or not file_storage.filename:
        raise UploadError(400, "No file selected", "Choose an image file before pressing Analyse.")
    name = secure_filename(file_storage.filename)[:120] or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Upload one of: "
                          + ", ".join(sorted(ALLOWED_EXTENSIONS)) + ".")
    data = file_storage.read(max_bytes + 1)  # read at most limit+1 bytes into memory
    if len(data) > max_bytes:
        raise UploadError(413, "File too large",
                          f"The limit is {max_bytes // (1024 * 1024)} MB. Resize or compress the image.")
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file contains no data.")
    try:
        img = open_image(data)
    except ImageDecodeError:
        raise UploadError(422, "Not a readable image",
                          "The file has an image extension but its contents could not be decoded. "
                          "It may be corrupt, truncated or a renamed non-image file.") from None
    return CheckedUpload(data=data, filename=name, sha256=hashlib.sha256(data).hexdigest(),
                         width=img.width, height=img.height, size_kb=round(len(data) / 1024, 1))
