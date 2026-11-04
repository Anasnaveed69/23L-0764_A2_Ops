import io

import pytest
from werkzeug.datastructures import FileStorage

from conftest import image_bytes
from visionguard.imaging import ImageDecodeError, decode_and_resize, open_image
from visionguard.validation import UploadError, check_upload

MB = 1024 * 1024


def fs(data: bytes, name: str) -> FileStorage:
    return FileStorage(stream=io.BytesIO(data), filename=name)


def test_accepts_valid_jpeg():
    up = check_upload(fs(image_bytes(), "cat photo.jpg"), MB)
    assert (up.width, up.height) == (64, 48)
    assert up.filename == "cat_photo.jpg" and len(up.sha256) == 64


@pytest.mark.parametrize("name", ["virus.exe", "notes.txt", "noextension", "img.svg"])
def test_rejects_wrong_extension(name):
    with pytest.raises(UploadError) as e:
        check_upload(fs(image_bytes(), name), MB)
    assert e.value.status == 415


def test_rejects_too_large():
    with pytest.raises(UploadError) as e:
        check_upload(fs(b"\xff" * (MB + 10), "big.jpg"), MB)
    assert e.value.status == 413


def test_rejects_renamed_non_image():
    with pytest.raises(UploadError) as e:
        check_upload(fs(b"this is not really a jpeg" * 20, "fake.jpg"), MB)
    assert e.value.status == 422


def test_rejects_truncated_image():
    data = image_bytes("JPEG", (256, 256))
    with pytest.raises(UploadError) as e:
        check_upload(fs(data[: len(data) // 2], "cut.jpg"), MB)
    assert e.value.status == 422


def test_sanitises_path_traversal_names():
    up = check_upload(fs(image_bytes("PNG"), "../../etc/passwd.png"), MB)
    assert "/" not in up.filename and ".." not in up.filename


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("PNG", "L"), ("WEBP", "RGB"), ("PNG", "P")])
def test_decode_handles_modes(fmt, mode):
    x = decode_and_resize(image_bytes(fmt, (80, 40), mode), 32)
    assert x.shape == (32, 32, 3) and x.dtype.name == "float32"
    assert 0 <= x.min() and x.max() <= 255


def test_pad_keeps_aspect():
    x = decode_and_resize(image_bytes("PNG", (100, 50)), 32, "pad")
    assert x.shape == (32, 32, 3)
    assert x[0].sum() == 0  # letterbox border is black


def test_open_image_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        open_image(b"GIF89a" + b"\x00" * 10)
