import io

from conftest import image_bytes


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_pages_render(client):
    for path in ("/", "/history"):
        r = client.get(path)
        assert r.status_code == 200 and b"VisionGuard" in r.data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["debug"] is False


def test_predict_logs_to_history(client):
    r = upload(client, image_bytes(), "photo.jpg")
    assert r.status_code == 302 and "/result/" in r.headers["Location"]
    page = client.get(r.headers["Location"])
    assert b"Likely AI-generated" in page.data and b"91.0%" in page.data
    hist = client.get("/history").data
    assert b"photo.jpg" in hist and b"fake-model" in hist
    items = client.get("/api/history").get_json()["items"]
    assert items[0]["label"] == "AI-generated" and items[0]["status"] == "ok"


def test_wrong_type_is_rejected_and_logged(client):
    r = upload(client, b"MZ\x90\x00", "tool.exe")
    assert r.status_code == 415 and b"Unsupported file type" in r.data
    items = client.get("/api/history").get_json()["items"]
    assert items[0]["status"] == "rejected"


def test_undecodable_image_rejected(client):
    r = upload(client, b"not an image at all" * 10, "holiday.png")
    assert r.status_code == 422 and b"Not a readable image" in r.data


def test_too_large_rejected(client):
    r = upload(client, b"\xff" * (2 * 1024 * 1024), "huge.jpg")
    assert r.status_code == 413 and b"File too large" in r.data


def test_missing_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_unknown_result_404(client):
    r = client.get("/result/99999")
    assert r.status_code == 404 and b"404" in r.data


def test_history_filter(client):
    upload(client, image_bytes(), "a.jpg")
    r = client.get("/history?label=Real")
    assert r.status_code == 200 and b"a.jpg" not in r.data
