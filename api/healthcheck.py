"""Docker HEALTHCHECK for the model API (slim images have no curl)."""
import os
import sys
import urllib.request

try:
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8080')}/health"
    sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)
except Exception:  # noqa: BLE001
    sys.exit(1)
