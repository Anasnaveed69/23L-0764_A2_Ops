"""Entry point for gunicorn (`gunicorn app.wsgi:app`) and `python -m app.wsgi`."""
import os

from . import create_app

app = create_app()

if __name__ == "__main__":
    # Development fallback only. debug stays OFF unless FLASK_DEBUG=1 is set explicitly:
    # the Werkzeug debugger allows arbitrary code execution from the browser.
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=app.config["DEBUG"])
