# VisionGuard model API — the image deployed to the cloud (Task 7).
# The champion model is BUILT IN: the image is immutable and self-contained, the server never
# needs DagsHub credentials, and every replica serves exactly the same model version.
#   python scripts/export_champion.py          # writes models/champion.keras + .json
#   docker build -f api/Dockerfile -t visionguard-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=/srv/src \
    MODEL_PATH=/srv/models/champion.keras PORT=8080

WORKDIR /srv
COPY requirements-serve.txt requirements-tf.txt ./
RUN pip install -r requirements-serve.txt -r requirements-tf.txt

RUN groupadd --system --gid 10001 vg && useradd --system --uid 10001 --gid vg --no-create-home vg
COPY --chown=vg:vg src ./src
COPY --chown=vg:vg api ./api
COPY --chown=vg:vg models/champion.keras models/champion.json ./models/

USER vg
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "/srv/api/healthcheck.py"]
# Shell form so $PORT (set by Cloud Run / Azure / Render) is expanded at runtime.
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60 \
    --access-logfile - api.wsgi:app
