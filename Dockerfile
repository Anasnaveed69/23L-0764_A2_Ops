# VisionGuard web app (Flask UI + database trail).
#   docker build -t visionguard-web .                         # local inference (TensorFlow inside)
#   docker build -t visionguard-web:remote --build-arg WITH_TF=false .   # remote inference only
FROM python:3.11-slim AS runtime

ARG WITH_TF=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app/src

WORKDIR /app

# 1) Dependencies FIRST. These layers only rebuild when a requirements file changes, so editing
#    Python/HTML code re-uses the cached ~1 GB TensorFlow layer and rebuilds in seconds.
COPY requirements-serve.txt requirements-tf.txt ./
RUN pip install -r requirements-serve.txt \
 && if [ "$WITH_TF" = "true" ]; then pip install -r requirements-tf.txt; fi

# 2) Unprivileged user: a compromised process cannot modify the image or bind low ports.
RUN groupadd --system --gid 10001 vg && useradd --system --uid 10001 --gid vg --no-create-home vg

# 3) Source code LAST (changes most often). Model weights are NOT copied: they are mounted
#    read-only by docker-compose so a new champion does not need an image rebuild.
COPY --chown=vg:vg src ./src
COPY --chown=vg:vg app ./app

USER vg
EXPOSE 8000

# python is used instead of curl because slim images do not ship curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# gunicorn: production WSGI server. 1 worker x 4 threads keeps ONE copy of the model in RAM.
# NOT --preload: TensorFlow initialised in the master and then forked deadlocks on the first
# predict(); without preload the model is loaded once inside the worker, after the fork.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "60", \
     "--access-logfile", "-", "app.wsgi:app"]
