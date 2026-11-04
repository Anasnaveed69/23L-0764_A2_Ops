# VisionGuard — AI-generated vs real image classifier

[![CI](https://github.com/Anasnaveed69/23L-0764_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Anasnaveed69/23L-0764_A2_Ops/actions/workflows/ci.yml)

MLOps Assignment 2 · **Anas Naveed Butt (23L-0764)** · project `23L-0764_A2_Ops`

| | |
|---|---|
| DagsHub (code, data, experiments, registry) | https://dagshub.com/Anasnaveed69/23L-0764_A2_Ops |
| Model API (cloud) | https://20-219-10-43.sslip.io — `GET /health`, `POST /v1/predict` |
| Report | `report/` (LaTeX) → `23L-0764_VisionGuard.pdf` |

## Architecture

```
 browser ──► Flask web app (localhost:5000, Docker) ──► PostgreSQL (internal network, named volume)
                   │  INFERENCE_MODE=local:  champion.keras loaded once in-process
                   └─ INFERENCE_MODE=remote: HTTPS + X-API-Key ──► Model API (Azure VM: Caddy ─► gunicorn/Flask + TF-CPU)

 Kaggle data ─► DVC (DagsHub storage) ─► tf.data ─► Keras (CNN / MobileNetV2) ─► MLflow on DagsHub ─► Registry @champion
```

## Layout

```
src/visionguard/   shared library: imaging (decode/resize used by BOTH training and serving),
                   data (tf.data), models, evaluation, tracking (MLflow), inference, validation
scripts/           00_env_check · 01_audit · 02_eda · 03_make_splits · 04_show_batch · train ·
                   run_experiments · tune · error_analysis · compare_runs · register_model ·
                   load_champion · export_champion · infer_cli · send_samples
app/               Flask UI (upload / result / history / health) + SQLAlchemy trail
api/               cloud model API + its Dockerfile
labs/              toy labs: DVC, MLflow, Keras Tuner, Docker warm-up, CI
deploy/            cloud compose (API + Caddy HTTPS), VM setup, Postman collection
notebooks/         Colab GPU training notebook
report/            LaTeX report; generated/ and figures/auto/ are written by the scripts
```

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
cp .env.example .env                       # fill in
dvc pull                                   # dataset from DagsHub
python scripts/run_experiments.py          # train + log everything to MLflow
python scripts/compare_runs.py && python scripts/register_model.py && python scripts/export_champion.py
docker compose up -d --build               # http://localhost:5000
```

Full, ordered instructions with every screenshot: **[RUNBOOK.md](RUNBOOK.md)**.

## Tests

`pytest` (27 tests: validation, web routes + DB trail, inference, API auth). CI runs them on Python 3.11
and 3.12, again against PostgreSQL, then builds both Docker images and smoke-tests the containers.
