# VisionGuard — step-by-step runbook (23L-0764_A2_Ops)

Everything is already coded and tested. This runbook is the order to run it in, with the exact
commands and the screenshot to take at each point. Screenshot file names match the report:
save each one into `report/figures/shots/<name>.png` and the report picks it up automatically
(missing ones show as a grey placeholder box, so you can see what's left).

> Screenshot rule from the assignment: every shot must show your terminal prompt / username,
> date-time, or the address bar with the project name. Keep the taskbar clock visible.

Suggested schedule: **Sat** steps 0–2 · **Sun** step 3 (Colab) · **Mon** steps 4–6 · **Tue** steps 7–8 · **Wed** report + submit.

Commands are for **Git Bash** on Windows (or any Linux/macOS terminal). PowerShell differences are noted.

---

## 0. Accounts (15 min)

| What | Name to use |
|---|---|
| GitHub repo | `23L-0764_A2_Ops` (public) |
| DagsHub repo | `23L-0764_A2_Ops` — **Create → Connect a repository → GitHub** (mirror) |
| DagsHub toy repo | `23L-0764_A2_Ops-dvc-toy` (empty) |
| DagsHub token | avatar → Your settings → Tokens → copy the default token |
| Azure | Azure for Students, sign up with your `@lhr.nu.edu.pk` email (no card) |
| Postman | desktop app, free account |

## 1. Environment — Task 1.1

```bash
# unzip the project, then:
cd 23L-0764_A2_Ops
py -3.11 -m venv .venv            # Python 3.11 or 3.12 (3.13 also works)
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1   Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                   # makes `visionguard` importable everywhere
cp .env.example .env               # fill DAGSHUB_USER, MLFLOW_* (token), passwords, SECRET_KEY
python scripts/00_env_check.py     # [SHOT s1_1_env]
```

Push to GitHub (first commit = the code you were given):

```bash
git init && git add . && git commit -m "VisionGuard scaffold"
git branch -M main
git remote add origin https://github.com/<github-user>/23L-0764_A2_Ops.git
git push -u origin main
git check-ignore -v .env && git status --short    # [SHOT s3_2_gitignore]  (.env ignored, not listed)
```

## 2. Data — Tasks 1.2, 1.3, 1.4

1. Download the Kaggle zip (https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) and extract it so that the class folders sit under `data/raw/` (e.g. `data/raw/AiArtData/...`, `data/raw/RealArt/...`). Folder names containing `ai`/`real` are mapped automatically (`params.yaml → data.class_map`).
2. Audit, EDA, splits:

```bash
python scripts/01_audit.py         # [SHOT s1_2_audit]  table of counts, min/median/max, corrupt, duplicates
python scripts/02_eda.py           # writes report/figures/auto/eda_*.png
python scripts/03_make_splits.py   # data/splits/{train,val,test}.csv (committed to Git)
```

3. DVC toy (Task 1.4) — follow `labs/dvc_toy.sh` block by block. Shots:
   `s1_4_toy_pointer` (cat scores.csv.dvc + git log), `s1_4_toy_checkout` (old version restored),
   `s1_4_toy_pushpull` (rm + dvc pull restores), `s1_4_toy_dagshub` (DagsHub toy repo page).
4. DVC for the real dataset:

```bash
dvc init
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<dagshub-user>/23L-0764_A2_Ops.s3
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
dvc add data/raw                   # creates data/raw.dvc (pointer) and adds data/raw to .gitignore
git add data/raw.dvc data/.gitignore .dvc/config .dvcignore data/splits report/generated report/figures/auto
git commit -m "Track dataset with DVC; audit, EDA and splits"
dvc push                           # [SHOT s1_4_dataset_dvc]  (terminal) 
git push                           # DagsHub repo page now shows data/raw with the DVC badge [SHOT s1_4_dagshub_data]
```

## 3. Training + tracking on Colab GPU — Tasks 2, 3.3–3.6

Local tool labs first (fast, CPU is fine):

```bash
python labs/mlflow_toy.py --lr 0.001 --local
python labs/mlflow_toy.py --lr 0.01 --local
mlflow ui --backend-store-uri ./mlruns_toy       # http://127.0.0.1:5000
#   [SHOT s3_1_mlflow_runs]  experiment list with both runs
#   [SHOT s3_1_mlflow_compare] select both -> Compare -> metrics chart
python labs/mlflow_toy.py --lr 0.001             # same toy run, now to DagsHub (.env)
#   [SHOT s3_2_dagshub_project] DagsHub repo page   [SHOT s3_2_dagshub_toy_run] Experiments tab with toy run
```

Then open `notebooks/01_data_colab.ipynb` (Task 1 in the cloud: Kaggle download, audit, EDA, splits, DVC) and afterwards `notebooks/02_train_colab.ipynb` in Colab (File → Upload, or open from GitHub), set the 3
secrets, **Runtime → T4 GPU → Run all**. It runs: augmented batch figure, 8 experiment runs,
Keras Tuner (10 trials) + toy tuner, comparison, registry + `champion` alias, export, error
analysis + Grad-CAM, and downloads `colab_results.zip`.

> No Colab? Everything runs locally too: `python scripts/run_experiments.py` then the same
> scripts in the order the notebook uses (slow on CPU: plan ~4–6 h).

Unzip `colab_results.zip` into the project root (it fills `report/generated`, `report/figures/auto`,
`models/champion.*`). Commit: `git add report && git commit -m "Training results" && git push`.

Shots from DagsHub → Experiments (or its MLflow UI):
- `s2_2_summary` — baseline model summary (Colab cell output of run_experiments, `baseline_cnn` part)
- `s2_7_toy_tuner` — toy tuner `results_summary` output in Colab
- `s2_7_tuner` — tune.py output table in Colab
- `s3_3_manual_run` — run `mnv2_default`: Parameters, Metrics charts (loss/val_loss), Tags (git_commit, dvc_data_version, seed)
- `s3_3_autolog_run` — run `mnv2_autolog` (+ its nested `_stage2` run) showing the autologged params like opt_learning_rate
- `s3_4_artifacts` — Artifacts tab: plots/, splits/, environment/, keras_model/, model/
- `s3_5_compare` — select the 8 runs → Compare (table / parallel coordinates)
- `s3_6_registry` — Models tab: `VisionGuard-23L-0764` with ≥2 versions and the `champion` alias
- `s3_6_load_alias` — the load_champion.py cell output

If DagsHub rejects aliases, `register_model.py` automatically falls back to a version tag `alias=champion`; say so in the report.

## 4. Flask app — Tasks 4, 5 (local, no Docker yet)

```bash
python scripts/infer_cli.py <an AI image from test.csv> <a real image>   # [SHOT s4_1_console]
export INFERENCE_MODE=local DATABASE_URL=sqlite:///visionguard.db        # PowerShell: $env:INFERENCE_MODE="local"
python -m app.wsgi                                                       # http://127.0.0.1:5000
```

Shots: `s4_3_upload`, `s4_3_result`, `s4_3_history`; failure cases in the browser:
`s4_2_wrong_type` (upload a .txt/.pdf), `s4_2_too_large` (a >5 MB photo), `s4_2_corrupt`
(rename a .txt to .jpg); `s4_4_health` (http://127.0.0.1:5000/health).

## 5. Docker + Compose — Task 6.1–6.5

```bash
# 6.1 warm-up: follow labs/docker_warmup.md  [SHOTS s6_1_run, s6_1_logs_exec, s6_1_cleanup]

docker compose up -d --build                    # first build ~5-10 min (TensorFlow layer)
docker compose ps                               # both "healthy"            [SHOT s6_3_ps_healthy]
docker compose exec web sh -c "whoami; id"      # vg, uid 10001             [SHOT s6_2_whoami]
docker compose port db 5432 || echo "no published port"
Test-NetConnection localhost -Port 5432         # PowerShell; Linux: nc -zv localhost 5432 -> refused   [SHOT s6_3_port_closed]
docker compose exec db psql -U visionguard -c "select id, created_at, label, round(confidence::numeric,3), model_version, inference_mode from predictions order by id desc limit 10;"
                                                #                           [SHOT s5_db_query]
# 6.4 image size before/after
docker build -f docker/Dockerfile.naive -t visionguard-web:naive .
docker images | grep visionguard                # [SHOT s6_4_sizes]
docker history visionguard-web:latest           # [SHOT s6_4_history]  largest layer = pip install tensorflow-cpu
docker build -t visionguard-web:remote --build-arg WITH_TF=false .   # bonus: no-TF web image for remote mode

# 6.5 operations
docker compose stop web && docker compose start web && docker compose restart db
docker compose logs --tail 30 web               # [SHOT s6_5_ops]
docker compose exec db sh -c "pg_isready; psql -U visionguard -c '\dt'"
# break it on purpose: wrong DB password
#   edit .env -> POSTGRES_PASSWORD=wrong ; then:
docker compose up -d --force-recreate web
docker compose logs --tail 40 web               # password authentication failed  [SHOT s6_5_failure_logs]
#   restore the password in .env
docker compose up -d --force-recreate web && docker compose ps   # [SHOT s6_5_fixed]
```

(Alternative failure: `mv models/champion.keras /tmp/` → logs show `Model file not found` → move it back.)

## 6. GitHub Actions — Task 6.6

Toy: create a throwaway GitHub repo `ci-toy-23L-0764`, copy `labs/ci_toy/*` (incl. `.github`) into it,
push → green. Edit `test_calc.py` to the failing assertion, push → red `[SHOT s6_6_toy_red]`, revert, push → green `[SHOT s6_6_toy_green]`.

Project: the workflow in `.github/workflows/ci.yml` already runs on every push to your repo:
lint + tests on Python 3.11/3.12, tests on PostgreSQL, Docker builds + container smoke test.
Add the badge line at the top of README.md (replace `<github-user>`), push, and screenshot the green
Actions run `[SHOT s6_6_project_green]` and the README badge `[SHOT s6_6_badge]`.

## 7. Cloud deployment of the model API — Task 7 (Azure for Students VM)

1. Azure portal → **Virtual machines → Create**: name `vm-23L-0764-A2-Ops`, region close to you
   (e.g. *UAE North* / *Central India*), image **Ubuntu Server 24.04 LTS x64**, size **B2s** (2 vCPU, 4 GB; TensorFlow needs >1 GB),
   authentication SSH key (download the .pem), inbound ports **SSH, HTTP, HTTPS**. Review + create. `[SHOT s7_vm_created]`
2. Networking tab of the VM → confirm NSG inbound rules 22/80/443 `[SHOT s7_nsg_rules]`.
   *Two firewalls:* the Azure NSG (cloud) and `ufw` (inside the VM). Both must allow 80/443.
3. From your laptop:

```bash
chmod 400 vm-23L-0764-A2-Ops_key.pem
ssh -i vm-23L-0764-A2-Ops_key.pem azureuser@<PUBLIC_IP>
# on the VM:
git clone https://github.com/<github-user>/23L-0764_A2_Ops.git && cd 23L-0764_A2_Ops
bash deploy/vm_setup.sh && exit          # log out so the docker group applies
# back on the laptop: copy the champion model (it's git-ignored)
scp -i vm-23L-0764-A2-Ops_key.pem models/champion.keras models/champion.json azureuser@<PUBLIC_IP>:~/23L-0764_A2_Ops/models/
ssh -i vm-23L-0764-A2-Ops_key.pem azureuser@<PUBLIC_IP>
cd 23L-0764_A2_Ops
sudo ufw status                           # [SHOT s7_ufw]
export API_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))"); echo $API_KEY   # save it in your laptop .env as MODEL_API_KEY
export API_DOMAIN=$(curl -s ifconfig.me | tr . -).sslip.io; echo $API_DOMAIN                     # e.g. 20-55-1-9.sslip.io
docker compose -f deploy/docker-compose.cloud.yml up -d --build
docker compose -f deploy/docker-compose.cloud.yml ps        # api healthy, caddy up   [SHOT s7_compose_up]
docker compose -f deploy/docker-compose.cloud.yml logs caddy | grep -i certificate   # TLS obtained
```

4. **API URL = `https://<API_DOMAIN>`**. Open `https://<API_DOMAIN>/health` on your **phone using mobile data**
   (a different network) `[SHOT s7_health_phone]`.
5. Postman: Import `deploy/VisionGuard.postman_collection.json`, set `base_url` and `api_key`, pick files
   in the `image` fields, run each request (or *Run collection*). Shots: `s7_postman_health`,
   `s7_postman_ai`, `s7_postman_real`, `s7_postman_401`, `s7_postman_415`, and the Postman Console
   log `[SHOT s7_postman_console]`.
6. Also: `python scripts/send_samples.py --target api --url https://<API_DOMAIN> --n 6` `[SHOT s7_send_samples]`.

> No Azure access? Same compose file works on any Ubuntu VM (GCP e2-medium, AWS t3.medium, Oracle Ampere A1).
> Only the firewall screens differ.

Stop the VM after grading to keep credit (**Stop**, not delete, keeps the IP if you set it to static).

## 8. Flask ↔ cloud API — Task 8

```bash
# laptop .env:
#   INFERENCE_MODE=remote
#   MODEL_API_URL=https://<API_DOMAIN>
#   MODEL_API_KEY=<the key>
#   WITH_TF=false            (optional: small web image, no TensorFlow needed locally)
docker compose up -d --build web
curl http://localhost:5000/health        # inference_mode: remote, model_backend ok   [SHOT s8_health_remote]
python scripts/send_samples.py --target flask --url http://localhost:5000 --n 8      # [SHOT s8_send_samples]
```

Then in the browser upload 3 different images → `s8_result_1`, `s8_result_2`, `s8_result_3`; open
History → `s8_history` (rows with mode = remote); on the VM `docker compose -f deploy/docker-compose.cloud.yml logs --tail 20 api`
shows the same requests arriving `[SHOT s8_vm_logs]`.

## 9. Report and submission

1. Fill the 4 links + commit hash in `report/main.tex` (top, `\newcommand` block). Get the hash with `git rev-parse --short HEAD`.
2. Read every chapter; anything in a yellow box `\yours{...}` is a spot where you add your own observation.
3. Compile: upload the `report/` folder to Overleaf (Compiler: pdfLaTeX) or run `latexmk -pdf main.tex`.
4. Rename the PDF to **`23L-0764_VisionGuard.pdf`**.
5. Submit: PDF, GitHub link, DagsHub link, API URL.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dvc push` 401/403 | token in `.dvc/config.local` wrong; rerun the two `--local` commands |
| MLflow `RestException` on `log_model` | make sure `mlflow==2.22.5` (DagsHub speaks the 2.x API) |
| `alias API failed` | expected on some DagsHub versions; fallback tag `alias=champion` is used |
| web container `unhealthy`, logs `Model file not found` | `models/champion.keras` missing; run `scripts/export_champion.py` |
| `password authentication failed` | `.env` password changed after the volume was created; `docker compose down -v` resets the DB (deletes history) |
| Caddy cannot get a certificate | ports 80/443 closed in the NSG or ufw; or API_DOMAIN doesn't match the VM IP |
| Prediction hangs forever under gunicorn | never use `--preload` with TensorFlow (fork after TF init deadlocks) |
