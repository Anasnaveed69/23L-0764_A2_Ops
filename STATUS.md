# Where things stand (4 Nov 2026, 23:59) — remaining steps

## Done
- GitHub repo + DagsHub (connected), DVC toy repo `23L-0764_A2_Ops-dvc-toy`
- Phase A (Colab): Kaggle data, audit, EDA, splits, DVC push of the dataset
- Phase B (Colab): 7 runs + Keras Tuner + MLflow toy runs + registry (`champion`), error analysis, Grad-CAM
- Follow-up `scripts/colab_finish.sh` was running: toy tuner done; re-runs of baseline_cnn,
  effb0_default, mnv2_tuned, mnv2_autolog; then compare/register/export -> /content/phaseB_results.zip
- Azure VM `vm-23L-0764-A2-Ops` (Standard_B2s_v2, Central India, 20.219.10.43): Docker, ufw, swap, NSG 22/80/443
- Task 6.1 warm-up done; screenshots in report/figures/shots/

## Remaining
1. **Colab**: in the 02 notebook (or its Terminal) check `ls /content/phaseB_results.zip`; if missing,
   run `!git pull -q && bash scripts/colab_finish.sh`. Download it:
   `from google.colab import files; files.download('/content/phaseB_results.zip')`.
   Unzip into the repo root, commit (`report/generated`, `report/figures/auto`, `models/champion.*`), push.
2. **VM** (Azure Cloud Shell -> `ssh azureuser@20.219.10.43`; if the key is gone:
   `az vm user update -g rg-23L-0764-A2-Ops -n vm-23L-0764-A2-Ops -u azureuser --ssh-key-value ~/.ssh/id_rsa.pub`):
   `cd ~/23L-0764_A2_Ops && git pull && bash ~/boot.sh` -> API at https://20-219-10-43.sslip.io.
   API key: `cat ~/23L-0764_A2_Ops/deploy/.env`.
3. **Task 6 evidence** on the VM: `bash deploy/t6.sh up`, then `whoami`, `port`, `sizes`, `history`,
   `ops`, `break`, `fix`, `db` (screenshot each).
4. **Laptop (Tasks 4, 5, 8)**: clone the repo, then
   `powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -ApiUrl https://20-219-10-43.sslip.io -ApiKey <key>`
   -> http://127.0.0.1:5000 : upload 3+ images, wrong type / >5 MB / renamed .txt, history, /health.
5. **Postman**: import `deploy/VisionGuard.postman_collection.json`, set `base_url` + `api_key`, run.
6. **Report**: save screenshots as `report/figures/shots/<name>.png` (names in RUNBOOK.md / grey boxes),
   set `\CommitHash` in report/main.tex, write the yellow boxes, compile on Overleaf (pdfLaTeX),
   rename to `23L-0764_VisionGuard.pdf`.
7. After grading: Azure portal -> VM -> **Stop** to save credit.
