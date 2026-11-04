#!/usr/bin/env bash
# Phase B follow-up (run in Colab after 02_train_colab.ipynb):  !git pull -q && bash scripts/colab_finish.sh
# Re-runs the two toy labs (fixed metric/dtype issue) and the baseline CNN (fixed early stopping),
# then recomputes comparison, registry, champion export, error analysis and the results zip.
set -uo pipefail
quiet() { grep -v ' step - \|^Epoch \|^$'; }
echo "== 3.1 MLflow toy (local, two learning rates)"
python labs/mlflow_toy.py --lr 0.001 --local 2>&1 | tail -3
python labs/mlflow_toy.py --lr 0.01 --local 2>&1 | tail -3
echo "== 3.2 MLflow toy -> DagsHub"
python labs/mlflow_toy.py --lr 0.001 2>&1 | tail -3
echo "== 2.7 toy tuner"
python labs/tuner_toy.py 2>&1 | quiet | tail -25
echo "== baseline CNN re-run"
python scripts/run_experiments.py --only baseline_cnn 2>&1 | grep -E '>>>|best epoch [0-9]+ \||TEST:|run id|Error'
echo "== compare / register / export"
python scripts/compare_runs.py
python scripts/register_model.py
python scripts/export_champion.py
BEST=$(python -c "import json;print(json.load(open('report/generated/best_run.json'))['run'])")
echo "== error analysis on $BEST"
python scripts/error_analysis.py --run-dir "outputs/$BEST" 2>&1 | grep -v Warning
echo "== 3.6 load by alias, 4.1 inference module"
AI=$(python -c "import pandas as pd;t=pd.read_csv('data/splits/test.csv');print(t[t.label==1].path.iloc[0])")
REAL=$(python -c "import pandas as pd;t=pd.read_csv('data/splits/test.csv');print(t[t.label==0].path.iloc[0])")
python scripts/load_champion.py "$AI" 2>&1 | grep -v WARNING | tail -8
python scripts/infer_cli.py "$AI" "$REAL" 2>&1 | grep -v WARNING | tail -4
rm -f /content/phaseB_results.zip
zip -qr /content/phaseB_results.zip report/generated report/figures/auto models/champion.keras \
  models/champion.json outputs/run_comparison.csv outputs/tuner
ls -la /content/phaseB_results.zip
