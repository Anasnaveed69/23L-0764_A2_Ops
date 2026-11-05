#!/usr/bin/env bash
# Phase C (fresh Colab runtime): rebuild the report assets that only existed in the expired runtime.
# All MLflow runs are already on DagsHub; this re-trains the two runs whose local files the report
# needs (baseline_cnn: report figures; mnv2_tuned: champion candidate for error analysis), re-runs
# the toy tuner, then compare -> register -> export -> error analysis -> results zip.
set -uo pipefail
quiet() { grep -v ' step - \|^Epoch \|^$'; }
echo "== 2.7 toy tuner"
python labs/tuner_toy.py 2>&1 | quiet | tail -25
echo "== baseline CNN (report figures) + tuned run"
python scripts/run_experiments.py --only baseline_cnn mnv2_tuned 2>&1 | grep -E '>>>|best epoch [0-9]+ \||TEST:|run id|Error|FAILED'
echo "== compare / register / export"
python scripts/compare_runs.py
python scripts/register_model.py
python scripts/export_champion.py
BEST=$(python -c "import json;print(json.load(open('report/generated/best_run.json'))['run'])")
if [ ! -d "outputs/$BEST" ]; then
  echo "== $BEST has no local outputs in this runtime; training it again for error analysis"
  python scripts/run_experiments.py --only "$BEST" 2>&1 | grep -E '>>>|best epoch [0-9]+ \||TEST:'
  python scripts/compare_runs.py && python scripts/register_model.py && python scripts/export_champion.py
  BEST=$(python -c "import json;print(json.load(open('report/generated/best_run.json'))['run'])")
fi
echo "== error analysis on $BEST"
python scripts/error_analysis.py --run-dir "outputs/$BEST" 2>&1 | grep -v Warning
echo "== 3.6 load by alias, 4.1 inference module"
AI=$(python -c "import pandas as pd;t=pd.read_csv('data/splits/test.csv');print(t[t.label==1].path.iloc[0])")
REAL=$(python -c "import pandas as pd;t=pd.read_csv('data/splits/test.csv');print(t[t.label==0].path.iloc[0])")
python scripts/load_champion.py "$AI" 2>&1 | grep -v WARNING | tail -8
python scripts/infer_cli.py "$AI" "$REAL" 2>&1 | grep -v WARNING | tail -4
rm -f /content/phaseC_results.zip
zip -qr /content/phaseC_results.zip report/generated report/figures/auto models/champion.keras \
  models/champion.json outputs/run_comparison.csv
ls -la /content/phaseC_results.zip
