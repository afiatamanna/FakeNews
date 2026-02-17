#!/usr/bin/env bash
# Full pipeline: data load, train classical + LSTM + BERT, evaluate, explain.
# Run from project root. Uses .venv if present; otherwise python3.

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [ -d "$PROJECT_ROOT/.venv" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

echo "=== Training classical models (LR, SVM, RF) ==="
"$PYTHON" src/train_classical.py

echo "=== Evaluating classical models ==="
"$PYTHON" src/evaluate.py

echo "=== Training LSTM ==="
"$PYTHON" src/train_lstm.py

echo "=== Training BERT ==="
"$PYTHON" src/train_bert.py

echo "=== Explainability (SHAP + LIME) ==="
"$PYTHON" src/explain.py

echo "=== Done. Results in outputs/results.csv and outputs/figures/ ==="
