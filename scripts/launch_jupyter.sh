#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=========================================================="
echo "   Deep Learning Environment Setup & JupyterLab Launcher   "
echo "=========================================================="

# 1. Create .venv if not already created
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "==> Found existing .venv."
fi

# 2. Activate virtual environment
echo "==> Activating .venv..."
source .venv/bin/activate

# 3. Install requirements
echo "==> Installing / checking requirements from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

# 4. Register ipykernel for Jupyter notebooks
python -m ipykernel install --user --name dl-env --display-name "Python 3 (DL Course)" --quiet 2>/dev/null || true

# 5. Launch Jupyter Lab
echo "==> Launching JupyterLab..."
exec jupyter lab
