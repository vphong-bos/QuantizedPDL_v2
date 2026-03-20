#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REQ_FILE="${SCRIPT_DIR}/requirements.txt"

WEIGHTS_DIR="${SCRIPT_DIR}/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/model_final_bd324a.pkl"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl"

echo "Installing Python requirements..."
python -m pip install -q -r "${REQ_FILE}"

echo "Installing Cityscapes scripts..."
python -m pip install -q git+https://github.com/mcordts/cityscapesScripts.git

echo "Preparing weights directory..."
mkdir -p "${WEIGHTS_DIR}"

if [ ! -f "${WEIGHTS_FILE}" ]; then
    echo "Downloading Panoptic-DeepLab weights..."
    curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_FILE}"
fi

echo "Setup complete."