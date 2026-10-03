#!/usr/bin/env bash
# ==============================================================================
# Script: update_kartik_worksheets.sh
# Purpose: Automatically force-sync and update Kartik Gupta's DL interactive
#          worksheets and competition materials from upstream, strip .git,
#          and run the collection locally via an interactive web server.
# Source:  https://github.com/kartikgupta98/dl-worksheets
# ==============================================================================

set -euo pipefail

# Determine repository root and target directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/03_HTML_Visualizations/Kartik_Gupta_Worksheets"
WORKSHEETS_DIR="${ROOT_DIR}/02_Worksheets"
HTML_VIS_DIR="${ROOT_DIR}/03_HTML_Visualizations"
COMPETITION_MAT_DIR="${ROOT_DIR}/04_Notebooks/Kaggle_Competition_01_Flagged_or_Fraud_14_Sep_2026/Materials"
UPSTREAM_CACHE_DIR="/tmp/kartik_dl_worksheets_upstream"
REPO_URL="https://github.com/kartikgupta98/dl-worksheets.git"
PID_FILE="/tmp/kartik_worksheets_server.pid"

# Default configuration
RUN_LOCAL=true
RUN_BACKGROUND=false
OPEN_BROWSER=true
PORT=8080
URL_PATH=""

# Helper: Find an available TCP port starting from the given port
find_available_port() {
    local port="${1:-8080}"
    while lsof -i ":${port}" >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo "${port}"
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cnn)
            URL_PATH="worksheet09_cnn.html"
            shift
            ;;
        --arch)
            URL_PATH="worksheet10_architectures.html"
            shift
            ;;
        --resnet)
            URL_PATH="worksheet11_inception_resnet.html"
            shift
            ;;
        --port|-p)
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                PORT="$2"
                shift 2
            else
                echo "Error: --port requires a port number argument." >&2
                exit 1
            fi
            ;;
        --bg|--background)
            RUN_BACKGROUND=true
            shift
            ;;
        --no-run|--update-only)
            RUN_LOCAL=false
            shift
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --stop)
            echo "==> Checking for active Kartik worksheets server..."
            STOPPED=false
            if [ -f "${PID_FILE}" ]; then
                OLD_PID=$(cat "${PID_FILE}")
                if kill -0 "${OLD_PID}" 2>/dev/null; then
                    echo "==> Stopping background server (PID ${OLD_PID})..."
                    kill "${OLD_PID}" 2>/dev/null || true
                    rm -f "${PID_FILE}"
                    STOPPED=true
                fi
            fi
            RUNNING_PIDS=$(pgrep -f "http.server.*${TARGET_DIR}" || true)
            if [ -n "${RUNNING_PIDS}" ]; then
                echo "==> Terminating server process(es): ${RUNNING_PIDS}"
                echo "${RUNNING_PIDS}" | xargs kill 2>/dev/null || true
                STOPPED=true
            fi
            if [ "${STOPPED}" = true ]; then
                echo "==> Kartik worksheets server stopped successfully."
            else
                echo "==> No active Kartik worksheets server found running."
            fi
            exit 0
            ;;
        --help|-h)
            echo "=================================================================="
            echo "    Kartik Gupta DL Interactive Worksheets Manager                "
            echo "=================================================================="
            echo "Usage: ./scripts/update_kartik_worksheets.sh [options]"
            echo ""
            echo "Options:"
            echo "  --cnn               Open Worksheet 09: CNN (Pixels to Patterns)"
            echo "  --arch              Open Worksheet 10: Architectures (LeNet, AlexNet, VGG-16)"
            echo "  --resnet            Open Worksheet 11: Inception & ResNet"
            echo "  --port, -p <PORT>   Specify custom port (default: 8080 or next free port)"
            echo "  --bg, --background  Run server in the background (detached)"
            echo "  --stop              Stop any currently running background server"
            echo "  --update-only       Only sync and strip .git, do not start local server"
            echo "  --no-browser        Start server without automatically opening browser"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help to see available options." >&2
            exit 1
            ;;
    esac
done

echo "=================================================================="
echo "    Kartik Gupta DL Interactive Worksheets Manager                "
echo "=================================================================="
echo "Target directory: ${TARGET_DIR}"
echo "Upstream repo:    ${REPO_URL}"
echo ""

# STEP 1: Fetch upstream repo
echo "==> Step 1/4: Fetching latest version from upstream..."
if [ -d "${UPSTREAM_CACHE_DIR}/.git" ]; then
    echo "    Reusing upstream cache at ${UPSTREAM_CACHE_DIR}..."
    cd "${UPSTREAM_CACHE_DIR}"
    git remote set-url origin "${REPO_URL}" 2>/dev/null || git remote add origin "${REPO_URL}"
    git fetch origin main --quiet
    git reset --hard origin/main --quiet
    git clean -fd --quiet
else
    echo "    Cloning fresh upstream repository to cache..."
    rm -rf "${UPSTREAM_CACHE_DIR}"
    git clone --depth 1 "${REPO_URL}" "${UPSTREAM_CACHE_DIR}" --quiet
fi

# Display upstream commit info
echo ""
echo "------------------------------------------------------------------"
echo "                      Latest Upstream Commit                      "
echo "------------------------------------------------------------------"
git -C "${UPSTREAM_CACHE_DIR}" log -1 --format="  Commit:  %h%n  Author:  %an <%ae>%n  Date:    %ad (%cr)%n  Message: %s" --date=short
echo "------------------------------------------------------------------"
echo ""

# STEP 2: Sync to target directory
echo "==> Step 2/4: Syncing files into local mirror directory..."
mkdir -p "${TARGET_DIR}"
rsync -a --delete --exclude='.git' "${UPSTREAM_CACHE_DIR}/" "${TARGET_DIR}/"

# STEP 3: Remove .git from target directory
echo "==> Step 3/4: Removing .git from target directory..."
rm -rf "${TARGET_DIR}/.git"
echo "    [OK] .git successfully removed. Folder is a clean static asset directory."

# STEP 4: Distribute updated worksheets into 02_Worksheets & 03_HTML_Visualizations
echo "==> Step 4/4: Organizing individual worksheets into course folders..."
mkdir -p "${WORKSHEETS_DIR}" "${HTML_VIS_DIR}" "${COMPETITION_MAT_DIR}"

# 02_Worksheets copies
[ -f "${TARGET_DIR}/worksheet09_regularisation.html" ] && cp "${TARGET_DIR}/worksheet09_regularisation.html" "${WORKSHEETS_DIR}/Worksheet_09_Regularisation_Interactive.html"
[ -f "${TARGET_DIR}/worksheet09_cnn.html" ] && cp "${TARGET_DIR}/worksheet09_cnn.html" "${WORKSHEETS_DIR}/Worksheet_09_CNN_From_Pixels_to_Patterns_Interactive.html"
[ -f "${TARGET_DIR}/worksheet10_architectures.html" ] && cp "${TARGET_DIR}/worksheet10_architectures.html" "${WORKSHEETS_DIR}/Worksheet_10_Inside_LeNet_AlexNet_VGG16_Interactive.html"
[ -f "${TARGET_DIR}/worksheet11_inception_resnet.html" ] && cp "${TARGET_DIR}/worksheet11_inception_resnet.html" "${WORKSHEETS_DIR}/Worksheet_11_Inception_and_ResNet_Interactive.html"
[ -f "${TARGET_DIR}/worksheet07_momentum_nag.html" ] && cp "${TARGET_DIR}/worksheet07_momentum_nag.html" "${WORKSHEETS_DIR}/Worksheet_07_Momentum_EWMA_and_NAG_Interactive.html"
[ -f "${TARGET_DIR}/worksheet08_adaptive_lr.html" ] && cp "${TARGET_DIR}/worksheet08_adaptive_lr.html" "${WORKSHEETS_DIR}/Worksheet_08_Adaptive_Learning_AdaGrad_RMSProp_and_Adam_Interactive.html"
[ -f "${TARGET_DIR}/backprop_worksheet_output_layer.html" ] && cp "${TARGET_DIR}/backprop_worksheet_output_layer.html" "${WORKSHEETS_DIR}/Worksheet_04_Backpropagation_Output_Layer_Interactive.html"

# 03_HTML_Visualizations copies
[ -f "${TARGET_DIR}/worksheet09_regularisation.html" ] && cp "${TARGET_DIR}/worksheet09_regularisation.html" "${HTML_VIS_DIR}/Module_07_Worksheet_09_Regularisation_Interactive.html"
[ -f "${TARGET_DIR}/worksheet09_cnn.html" ] && cp "${TARGET_DIR}/worksheet09_cnn.html" "${HTML_VIS_DIR}/Module_10_Worksheet_09_CNN_Pixels_to_Patterns_Interactive.html"
[ -f "${TARGET_DIR}/worksheet10_architectures.html" ] && cp "${TARGET_DIR}/worksheet10_architectures.html" "${HTML_VIS_DIR}/Module_11_Worksheet_10_Inside_LeNet_AlexNet_VGG16_Interactive.html"
[ -f "${TARGET_DIR}/worksheet11_inception_resnet.html" ] && cp "${TARGET_DIR}/worksheet11_inception_resnet.html" "${HTML_VIS_DIR}/Module_12_Worksheet_11_Inception_and_ResNet_Interactive.html"

# Competition materials copies
if [ -d "${TARGET_DIR}/fraud_competition/materials" ]; then
    rsync -a "${TARGET_DIR}/fraud_competition/materials/" "${COMPETITION_MAT_DIR}/"
fi
echo "    [OK] All worksheets and competition materials organized."
echo ""

# Local Server Launch
if [ "${RUN_LOCAL}" = true ]; then
    ACTUAL_PORT=$(find_available_port "${PORT}")
    BASE_URL="http://localhost:${ACTUAL_PORT}"
    FULL_URL="${BASE_URL}/${URL_PATH}"
    
    echo "=================================================================="
    echo "  🚀 Starting Kartik Gupta Worksheets Local Server                "
    echo "=================================================================="
    echo "  Local URL:        ${FULL_URL}"
    echo "  Root Hub:         ${BASE_URL}/"
    echo "  Serving from:     ${TARGET_DIR}"
    
    if [ "${RUN_BACKGROUND}" = true ]; then
        if [ -f "${PID_FILE}" ]; then
            OLD_PID=$(cat "${PID_FILE}")
            if kill -0 "${OLD_PID}" 2>/dev/null; then
                kill "${OLD_PID}" 2>/dev/null || true
            fi
            rm -f "${PID_FILE}"
        fi
        
        LOG_FILE="/tmp/kartik_worksheets_server.log"
        nohup python3 -m http.server "${ACTUAL_PORT}" --directory "${TARGET_DIR}" >"${LOG_FILE}" 2>&1 &
        SERVER_PID=$!
        echo "${SERVER_PID}" > "${PID_FILE}"
        sleep 1
        
        echo "  Server PID:       ${SERVER_PID} (Running in background)"
        echo "  Logs:             ${LOG_FILE}"
        echo "  Stop command:     ./scripts/update_kartik_worksheets.sh --stop"
        echo "=================================================================="
        
        if [ "${OPEN_BROWSER}" = true ]; then
            echo "==> Opening ${FULL_URL} in browser..."
            sleep 0.5
            if command -v open >/dev/null 2>&1; then
                open "${FULL_URL}"
            elif command -v xdg-open >/dev/null 2>&1; then
                xdg-open "${FULL_URL}"
            fi
        fi
    else
        echo "  Mode:             Foreground (Press Ctrl+C to stop)"
        echo "=================================================================="
        echo ""
        
        if [ "${OPEN_BROWSER}" = true ]; then
            (
                sleep 1
                if command -v open >/dev/null 2>&1; then
                    open "${FULL_URL}"
                elif command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "${FULL_URL}"
                fi
            ) &
        fi
        
        trap 'echo -e "\n\n==> Local worksheets server stopped. Goodbye!"; exit 0' INT TERM
        python3 -m http.server "${ACTUAL_PORT}" --directory "${TARGET_DIR}"
    fi
else
    echo "==> Update complete. Local server skipped (--no-run / --update-only specified)."
fi
