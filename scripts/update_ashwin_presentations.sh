#!/usr/bin/env bash
# ==============================================================================
# Script: update_ashwin_presentations.sh
# Purpose: Automatically force-sync and update Ashwin Tewary's DL lecture
#          presentations and practice sheets from upstream, strip .git so
#          the folder remains a clean static project, and run the project
#          locally with an interactive web server.
# Source:  https://github.com/ashwin-tewary/dl-worksheets
# ==============================================================================

set -euo pipefail

# Determine repository root and target directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/03_HTML_Visualizations/Ashwin_Tewary_Presentations"
UPSTREAM_CACHE_DIR="/tmp/ashwin_dl_worksheets_upstream"
REPO_URL="https://github.com/ashwin-tewary/dl-worksheets.git"
PID_FILE="/tmp/ashwin_presentations_server.pid"

# Default configuration
RUN_LOCAL=true
RUN_BACKGROUND=false
OPEN_BROWSER=true
PORT=8000
URL_PATH=""

# Helper: Find an available TCP port starting from the given port
find_available_port() {
    local port="${1:-8000}"
    while lsof -i ":${port}" >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo "${port}"
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cnn|-c)
            URL_PATH="presentations/cnn/index.html#field"
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
            echo "==> Checking for active presentation servers..."
            STOPPED=false
            if [ -f "${PID_FILE}" ]; then
                OLD_PID=$(cat "${PID_FILE}")
                if kill -0 "${OLD_PID}" 2>/dev/null; then
                    echo "==> Stopping background presentation server (PID ${OLD_PID})..."
                    kill "${OLD_PID}" 2>/dev/null || true
                    rm -f "${PID_FILE}"
                    STOPPED=true
                fi
            fi
            # Check for any lingering server process serving TARGET_DIR
            RUNNING_PIDS=$(pgrep -f "http.server.*${TARGET_DIR}" || true)
            if [ -n "${RUNNING_PIDS}" ]; then
                echo "==> Terminating server process(es): ${RUNNING_PIDS}"
                echo "${RUNNING_PIDS}" | xargs kill 2>/dev/null || true
                STOPPED=true
            fi
            if [ "${STOPPED}" = true ]; then
                echo "==> Presentation server stopped successfully."
            else
                echo "==> No active presentation server found running."
            fi
            exit 0
            ;;
        --help|-h)
            echo "=================================================================="
            echo "    Ashwin Tewary DL Presentations & Worksheets Manager           "
            echo "=================================================================="
            echo "Usage: ./scripts/update_ashwin_presentations.sh [options]"
            echo ""
            echo "Default behavior:"
            echo "  Updates the project from GitHub, strips .git to keep workspace clean,"
            echo "  and launches a local HTTP server opening the presentations hub in browser."
            echo ""
            echo "Options:"
            echo "  --cnn, -c           Launch directly into the CNN & Receptive Field presentation"
            echo "  --port, -p <PORT>   Specify custom port (default: 8000 or next free port)"
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
echo "    Ashwin Tewary DL Presentations & Worksheets Manager           "
echo "=================================================================="
echo "Target directory: ${TARGET_DIR}"
echo "Upstream repo:    ${REPO_URL}"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Fetch latest changes from upstream repository
# ------------------------------------------------------------------------------
echo "==> Step 1/3: Fetching latest version from upstream..."
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

# Display upstream commit information
echo ""
echo "------------------------------------------------------------------"
echo "                      Latest Upstream Commit                      "
echo "------------------------------------------------------------------"
git -C "${UPSTREAM_CACHE_DIR}" log -1 --format="  Commit:  %h%n  Author:  %an <%ae>%n  Date:    %ad (%cr)%n  Message: %s" --date=short
echo "------------------------------------------------------------------"
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Sync files to target project directory
# ------------------------------------------------------------------------------
echo "==> Step 2/3: Syncing files into local project directory..."
mkdir -p "${TARGET_DIR}"
rsync -a --delete --exclude='.git' "${UPSTREAM_CACHE_DIR}/" "${TARGET_DIR}/"

# ------------------------------------------------------------------------------
# STEP 3: Remove .git from target project directory
# ------------------------------------------------------------------------------
echo "==> Step 3/3: Removing .git from target project directory..."
rm -rf "${TARGET_DIR}/.git"

if [ ! -d "${TARGET_DIR}/.git" ]; then
    echo "    [OK] .git successfully removed. Folder is a clean static asset directory."
else
    echo "    [WARNING] Could not remove .git directory." >&2
fi
echo ""

# List available presentations
echo "Available Interactive Presentations:"
if [ -d "${TARGET_DIR}/presentations" ]; then
    for p in "${TARGET_DIR}"/presentations/*; do
        if [ -d "$p" ] && [ -f "$p/index.html" ]; then
            p_name="$(basename "$p")"
            echo "  * [${p_name}] -> presentations/${p_name}/index.html"
        fi
    done
fi
echo ""
echo "Main Presentation Hub: index.html"
echo ""

# ------------------------------------------------------------------------------
# STEP 4: Run project locally if enabled
# ------------------------------------------------------------------------------
if [ "${RUN_LOCAL}" = true ]; then
    # Check if a server is already running on this port
    ACTUAL_PORT=$(find_available_port "${PORT}")
    if [ "${ACTUAL_PORT}" != "${PORT}" ]; then
        echo "==> Port ${PORT} is in use; automatically using free port ${ACTUAL_PORT}."
    fi
    
    BASE_URL="http://localhost:${ACTUAL_PORT}"
    FULL_URL="${BASE_URL}/${URL_PATH}"
    
    echo "=================================================================="
    echo "  🚀 Starting Local Presentation Server                          "
    echo "=================================================================="
    echo "  Local URL:        ${FULL_URL}"
    echo "  Root Hub:         ${BASE_URL}/"
    echo "  Serving from:     ${TARGET_DIR}"
    
    if [ "${RUN_BACKGROUND}" = true ]; then
        # Terminate any previously recorded background server
        if [ -f "${PID_FILE}" ]; then
            OLD_PID=$(cat "${PID_FILE}")
            if kill -0 "${OLD_PID}" 2>/dev/null; then
                kill "${OLD_PID}" 2>/dev/null || true
            fi
            rm -f "${PID_FILE}"
        fi
        
        LOG_FILE="/tmp/ashwin_presentations_server.log"
        nohup python3 -m http.server "${ACTUAL_PORT}" --directory "${TARGET_DIR}" >"${LOG_FILE}" 2>&1 &
        SERVER_PID=$!
        echo "${SERVER_PID}" > "${PID_FILE}"
        sleep 1
        
        echo "  Server PID:       ${SERVER_PID} (Running in background)"
        echo "  Logs:             ${LOG_FILE}"
        echo "  Stop command:     ./scripts/update_ashwin_presentations.sh --stop"
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
        
        # Schedule browser opening after server binds
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
        
        # Handle graceful shutdown on Ctrl+C
        trap 'echo -e "\n\n==> Local presentation server stopped. Goodbye!"; exit 0' INT TERM
        
        python3 -m http.server "${ACTUAL_PORT}" --directory "${TARGET_DIR}"
    fi
else
    echo "==> Update complete. Local server skipped (--no-run / --update-only specified)."
fi
