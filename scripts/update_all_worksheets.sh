#!/usr/bin/env bash
# ==============================================================================
# Script: update_all_worksheets.sh
# Purpose: Master updater to sync, strip .git, and organize all instructor
#          materials from both:
#          1. Ashwin Tewary's Presentations (ashwin-tewary/dl-worksheets)
#          2. Kartik Gupta's Interactive Worksheets (kartikgupta98/dl-worksheets)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo "    Master Deep Learning Course Materials Updater                 "
echo "=================================================================="
echo ""

echo ">>> [1/2] Updating Ashwin Tewary Lecture Presentations..."
"${SCRIPT_DIR}/update_ashwin_presentations.sh" --update-only
echo ""

echo ">>> [2/2] Updating Kartik Gupta Interactive Worksheets..."
"${SCRIPT_DIR}/update_kartik_worksheets.sh" --update-only
echo ""

echo "=================================================================="
echo "  All course materials updated successfully!"
echo "  - 02_Worksheets/ has all latest interactive worksheets"
echo "  - 03_HTML_Visualizations/ has both Ashwin and Kartik mirrors"
echo "  - 04_Notebooks/ has latest Kaggle materials & solution notebooks"
echo "  - All .git directories stripped cleanly"
echo "=================================================================="
