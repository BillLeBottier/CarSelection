#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="${SCRIPT_DIR}/cv-axa.html"
PDF_FILE="${SCRIPT_DIR}/Robin_Vassaux_CV_AXA_CAT_Analytics.pdf"

google-chrome \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --user-data-dir=/tmp/chrome-cv-axa \
  --virtual-time-budget=5000 \
  --print-to-pdf="${PDF_FILE}" \
  --print-to-pdf-no-header \
  --no-pdf-header-footer \
  "file://${HTML_FILE}"

echo "PDF généré : ${PDF_FILE}"
