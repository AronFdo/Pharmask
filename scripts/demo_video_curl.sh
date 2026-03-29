#!/usr/bin/env bash
# Optional: curl /query for demo recording (same behaviour as UI).
# Usage: bash scripts/demo_video_curl.sh http://localhost:8000
# Requires: curl, python3 (for JSON escaping). jq optional for pretty output.

set -euo pipefail
BASE="${1:-http://localhost:8000}"

json_body() {
  python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$1"
}

query() {
  local name="$1"
  local q="$2"
  echo ""
  echo "=== ${name} ==="
  resp="$(curl -sS -X POST "${BASE}/query" \
    -H "Content-Type: application/json" \
    -d "$(json_body "$q")")"
  if command -v jq >/dev/null 2>&1; then
    echo "$resp" | jq '{classification: .classification.query_type, answer_preview: (.answer | .[0:400]), source_count: (.sources | length)}'
  else
    echo "$resp"
  fi
}

echo "Base URL: ${BASE}"
echo "Health:"
curl -sS "${BASE}/health"
echo ""

echo ""
echo "--- Positive (P1–P3) ---"
query "P1 SQL" "What is 911 Stress and Anxiousness indicated for according to the database?"
query "P2 TEXT" "What side effects and safety warnings are commonly discussed for prescription medicines in the retrieved literature?"
query "P3 HYBRID" "What is Good Mood Enhancer indicated for in the database, and what does the biomedical text add about mood, stress, or nervousness?"

echo ""
echo "--- Negative (N1–N3) ---"
query "N1 fake drug" "What is the recommended dose of COMPLETELY_FAKE_DRUG_XYZ for adults according to the database?"
query "N2 off-domain" "What is the weather in Paris tomorrow?"
query "N3 limitation" "What is the single best treatment across all diseases in the corpus?"

echo ""
echo "Done."
