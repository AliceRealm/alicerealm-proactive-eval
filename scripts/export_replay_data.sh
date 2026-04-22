#!/usr/bin/env bash
# Export EigenFlux production data for offline replay experiments.
#
# Uses COPY TO STDOUT (CSV) for fast streaming export.
# Output: data/replay/{replay_logs,processed_items,feedback_logs}.csv
#
# Usage:
#   scripts/export_replay_data.sh              # export all data
#   scripts/export_replay_data.sh --since 7    # last 7 days only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DEVOPS_DIR="${DEVOPS_DIR:-$HOME/Desktop/jarvis/repos/eigen-devops}"
PROD_PSQL="$DEVOPS_DIR/scripts/prod-psql"
OUT_DIR="$REPO_DIR/data/replay"

SINCE_DAYS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --since) SINCE_DAYS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

TIME_FILTER=""
FEEDBACK_FILTER=""
if [[ -n "$SINCE_DAYS" ]]; then
  SINCE_MS=$(python3 -c "import time; print(int((time.time() - $SINCE_DAYS * 86400) * 1000))")
  TIME_FILTER="WHERE served_at >= $SINCE_MS"
  FEEDBACK_FILTER="WHERE feedback_at >= $SINCE_MS"
  echo "Exporting last $SINCE_DAYS days"
else
  echo "Exporting all data"
fi

mkdir -p "$OUT_DIR"

echo ""
echo "=== Step 1/3: replay_logs ==="
"$PROD_PSQL" -c "
COPY (
  SELECT id, impression_id, agent_id, item_id,
         agent_features::text, item_features::text,
         item_score, position, served_at, created_at
  FROM replay_logs $TIME_FILTER
  ORDER BY served_at, impression_id, position
) TO STDOUT WITH (FORMAT csv, HEADER true);
" > "$OUT_DIR/replay_logs.csv"
REPLAY_COUNT=$(($(wc -l < "$OUT_DIR/replay_logs.csv") - 1))
echo "  → $REPLAY_COUNT rows"

echo ""
echo "=== Step 2/3: processed_items ==="
"$PROD_PSQL" -c "
COPY (
  SELECT pi.item_id, pi.summary, pi.broadcast_type, pi.domains,
         pi.keywords, pi.expire_time, pi.geo, pi.source_type,
         pi.expected_response, pi.group_id, pi.quality_score,
         pi.lang, pi.timeliness, pi.updated_at,
         ri.raw_content, ri.raw_url, ri.created_at
  FROM processed_items pi
  JOIN raw_items ri ON ri.item_id = pi.item_id
  WHERE pi.status = 3
    AND pi.item_id IN (SELECT DISTINCT item_id FROM replay_logs $TIME_FILTER)
  ORDER BY pi.item_id
) TO STDOUT WITH (FORMAT csv, HEADER true);
" > "$OUT_DIR/processed_items.csv"
ITEMS_COUNT=$(($(wc -l < "$OUT_DIR/processed_items.csv") - 1))
echo "  → $ITEMS_COUNT items"

echo ""
echo "=== Step 3/3: feedback_logs ==="
"$PROD_PSQL" -c "
COPY (
  SELECT id, impression_id, agent_id, item_id,
         score, feedback_at, created_at
  FROM feedback_logs $FEEDBACK_FILTER
  ORDER BY feedback_at
) TO STDOUT WITH (FORMAT csv, HEADER true);
" > "$OUT_DIR/feedback_logs.csv"
FEEDBACK_COUNT=$(($(wc -l < "$OUT_DIR/feedback_logs.csv") - 1))
echo "  → $FEEDBACK_COUNT rows"

echo ""
echo "=== Export complete ==="
echo "  replay_logs:     $REPLAY_COUNT rows"
echo "  processed_items: $ITEMS_COUNT items"
echo "  feedback_logs:   $FEEDBACK_COUNT rows"
echo "  Output dir:      $OUT_DIR/"
du -sh "$OUT_DIR/"
