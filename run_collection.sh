#!/bin/bash
# Cron wrapper for search collection.
#
# To install cron (every 5 min):
#   crontab -e
#   */5 * * * * /path/to/proactive-eval/run_collection.sh
#
# To stop after expiry: remove the cron entry, or the script auto-expires.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/data/logs"
EXPIRE_FILE="$SCRIPT_DIR/data/.expire_at"

mkdir -p "$LOG_DIR"

# Auto-expire: stop running after the expiry timestamp
if [ -f "$EXPIRE_FILE" ]; then
    expire_at=$(cat "$EXPIRE_FILE")
    now=$(date +%s)
    if [ "$now" -gt "$expire_at" ]; then
        echo "$(date -Iseconds) Collection expired. Removing cron entry." >> "$LOG_DIR/cron.log"
        crontab -l 2>/dev/null | grep -v "run_collection.sh" | crontab -
        exit 0
    fi
fi

# Run the collector
echo "$(date -Iseconds) Starting collection cycle" >> "$LOG_DIR/cron.log"
/opt/homebrew/bin/python3 -m proactive_eval collect --source search --once \
    >> "$LOG_DIR/collection_$(date +%Y%m%d).log" 2>&1
echo "$(date -Iseconds) Cycle complete" >> "$LOG_DIR/cron.log"
