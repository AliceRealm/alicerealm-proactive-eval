#!/bin/bash
# Cron wrapper for baseline search collection.
# Runs collect_searches.py and logs output.
#
# To install cron (every 5 min):
#   crontab -e
#   */5 * * * * /Users/pascal/Desktop/jarvis/proactive-eval/run_collection.sh
#
# To stop after 2 days: remove the cron entry, or the script auto-expires.

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
        # Remove ourselves from crontab
        crontab -l 2>/dev/null | grep -v "run_collection.sh" | crontab -
        exit 0
    fi
fi

# Run the collector
echo "$(date -Iseconds) Starting collection cycle" >> "$LOG_DIR/cron.log"
/opt/homebrew/bin/python3 "$SCRIPT_DIR/collect_searches.py" >> "$LOG_DIR/collection_$(date +%Y%m%d).log" 2>&1
echo "$(date -Iseconds) Cycle complete" >> "$LOG_DIR/cron.log"
