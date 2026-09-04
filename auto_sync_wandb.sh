#!/usr/bin/env bash
# Background loop to auto-sync runs to W&B every 5 minutes
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INTERVAL="${SYNC_INTERVAL:-300}" # default 300s = 5 minutes

echo "🔄 Starting W&B Auto-Sync Daemon (interval: ${INTERVAL}s)..."
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running sync..."
    bash "$ROOT_DIR/sync_wandb.sh"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for ${INTERVAL}s..."
    sleep "$INTERVAL"
done
