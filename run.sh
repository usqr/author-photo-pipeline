#!/bin/bash
set -e
cd "$(dirname "$0")"

# Kill any existing server on port 8787
lsof -ti:8787 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

# Start API server in background
python3 server.py &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
sleep 1

# Open compare page (before pipeline starts — progress bar will show)
open "http://localhost:8787/compare.html"

# Run pipeline
echo "Running pipeline..."
python3 rainbow_convert.py "$@"

# Archive to timestamped folder
RUN_NAME=$(date +%y%m%d_%H%M)
mkdir -p "$RUN_NAME"
cp -r step1_upscaled step2_nobg step3_bw step4_rainbow "$RUN_NAME/"
echo "Archived to $RUN_NAME/"

# Notify completion (macOS notification)
osascript -e 'display notification "Pipeline complete. Refresh compare page." with title "Author Photo Pipeline"' 2>/dev/null || true

echo ""
echo "=== Pipeline complete ==="
echo "Compare page: http://localhost:8787/compare.html"
echo "Server PID: $SERVER_PID — use 'End All' button in browser to stop"
