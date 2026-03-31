#!/bin/bash
set -e
cd "$(dirname "$0")"

# Kill any existing server on port 8787
lsof -ti:8787 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

# Start API server — nohup keeps it alive after this script exits
nohup python3 server.py > server.log 2>&1 &
SERVER_PID=$!
disown $SERVER_PID
echo "Server started (PID: $SERVER_PID)"
echo "$SERVER_PID" > .server.pid
sleep 1

# Open compare page (before pipeline starts — progress bar will show)
open "http://localhost:8787/compare.html"

# Run pipeline (archives to timestamped folder + updates run_info.json)
echo "Running pipeline..."
python3 rainbow_convert.py "$@"

# Notify completion (macOS notification)
osascript -e 'display notification "Pipeline complete. Refresh compare page." with title "Author Photo Pipeline"' 2>/dev/null || true

echo ""
echo "=== Pipeline complete ==="
echo "Compare page: http://localhost:8787/compare.html"
echo "Server running in background — use 'End All' button in browser to stop"
