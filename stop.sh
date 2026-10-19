

#!/bin/bash

# Stop all processes started by start.sh + any lingering crawl browsers
# Usage: ./stop.sh

echo "Stopping servers..."
pkill -f "uvicorn api:app" 2>/dev/null
pkill -f "next dev" 2>/dev/null
pkill -f "python.*main.py" 2>/dev/null
pkill -f "python.*main_v2.py" 2>/dev/null

# Kill anything still holding ports 8000/3000
lsof -ti:8000 | xargs kill 2>/dev/null
lsof -ti:3000 | xargs kill 2>/dev/null

echo "Stopping crawl browsers..."
pkill -f "playwright" 2>/dev/null
pkill -f "chromium" 2>/dev/null

echo "All stopped."
