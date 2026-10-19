#!/bin/bash

# Start both backend and frontend
# Usage: ./start.sh

trap "kill 0" EXIT

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate thesis || { echo "conda env 'thesis' not found"; exit 1; }

# Free ports from any stale run before starting (prevents "Address already in use")
for port in 8000 3000; do
  pids=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
  [ -n "$pids" ] && echo "Freeing port $port (killing $pids)..." && kill $pids 2>/dev/null
done

echo "Starting backend on :8000..."
(cd backend/phase1_job_site_discovery && python -m uvicorn api:app --reload --reload-dir . --reload-dir ../phase5_curriculum_alignment --reload-dir ../archive/phase5_legacy --port 8000) &

echo "Starting frontend on :3000..."
# Use nvm's Node 20 when available; otherwise fall back to the Node on PATH
(export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && nvm use 20; cd frontend && npm run dev) &

wait
