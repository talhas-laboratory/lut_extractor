#!/bin/bash

# Function to kill processes on ports
cleanup() {
    echo "Cleaning up..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:5173 | xargs kill -9 2>/dev/null
}

# Cleanup existing processes
cleanup

# Start Backend (API)
echo "Starting Backend..."
source venv/bin/activate
nohup python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend (Vite)
echo "Starting Frontend..."
cd frontend
nohup npm run dev -- --host > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "------------------------------------------------"
echo "HCGE Application Started"
echo "------------------------------------------------"
echo "Backend PID: $BACKEND_PID | Logs: backend.log"
echo "Frontend PID: $FRONTEND_PID | Logs: frontend/frontend.log"
echo ""
echo "Access the app at: http://localhost:5173"
echo "------------------------------------------------"
