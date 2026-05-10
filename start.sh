#!/bin/bash
echo "Starting InvestIQ..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "Frontend: open frontend/index.html in your browser"
echo "Press Ctrl+C to stop"
echo ""
source venv/bin/activate
uvicorn api.routes:app --host 0.0.0.0 --port 8000
