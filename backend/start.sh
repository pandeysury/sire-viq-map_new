#!/bin/bash

# VIQ AI System Startup Script

echo "🚢 Starting VIQ AI Matching System..."

# Run from backend/ regardless of caller's cwd (venv, requirements.txt, .env live here)
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your OpenAI API key."
    echo "Example: OPENAI_API_KEY=your_api_key_here"
fi

# Start the backend server
# Run from the repo root (one level up) so the app's "./data" and "./logs"
# paths resolve to the top-level data/ and backend/logs/ folders; PYTHONPATH
# points back at backend/ so the `app` package still resolves.
echo "🚀 Starting backend server..."
cd ..
PYTHONPATH=backend python -m app.main