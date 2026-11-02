#!/bin/bash

# VIQ AI System Startup Script

echo "🚢 Starting VIQ AI Matching System..."

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
echo "🚀 Starting backend server..."
python run.py