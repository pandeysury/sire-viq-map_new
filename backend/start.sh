#!/bin/bash

# VIQ RAG System Startup Script

echo "🚀 Starting VIQ RAG System..."

# Check if we're in the correct directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please configure your OpenAI API key."
    echo "OPENAI_API_KEY=your_key_here" > .env
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/pdfs data/vectordb logs

# Start the application
echo "🎯 Starting VIQ RAG System..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""

python3 -m app.main