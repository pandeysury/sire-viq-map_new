#!/bin/bash

# VIQ RAG System Production Deployment Script
set -e

echo "🚢 Starting VIQ RAG System Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   exit 1
fi

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.production .env
    print_warning "Please edit .env file with your OpenAI API key before continuing."
    read -p "Press Enter after updating .env file..."
fi

# Validate OpenAI API key
if grep -q "sk-proj-your-production-key-here" .env; then
    print_error "Please update your OpenAI API key in .env file"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p backend/data/pdfs
mkdir -p backend/data/vectordb
mkdir -p backend/logs

# Set proper permissions
chmod 755 backend/data
chmod 755 backend/logs

# Stop existing container if running
print_status "Stopping existing container..."
docker-compose down || true

# Build and start the application
print_status "Building Docker image..."
docker-compose build --no-cache

print_status "Starting VIQ RAG System..."
docker-compose up -d

# Wait for container to be healthy
print_status "Waiting for container to be healthy..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose ps | grep -q "healthy"; then
        print_success "Container is healthy!"
        break
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    print_error "Container failed to become healthy within $timeout seconds"
    docker-compose logs
    exit 1
fi

# Test the API
print_status "Testing API endpoints..."
sleep 5

# Test health endpoint
if curl -f http://localhost:8003/health > /dev/null 2>&1; then
    print_success "Health endpoint is working"
else
    print_error "Health endpoint is not responding"
    docker-compose logs
    exit 1
fi

# Test stats endpoint
if curl -f http://localhost:8003/api/v1/stats > /dev/null 2>&1; then
    print_success "Stats endpoint is working"
else
    print_warning "Stats endpoint might not be ready yet"
fi

print_success "VIQ RAG System deployed successfully!"
print_status "Container is running on port 8003"
print_status "API Documentation: http://localhost:8003/docs"
print_status "Health Check: http://localhost:8003/health"

# Show container status
echo ""
print_status "Container Status:"
docker-compose ps

# Show logs
echo ""
print_status "Recent logs:"
docker-compose logs --tail=20

echo ""
print_success "Deployment completed! 🎉"
print_status "Next steps:"
echo "1. Update your nginx configuration"
echo "2. Add VIQ documents to backend/data/pdfs/"
echo "3. Test the system with sample queries"
echo "4. Monitor logs: docker-compose logs -f"