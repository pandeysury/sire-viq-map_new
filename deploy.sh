#!/bin/bash

echo "🚢 Deploying VIQ RAG System..."

# Stop and remove existing container if it exists
echo "Stopping existing container..."
docker-compose down 2>/dev/null || true

# Build and start the new container
echo "Building and starting new container..."
docker-compose up -d --build

# Wait for container to be healthy
echo "Waiting for container to be ready..."
sleep 10

# Check if container is running
if docker ps | grep -q "viq-rag-system-new"; then
    echo "✅ VIQ RAG System deployed successfully!"
    echo "🌐 Access the API at: http://localhost:8003"
    echo "📚 API Documentation: http://localhost:8003/docs"
    echo "❤️  Health Check: http://localhost:8003/health"
    
    # Show container status
    echo ""
    echo "Container Status:"
    docker ps | grep viq-rag-system-new
else
    echo "❌ Deployment failed!"
    echo "Container logs:"
    docker-compose logs
fi