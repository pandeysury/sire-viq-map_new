#!/bin/bash

# ====================================================================
# VIQ RAG System - Automated Git Pull and Redeploy Script
# ====================================================================

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/opt/sire-viq-map_new"
CONTAINER_NAME="viq-rag-system-new"
SERVICE_NAME="viq-rag-system"
HOST_PORT=8003
GIT_BRANCH="main"  # Change this to your branch name

echo -e "${BLUE}=========================================="
echo "🔄 VIQ RAG - Git Pull & Redeploy"
echo -e "==========================================${NC}"
echo ""

# ====================================================================
# Step 1: Navigate to Project Directory
# ====================================================================
echo -e "${YELLOW}Step 1: Navigating to project directory...${NC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Project directory $PROJECT_DIR does not exist${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}✓ In directory: $(pwd)${NC}"
echo ""

# ====================================================================
# Step 2: Check Git Status
# ====================================================================
echo -e "${YELLOW}Step 2: Checking Git status...${NC}"

# Show current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Show current commit
CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo "Current commit: $CURRENT_COMMIT"

# Check for local changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted local changes${NC}"
    echo "Stashing local changes..."
    git stash
fi

echo -e "${GREEN}✓ Git status checked${NC}"
echo ""

# ====================================================================
# Step 3: Pull Latest Changes
# ====================================================================
echo -e "${YELLOW}Step 3: Pulling latest changes from Git...${NC}"

# Fetch latest changes
git fetch origin

# Check if there are new changes
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/$GIT_BRANCH)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}✓ Already up to date. No new changes.${NC}"
    echo ""
    read -p "Continue with rebuild anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
else
    echo "New changes available:"
    git log HEAD..origin/$GIT_BRANCH --oneline | head -5
    echo ""
    
    # Pull changes
    git pull origin $GIT_BRANCH
    
    NEW_COMMIT=$(git rev-parse --short HEAD)
    echo -e "${GREEN}✓ Updated from $CURRENT_COMMIT to $NEW_COMMIT${NC}"
fi
echo ""

# ====================================================================
# Step 4: Backup Current State
# ====================================================================
echo -e "${YELLOW}Step 4: Creating backup...${NC}"

BACKUP_DIR="/tmp/viq-rag-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Save current logs
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "Saving current container logs..."
    docker logs "$CONTAINER_NAME" > "$BACKUP_DIR/logs-before-deploy.txt" 2>&1 || true
fi

# Save current image info
docker images | grep sire-viq-map_new > "$BACKUP_DIR/images-before-deploy.txt" 2>&1 || true

echo -e "${GREEN}✓ Backup saved to: $BACKUP_DIR${NC}"
echo ""

# ====================================================================
# Step 5: Stop Current Container
# ====================================================================
echo -e "${YELLOW}Step 5: Stopping current container...${NC}"

if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Container is running. Stopping..."
    docker-compose down
    echo -e "${GREEN}✓ Container stopped${NC}"
else
    echo "Container not running. Cleaning up..."
    docker-compose down 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup done${NC}"
fi
echo ""

# ====================================================================
# Step 6: Remove Old Docker Image
# ====================================================================
echo -e "${YELLOW}Step 6: Removing old Docker image...${NC}"

# Get old image ID
OLD_IMAGE_ID=$(docker images | grep sire-viq-map_new-viq-rag-system | awk '{print $3}' | head -1)

if [ ! -z "$OLD_IMAGE_ID" ]; then
    echo "Removing old image: $OLD_IMAGE_ID"
    docker rmi "$OLD_IMAGE_ID" --force 2>/dev/null || true
    echo -e "${GREEN}✓ Old image removed${NC}"
else
    echo "No old image found"
fi

# Clean up dangling images
echo "Cleaning up dangling images..."
docker image prune -f
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# ====================================================================
# Step 7: Build New Docker Image
# ====================================================================
echo -e "${YELLOW}Step 7: Building new Docker image...${NC}"

echo "This may take a few minutes..."
if docker-compose build --no-cache; then
    echo -e "${GREEN}✓ New image built successfully${NC}"
else
    echo -e "${RED}✗ Image build failed!${NC}"
    echo "Check build logs above for errors"
    exit 1
fi
echo ""

# ====================================================================
# Step 8: Start New Container
# ====================================================================
echo -e "${YELLOW}Step 8: Starting new container...${NC}"

if docker-compose up -d; then
    echo -e "${GREEN}✓ Container started${NC}"
else
    echo -e "${RED}✗ Container start failed!${NC}"
    exit 1
fi
echo ""

# ====================================================================
# Step 9: Wait for Container to be Healthy
# ====================================================================
echo -e "${YELLOW}Step 9: Waiting for container to be healthy...${NC}"

echo "Waiting 30 seconds for initialization..."
sleep 30

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}✗ Container is not running!${NC}"
    echo "Checking logs:"
    docker logs "$CONTAINER_NAME" --tail 50
    exit 1
fi

echo -e "${GREEN}✓ Container is running${NC}"
echo ""

# ====================================================================
# Step 10: Health Check
# ====================================================================
echo -e "${YELLOW}Step 10: Running health checks...${NC}"

MAX_RETRIES=12
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:$HOST_PORT/health &> /dev/null; then
        echo -e "${GREEN}✓ Health check passed!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}✗ Health check failed after $MAX_RETRIES attempts${NC}"
        echo ""
        echo "Container logs (last 50 lines):"
        docker logs "$CONTAINER_NAME" --tail 50
        exit 1
    fi
    
    echo "Health check attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 5
done
echo ""

# ====================================================================
# Step 11: Verify Deployment
# ====================================================================
echo -e "${YELLOW}Step 11: Verifying deployment...${NC}"

# Get new image info
NEW_IMAGE_ID=$(docker images | grep sire-viq-map_new-viq-rag-system | awk '{print $3}' | head -1)
NEW_CONTAINER_ID=$(docker ps | grep "$CONTAINER_NAME" | awk '{print $1}')

echo "New Image ID: $NEW_IMAGE_ID"
echo "New Container ID: $NEW_CONTAINER_ID"

# Check container status
CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
CONTAINER_HEALTH=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "no-healthcheck")

echo "Container Status: $CONTAINER_STATUS"
echo "Container Health: $CONTAINER_HEALTH"

echo -e "${GREEN}✓ Deployment verified${NC}"
echo ""

# ====================================================================
# Step 12: Test Endpoints
# ====================================================================
echo -e "${YELLOW}Step 12: Testing endpoints...${NC}"

echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:$HOST_PORT/health)
echo "Response: $HEALTH_RESPONSE"

echo ""
echo "Testing API docs..."
if curl -f http://localhost:$HOST_PORT/docs &> /dev/null; then
    echo -e "${GREEN}✓ API docs accessible${NC}"
else
    echo -e "${YELLOW}⚠️  API docs not accessible (might be normal)${NC}"
fi

echo ""

# ====================================================================
# Step 13: Show Recent Logs
# ====================================================================
echo -e "${YELLOW}Step 13: Showing recent logs...${NC}"
echo "----------------------------------------"
docker logs "$CONTAINER_NAME" --tail 20
echo "----------------------------------------"
echo ""

# ====================================================================
# Summary
# ====================================================================
echo -e "${BLUE}=========================================="
echo "✅ Deployment Successful!"
echo -e "==========================================${NC}"
echo ""
echo "📊 Deployment Summary:"
echo "  Old Commit: $CURRENT_COMMIT"
echo "  New Commit: $(git rev-parse --short HEAD)"
echo "  Old Image: ${OLD_IMAGE_ID:-none}"
echo "  New Image: $NEW_IMAGE_ID"
echo "  Container: $NEW_CONTAINER_ID"
echo "  Status: $CONTAINER_STATUS"
echo "  Health: $CONTAINER_HEALTH"
echo ""
echo "🌐 Access URLs:"
echo "  Frontend: http://devsmsragai.com/viq-rag/"
echo "  Health: http://13.250.51.71:$HOST_PORT/health"
echo "  API Docs: http://13.250.51.71:$HOST_PORT/docs"
echo ""
echo "📁 Backup Location: $BACKUP_DIR"
echo ""
echo "📝 Useful Commands:"
echo "  View logs: docker logs -f $CONTAINER_NAME"
echo "  Check status: docker ps | grep $CONTAINER_NAME"
echo "  Restart: docker-compose restart"
echo ""
echo -e "${GREEN}🎉 All done!${NC}"
echo ""
