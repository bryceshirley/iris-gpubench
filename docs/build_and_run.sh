#!/bin/bash

# Define Docker image and container names
IMAGE_NAME="mkdocs_docs"
CONTAINER_NAME="mkdocs_container"

# Check if the Docker image exists locally
echo "Building Docker image $IMAGE_NAME..."
docker build -f Dockerfile.build_docs -t $IMAGE_NAME .

# Run or restart the Docker container
CONTAINER_ID=$(docker ps -a -q -f name=$CONTAINER_NAME)

if [ -n "$CONTAINER_ID" ]; then
    if docker ps -q -f id=$CONTAINER_ID > /dev/null; then
        echo "Container $CONTAINER_NAME is already running."
    else
        echo "Starting stopped container $CONTAINER_NAME..."
        docker start $CONTAINER_ID
    fi
else
    echo "Starting new container $CONTAINER_NAME..."
    docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME
    if [ $? -ne 0 ]; then
        echo "Error starting Docker container $CONTAINER_NAME. Exiting."
        exit 1
    fi
fi

# Output the website URL
echo "Documentation is being served at http://localhost:8000"
