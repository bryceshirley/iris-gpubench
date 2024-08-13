#!/bin/bash

# Define image and container names
IMAGE_NAME="mkdocs_docs"
CONTAINER_NAME="mkdocs_container"

# Build the Docker image using the Dockerfile.build_docs
docker build -f Dockerfile.build_docs -t $IMAGE_NAME .

# Remove any existing container with the same name
if docker ps -a -q -f name=$CONTAINER_NAME; then
    echo "Removing existing container $CONTAINER_NAME..."
    docker rm -f $CONTAINER_NAME
fi

# Run the Docker container
docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME

# Output the website URL
echo "Documentation is being served at http://localhost:8000"

