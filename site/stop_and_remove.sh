#!/bin/bash

# Define the Docker container name
CONTAINER_NAME="mkdocs_container"

# Check if the container exists (including stopped containers)
CONTAINER_ID=$(docker ps -a -q -f name=$CONTAINER_NAME)

if [ -n "$CONTAINER_ID" ]; then
    # Check if the container is running
    if docker ps -q -f id=$CONTAINER_ID > /dev/null; then
        echo "Stopping container $CONTAINER_NAME..."
        if docker stop $CONTAINER_ID > /dev/null; then
            echo "Stopped container $CONTAINER_NAME."
        else
            echo "Error stopping container $CONTAINER_NAME."
            exit 1
        fi
    fi

    echo "Removing container $CONTAINER_NAME..."
    if docker rm $CONTAINER_ID > /dev/null; then
        echo "Removed container $CONTAINER_NAME."
    else
        echo "Error removing container $CONTAINER_NAME."
        exit 1
    fi
else
    echo "Container $CONTAINER_NAME does not exist."
fi
