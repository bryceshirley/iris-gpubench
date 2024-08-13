#!/bin/bash

# Name of the Docker container
CONTAINER_NAME="mkdocs_container"

# Stop the Docker container if it is running
if docker ps -q -f name=$CONTAINER_NAME; then
    echo "Stopping container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME is not running."
fi

# Remove the Docker container
if docker ps -a -q -f name=$CONTAINER_NAME; then
    echo "Removing container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME does not exist."
fi

