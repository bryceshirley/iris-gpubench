#!/bin/bash

# ==============================================================================
# Build Docker Images Locally
# ==============================================================================
#
# This script builds Docker images from Dockerfiles located in the specified
# directories. Each Dockerfile's extension is used to determine the corresponding
# image name. Base images are built first, followed by app images.
#
# Requirements:
# - Docker must be installed and running.
#
# Usage:
# Run the script: ./build_images.sh
#
# Author: Bryce Shirley
# Date: 19.08.2024
# ==============================================================================

# Directory containing Dockerfiles
BASE_IMAGES_DIR="dockerfiles/base_images"
APP_IMAGES_DIR="dockerfiles/app_images"

# Build base images first
echo "Building base images..."
for DOCKERFILE in ${BASE_IMAGES_DIR}/Dockerfile.*; do
  # Extract the image name from the Dockerfile extension
  IMAGE_NAME=$(basename $DOCKERFILE | sed 's/Dockerfile\.//')

  # Construct the full image tag for base images
  IMAGE_TAG="${IMAGE_NAME}_base:latest"

  echo "Building base image: ${IMAGE_TAG}..."
  docker build -f $DOCKERFILE -t ${IMAGE_TAG} .
done

# Build app images
echo "Building app images..."
for DOCKERFILE in ${APP_IMAGES_DIR}/Dockerfile.*; do
  # Extract the image name from the Dockerfile extension
  IMAGE_NAME=$(basename $DOCKERFILE | sed 's/Dockerfile\.//')

  # Construct the full image tag for app images
  IMAGE_TAG="${IMAGE_NAME}_app:latest"

  echo "Building app image: ${IMAGE_TAG}..."
  docker build -f $DOCKERFILE -t ${IMAGE_TAG} .
done

echo -e "Build process completed.\n"
