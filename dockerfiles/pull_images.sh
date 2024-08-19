#!/bin/bash

# ==============================================================================
# Pull Docker Images from GitHub Container Registry
# ==============================================================================
#
# This script pulls Docker images from GitHub Container Registry based on the
# Dockerfiles located in the specified directory. Each Dockerfile's extension
# is used to determine the corresponding image name.
#
# Requirements:
# - Docker must be installed and running.
# - Access to GitHub Container Registry with appropriate permissions.
#
# Usage:
# Run the script: ./pull_images.sh
#
# Author: Bryce Shirley
# Date: 19.08.2024
# ==============================================================================

# Set the GitHub Container Registry base URL and repository
REGISTRY_URL="ghcr.io"
REPOSITORY="bryceshirley/iris-gpubench"

# Directory containing Dockerfiles
DOCKERFILE_DIR="dockerfiles/app_images"

# Find all Dockerfiles in the directory
DOCKERFILES=$(ls ${DOCKERFILE_DIR}/Dockerfile.*)

echo "Pulling images from GitHub Container Registry..."

# Loop through each Dockerfile, extract image names, and pull the images
for DOCKERFILE in $DOCKERFILES; do
  # Extract the image name from the Dockerfile extension
  IMAGE_NAME=$(basename $DOCKERFILE | sed 's/Dockerfile\.//')

  # Construct the full image tag
  IMAGE_TAG="${REGISTRY_URL}/${REPOSITORY}/${IMAGE_NAME}:latest"

  echo "Pulling image: ${IMAGE_TAG}..."
  docker pull ${IMAGE_TAG}
done

echo -e "Pull process completed.\n"