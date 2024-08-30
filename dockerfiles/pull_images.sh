#!/bin/bash

# ==============================================================================
# Pull Docker Images from Harbor Container Registry
# ==============================================================================
#
# This script pulls Docker images from Harbor Container Registry based on 
# hardcoded lists of image names. It then retags the images to remove the 
# registry URL.
#
# Requirements:
# - Docker must be installed and running.
# - Access to Harbor Container Registry with appropriate permissions.
#
# Usage:
# Run the script: ./pull_images.sh
#
# Author: Bryce Shirley
# Date: 19.08.2024 (Updated on 30.08.2024)
# ==============================================================================

# Set the Harbor Container Registry base URL
REGISTRY_URL="harbor.stfc.ac.uk/stfc-cloud-staging/iris-bench"

# Hardcoded lists of images
BASE_IMAGES=("mantid_base" "sciml_base")
SCIML_IMAGES=("mnist_tf_keras" "stemdl_classification" "synthetic_regression")
MANTID_IMAGES=("mantid_run_1" "mantid_run_8")
OTHER_IMAGES=("dummy")

# Perform Docker login using environment variables
echo "Logging in to Harbor Container Registry..."
docker login harbor.stfc.ac.uk -u "$HARBOR_USERNAME" -p "$HARBOR_PASSWORD"

if [ $? -ne 0 ]; then
  echo "Docker login failed. Exiting..."
  exit 1
fi

echo "Pulling images from Harbor Container Registry..."

# Function to pull, retag, and remove original tag for an image
pull_and_retag() {
  IMAGE_NAME=$1
  IMAGE_TAG="${REGISTRY_URL}/${IMAGE_NAME}:latest"
  
  echo "Pulling image: ${IMAGE_TAG}..."
  docker pull ${IMAGE_TAG}
  
  echo "Retagging image to: ${IMAGE_NAME}:latest..."
  docker tag ${IMAGE_TAG} ${IMAGE_NAME}:latest
  
  echo "Removing original image: ${IMAGE_TAG}..."
  docker rmi ${IMAGE_TAG}
}

# Pull and retag base images
for IMAGE in "${BASE_IMAGES[@]}"; do
  pull_and_retag $IMAGE
done

# Pull and retag sciml images
for IMAGE in "${SCIML_IMAGES[@]}"; do
  pull_and_retag $IMAGE
done

# Pull and retag mantid images
for IMAGE in "${MANTID_IMAGES[@]}"; do
  pull_and_retag $IMAGE
done

# Pull and retag other images
for IMAGE in "${OTHER_IMAGES[@]}"; do
  pull_and_retag $IMAGE
done

echo -e "Pull and retag process completed.\n"