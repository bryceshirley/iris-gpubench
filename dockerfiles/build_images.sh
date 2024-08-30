#!/bin/bash

# ==============================================================================
# Build Docker Images Locally
# ==============================================================================
#
# This script builds Docker images from Dockerfiles located in the specified
# directories. Base images are built first, followed by app images.
#
# Requirements:
# - Docker must be installed and running.
#
# Usage:
# Run the script: ./build_images.sh
#
# Author: Bryce Shirley
# Date: 19.08.2024 (Updated on 30.08.2024)
# ==============================================================================

# Directory containing Dockerfiles
BASE_IMAGES_DIR="base_images"
APP_IMAGES_DIR="app_images"

# List of base images to build
BASE_IMAGES=("mantid_base" "sciml_base")

# List of app images to build
SCIML_IMAGES=("mnist_tf_keras" "stemdl_classification" "synthetic_regression")
MANTID_IMAGES=("mantid_run_1" "mantid_run_8")

# Build base images first
echo "Building base images..."
for IMAGE in "${BASE_IMAGES[@]}"; do
    DOCKERFILE="${BASE_IMAGES_DIR}/Dockerfile.${IMAGE}"
    IMAGE_TAG="${IMAGE}:latest"
    echo "Building base image: ${IMAGE_TAG}..."
    docker build -f $DOCKERFILE -t ${IMAGE_TAG} .
done

# Build sciml_bench images
echo "Building sciml_bench images..."
for IMAGE in "${SCIML_IMAGES[@]}"; do
    DOCKERFILE="${APP_IMAGES_DIR}/sciml_bench/Dockerfile.${IMAGE}"
    IMAGE_TAG="${IMAGE}:latest"
    echo "Building app image: ${IMAGE_TAG}..."
    docker build -f $DOCKERFILE -t ${IMAGE_TAG} --build-arg BASE_IMAGE=sciml_base:latest .
done

# Build mantid_bench images
echo "Building mantid_bench images..."
for IMAGE in "${MANTID_IMAGES[@]}"; do
    DOCKERFILE="${APP_IMAGES_DIR}/mantid_bench/Dockerfile.${IMAGE}"
    IMAGE_TAG="${IMAGE}:latest"
    echo "Building app image: ${IMAGE_TAG}..."
    docker build -f $DOCKERFILE -t ${IMAGE_TAG} --build-arg BASE_IMAGE=mantid_base:latest .
done

# Build dummy image
echo "Building dummy image..."
DOCKERFILE="${APP_IMAGES_DIR}/Dockerfile.dummy"
IMAGE_TAG="dummy:latest"
echo "Building app image: ${IMAGE_TAG}..."
docker build -f $DOCKERFILE -t ${IMAGE_TAG} .

echo -e "Build process completed.\n"