#!/bin/bash

# ~~~~~~~~~ Pull Pre-Built Images from GitHub Container Registry ~~~~~~~~~

# Pull gpu_base image
echo "Pulling gpu_base image..."
docker pull ghcr.io/yourusername/gpu_base:latest

# Pull sciml_base image
echo "Pulling sciml_base image..."
docker pull ghcr.io/bryceshirley/sciml_base:latest

# Pull mantid_base image
echo "Pulling mantid_base image..."
docker pull ghcr.io/bryceshirley/mantid_base:latest

# Pull stemdl_classification_base image
echo "Pulling stemdl_classification_base image..."
docker pull ghcr.io/bryceshirley/stemdl_classification_base:latest

# Pull synthetic_regression image
echo "Pulling synthetic_regression image..."
docker pull ghcr.io/bryceshirley/synthetic_regression:latest

# Pull stemdl_classification_1gpu image
echo "Pulling stemdl_classification_1gpu image..."
docker pull ghcr.io/bryceshirley/stemdl_classification_1gpu:latest

# Pull stemdl_classification_2gpu image
echo "Pulling stemdl_classification_2gpu image..."
docker pull ghcr.io/bryceshirley/stemdl_classification_2gpu:latest

# Pull mnist_tf_keras image
echo "Pulling mnist_tf_keras image..."
docker pull ghcr.io/bryceshirley/mnist_tf_keras:latest

# Pull mantid_run_8 image
echo "Pulling mantid_run_8 image..."
docker pull ghcr.io/bryceshirley/mantid_run_8:latest

# Pull mantid_run_1 image
echo "Pulling mantid_run_1 image..."
docker pull ghcr.io/bryceshirley/mantid_run_1:latest

# Pull dummy image
echo "Pulling dummy image..."
docker pull ghcr.io/bryceshirley/dummy:latest

echo -e "Pull process completed.\n"
