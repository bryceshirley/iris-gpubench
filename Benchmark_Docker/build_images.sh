#!/bin/bash

## ~~~~~~~~~ Build Base Images ~~~~~~~~~

# Build gpu_base image
echo "Building gpu_base image..."
docker build -t gpu_base -f Dockerfiles/Dockerfile.gpu_base .

# Build sciml_base image using gpu_base as base
echo "Building sciml_base image..."
docker build -t sciml_base -f Dockerfiles/benchmark_dockerfiles/sciml_benchmarks/Dockerfile.sciml_base .

# # Build mantid_base image using gpu_base as base
# echo "Building mantid_base image..."
# docker build -t mantid_base -f Dockerfiles/benchmark_dockerfiles/mantid_imaging_benchmarks/Dockerfile.mantid_base .

## ~~~~~~~~~ Build GPU Monitor Image ~~~~~~~~~

echo "Building gpu_monitor image..."
docker build -t gpu_monitor -f Dockerfiles/gpu_monitor_dockerfiles/Dockerfile.gpu_monitor .

## ~~~~~~~~~ Build Sciml Benchmark Images ~~~~~~~~~
echo "Building Sciml Benchmark images..."

# Build synthetic_regression image using sciml_base as base
echo "Building Sciml Benchmark: synthetic_regression image..."
docker build -t synthetic_regression -f Dockerfiles/benchmark_dockerfiles/sciml_benchmarks/Dockerfile.synthetic_regression .

# Build stemdl_classification image using sciml_base as base
echo "Building Sciml Benchmark: stemdl_classification_2gpu image..."
docker build -t stemdl_classification_2gpu -f Dockerfiles/benchmark_dockerfiles/sciml_benchmarks/Dockerfile.stemdl_classification_2gpu .

# Build mnist_tf_keras image using sciml_base as base
echo "Building Sciml Benchmark: mnist_tf_keras image..."
docker build -t mnist_tf_keras -f Dockerfiles/benchmark_dockerfiles/sciml_benchmarks/Dockerfile.mnist_tf_keras .

# ## ~~~~~~~~~ Build Mantid Imaging Benchmark Images ~~~~~~~~~
# echo "Building Mantid Imaging Benchmark images..."

# # Build mantid_run_8 image using sciml_base as base
# echo "Building Mantid Imaging Benchmark: mantid_run_8 image..."
# docker build -t mantid_run_1 -f Dockerfiles/benchmark_dockerfiles/mantid_imaging_benchmarks/Dockerfile.mantid_run_8 .

# # Build mantid_run_1 image using sciml_base as base
# echo "Building Mantid Imaging Benchmark: mantid_run_1 image..."
# docker build -t mantid_run_8 -f Dockerfiles/benchmark_dockerfiles/mantid_imaging_benchmarks/Dockerfile.mantid_run_1 .

echo -e "Build process completed.\n"