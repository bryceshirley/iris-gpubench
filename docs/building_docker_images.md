Here’s the updated documentation with a new section added for building images locally and updated references for Harbor:

## Available Benchmark Images

The repository includes various Docker images for benchmarking. Below is a list of available images and their purposes:

### Mantid Imaging Benchmarks
- **`mantid_run_1`**: Dockerfile for a Mantid benchmark with 1GB of data.
- **`mantid_run_8`**: Dockerfile for a Mantid benchmark with 8GB of data.
  - For more details, visit the [Mantid Imaging Benchmarks Repository](https://github.com/samtygier-stfc/mantid_imaging_cloud_bench).

### SciML Benchmarks
- **`mnist_tf_keras`**: Dockerfile for MNIST classification using TensorFlow/Keras.
- **`stemdl_classification_1gpu`**: Dockerfile for STEMDL classification using a single GPU.
- **`stemdl_classification_2gpu`**: Dockerfile for STEMDL classification using two GPUs (ensure two GPUs are available).
- **`synthetic_regression`**: Dockerfile for synthetic regression benchmarks with parameters `-b hidden_size 9000 -b epochs 10`.
  - For more information, check the [SCIML Bench Repository](https://github.com/stfc-sciml/sciml-bench). These Dockerfiles clone a [forked version](https://github.com/bryceshirley/sciml-bench) of the SCIML-Bench repo.

### Dummy Benchmark Container
- **`dummy`**: A Dockerfile for a container used to profile GPU resource usage. This container runs for 5 minutes and helps assess the impact of the monitoring tool. For more details on profiling tests, see [Profiling the Monitor's Impact on GPU Resources](considerations_on_accuracy.md#profiling-the-monitors-impact-on-gpu-resources).

### Base Images
- **`sciml_base`**: Base image for SciML (Scientific Machine Learning) benchmarks.
- **`mantid_base`**: Base image tailored for Mantid imaging benchmarks.

## Directory Structure

The repository's Dockerfiles are organized as follows:

```
dockerfiles
├── app_images
│   ├── Dockerfile.dummy
│   ├── Dockerfile.mantid_run_1
│   ├── Dockerfile.mantid_run_8
│   ├── Dockerfile.mnist_tf_keras
│   ├── Dockerfile.stemdl_classification_1gpu
│   ├── Dockerfile.stemdl_classification_2gpu
│   └── Dockerfile.synthetic_regression
├── base_images
│   ├── Dockerfile.mantid_base
│   └── Dockerfile.sciml_base
└── pull_images.sh
```

### Explanation:
- **`app_images`**: Contains Dockerfiles for specific benchmark applications.
- **`base_images`**: Contains Dockerfiles for base images used to build other Docker images.
- **`pull_images.sh`**: Script to pull all Docker images from Harbor to your local environment.

## Pulling Docker Images From Harbor

To pull all images from Harbor to your local machine, execute the `pull_images.sh` script:

```sh
./pull_images.sh
```

## Building Docker Images Locally

To build Docker images locally instead of pulling them from Harbor, execute the `build_images.sh` script: 

   ```sh
   ./build_images.sh
   ```


## Adding New Benchmarks

### Steps to Add New Benchmarks:
1. **Add Dockerfiles**: Place Dockerfiles for new benchmarks in the `dockerfiles/app_images` directory. Ensure the Dockerfile extension names reflect the image names you want (e.g., `Dockerfile.synthetic_regression` will create an image named `synthetic_regression`).

2. **Custom Base Images**: Store Dockerfiles for custom base images in the `dockerfiles/base_images` directory. These base images will be built first by the GitHub Actions workflows.

3. **Push Changes**: Commit and push your changes to the GitHub repository. The GitHub Actions workflows will automatically build and push the Docker images to Harbor.

4. **Pull Images**: Use `pull_images.sh` to download the new images to your local environment or use `build_images.sh` to build them locally.

5. **Check Build Status:** Verify that the images have been built successfully by listing the local Docker images using `docker images`

With this streamlined setup, you can efficiently manage and deploy Docker images for benchmarking using GitHub Actions and Harbor.
