Here’s the updated documentation with a new section added for building images locally and updated references for Harbor:

## Available Benchmark Images

The repository includes various Docker images for benchmarking. Below is a list of available images and their purposes:

### Mantid Imaging Benchmarks
- **`mantid_run_1`**: Dockerfile for a Mantid benchmark with 1GB of data.
- **`mantid_run_4`**: Dockerfile for a Mantid benchmark with 4GB of data.
- **`mantid_run_5`**: Dockerfile for a Mantid benchmark with 5GB of data.
- **`mantid_run_8`**: Dockerfile for a Mantid benchmark with 8GB of data.

For more details, visit the [Mantid Imaging Benchmarks Repository](https://github.com/samtygier-stfc/mantid_imaging_cloud_bench).

### SciML Benchmarks
- **`mnist_tf_keras`**: Dockerfile for MNIST classification using TensorFlow/Keras. (NOT WORKING)
- **`stemdl_classification`**: Dockerfile for STEMDL classification, this benchmark will utilize multiple gpus, if available. Uses parameters `-b epochs 3`.
- **`synthetic_regression`**: Dockerfile for synthetic regression benchmarks with parameters `-b hidden_size 9000 -b epochs 3`.
  
For more information, check the [SCIML Bench Repository](https://github.com/stfc-sciml/sciml-bench). These Dockerfiles clone a [forked version](https://github.com/bryceshirley/sciml-bench) of the SCIML-Bench repo.

### Dummy Benchmark Container
- **`dummy`**: A Dockerfile for a container used to profile GPU resource usage. This container runs for 5 minutes and helps assess the impact of the monitoring tool. For more details on profiling tests, see [Profiling the Monitor's Impact on GPU Resources](considerations_on_accuracy.md#profiling-the-monitors-impact-on-gpu-resources).

### Base Images
- **`sciml_base`**: Base image for SciML (Scientific Machine Learning) benchmarks.
- **`mantid_base`**: Base image tailored for Mantid imaging benchmarks.

## Directory Structure

The repository's Dockerfiles are organized as follows:

```
dockerfiles/
├── app_images
│   ├── Dockerfile.dummy
│   ├── mantid_bench
│   │   ├── Dockerfile.mantid_run_1
│   │   ├── Dockerfile.mantid_run_4
│   │   ├── Dockerfile.mantid_run_5
│   │   └── Dockerfile.mantid_run_8
│   └── sciml_bench
│       ├── Dockerfile.mnist_tf_keras
│       ├── Dockerfile.stemdl_classification
│       └── Dockerfile.synthetic_regression
├── base_images
│   ├── Dockerfile.mantid_base
│   └── Dockerfile.sciml_base
├── build_images.sh
└── pull_images.sh
```

### Explanation:
- **`app_images`**: Contains Dockerfiles for specific benchmark applications.
- **`base_images`**: Contains Dockerfiles for base images used to build the app Docker images.

## Pulling Docker Images From Harbor

To pull all images from Harbor to your local machine, execute the `pull_images.sh` script (login details are requested):

```sh
./pull_images.sh
```

If the above fails, running the command below is recommended.

## Building Docker Images Locally

To build Docker images locally instead of pulling them from Harbor, execute the `build_images.sh` script: 

   ```sh
   ./build_images.sh
   ```

## Running Your Own Containerized Benchmarks

`iris-gpubench` allows you to evaluate your own containerized workloads. To use this feature:

1. Build your benchmark image locally.
2. Use `iris-gpubench` to run your image and collect performance metrics.

For detailed instructions on running your own images, refer to the Command Line Interface section (Next Page).

## Adding New Benchmarks (For Developers)

### Steps to Add New Benchmarks:
1. **Add Dockerfiles**: Place Dockerfiles for new benchmarks in the `dockerfiles/app_images` directory. Ensure the Dockerfile extension names reflect the image names you want (e.g., `Dockerfile.synthetic_regression` will create an image named `synthetic_regression`).

2. **Custom Base Images**: Store Dockerfiles for custom base images in the `dockerfiles/base_images` directory. These base images will be built first by the GitHub Actions workflows or by `build_images.sh`.

3. **Update `build_images.sh` so the can easily be built locally**

4. **To add to Harbor**: Update github workflow `docker-build.yml` to include them (on push/pull request to main they will be built) and as add them to `pull_images.sh`.

5. **Ensure Benchmark Specific Results are Stored Correctly**: In the dockerfile save benchmark specific results to the `/root/results/` directory within the container. Additionally, record the benchmark's time score (in seconds) in a `metrics.yml` file using the key `time: 123`. This ensures that IRIS Bench can copy these results from the container before it is removed.

---

[Previous Page](installation.md) | [Next Page](command_line_interface.md)
