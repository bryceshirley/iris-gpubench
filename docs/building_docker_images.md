### Script to Build The Docker Images (Currently under development to integrate CI so may not work)

To build Docker images for benchmarking, navigate to the `Benchmark_Docker` directory and execute the build script:

```sh
cd Benchmark_Docker
./build_images.sh
```

This script will build Docker images from the Dockerfiles located in `Benchmark_Docker/Benchmark_Dockerfiles`. Feel Free to add your own here too. 

### Available Dockerfiles and Their Purposes

#### Base Images
- **`Dockerfile.gpu_base`**: Base image for GPU-based benchmarks.
- **`Dockerfile.sciml_base`**: Base image for SciML benchmarks.
- **`Dockerfile.mantid_base`**: Base image for Mantid imaging benchmarks.

#### Mantid Imaging Benchmarks
More details about can be found on the [Mantid Imaging Benchmarks Repo](https://github.com/samtygier-stfc/mantid_imaging_cloud_bench).

- **`Dockerfile.mantid_run_1`**: Dockerfile for Mantid benchmark run 1GB.
- **`Dockerfile.mantid_run_8`**: Dockerfile for Mantid benchmark run 8GB.

#### SciML Benchmarks
The Dockerfiles for SciML benchmarks from a [fork of the sciml-bench repo](https://github.com/bryceshirley/sciml-bench).

- **`Dockerfile.mnist_tf_keras`**: Dockerfile for MNIST classification Benchmark using TensorFlow/Keras.
- **`Dockerfile.stemdl_classification_2gpu`**: Dockerfile for STEMDL classification using 1 GPU.
- **`Dockerfile.stemdl_classification_2gpu`**: Dockerfile for STEMDL classification using 2 GPUs (ensure 2 GPUs are available).
- **`Dockerfile.synthetic_regression`**: Dockerfile for synthetic regression benchmarks using the options `-b hidden_size 9000 -b epochs 10`.

For more information on these benchmarks and their options, refer to the [sciml-bench docs](https://github.com/stfc-sciml/sciml-bench/tree/master/sciml_bench/docs).

#### Dummy Benchmark Container
- **`Dockerfile.dummy`**: This Dockerfile is intended for testing and profiling purposes. It sets up a container that runs for 5 minutes before automatically terminating. The primary goal of this container is to profile the GPU resource usage of the monitoring tool itself. Ideally, the monitor should operate in isolation from benchmark containers to prevent interference and ensure accurate profiling. However, since the monitor currently runs on the same VM as the benchmark containers, this setup has scalability limitations. For a more details and results from profiling the monitor’s impact on GPU resources, see [Profiling the Monitor's Impact on GPU Resources](considerations_on_accuracy.md#profiling-the-monitors-impact-on-gpu-resources). Future improvements could involve using Docker Compose to manage multiple containers simultaneously, potentially requiring an SSH-based solution to monitor them from an external VM.
  
This setup will prepare the environment and Docker images required for running your benchmarks effectively.

### Adding New Benchmarks Dockerfiles

To add new benchmarks, place their setup Dockerfiles in the `Benchmark_Docker/Benchmark_Dockerfiles` directory. Each Dockerfile should include an entry point that runs the benchmark, such as:

```sh
ENTRYPOINT ["/bin/bash", "-c", "cd /root/mantid_imaging_cloud_bench && conda activate mantidimaging && ./run_8.sh"]
```

Additionally, update `Benchmark_Docker/build_images.sh` to include your new image, or you can manually build the image using the following command:

```sh
docker build -t <image_name> -f path/to/Dockerfile.<image_name> .
```

With the `iris-gpubench` package installed and the Benchmark Docker images built, you can now monitor the benchmarks using `iris-gpubench`. For details on how to use it, refer to the next page Command-Line Arguments.

---

[Previous Page](installation.md) | [Next Page](command_line_interface.md)
