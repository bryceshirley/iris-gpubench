If you need to build Docker images for benchmarking, you can use the provided `build_images.sh` script. This script will build images defined in the `Benchmark_Docker` directory. Here’s how to use it:

 1.**Navigate to the Docker Directory**:
   Go to the `Benchmark_Docker` directory:
```sh
cd Benchmark_Docker
```

 2.**Run the Build Script**:
   Execute the build script to build all Docker images:
```sh
./build_images.sh
```

   This script will build Docker images from the Dockerfiles located in `Benchmark_Docker/Benchmark_Dockerfiles`. Feel Free to add your own. The available Dockerfiles and their purposes are:

   - **Base Images**:
     - `Dockerfile.gpu_base`: Base image for GPU-based benchmarks.
     - `Dockerfile.sciml_base`: Base image for SciML benchmarks.
     - `Dockerfile.mantid_base`: Base image for Mantid imaging benchmarks.
   
   - **Mantid Imaging Benchmarks**:
     - `Dockerfile.mantid_run_1`: Dockerfile for Mantid benchmark run 1GB.
     - `Dockerfile.mantid_run_8`: Dockerfile for Mantid benchmark run 8GB.
   
   - **SciML Benchmarks**:
     The docker files are run on a [fork of the sciml-bench repo.](https://github.com/bryceshirley/sciml-bench)
     - `Dockerfile.mnist_tf_keras`: Dockerfile for MNIST classification Benchmark using TensorFlow/Keras.
     - `Dockerfile.stemdl_classification_2gpu`: Dockerfile for STEMDL classification using 1 GPU
     - `Dockerfile.stemdl_classification_2gpu`: Dockerfile for STEMDL classification using 2 GPUs (There must be 2 available).
     - `Dockerfile.synthetic_regression`: Dockerfile for synthetic regression benchmarks using
     the options `-b hidden_size 9000 -b epochs 10`. 
     
     See [sciml-bench docs](https://github.com/stfc-sciml/sciml-bench/tree/master/sciml_bench/docs) for more information on these benchmarks and their options.
    
   - **Dummy Benchmark Container**:
     - `Dockerfile.dummy`: Designed for testing purposes, this Dockerfile sets up a container that runs for 5 minutes before terminating. It is primarily used to profile the GPU resource usage of this monitoring tool. Ideally, the monitor should operate in isolation from the benchmarks to avoid interference. However, currently, the monitor runs in the background on the same VM as the benchmark containers, which poses scalability limitations. To address this in the future, Docker Compose could be used to manage multiple containers simultaneously, but this would require an SSH-based solution to monitor them from an external VM.
  
  This setup will prepare the environment and Docker images required for running your benchmarks effectively.

## Adding New Benchmarks

To add new benchmarks, place their setup Dockerfiles in the `Benchmark_Docker/Benchmark_Dockerfiles` directory. Each Dockerfile should include an entry point that runs the benchmark, such as:

```sh
ENTRYPOINT ["/bin/bash", "-c", "cd /root/mantid_imaging_cloud_bench && conda activate mantidimaging && ./run_8.sh"]
```

Additionally, update `Benchmark_Docker/build_images.sh` to include your new image, or you can manually build the image using the following command:

```sh
docker build -t <image_name> -f path/to/Dockerfile.<image_name> .
```

With the `iris-gpubench` package installed and the Benchmark Docker images built, you can now monitor the benchmarks using `iris-gpubench`. For details on how to use it, refer to the next page Command-Line Arguments.

[Previous Page](installation.md) | [Next Page](command_line_interface.md)