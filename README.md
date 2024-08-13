---

# GPU Monitoring And Carbon Calculation Tool For Containerized Benchmarks

## Table of Contents
1. [Installation](#installation)
2. [Building Docker Images](#building-docker-images)
3. [Command-Line Arguments](#command-line-arguments)
4. [Example Commands](#example-commands)
5. [Help Option](#help-option)
6. [Results](#results)
7. [Live Monitoring](#live-monitoring)
8. [Considerations On Accuracy](#considerations-on-accuracy)
9. [Work To Do](#work-to-do)

---

## Overview

This tool monitors GPU metrics using the `GPUMonitor` class and optionally exports the collected data to VictoriaMetrics. 

Here's the updated installation section including instructions for building Docker images:

---

## Installation

To set up the project, follow these steps:

1. **Set Up Virtual Environment**:
   Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. **Install the Package**:
   Install the package in editable mode:
   ```bash
   pip install wheel
   pip install .
   ```

## Building Docker Images

If you need to build Docker images for benchmarking, you can use the provided `build_images.sh` script. This script will build images defined in the `Benchmark_Docker` directory. Here’s how to use it:

1. **Navigate to the Docker Directory**:
   Go to the `Benchmark_Docker` directory:
   ```bash
   cd Benchmark_Docker
   ```

2. **Run the Build Script**:
   Execute the build script to build all Docker images:
   ```bash
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
     - `Dockerfile.mnist_tf_keras`: Dockerfile for MNIST classification Benchmark using TensorFlow/Keras.
     - `Dockerfile.stemdl_classification_2gpu`: Dockerfile for STEMDL classification using 1 GPUs
     - `Dockerfile.stemdl_classification_2gpu`: Dockerfile for STEMDL classification using 2 GPUs (There must be 2 available).
     - `Dockerfile.synthetic_regression`: Dockerfile for synthetic regression benchmarks.
    
   - **Dummy Benchmark Container**:
     - `Dockerfile.dummy`: Designed for testing purposes, this Dockerfile sets up a container that runs for 5 minutes before terminating. It is primarily used to profile the GPU resource usage of this monitoring tool. Ideally, the monitor should operate in isolation from the benchmarks to avoid interference. However, currently, the monitor runs in the background on the same VM as the benchmark containers, which poses scalability limitations. To address this in the future, Docker Compose could be used to manage multiple containers simultaneously, but this would require an SSH-based solution to monitor them from an external VM.

This setup will prepare the environment and Docker images required for running your benchmarks effectively.

---

## Command-Line Arguments

The following optional arguments are supported:

- `--no_live_monitor`: Disable live monitoring of GPU metrics. Default is enabled.
- `--interval <seconds>`: Set the interval for collecting GPU metrics. Default is `5` seconds.
- `--carbon_region <region>`: Specify the carbon region for the National Grid ESO Regional Carbon Intensity API. Default is `"South England"`.
- `--no_plot`: Disable plotting of GPU metrics. Default is enabled.
- `--live_plot`: Enable live plotting of GPU metrics.
- `--export_to_victoria`: Enable exporting of collected data to VictoriaMetrics.
- `--benchmark_image <image>`: Docker container image to run as a benchmark (required).
- `--monitor_benchmark_logs`: Enable monitoring of container logs in addition to GPU metrics.

## Example Commands

1. **Basic Monitoring with Completion Plot**:
   ```bash
   iris-gpubench --benchmark_image "synthetic_regression"
   ```

2. **Exporting Data to VictoriaMetrics**:
   ```bash
   iris-gpubench --benchmark_image "synthetic_regression" --export_to_victoria
   ```

3. **Full Command with All Options**:
   ```bash
   iris-gpubench --benchmark_image "synthetic_regression" --interval 10 --carbon_region "South England" --live_plot --export_to_victoria --monitor_benchmark_logs
   ```

## Help Option

To display the help message with available options, run:

```bash
iris-gpubench --help
```

### Notes:
- The `--benchmark_image` argument is required for specifying the Docker container image.
- live gpu metrics monitoring and saving a final plot are enabled by default; use `--no_live_monitor` and `--no_plot` to disable them, respectively.
- To view the available carbon regions, use `--carbon_region ""` to get a list of all regions.
- To list available Docker images, use `--benchmark_image ""` for a list of images.

-----------

# Results 

By default Results are saved to iris-gpubench/results.

## 1. Formatted Results

* formatted_metrics.txt : Formatted version of metrics.yml, see example below.

```bash
GPU and Carbon Performance Results

+---------------------------------------+------------------------+
| Metric                                | Value                  |
+=======================================+========================+
| Benchmark Image Name                  | synthetic_regression   |
+---------------------------------------+------------------------+
| Elapsed Monitor Time of Container (s) | 245.627                |
+---------------------------------------+------------------------+
| Total GPU Energy Consumed (kWh)       | 0.00993                |
+---------------------------------------+------------------------+
| Total GPU Carbon Emissions (gCO2)     | 1.4196                 |
+---------------------------------------+------------------------+

+---------------------------------------+-----------+
| Metric                                |     Value |
+=======================================+===========+
| Elapsed Monitor Time of Container (s) | 245.627   |
+---------------------------------------+-----------+
| Total GPU Energy Consumed (kWh)       |   0.00993 |
+---------------------------------------+-----------+
| Total GPU Carbon Emissions (gCO2)     |   1.4196  |
+---------------------------------------+-----------+

Carbon Information

+------------------------------------+---------------------+
| Metric                             | Value               |
+====================================+=====================+
| Average Carbon Forecast (gCO2/kWh) | 143.0               |
+------------------------------------+---------------------+
| Carbon Forecast Start Time         | 2024-08-12 17:35:12 |
+------------------------------------+---------------------+
| Carbon Forecast End Time           | 2024-08-12 17:39:18 |
+------------------------------------+---------------------+

GPU Information

+------------------------------------------------+------------------------------------+
| Metric                                         | Value                              |
+================================================+====================================+
| GPU Name                                       | Tesla V100-PCIE-32GB               |
+------------------------------------------------+------------------------------------+
| Average GPU Util. (for >0.00% GPU Util.) (%)   | 92.90476                           |
+------------------------------------------------+------------------------------------+
| Avg GPU Power (for >0.00% GPU Util.) (W)       | 167.42962 (Power Limit: 250)       |
+------------------------------------------------+------------------------------------+
| Avg GPU Temperature (for >0.00% GPU Util.) (C) | 52.33333                           |
+------------------------------------------------+------------------------------------+
| Avg GPU Memory (for >0.00% GPU Util.) (MiB)    | 6500.00595 (Total Memory: 32768.0) |
+------------------------------------------------+------------------------------------+
```
## 2. GPU Metric png Plots

* metrics_plot.png: Time series plots for gpu utilization, power usage, temperature and Memory. See example below:
 
  <img src="docs/docs_image_multigpu.png" alt="GPU Metrics Output" width="500"/>

## 2. GPU Metric Grafana Plots (--export_to_victoria) (NOT WORKING)

INSERT GRAFANA LINK HERE

## 3. Result Metrics

* metrics.yml: yml with the Benchmark Score and GPU Energy Performance results.
  
-----------

# Live Monitoring

## 1. Monitor GPU Metrics

```bash
Current GPU Metrics as of 2024-08-01 23:32:47:
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|   GPU Index (Tesla V100-PCIE-32GB) |   Utilization (%) |   Power (W) / Max 250.0 W |   Temperature (C) |   Memory (MiB) / Total 32768.0 MiB |
+====================================+===================+===========================+===================+====================================+
|                                  0 |                60 |                    87.027 |                34 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|                                  1 |                63 |                    87.318 |                40 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
```

## 2. Monitor Benchmark Container logs
  
```bash
gpu_monitor --benchmark_image "synthetic_regression" --monitor_benchmark_logs

Container Logs:
<BEGIN> Running benchmark synthetic_regression in training mode
....<BEGIN> Parsing input arguments
....<ENDED> Parsing input arguments [ELAPSED = 0.000035 sec]
....<BEGIN> Creating dataset
....<ENDED> Creating dataset [ELAPSED = 12.436691 sec]
....<MESSG> Number of samples: 1024000
....<MESSG> Total number of batches: 8000, 8000.0
....<BEGIN> Training model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/root/anaconda3/envs/bench/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name | Type       | Params | Mode 
--------------------------------------------
0 | net  | Sequential | 250 M  | train
--------------------------------------------
250 M     Trainable params
0         Non-trainable params
250 M     Total params
1,000.404 Total estimated model params size (MB)
11        Modules in train mode
0         Modules in eval mode
/root/anaconda3/envs/bench/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
Epoch 0:   4%|▍         | 350/8000 [00:09<03:23, 37.56it/s, v_num=0]
```
(Long container logs containing "\r" to clear the line for progress bars are not a very efficiently processed, as it captures a snapshot of the entire container log at that moment.
Potiential solution: use asynico package to capture and process the logs whilst the monitor is paused between intervals)
## 3. Save png Timeseries Live
  
```bash
gpu_monitor --benchmark_image "synthetic_regression" --plot_live
```

Gives you saves plot png during every reading so that the metrics can be viewed live.

----------- 

## Considerations On Accuracy
### Carbon Metrics Accuracy Limitations
* The Carbon Data is collected in real-time from the [National Grid ESO Regional Carbon Intensity API.](https://api.carbonintensity.org.uk/regional)
* The Carbon Forecast Readings are updated every 30 minutes. The monitor records the index values at the start and end of each interval and calculates an average. Therefore, the accuracy may be limited for containers that run longer than 30 minutes, as the index can fluctuate significantly over time.
* The Carbon Forecast can vary based on factors such as weather, time of day/year, and energy demand, resulting in fluctuations in total carbon emissions from one run to the next. Therefore, it serves as a real-time estimate. For a broader perspective, you can multiply the total energy by the average Carbon Emission Rate in the UK, which was [162 gCO2/kWh in 2023.](https://www.carbonbrief.org/analysis-uk-electricity-from-fossil-fuels-drops-to-lowest-level-since-1957/#:~:text=Low%2Dcarbon%20sources%20made%20up,fully%20decarbonised%20grid%20by%202035.)

### GPU Metrics Accuracy Limitions
* The GPU metrics come from pynvml which is a python interface for NVML and "nvidia-smi" results.
* The ["error in nvidia-smi's power draw is ± 5%".](https://arxiv.org/html/2312.02741v2#:~:text=The%20error%20in%20nvidia%2Dsmi's,%C2%B1%2030W%20of%20over%2Funderestimation.>)
* Total energy is calculated by integrating power readings over time using the trapezoidal integration method. The accuracy of this calculation depends on the monitoring interval: a smaller interval results in more accurate energy estimates.

### Profiling the Monitors Impact on GPU Resources
* (Minimal) GPU Resource Usage by the Monitor: The monitoring tool consumes a small portion of GPU resources. For instance, a ~5-minute test with a dummy container shows negligible GPU usage, see below. CPU resources are also utilized, though profiling tests to determine exact CPU usage have not yet been conducted.

```bash
GPU and Carbon Performance Results

+---------------------------------------+---------+
| Metric                                | Value   |
+=======================================+=========+
| Benchmark Image Name                  | dummy   |
+---------------------------------------+---------+
| Elapsed Monitor Time of Container (s) | 320.177 |
+---------------------------------------+---------+
| Total GPU Energy Consumed (kWh)       | 0.00183 |
+---------------------------------------+---------+
| Total GPU Carbon Emissions (gCO2)     | 0.36118 |
+---------------------------------------+---------+
```

  <img src="docs/docs_image_dummy.png" alt="GPU Metrics Output" width="500"/>

* These are idle usage levels, so monitoring the GPUs has a negligible impact on GPU resources.

-----------

## Work To Do
- Allow users to save container results
- Fix victoria_metrics exporter (username and password needed) and Test with grafana (add grafana link to docs)
- Improve live monitoring of container ie by threading

- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- using shell check from bash script (similar to pylint) on bash script
- Add CI tests for python scripts

