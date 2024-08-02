GPU Energy and Carbon Performance Benchmarking
==============================================

-----------
## Table of Contents
* [The Command](https://github.com/bryceshirley/gpu_benchmark_metrics#the-command)

* [Usage Instructions](https://github.com/bryceshirley/gpu_benchmark_metrics#usage-instructions)

* [Benchmark Results](https://github.com/bryceshirley/gpu_benchmark_metrics#results)

* [Requirements](https://github.com/bryceshirley/gpu_benchmark_metrics#requirements)

* [Work To Do](https://github.com/bryceshirley/gpu_benchmark_metrics#work-to-do)

-----------

# The Command
The command produces a summary of a benchmark or workloads GPU power and real-time carbon performance. It's currently compatible with sciml-benchmarks but can be generalized to any benchmark that utilizes NVIDIA GPUs:

```bash
./multi_gpu_monitor.sh <--run_options sciml_benchmark>
```

Or 

BASH CODE NEEDS UPDATING FOR THESE TAGS TO WORK (THEY EXIST VIA THE PYTHON AND ARE CORRECTLY ALL SET TO DEFAULTS)
```bash
./multi_gpu_monitor.sh <--run_options sciml_benchmark> [--interval INTERVAL] [--carbon_region REGION] [--live_plot]
```
Parameters:

    --interval INTERVAL: Sets the interval (in seconds) for collecting GPU
    metrics. Default is 1 second.

    --carbon_region REGION: Specifies the region shorthand for the National
    Grid ESO Regional Carbon Intensity API. Default is 'South England'.

    --live_plot: Enables live plotting of GPU metrics via continuously save png file through run. Note: Live plotting is not
    recommended as errors may arise if the code is interrupted during the plot saving
    process. The plots will be saved at the end anyway. [Plot Results](https://github.com/bryceshirley/gpu_benchmark_metrics/edit/main/README.md#4-gpu-power-and-utilization-plots).

Example:
```bash
./multi_gpu_monitor.sh <--run_options sciml_benchmark> --interval 30 --carbon_region "North Scotland" --plot_live
```
Sets the monitoring interval to 30 seconds, uses "North Scotland" as the carbon intensity region, and generates plots for metrics throughout.

-----------

# Live Monitoring


### a. The Output of the Benchmark and GPU Metrics Are Tracked Live By Copying over The Tmux Outputs. Example:

This example uses the "stemdl_classification" benchmark with the "-b epochs 1" option for two epochs and "-b gpus 2" too utilize both gpus available (see sciml-bench docs for more options)

```bash
(bench) dnz75396@bs-scimlbench-a100:~/gpu_benchmark_metrics$ ./multi_gpu_monitor.sh '-b epochs 1 -b gpus 2 stemdl_classification'

Live Monitor: GPU Metrics

Current GPU Metrics as of 2024-08-01 23:32:47:
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|   GPU Index (Tesla V100-PCIE-32GB) |   Utilization (%) |   Power (W) / Max 250.0 W |   Temperature (C) |   Memory (MiB) / Total 32768.0 MiB |
+====================================+===================+===========================+===================+====================================+
|                                  0 |                60 |                    87.027 |                34 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|                                  1 |                63 |                    87.318 |                40 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+



Live Monitor: Benchmark Output

2 | f1_score | MulticlassF1Score  | 0      | train
--------------------------------------------------------
24.0 M    Trainable params
0         Non-trainable params
24.0 M    Total params
95.925    Total estimated model params size (MB)
Sanity Checking DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.98it/s]
/root/anaconda3/envs/bench/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:439: It is recommended to
use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
Epoch 0:  15%|█████████████                                                                            | 77/525 [00:05<00:29, 15.21it/s, v_num=0]

```


### b. (Optional) Live Timeseries Using the --plot_live Option (TODO: WORKS FOR PYTHON BUT NEEDS ADDING TO BASH SCRIPT)
  
```bash
./multi_gpu_monitor.sh <sciml benchmark command> --plot_live
```

Gives you saves plot png during every reading so that the metrics can be viewed live. They can be found there afterwards too. See [Plot Results](https://github.com/bryceshirley/gpu_benchmark_metrics/edit/main/README.md#4-gpu-power-and-utilization-plots).

### 3. If you need to terminate the tool for any reason (ie press CTRL+C) then you must kill the tmux session by running:

```bash
tmux kill-session
```

-----------

# Benchmark Results 

## Results are saved to gpu_benchmark_metrics/results (these include):

### 1. Formatted Results

* formatted_metrics.txt : Formatted version of metrics.yml, see example below.

```bash
Benchmark Score and GPU Energy Performance

+-----------------------------------+----------+
| Metric                            |    Value |
+===================================+==========+
| Benchmark Score (s)               | 44.2312  |
+-----------------------------------+----------+
| Total GPU Energy Consumed (kWh)   |  0.00206 |
+-----------------------------------+----------+
| Total GPU Carbon Emissions (gC02) |  0.426   |
+-----------------------------------+----------+

Additional Information

+--------------------------------------------------+------------------------------------+
| Metric                                           | Value                              |
+==================================================+====================================+
| Average GPU Util. (for >0.00% GPU Util.) (%)     | 56.44186                           |
+--------------------------------------------------+------------------------------------+
| Avg GPU Power (for >0.00% GPU Util.) (W)         | 78.89320 (Power Limit: 250)        |
+--------------------------------------------------+------------------------------------+
| Avg GPU Temperature (for >0.00% GPU Util.) (C)   | 38.62791                           |
+--------------------------------------------------+------------------------------------+
| Avg GPU Memory (for >0.00% GPU Util.) (MiB)      | 2032.53198 (Total Memory: 32768.0) |
+--------------------------------------------------+------------------------------------+
| Average Carbon Forcast from start/end (gCO2/kWh) | 207.0                              |
+--------------------------------------------------+------------------------------------+
```

### 2. GPU Metric Plots 

* metrics_plot.png: Time series plots for gpu utilization, power usage, temperature and Memory. See example below:
 
  <img src="docs_image.png" alt="GPU Metrics Output" width="500"/>

### 3. Result Metrics

* metrics.yml: yml with the Benchmark Score and GPU Energy Performance results.

### 4. Benchmark Specific

* benchmark_specific/: directory containing all the results output by the sciml-bench benchmark. 

#### Please Note:
* The Carbon Data is collected in real-time from the National Grid ESO Regional Carbon Intensity API:
  <https://api.carbonintensity.org.uk/regional>
* The Carbon Forcast and Index Readings are updated every 30 minutes.
* Set your region in gpu_monitor.py: CARBON_INSTENSITY_REGION_SHORTHAND="South England"

* The GPU power metrics and GPU utilization come from "nvidia-smi" results.
* The "error in nvidia-smi's power draw is ± 5%" according to:
  <https://arxiv.org/html/2312.02741v2#:~:text=The%20error%20in%20nvidia%2Dsmi's,%C2%B1%2030W%20of%20over%2Funderestimation.>  

-----------

# Requirements (Needs updating)

* **Python Script (gpu_monitor.py):**
	* Python interpreter.
	* Required Python modules: subprocess, csv, time, os, datetime, argparse, matplotlib, tabulate.
	* Dependency on nvidia-smi for GPU metrics.
* **Bash Script (monitor.sh):**
	* Bash shell.
 	* pip install jq
	* External commands: tmux, conda (optional).
	* Ensure correct paths for scripts (gpu_monitor.py) and temporary files.
* **Dependencies:**
	* Python dependencies (matplotlib, tabulate) must be installed.
	* Availability of nvidia-smi for GPU metrics.
* **Configuration:**
	* Set environment variables (PATH, PYTHONPATH) appropriately.
	* Verify paths and environment configurations to prevent command not found errors.

-----------

# Work To Do
- Fix GPU score so time comes up for stemdl_classification when different epochs are used
- Get my own metric for time
- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- using shell check from bash script (similar to pylint) on bash script
- Add CI tests for python scripts
- Make monitor.sh collect errors from sciml-bench command

