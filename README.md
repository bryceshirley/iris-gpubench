- FIX THE SAVING BUG for save plots/ metrics/ figs into the correct directory
- FIX THE CARBON METRIC BUG for end carbon
- Fix victoria_metrics exporter (username and password needed) and Test with grafana (add grafana link to docs)
- Edit Formatted Results so that the time is the total time it took for the benchmark container to run.

# GPU Monitoring Tool Usage

## Overview

This tool monitors GPU metrics using the `GPUMonitor` class and optionally exports the collected data to VictoriaMetrics. 

## Installation

To set up the project, you need to install the package in editable mode. Run the following command in your terminal with the setup.py file:

```bash
pip install .
```

## Command-Line Arguments

The following optional arguments are supported:

- `--live_monitor`: Enable live monitoring of GPU metrics.
- `--interval <seconds>`: Set the interval for collecting GPU metrics. Default is `1` second.
- `--carbon_region <region>`: Specify the carbon region for the National Grid ESO Regional Carbon Intensity API. Default is `"South England"`.
- `--plot`: Enable plotting of GPU metrics.
- `--live_plot`: Enable live plotting of GPU metrics.
- `--export_to_victoria`: Enable exporting of collected data to VictoriaMetrics.

## Example Commands

1. **Basic Monitoring**:
   ```bash
   gpu_monitor --live_monitor --interval 5
   ```

2. **Monitoring with Plotting**:
   ```bash
   gpu_monitor --live_monitor --interval 5 --plot
   ```

3. **Exporting Data to VictoriaMetrics**:
   ```bash
   gpu_monitor --live_monitor --interval 5 --export_to_victoria
   ```

4. **Full Command with All Options**:
   ```bash
   gpu_monitor --live_monitor --interval 10 --carbon_region "South England" --plot --live_plot --export_to_victoria
   ```

## Help Option

To display the help message with available options, run:

```bash
gpu_monitor --help
```

-----------

# Results 

Results are saved to gpu_benchmark_metrics/results (these include):

## 1. Formatted Results

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
## 2. GPU Metric png Plots (--plot)

* metrics_plot.png: Time series plots for gpu utilization, power usage, temperature and Memory. See example below:
 
  <img src="docs_image.png" alt="GPU Metrics Output" width="500"/>

## 2. GPU Metric Grafana Plots (--export_to_victoria) (NOT WORKING)

INSERT GRAFANA LINK HERE

## 3. Result Metrics

* metrics.yml: yml with the Benchmark Score and GPU Energy Performance results.
  
-----------

# Live Monitoring

## 1. Monitor GPU Metrics (--live_monitor)

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

## 2. Save png Timeseries Live (--plot_live)
  
```bash
gpu_monitor --plot_live
```

Gives you saves plot png during every reading so that the metrics can be viewed live.

----------- 

## Please Note:
* The Carbon Data is collected in real-time from the National Grid ESO Regional Carbon Intensity API:
  <https://api.carbonintensity.org.uk/regional>
* The Carbon Forcast and Index Readings are updated every 30 minutes.
* Set your region in gpu_monitor.py: CARBON_INSTENSITY_REGION_SHORTHAND="South England"

* The GPU power metrics and GPU utilization come from "nvidia-smi" results.
* The "error in nvidia-smi's power draw is ± 5%" according to:
  <https://arxiv.org/html/2312.02741v2#:~:text=The%20error%20in%20nvidia%2Dsmi's,%C2%B1%2030W%20of%20over%2Funderestimation.>  


-----------

# Work To Do
- Fix GPU score so time comes up for stemdl_classification when different epochs are used
- Get my own metric for time
- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- using shell check from bash script (similar to pylint) on bash script
- Add CI tests for python scripts
- Make monitor.sh collect errors from sciml-bench command
