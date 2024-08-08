- FIX THE SAVING BUG for save plots/ metrics/ figs into the correct directory
- FIX THE CARBON METRIC BUG for end carbon
- Fix victoria_metrics exporter (username and password needed) and Test with grafana

# GPU Monitoring Tool Usage

## Overview

This tool monitors GPU metrics using the `GPUMonitor` class and optionally exports the collected data to VictoriaMetrics. 

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
