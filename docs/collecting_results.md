# Collecting Results 

## Summary

By default, results are saved to a `results` folder within the current directory where the `iris-gpubench` command is executed. If the folder does not already exist, it will be created automatically. The folder will contain the following files:

- **Formatted Results Text**: `formatted_metrics.txt`
- **GPU Metrics Timeseries Plot png**: `metrics_plot.png`
- **Results YAML**: `metrics.yml`

NOT YET WORKING: If the --export-to-victoria tag is used, the timeseries data will be sent to a Grafana dashboard. This dashboard can be used to visualize and analyze GPU metrics and performance data in a more interactive and detailed way.


## Formatted Results

- **File:** `formatted_metrics.txt`  
- **Description:** A human-readable version of the `metrics.yml` file. Provides a tabular summary of GPU and carbon performance metrics, including benchmark image name, elapsed monitor time, energy consumption, carbon emissions, and detailed GPU performance data.  
- **Example:**  
```sh
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

## GPU Metrics Timeseries Plot png

- **File:** `metrics_plot.png`  
- **Description:** Time series plots showing GPU utilization, power usage, temperature, and memory. This plot aggregates data from multiple GPUs, including maximum power limits, peak memory usage, and total energy consumption calculated from the power usage timeseries.

- **Example:**  
![GPU Metrics Output](docs_image_multigpu.png)

## Result Metrics YAML

- **File:** `metrics.yml`  
- **Description:** Contains formatted data on GPU and carbon performance results. Includes metrics such as benchmark image name, elapsed monitor time, total GPU energy consumed, total carbon emissions, carbon forecast information, and detailed GPU performance data.  


## GPU Metric Grafana Plots (--export_to_victoria) (NOT WORKING)

INSERT GRAFANA LINK HERE

---

[Previous Page](example_commands.md) | [Next Page](live_monitoring.md)