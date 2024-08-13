# Results 

By default Results are saved to iris-gpubench/results.

## 1. Formatted Results

* formatted_metrics.txt : Formatted version of metrics.yml, see example below.

```
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

![GPU Metrics Output](docs_image_multigpu.png)

## 2. GPU Metric Grafana Plots (--export_to_victoria) (NOT WORKING)

INSERT GRAFANA LINK HERE

## 3. Result Metrics

* metrics.yml: yml with the Benchmark Score and GPU Energy Performance results.