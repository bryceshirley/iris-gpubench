![STFC Logo](stfc_logo.png)
![IRIS Logo](iris_logo.png)

# Overview

The GPU Monitoring and Carbon Calculation Tool `iris-gpubench` monitors GPU performance and estimates carbon emissions during benchmarks.

## Key Features

- **GPU Monitoring**: Tracks real-time metrics such as utilization, power, temperature, and memory usage.
  
- **Carbon Emissions Calculation**: Estimates emissions using the National Grid ESO Regional Carbon Intensity API based on GPU energy consumption.

- **Export to VictoriaMetrics**: Optionally exports data to Grafana for long-term storage and analysis.

- **Flexible Benchmarking**:  
  - **Docker**: Run benchmarks in a consistent, isolated environment.
  - **Tmux**: Execute benchmarks directly on the host system for quick setups.

- **Command-Line Interface**: Customizable options for intervals, carbon calculations, live plotting, and more.

- **Live Monitoring and Logging**: Supports real-time GPU monitoring and logging from Docker containers and Tmux sessions.

## Use Cases

- **Informed Purchasing**: Evaluate and compare performance across different hardware configurations before making purchasing decisions.

- **Benchmarking**: Assess GPU-accelerated applications' performance.
  
- **Environmental Impact**: Quantify carbon emissions from GPU usage.
  
- **Performance Optimization**: Identify and resolve performance bottlenecks.

---

[Previous Page](index.md) | [Next Page](installation.md)
