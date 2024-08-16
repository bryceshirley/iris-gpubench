# Overview

The GPU Monitoring and Carbon Calculation Tool for Containerized Benchmarks is designed to monitor GPU performance and calculate associated carbon emissions during benchmarking tasks involving GPU operations.

## Key Features

- **Comprehensive GPU Monitoring**: Utilizing the `GPUMonitor` class, the tool captures real-time GPU metrics such as utilization, power draw, temperature, and memory usage. These metrics are essential for evaluating the efficiency and performance of GPU-based applications.

- **Carbon Emissions Calculation**: The tool integrates with the National Grid ESO Regional Carbon Intensity API to estimate carbon emissions based on the energy consumed by the GPU during benchmarks. This feature is particularly useful for understanding the environmental impact of high-performance computing tasks.

- **Export to VictoriaMetrics**: Collected data can be optionally exported to VictoriaMetrics, allowing for long-term storage and analysis of GPU performance metrics. This is useful for tracking performance trends over time and integrating with other monitoring systems.

- **Docker Integration**: The tool is designed to work seamlessly with containerized environments. It includes scripts for building Docker images tailored to specific benchmarking tasks, ensuring that the monitoring tool can be easily integrated into existing workflows.

- **Flexible Command-Line Interface**: Users can customize the monitoring process with a variety of command-line arguments. These options allow for control over monitoring intervals, region-specific carbon calculations, live plotting of metrics, and more.

- **Live Monitoring and Logging**: The tool supports live monitoring of GPU metrics during benchmark execution, with the ability to log and display container logs in real-time. This feature is crucial for debugging and optimizing benchmark runs.

## Use Cases

- **Benchmarking GPU-Intensive Applications**: Ideal for researchers, developers, and engineers who need to assess the performance of GPU-accelerated applications under various conditions.

- **Environmental Impact Assessment**: Provides a means to quantify the carbon emissions associated with GPU usage, which is important for organizations focused on sustainability and reducing their carbon footprint.

- **Performance Optimization**: By analyzing the collected GPU metrics, users can identify bottlenecks and optimize their applications to run more efficiently on GPUs.

---

[Previous Page](index.md) | [Next Page](installation.md)