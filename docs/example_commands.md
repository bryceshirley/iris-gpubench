Here's an updated version of the example commands section in the documentation, including detailed explanations for each example:

---

## Example Usage

### Example 1: Basic Monitoring with Completion Plot:

```sh
iris-gpubench --benchmark_image "synthetic_regression"
```

- **Explanation**: This command runs GPU monitoring while executing the benchmark specified by the Docker image `synthetic_regression`. The system will collect GPU metrics and generate a completion plot at the end. Live monitoring of GPU metrics is enabled by default.

### Example 2: Exporting Data to VictoriaMetrics:

```sh
iris-gpubench --benchmark_image "synthetic_regression" --export_to_victoria
```

- **Explanation**: Similar to the first example, this command runs the `synthetic_regression` Docker image benchmark and collects GPU metrics. Additionally, the collected data is exported to VictoriaMetrics for long-term storage and further analysis. This is useful when you need to monitor metrics over time and visualize them later using external tools.

### Example 3: Full Command with All Options:

```sh
iris-gpubench --benchmark_image "stemdl_classification_2gpu" --interval 10 --carbon_region "South England" --live_plot --export_to_victoria --monitor_logs
```

- **Explanation**: This is a comprehensive example that runs the `stemdl_classification_2gpu` benchmark in a Docker container and collects GPU metrics at a 10-second interval. The `--carbon_region` flag specifies the carbon intensity region as "South England" to track the carbon emissions impact. Live plotting of GPU metrics is enabled (`--live_plot`), and data will be exported to VictoriaMetrics (`--export_to_victoria`). The `--monitor_logs` flag enables logging of both GPU metrics and the Docker container logs, allowing for deeper analysis of benchmark performance.

### Example 4: Run and Monitor Benchmark in the Background without the Need for a Container:

```sh
/mantid_imaging_cloud_bench$ iris-gpubench --benchmark_command "./run_1.sh" --live_plot --interval 1
```

- **Explanation**: In this example, a benchmark command (`./run_1.sh`) is executed in the background using `tmux` instead of a Docker container. GPU metrics are collected at 1-second intervals, and live plotting of these metrics is enabled. This is useful when you have a script or binary that doesn't require containerization and want to monitor the system's GPU usage in real-time. Running benchmarks in `tmux` allows the process to continue in the background, making it ideal for long-running benchmarks that don't need constant attention.
- **Important**: For this example, you'll need to install you benchmark on the VM and the iris-gpubench package.

---

[Previous Page](command_line_interface.md) | [Next Page](collecting_results.md)

--- 
