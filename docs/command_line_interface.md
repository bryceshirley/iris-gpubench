```sh
iris-gpubench [--interval INTERVAL] [--carbon_region CARBON_REGION] [--live_plot] [--export_to_victoria] [--benchmark_image BENCHMARK_IMAGE] [--monitor_benchmark_logs]
```

The following optional arguments are supported:

- `--no_live_monitor`: Disable live monitoring of GPU metrics. Default is enabled.
- `--interval <seconds>`: Set the interval for collecting GPU metrics. Default is `5` seconds.
- `--carbon_region <region>`: Specify the carbon region for the National Grid ESO Regional Carbon Intensity API. Default is `"South England"`.
- `--no_plot`: Disable plotting of GPU metrics. Default is enabled.
- `--live_plot`: Enable live plotting of GPU metrics.
- `--export_to_victoria`: Enable exporting of collected data to VictoriaMetrics.
- `--benchmark_image <image>`: Docker container image to run as a benchmark (required).
- `--monitor_benchmark_logs`: Enable monitoring of container logs in addition to GPU metrics.

## Help Option

To display the help message with available options, run:

```sh
iris-gpubench --help
```

### Notes:
- The `--benchmark_image` argument is required for specifying the Docker container image.
- live gpu metrics monitoring and saving a final plot are enabled by default; use `--no_live_monitor` and `--no_plot` to disable them, respectively.
- To view the available carbon regions, use `--carbon_region ""` to get a list of all regions.
- To list available Docker images, use `--benchmark_image ""` for a list of images.

For example, commands please see the next page.

---

[Previous Page](building_docker_images.md) | [Next Page](example_commands.md)
