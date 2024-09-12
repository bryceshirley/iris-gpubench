
```sh
iris-gpubench [--benchmark_image BENCHMARK_IMAGE | --benchmark_command BENCHMARK_COMMAND] [--interval INTERVAL] [--carbon_region CARBON_REGION] [--live_plot] [--export_to_meerkat] [--monitor_logs]
```

The following optional arguments are supported:

- `--no_live_monitor`: Disable live monitoring of GPU metrics. Default is enabled.
- `--interval <seconds>`: Set the interval for collecting GPU metrics. Default is `5` seconds.
- `--carbon_region <region>`: Specify the carbon region for the National Grid ESO Regional Carbon Intensity API. Default is `"South England"`.
- `--no_plot`: Disable plotting of GPU metrics, saves as png. Default is enabled.
- `--live_plot`: Enable live plotting of GPU metrics and saves plot to png.
- `--export_to_meerkat`: Enable exporting of collected data to Meerkat DB to be scrapped by the Grafana Dashboard.
- `--benchmark_image <image>`: Docker container image to run as a benchmark.
- `--benchmark_command <command>`: Command to run as a benchmark in a `tmux` session. This option allows running benchmarks without Docker.
- `--monitor_logs`: Enable monitoring of container or tmux logs in addition to GPU metrics.

## Help Option

To display the help message with available options, run:

```sh
iris-gpubench --help
```

## Using the Meerkat Exporter

For the `--export_to_meerkat` option, the Meerkat username, password, and URL must be set as environment variables. Use the following commands:

```bash
export MEERKAT_USERNAME='insert_username'
export MEERKAT_PASSWORD='insert_password'
export MEERKAT_URL='https://172.16.101.182:8247/write' 
```

(As of 12/09/24, this is the correct URL.)

## Useful to know
- Either `--benchmark_image` or `--benchmark_command` must be provided, but not both. If both options are specified, an error will be raised.
- Live GPU metrics monitoring and saving a final plot are enabled by default; use `--no_live_monitor` and `--no_plot` to disable them, respectively.
- To view the available carbon regions, use `--carbon_region ""` to get a list of all regions.
- To list available Docker images, use `--benchmark_image ""` for a list of images.

For example commands please see the next page.

---

[Previous Page](building_docker_images.md) | [Next Page](example_commands.md)
