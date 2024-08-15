"""
This script is the entry point for monitoring GPU metrics using the GPUMonitor class.
It parses command-line arguments, validates them, and initializes the GPU monitoring process.
Additionally, it handles exporting the collected data to VictoriaMetrics if specified.

Dependencies:
- argparse: For parsing command-line arguments.
- os: For interacting with the operating system, e.g., creating directories.
- sys: For system-specific parameters and functions, e.g., exiting with error codes.
- GPUMonitor: Class for monitoring GPU metrics.
- VictoriaMetricsExporter: Class for exporting data to VictoriaMetrics.
- format_metrics: Utility function for formatting and saving metrics.

Usage:
- Run the script with appropriate arguments to monitor GPU metrics and optionally export data.
"""

import argparse
import os
import sys

from .carbon_metrics import get_carbon_region_names
from .gpu_monitor import GPUMonitor
from .gpu_victoria_exporter import VictoriaMetricsExporter
from .utils.globals import RESULTS_DIR, LOGGER, MONITOR_INTERVAL
from .utils.metric_utils import format_metrics
from .utils.docker_utils import image_exists, list_available_images

def parse_arguments():
    """
    Parses command-line arguments for the GPU monitoring script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Monitor GPU metrics and optionally export data to VictoriaMetrics.'
    )

    # Argument for enabling or disabling live monitoring
    parser.add_argument('--no_live_monitor', action='store_true',
                        help='Disable live monitoring of GPU metrics (default is enabled).')

    # Argument for setting the monitoring interval
    parser.add_argument('--interval', type=int, default=MONITOR_INTERVAL,
                        help='Interval in seconds for collecting GPU metrics (default is 5 seconds).')

    # Argument for specifying the carbon region
    parser.add_argument(
        '--carbon_region',
        type=str,
        default='South England',
        help='Region shorthand for The National Grid ESO Regional Carbon Intensity API (default is "South England").'
    )

    # Argument for enabling or disabling plotting
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting of GPU metrics (default is enabled).')

    # Argument for enabling live plotting
    parser.add_argument('--live_plot', action='store_true',
                        help='Enable live plotting of GPU metrics.')

    # Argument for enabling data export to VictoriaMetrics
    parser.add_argument('--export_to_victoria', action='store_true',
                        help='Enable exporting of collected data to VictoriaMetrics.')

    # Argument for specifying the Docker container image for benchmarking
    parser.add_argument('--benchmark_image', type=str, required=True,
                        help='Docker container image to run as a benchmark.')

    # Argument for monitoring both metrics and container logs
    parser.add_argument('--monitor_benchmark_logs', action='store_true',
                        help='Enable monitoring of container logs in addition to GPU metrics.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Validate the interval argument
    if args.interval <= 0:
        error_message = f"Monitoring interval must be a positive integer. Provided value: {args.interval}"
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Validate the carbon region argument
    valid_regions = get_carbon_region_names()
    if args.carbon_region not in valid_regions:
        error_message = (f"Invalid carbon region: {args.carbon_region}. Valid regions are: {', '.join(valid_regions)}")
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Check if the benchmark image exists
    if not image_exists(args.benchmark_image) or args.benchmark_image=="":
        print(f"Image '{args.benchmark_image}' is not valid.")
        LOGGER.error("Image '%s' does not exist.", args.benchmark_image)

        # List available images excluding those with "base" in their name
        available_images = list_available_images(exclude_base=True, exclude_none=True)
        print("Available images (excluding 'base' images):")
        for image in available_images:
            print(f"  - {image}")
        sys.exit(1)  # Exit with error code 1

    return args

def main():
    """
    Main function for running the GPU monitoring process.

    Parses command-line arguments, validates them, initializes the GPUMonitor,
    and handles data exporting to VictoriaMetrics if specified.
    """
    # Parse the command-line arguments
    args = parse_arguments()

    # Create an instance of GPUMonitor
    gpu_monitor = GPUMonitor(monitor_interval=args.interval,
                             carbon_region_shorthand=args.carbon_region)

    try:
        # Run the Monitoring process
        LOGGER.info("Starting GPU monitoring...")
        gpu_monitor.run(
            live_monitoring=not args.no_live_monitor,
            plot=not args.no_plot,
            live_plot=args.live_plot,
            benchmark_image=args.benchmark_image,
            monitor_logs=args.monitor_benchmark_logs
        )
        LOGGER.info("GPU monitoring completed.")

        # Save Monitor Results
        LOGGER.info("Saving monitoring results...")
        gpu_monitor.save_stats_to_yaml()
        LOGGER.info("Saving monitoring completed.")

        # Export data to VictoriaMetrics if specified
        if args.export_to_victoria:
            LOGGER.info("Starting data export to VictoriaMetrics...")
            exporter = VictoriaMetricsExporter(gpu_monitor.time_series_data)
            exporter.send_to_victoria()
            LOGGER.info("Data export to VictoriaMetrics completed.")

    except ValueError as value_error:
        LOGGER.error("Value error occurred: %s", value_error)
        print(f"Value error occurred: {value_error}")
        sys.exit(1)
    except FileNotFoundError as file_not_found_error:
        LOGGER.error("File not found: %s", file_not_found_error)
        print(f"File not found: {file_not_found_error}")
        sys.exit(1)
    except ConnectionError as connection_error:
        LOGGER.error("Connection error occurred: %s", connection_error)
        print(f"Connection error occurred: {connection_error}")
        sys.exit(1)
    except OSError as os_error:
        LOGGER.error("OS error occurred: %s", os_error)
        print(f"OS error occurred: {os_error}")
        sys.exit(1)

    # Output Formatted Results
    LOGGER.info("Formatting metrics...")
    #os.system('clear')
    format_metrics(results_dir=RESULTS_DIR)
    LOGGER.info("Metrics formatting completed.")

if __name__ == "__main__":
    main()
