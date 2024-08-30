"""
This script is the entry point for monitoring GPU metrics using the GPUMonitor class.
It uses the argument parser from utils.cli to handle command-line arguments,
initializes the GPU monitoring process, and optionally exports the collected data to MeerkatDB.

Dependencies:
- utils.cli: For parsing command-line arguments.
- gpu_monitor: Contains the GPUMonitor class for monitoring GPU metrics.
- gpu_meerkat_exporter: Contains the MeerkatExporter class for exporting data to MeerkatDB.
- utils.globals: For accessing global constants.
- utils.metric_utils: For formatting and saving metrics.

Usage:
- Run the script with appropriate arguments to monitor GPU metrics and optionally export data.
"""

import sys

from .gpu_monitor import GPUMonitor
from .carbon_metrics import get_carbon_region_names
from .utils.cli import parse_arguments
from .utils.globals import RESULTS_DIR, LOGGER
from .utils.metric_utils import format_metrics


def main():
    """
    Main function for Iris-gpubench for running the GPU monitoring process.

    Parses command-line arguments, validates them, initializes the GPUMonitor,
    and handles data exporting to MeerkatDB if specified.
    """
    # Parse the command-line arguments
    args = parse_arguments(get_carbon_region_names)

    # Create an instance of GPUMonitor
    gpu_monitor = GPUMonitor(monitor_interval=args.interval,
                             carbon_region_shorthand=args.carbon_region)

    try:
        # Run the monitoring process
        LOGGER.info("Starting GPU monitoring...")
        gpu_monitor.run(
            benchmark_command=args.benchmark_command,
            benchmark_image=args.benchmark_image,
            live_monitoring=not args.no_live_monitor,
            plot=not args.no_plot,
            live_plot=args.live_plot,
            monitor_logs=args.monitor_logs,
            export_to_meerkat=args.export_to_meerkat
        )
        LOGGER.info("GPU monitoring completed.")

        # Save monitoring results
        LOGGER.info("Saving monitoring results...")
        gpu_monitor.save_stats_to_yaml()
        LOGGER.info("Saving monitoring completed.")

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

    # Output formatted results
    LOGGER.info("Formatting metrics...")
    format_metrics(results_dir=RESULTS_DIR)
    LOGGER.info("Metrics formatting completed.")

if __name__ == "__main__":
    main()
