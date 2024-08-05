# main.py

import argparse
import logging
import sys
from gpu_monitor import GPUMonitor
from gpu_benchmark_metrics.gpu_monitoring.gpu_victoria_exporter import VictoriaMetricsExporter
from utils import setup_logging
from carbon_metrics import fetch_carbon_region_names

# Ensure the results directory exists
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    """
    Main function for parsing command-line arguments and running the GPUMonitor.
    """
    # Setup logging
    LOGGER = setup_logging()

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Monitor GPU metrics')

    # Argument for enabling live monitoring
    parser.add_argument('--live_monitor', action='store_true',
                        help='Enable live monitoring of GPU metrics.')

    # Argument for setting the monitoring interval
    parser.add_argument('--interval', type=int, default=1,
                        help='Interval in seconds for collecting GPU metrics (default is 1 second).')

    # Argument for specifying the carbon region
    parser.add_argument(
        '--carbon_region',
        type=str,
        default='South England',
        help='Region shorthand for The National Grid ESO Regional Carbon Intensity API (default is "South England").'
    )

    # Argument for enabling plotting
    parser.add_argument('--plot', action='store_true',
                        help='Enable plotting of GPU metrics.')

    # Argument for enabling live plotting
    parser.add_argument('--live_plot', action='store_true',
                        help='Enable live plotting of GPU metrics.')

    # Argument for enabling data export to VictoriaMetrics
    parser.add_argument('--export_to_victoria', action='store_true',
                        help='Enable exporting of collected data to VictoriaMetrics.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Validate the interval argument
    if args.interval <= 0:
        error_message = f"Monitoring interval must be a positive integer. Provided value: {args.interval}"
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Validate the carbon region argument
    valid_regions = fetch_carbon_region_names()
    if args.carbon_region not in valid_regions:
        error_message = (f"Invalid carbon region: {args.carbon_region}. Valid regions are: {', '.join(valid_regions)}")
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Create an instance of GPUMonitor
    gpu_monitor = GPUMonitor(monitor_interval=args.interval, carbon_region_shorthand=args.carbon_region)

    try:
        # Run the GPU monitoring process
        gpu_monitor.run(live_monitoring=args.live_monitor, plot=args.plot, live_plot=args.live_plot)

        # Export data to VictoriaMetrics if specified
        if args.export_to_victoria:
            exporter = VictoriaMetricsExporter(gpu_monitor.get_time_series_data())
            exporter.send_to_victoria()

    except Exception as e:
        LOGGER.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

