"""
Collects live GPU metrics using nvidia-smi and carbon data produced by the
National Grid ESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python3 multi_gpu_monitor.py [--live_monitor] [--interval INTERVAL] [--carbon_region REGION] [--plot] [--live_plot]

Parameters:
    --live_monitor: Enables live monitoring of GPU metrics. If not specified,
    live monitoring is disabled.

    --interval INTERVAL: Sets the interval (in seconds) for collecting GPU
    metrics. Default is 1 second.

    --carbon_region REGION: Specifies the region shorthand for the National
    Grid ESO Regional Carbon Intensity API. Default is 'South England'.

    --plot: Enables plotting of GPU metrics after monitoring. If not specified,
    plotting is disabled.

    --live_plot: Enables live plotting of GPU metrics. Note: Live plotting is not
    recommended as errors may arise if the code is interrupted during the plot saving
    process.

Example:
    python3 multi_gpu_monitor.py --live_monitor --interval 30 --carbon_region "North Scotland" --plot

    This command enables live monitoring, sets the monitoring interval to 30
    seconds, uses "North Scotland" as the carbon intensity region, and generates
    plots after collecting metrics.
"""

import argparse
import os
import sys
import time
from datetime import datetime
import logging
from typing import Optional

import matplotlib.figure as figure
import matplotlib.backends.backend_agg as agg
import matplotlib.ticker as ticker

import pynvml
import requests
import yaml
from tabulate import tabulate

# Ensure the results directory exists
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Constants
CARBON_INTENSITY_URL = (
    "https://api.carbonintensity.org.uk/regional"
)
SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = os.path.join(RESULTS_DIR, 'metrics.yml')
METRIC_PLOT_PATH = os.path.join(RESULTS_DIR, 'metric_plot.png')
TIMEOUT_SECONDS = 30

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./results/gpu_monitor.log',  # Specify the log file name
    filemode='w'  # Overwrite the file on each run; use 'a' to append
)
LOGGER = logging.getLogger(__name__)

class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML and collects carbon metrics from the
    National Grid ESO Regional Carbon Intensity API.
    """

    def __init__(self,
                 monitor_interval=1,
                 carbon_region_shorthand="South England"):
        """
        Initializes the GPUMonitor class.

        Args:
            monitor_interval (int): Interval in seconds for collecting GPU metrics.
            carbon_region_shorthand (str): Region for carbon intensity API.
        """
        self.monitor_interval = monitor_interval
        self.carbon_region_shorthand = carbon_region_shorthand

        # Initialize time series data for GPU metrics
        self._time_series_data = {
            'timestamp': [],
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': [],
            # 'total_energy': []  # Added for storing total energy time series
        }

        # Initialize private GPU metrics as a dict of Lists
        self.current_gpu_metrics = {
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': []
        }

        # Initialize Previous Power
        self.previous_power = []

        # Initialize Previous Power
        self.previous_power = []

        # Initialize stats
        self._stats = {}

        # Initialize pynvml
        pynvml.nvmlInit()
        LOGGER.info("NVML initialized")

        # Number of GPUS
        self.device_count = pynvml.nvmlDeviceGetCount()

    def __setup_stats(self) -> None:
        """
        Initializes and returns a dictionary with GPU statistics and default values.
        """
        # Find The First GPU's Name and Max Power
        first_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(first_handle)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(first_handle)
        power_limit = power_limit / 1000.0  # Convert from mW to W
        total_memory_info = pynvml.nvmlDeviceGetMemoryInfo(first_handle)
        total_memory = total_memory_info.total / (1024 ** 2)  # Convert bytes to MiB

        # Collect initial carbon forecast
        carbon_forecast = self.__update_carbon_forecast()

        self._stats = {
            "av_temp": 0.0,
            "av_util": 0.0,
            "av_mem": 0.0,
            "av_power": 0.0,
            "av_carbon_forecast": 0.0,
            "end_datetime": '',
            "end_carbon_forecast": 0.0,
            "max_power_limit": power_limit,
            "name": gpu_name,
            "start_carbon_forecast": carbon_forecast,
            "start_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_carbon": 0.0,
            "total_energy": 0.0,
            "total_mem": total_memory,
        }
        LOGGER.info("Statistics initialized: %s", self._stats)

    def __update_gpu_metrics(self) -> None:
        """
        Retrieves the current GPU metrics for all GPUs and updates the internal time series data.

        This method updates `self._time_series_data` with the following information:
            - 'timestamp': List of timestamps
            - 'gpu_idx': List of GPU indices
            - 'util': List of GPU utilization percentages
            - 'power': List of GPU power usage in watts
            - 'temp': List of GPU temperatures in degrees Celsius
            - 'mem': List of used GPU memory in MiB
            - 'total_energy': List of total energy consumed in kWh
        """
        try:

            # Store previous power readings
            # Store previous power readings
            self.previous_power = self.current_gpu_metrics['power']

            # Retrieve metrics for each GPU
            timestamps = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_gpu_metrics['gpu_idx'] = []
            self.current_gpu_metrics['util'] = []
            self.current_gpu_metrics['power'] = []
            self.current_gpu_metrics['temp'] = []
            self.current_gpu_metrics['mem'] = []

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                # Append metrics directly to the dictionary lists
                self.current_gpu_metrics['gpu_idx'].append(i)
                self.current_gpu_metrics['util'].append(utilization)
                self.current_gpu_metrics['power'].append(power_usage)
                self.current_gpu_metrics['temp'].append(temperature)
                self.current_gpu_metrics['mem'].append(memory)

            # Append new data to time series
            self._time_series_data['timestamp'].append(timestamps)
            self._time_series_data['gpu_idx'].append(self.current_gpu_metrics['gpu_idx'])
            self._time_series_data['util'].append(self.current_gpu_metrics['util'])
            self._time_series_data['power'].append(self.current_gpu_metrics['power'])
            self._time_series_data['temp'].append(self.current_gpu_metrics['temp'])
            self._time_series_data['mem'].append(self.current_gpu_metrics['mem'])

            # Update total energy and append to time series
            if len(self.previous_power) > 0:
                self.__update_total_energy()
                # self._time_series_data['total_energy'].append(self._stats['total_energy'])
            LOGGER.info("Updated GPU metrics: %s", self._time_series_data)

        except pynvml.NVMLError as error_message:
            LOGGER.error("NVML Error: %s", error_message)

    def __update_total_energy(self) -> None:
        """
        Computes the total energy consumed by all GPUs.

        Updates the total energy consumed and stores it in `self._stats`.
        """
        
        current_power = self.current_gpu_metrics['power']

        # Check if previous_power and current_power have the same length
        if len(self.previous_power) != self.device_count or len(current_power) != self.device_count:
            LOGGER.error(
                "Length of previous_power or current_power does not match the number of devices."
            )
            raise ValueError(
                "Length of previous_power or current_power does not match the number of devices."
            )

        # Convert Collection Interval from Seconds to Hours
        collection_interval_h = self.monitor_interval / SECONDS_IN_HOUR

        # Calculate total energy consumed in kWh
        energy_wh = sum(
            ((prev + curr) / 2) * collection_interval_h
            for prev, curr in zip(self.previous_power, current_power)
        )
        energy_kwh = energy_wh / 1000.0  # Convert Wh to kWh

        # Update total energy in stats
        self._stats["total_energy"] += energy_kwh
        LOGGER.info("Updated total energy: %f kWh", self._stats['total_energy'])

        # Update previous power for the next interval
        self.previous_power = current_power


    def __completion_stats(self) -> None:
        """
        Calculates and Updates completion metrics.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        # First Fill Carbon Forecast End time and End Forecast
        self._stats["end_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stats["end_carbon_forecast"] = self.update_carbon_forecast()

        # Fill Total Energy and Total Carbon Estimations
        self._stats["av_carbon_forecast"] = (
            self._stats["start_carbon_forecast"] +
            self._stats["end_carbon_forecast"]
        ) / 2
        self._stats["total_carbon"] = (
            self._stats["total_energy"] * self._stats["av_carbon_forecast"]
        )

        # Number of Readings where GPUs are being utilized
        n_utilized = 0

        # Compute average power/ utilization/ memory and temperature whilst gpus
        # are being utilized
        util = self._time_series_data["util"]
        power = self._time_series_data["power"]
        temp = self._time_series_data["temp"]
        mem = self._time_series_data["mem"]

        for gpu_idx, gpu_util in enumerate(util):  # Iterate over each GPU index and its utilization
            for reading_idx, util_reading in enumerate(gpu_util):  # Iterate over each reading
                if util_reading > 0:
                    self._stats["av_util"] += util_reading
                    self._stats["av_power"] += power[gpu_idx][reading_idx]
                    self._stats["av_mem"] += mem[gpu_idx][reading_idx]
                    self._stats["av_temp"] += temp[gpu_idx][reading_idx]
                    n_utilized += 1

        # Avoid division by zero
        if n_utilized > 0:
            self._stats["av_util"] /= n_utilized
            self._stats["av_power"] /= n_utilized
            self._stats["av_mem"] /= n_utilized
            self._stats["av_temp"] /= n_utilized

        LOGGER.info("Completion stats updated: %s", self._stats)

        LOGGER.info("Completion statistics updated: %s", self._stats)

    def save_stats_to_yaml(self, file_path: str):
        """
        Saves stats to a YAML file.

        Args:
            file_path (str): Path to the YAML file.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as yaml_file:
                yaml.dump(self._stats, yaml_file, default_flow_style=False)
            LOGGER.info("Stats saved to YAML file: %s", file_path)
        except IOError as io_error:
            LOGGER.error("Failed to save stats to YAML file: %s. Error: %s",
                         file_path, io_error)

    @staticmethod
    def plot_metric(ax, data, line_info=None, ylim=None):
        """
        Helper function to plot a GPU metric on a given axis.
        """
        timestamps, y_data, title, ylabel, xlabel = data

        for i, gpu_data in enumerate(y_data):
            ax.plot(timestamps, gpu_data, label=f'GPU {i}', marker='*')

        if line_info:
            yline, yline_label = line_info
            ax.axhline(y=yline, color='r', linestyle='--', label=yline_label)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(axis='x', rotation=45)

        if ylim:
            ax.set_ylim(ylim)

    def plot_metrics(self):
        """
        Plot and save the GPU metrics to a file.
        """
        timestamps = self._time_series_data['timestamp']

        # Prepare data for plotting
        power_data = [[p[i] for p in self._time_series_data['power']] for i in range(self.device_count)]
        util_data = [[u[i] for u in self._time_series_data['util']] for i in range(self.device_count)]
        temp_data = [[t[i] for t in self._time_series_data['temp']] for i in range(self.device_count)]
        mem_data = [[m[i] for m in self._time_series_data['mem']] for i in range(self.device_count)]

        # Create a new figure and axes
        fig = figure.Figure(figsize=(20, 15))
        axes = fig.subplots(nrows=2, ncols=2)

        # Create a backend for rendering the plot
        canvas = agg.FigureCanvasAgg(fig)

        # Plot each metric using the helper function
        self.plot_metric(
            axes[0, 0],
            (timestamps,
             power_data,
             f'GPU Power Usage, Total Energy: {self._stats["total_energy"]:.3g}kWh',
             'Power (W)', 'Timestamp'),
            (self._stats["max_power_limit"], 'Power Limit')
        )
        self.plot_metric(
            axes[0, 1],
            (timestamps, util_data, 'GPU Utilization', 'Utilization (%)', 'Timestamp'),
            ylim=(0, 100)  # Set y-axis limits for utilization
        )
        self.plot_metric(
            axes[1, 0],
            (timestamps, temp_data, 'GPU Temperature', 'Temperature (C)', 'Timestamp')
        )
        self.plot_metric(
            axes[1, 1],
            (timestamps, mem_data, 'GPU Memory Usage', 'Memory (MiB)', 'Timestamp'),
            (self._stats["total_mem"], 'Total Memory')
        )

        # Ensure tight_layout is applied
        fig.tight_layout(pad=3.0)

        # Save the 2x2 grid plot
        canvas.draw()  # Render the figure to the canvas
        canvas.figure.savefig(METRIC_PLOT_PATH, bbox_inches='tight')  # Save the plot as PNG


        # Close the figure to free memory
        canvas.draw()  # Ensure the figure is fully rendered
        del fig  # Remove reference to figure to free memory

    def _live_monitor(self) -> None:
        """
        Clears the terminal and prints the current GPU metrics in a formatted table.

        This method clears the terminal screen, fetches the current date and time,
        and then prints the GPU metrics as a grid table with headers.
        """
        os.system('clear')
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Define headers for the table
        headers = [
            f'GPU Index ({self._stats["name"]})',
            'Utilization (%)',
            f"Power (W) / Max {self._stats['max_power_limit']} W",
            'Temperature (C)',
            f"Memory (MiB) / Total {self._stats['total_mem']} MiB"
        ]

        # Print current GPU metrics with date and time
        print(
            f"Current GPU Metrics as of {current_time}:\n"
            f"{tabulate(self.current_gpu_metrics, headers=headers, tablefmt='grid')}"
        )


    def run(self, live_monitoring: bool = False, plot: bool = False,
            live_plot: bool = False) -> None:
        """
        Runs the GPU monitoring and plotting process.
        """
        # Print the metrics as a formatted table
        try:
            # Initialize statistics and metrics
            self.__setup_stats()

            # Start the monitoring loop
            while True:
                # Update GPU metrics
                self.__update_gpu_metrics()

                # Update the plot every iteration or after a certain interval
                if live_plot:
                    try:
                        self.plot_metrics()
                    except (FileNotFoundError, IOError) as plot_error:
                        LOGGER.error("Error during plotting: %s", plot_error)
                        continue  # Skip plotting and continue monitoring

                if live_monitoring:
                    self._live_monitor()

                # Wait for the defined collection interval before the next iteration
                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            # Handle interruption (e.g., Ctrl+C) by storing completion stats
            self.__completion_stats()
            LOGGER.info("Monitoring stopped.")
            print("\nMonitoring stopped.")

            if plot:
                try:
                    self.plot_metrics()
                except (FileNotFoundError, IOError) as plot_error:
                    LOGGER.error("Error during plotting: %s", plot_error)

        finally:
            # Properly shut down pynvml
            pynvml.nvmlShutdown()
            LOGGER.info("NVML shutdown")

def main():
    """
    Main function for parsing command-line arguments and running the GPUMonitor.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Monitor GPU metrics')

    # Argument for enabling live monitoring
    parser.add_argument('--live_monitor', action='store_true',
                        help='Enable live monitoring of GPU metrics.')

    # Argument for setting the monitoring interval
    parser.add_argument('--interval', type=int, default=1,
                        help='Interval in seconds for collecting GPU metrics'
                             '(default is 1 seconds).')

    # Argument for specifying the carbon region
    parser.add_argument(
        '--carbon_region',
        type=str,
        default='South England',
        help='Region shorthand for The National Grid ESO Regional Carbon Intensity API '
             '(default is "South England").'
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
        error_message = f"Monitoring interval must be a positive integer. " \
                        f"Provided value: {args.interval}"
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Validate the carbon region argument
    valid_regions = fetch_carbon_region_names()
    if args.carbon_region not in valid_regions:
        error_message = (f"Invalid carbon region. Provided value: '{args.carbon_region}'. "
                         f"Valid options are: {', '.join(valid_regions)}")
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)  # Exit with error code 1

    # Initialize the GPUMonitor with parsed arguments
    monitor = GPUMonitor(
        monitor_interval=args.interval,
        carbon_region_shorthand=args.carbon_region
    )

    # Start monitoring with the specified options
    monitor.run(live_monitoring=args.live_monitor,
                plot=args.plot,
                live_plot=args.live_plot
                )

    # Save collected metrics to a YAML file
    monitor.save_stats_to_yaml(METRICS_FILE_PATH)

    # Check if data export to VictoriaMetrics is enabled
    if args.export_to_victoria:
        LOGGER.info("Exporting data to VictoriaMetrics")
        exporter = VictoriaMetricsExporter(monitor._time_series_data)
        #exporter.send_to_victoria()
        LOGGER.info("Data export to VictoriaMetrics completed")

if __name__ == "__main__":
    main()
