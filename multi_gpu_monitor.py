"""
Collects live GPU metrics using nvidia-smi and carbon data produced by the
National Grid ESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python3 mutli_gpu_monitor.py [--live_monitor] [--interval INTERVAL] [--carbon_region REGION]

Parameters:
    --live_monitor: Enables live monitoring of GPU metrics. If not specified,
    live monitoring is disabled.
    --interval INTERVAL: Sets the interval (in seconds) for collecting GPU
    metrics. Default is 10 seconds.
    --carbon_region REGION: Specifies the region shorthand for the National
    Grid ESO Regional Carbon Intensity API. Default is 'South England'.

Example:
    python3 multi_gpu_monitor_class.py --live_monitor --interval 30 --carbon_region "North Scotland"

    This command enables live monitoring, sets the monitoring interval to 30
    seconds, and uses "North Scotland" as the carbon intensity region.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pynvml
import requests
import yaml
from tabulate import tabulate

# Ensure the results directory exists
os.makedirs("./results", exist_ok=True)

# Constants
CARBON_INTENSITY_URL = (
    "https://api.carbonintensity.org.uk/regional"
)
SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = './results/metrics.yml'
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
            'total_energy': []  # Added for storing total energy time series
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

        # Number of GPUS
        self.device_count = pynvml.nvmlDeviceGetCount()

        # Initialize pynvml
        pynvml.nvmlInit()
        LOGGER.info("NVML initialized")

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

            for i in range(self.self.device_count):
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
                self._time_series_data['total_energy'].append(self._stats['total_energy'])
            LOGGER.info("Updated GPU metrics: %s", self._time_series_data)

        except pynvml.NVMLError as error_message:
            LOGGER.error("NVML Error: %s", error_message)


    def __update_carbon_forecast(self) -> Optional[float]:
        """
        Uses The nationalgridESO Regional Carbon Intensity API to collect current carbon emissions.

        Returns:
            Optional[float]: Current carbon intensity.
        """
        try:
            response = requests.get(CARBON_INTENSITY_URL,
                                    headers={'Accept': 'application/json'},
                                    timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            regions = data['data'][0]['regions']

            for region in regions:
                if region['shortname'] == self.carbon_region_shorthand:
                    intensity = region['intensity']
                    carbon_forecast = float(intensity['forecast'])
                    LOGGER.info("Carbon forecast for '%s': %f",
                                self.carbon_region_shorthand, carbon_forecast)
                    return carbon_forecast

        except requests.exceptions.RequestException as error_message:
            LOGGER.error("Error request timed out (30s): %s", error_message)

        return None

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
        self._stats["end_carbon_forecast"] = self.__update_carbon_forecast()

        # Fill Total Energy and Total Carbon Estimations
        self._stats["av_carbon_forecast"] = (
            self._stats["start_carbon_forecast"] +
            self._stats["end_carbon_forecast"]
        ) / 2
        self._stats["total_carbon"] = (
            self._stats["total_energy"] * self._stats["av_carbon_forecast"]
        )

        # Compute average power/ utilization/ memory and temperature
        util = self._time_series_data["util"]
        power = self._time_series_data["power"]
        temp = self._time_series_data["temp"]
        mem = self._time_series_data["mem"]

        n_utilized = 0
        for gpu_idx in util:
            for reading in gpu_idx:
                if reading > 0:
                    self._stats["av_util"] += util[gpu_idx][reading]
                    self._stats["av_power"] += power[gpu_idx][reading]
                    self._stats["av_mem"] += temp[gpu_idx][reading]
                    self._stats["av_temp"] += mem[gpu_idx][reading]
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

    def plot_metrics(self):
        """
        Plot and save the GPU metrics to a file.
        """
        timestamps = self._time_series_data['timestamp']
        total_energy_data = self._time_series_data['total_energy']

        timestamps = [
            datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S') for ts in timestamps
        ]

        # Plot 2x2 grid of power, utilization, temperature, and memory
        fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
        fig1.tight_layout(pad=4.0)

        # Plot GPU power usage
        for i in range(self.device_count):
            axes[0, 0].plot(timestamps, [p[i] for p in self._time_series_data['power']], label=f'GPU {i}')
        axes[0, 0].axhline(y=self._stats["max_power_limit"], color='r',
                           linestyle='--', label='Power Limit')
        axes[0, 0].set_title('GPU Power Usage')
        #axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].legend()
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot GPU utilization
        for i in range(self.device_count):
            axes[0, 1].plot(timestamps, [u[i] for u in self._time_series_data['util']],
                            label=f'GPU {i}')
        axes[0, 1].set_title('GPU Utilization')
        # axes[0, 1].set_xlabel('Timestamp')
        axes[0, 1].set_ylabel('Utilization (%)')
        axes[0, 1].legend()
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot GPU temperature
        for i in range(self.device_count):
            axes[1, 0].plot(timestamps, [t[i] for t in self._time_series_data['temp']], label=f'GPU {i}')
        axes[1, 0].set_title('GPU Temperature')
        axes[1, 0].set_xlabel('Timestamp')
        axes[1, 0].set_ylabel('Temperature (C)')
        axes[1, 0].legend()
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot GPU memory usage
        for i in range(self.device_count):
            axes[1, 1].plot(timestamps, [m[i] for m in self._time_series_data['mem']], label=f'GPU {i}')
        axes[1, 1].axhline(y=self._stats["total_mem"], color='r',
                           linestyle='--', label='Total Memory')
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Timestamp')
        axes[1, 1].set_ylabel('Memory (MiB)')
        axes[1, 1].legend()
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Save the 2x2 grid plot
        plot_filename_1 = "./results/metrics_grid_plot.png"
        plt.savefig(plot_filename_1, bbox_inches='tight')
        plt.close(fig1)

        # Plot total energy
        fig2, ax = plt.subplots(figsize=(12, 6))
        if len(timestamps) > 1 and len(total_energy_data) == len(timestamps) - 1:
            ax.plot(timestamps[1:], total_energy_data, marker='o', color='blue')
            ax.set_title('Total Energy Consumed')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Energy (kWh)')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Data mismatch for total energy.', ha='center', va='center')

        # Save the total energy plot
        plot_filename_2 = "./results/total_energy_plot.png"
        plt.savefig(plot_filename_2, bbox_inches='tight')
        plt.close(fig2)

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



    def run(self, live_monitoring: bool = False, plot: bool = False) -> None:
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
                if plot:
                    self.plot_metrics()

                if live_monitoring:
                    self._live_monitor()

                # Wait for the defined collection interval before the next iteration
                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            # Handle interruption (e.g., Ctrl+C) by storing completion stats
            self.__completion_stats()
            LOGGER.info("Monitoring stopped.")
            print("\nMonitoring stopped.")

        finally:
            # Properly shut down pynvml
            pynvml.nvmlShutdown()
            LOGGER.info("NVML shutdown")

def fetch_carbon_region_names():
    """
    Retrieves and returns the short names of all regions from the carbon intensity API.

    Returns:
        List[str]: Short names of the regions.
    """
    try:
        response = requests.get(CARBON_INTENSITY_URL,
                                headers={'Accept': 'application/json'},
                                timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        # Extract the list of regions
        regions = data['data'][0]['regions']

        # Extract short names of all regions
        region_names = [region['shortname'] for region in regions]

        LOGGER.info("Extracted region names: %s", region_names)
        return region_names

    except requests.exceptions.RequestException as error_message:
        LOGGER.error("Error occurred during request (timeout %ds): %s",
                     TIMEOUT_SECONDS, error_message)
        return []

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
    monitor.run(live_monitoring=args.live_monitor,plot=args.plot)

    # Save collected metrics to a YAML file
    monitor.save_stats_to_yaml(METRICS_FILE_PATH)

if __name__ == "__main__":
    main()
