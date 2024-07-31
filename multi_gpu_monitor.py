"""
Collects live GPU metrics using nvidia-smi and carbon data produced by the
National Grid ESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python gpu_monitor.py [--live_monitor] [--interval INTERVAL] [--carbon_region REGION]

Parameters:
    --live_monitor: Enables live monitoring of GPU metrics. If not specified,
    live monitoring is disabled.
    --interval INTERVAL: Sets the interval (in seconds) for collecting GPU
    metrics. Default is 10 seconds.
    --carbon_region REGION: Specifies the region shorthand for the National
    Grid ESO Regional Carbon Intensity API. Default is 'South England'.

Example:
    python gpu_monitor_class.py --live_monitor --interval 30 --carbon_region "North Scotland"

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

import pynvml
import requests
import yaml
from prometheus_client import CollectorRegistry, Gauge, start_http_server
from tabulate import tabulate

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

class PrometheusServer:
    """
    Manages the setup and operation of a Prometheus HTTP server.
    """

    DEFAULT_PROMETHEUS_SERVER_CONFIG = {
        'port': 8000,
        'addr': '0.0.0.0'
    }

    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initializes the PrometheusServer class.

        Args:
            config (dict, optional): Configuration for the Prometheus server.
        """
        self.config = config or self.DEFAULT_PROMETHEUS_SERVER_CONFIG
        self.registry = CollectorRegistry()
        self.gauges = {}

    def start(self):
        """
        Starts the Prometheus HTTP server.
        """
        self.__validate_config()
        start_http_server(
            port=self.config['port'],
            addr=self.config['addr'],
            registry=self.registry
        )
        LOGGER.info("Prometheus server started on %s:%s",
                    self.config['addr'], self.config['port'])

    def add_metric(self, name: str, description: str, labels: List[str]) -> None:
        """
        Adds a new metric to the Prometheus server.

        Args:
            name (str): The name of the metric.
            description (str): A description of the metric.
            labels (list of str): List of labels for the metric.
        """
        self.gauges[name] = Gauge(
            name, description, labelnames=labels, registry=self.registry
        )
        LOGGER.info("Added metric '%s' with description '%s' and labels %s",
                    name, description, labels)

    def update_metric(self, name: str, value: float, **labels) -> None:
        """
        Updates the value of an existing metric.

        Args:
            name (str): The name of the metric to update.
            value (float): The new value for the metric.
            **labels: The labels for the metric.
        """
        if name in self.gauges:
            self.gauges[name].labels(**labels).set(value)
            LOGGER.info("Updated metric '%s' to value %f with labels %s",
                        name, value, labels)
        else:
            LOGGER.error("Metric '%s' does not exist.", name)
            raise ValueError(f"Metric '{name}' does not exist.")

    def __validate_config(self) -> None:
        """
        Validates the configuration settings for the Prometheus server.
        """
        if not isinstance(self.config['port'], int) or not 1 <= self.config['port'] <= 65535:
            LOGGER.error("Port must be an integer between 1 and 65535.")
            raise ValueError("Port must be an integer between 1 and 65535.")
        if not isinstance(self.config['addr'], str) or not self.config['addr']:
            LOGGER.error("Address must be a non-empty string.")
            raise ValueError("Address must be a non-empty string.")

class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML and collects carbon metrics from the
    The nationalgridESO Regional Carbon Intensity API.
    """

    def __init__(self,
                 monitor_interval=1,
                 carbon_region_shorthand="South England",
                 prometheus_config=None):
        """
        Initializes the GPUMonitor class.

        Args:
            monitor_interval (int): Interval in seconds for collecting GPU metrics.
            carbon_region_shorthand (str): Region for carbon intensity API.
            prometheus_config (dict, optional): Configuration for the Prometheus server.
        """
        self.monitor_interval = monitor_interval
        self.carbon_region_shorthand = carbon_region_shorthand

        # Initialize private GPU metrics as a dict of Lists
        self._gpu_metrics = {
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': []
        }

        # Initialize Previous Power
        self.previous_power = []

        # Initialize stats
        self._stats = {}

        # Initialize pynvml
        pynvml.nvmlInit()
        LOGGER.info("NVML initialized")

        # Set up Prometheus server
        self.prometheus_server = PrometheusServer(config=prometheus_config)
        LOGGER.info("Prometheus server configured")

        # Set up Prometheus metrics
        self.__setup_prometheus_metrics()

        # Start the Prometheus HTTP server
        self.prometheus_server.start()

    def __setup_stats(self) -> None:
        """
        Initializes and returns a dictionary with GPU statistics and default values.

        The dictionary has the following keys and default values:

        - 'av_carbon_forecast': Average carbon forecast in grams of CO2 per kWh (default 0.0).
        - 'end_datetime': End date and time of the measurement period (default '').
        - 'end_carbon_forecast': Forecasted carbon intensity in grams of CO2 per
            kWh at the end time (default 0.0).
        - 'max_power_limit': Maximum power limit of the GPU in watts (retrieved from GPU).
        - 'name': Name of the GPU device (retrieved from GPU).
        - 'start_carbon_forecast': Forecasted carbon intensity in grams of CO2
            per kWh at the start time (retrieved during setup).
        - 'start_datetime': Start date and time of the measurement period
            (default to current datetime).
        - 'total_carbon': Total carbon emissions in grams of CO2 (default 0.0).
        - 'total_energy': Total energy consumed in kilowatt-hours (default 0.0).
        - 'total_mem': Total memory of the GPU in MiB (retrieved from GPU).
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

    def __setup_prometheus_metrics(self):
        """
        Sets up Prometheus metrics.
        """
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            # Create metrics for each GPU
            self.prometheus_server.add_metric(
                name=f'gpu_{i}_power_usage',
                description='GPU Power Usage (W)',
                labels=['gpu_index']
            )
            self.prometheus_server.add_metric(
                name=f'gpu_{i}_utilization',
                description='GPU Utilization (%)',
                labels=['gpu_index']
            )
            self.prometheus_server.add_metric(
                name=f'gpu_{i}_temperature',
                description='GPU Temperature (C)',
                labels=['gpu_index']
            )
            self.prometheus_server.add_metric(
                name=f'gpu_{i}_memory_usage',
                description='GPU Memory Usage (MiB)',
                labels=['gpu_index']
            )
        LOGGER.info("Prometheus metrics setup complete")

    def __update_gpu_metrics(self) -> None:
        """
        Retrieves the current GPU metrics for all GPUs and updates the internal dictionary.

        This method updates `self._gpu_metrics` with the following information:
            - 'gpu_idx': List of GPU indices
            - 'util': List of GPU utilization percentages
            - 'power': List of GPU power usage in watts
            - 'temp': List of GPU temperatures in degrees Celsius
            - 'mem': List of used GPU memory in MiB
        """
        try:
            device_count = pynvml.nvmlDeviceGetCount()

            # Store previous power readings
            self.previous_power = self._gpu_metrics['power']

            # Clear existing lists in the dictionary to start fresh
            self._gpu_metrics['gpu_idx'] = []
            self._gpu_metrics['util'] = []
            self._gpu_metrics['power'] = []
            self._gpu_metrics['temp'] = []
            self._gpu_metrics['mem'] = []

            # Retrieve metrics for each GPU and append to the dictionary lists
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                # Append metrics directly to the dictionary lists
                self._gpu_metrics['gpu_idx'].append(i)
                self._gpu_metrics['util'].append(utilization)
                self._gpu_metrics['power'].append(power)
                self._gpu_metrics['temp'].append(temperature)
                self._gpu_metrics['mem'].append(memory)

            # Call a method to perform calculations based on updated metrics
            if len(self.previous_power) > 0:
                self.__update_total_energy()
            LOGGER.info("Updated GPU metrics: %s", self._gpu_metrics)

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
        """
        device_count = pynvml.nvmlDeviceGetCount()
        current_power = self._gpu_metrics['power']

        # Check if previous_power and current_power have the same length
        if len(self.previous_power) != device_count or len(current_power) != device_count:
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

    def __update_prometheus_metrics(self):
        """
        Updates Prometheus metrics with the latest GPU data.
        """
        # Retrieve GPU indices
        gpu_indices = self._gpu_metrics['gpu_idx']

        # Update all metrics for each GPU index
        for gpu_index in gpu_indices:
            gpu_str = str(gpu_index)
            # Retrieve the metrics for the current GPU index
            utilization = self._gpu_metrics['util'][gpu_index]
            power = self._gpu_metrics['power'][gpu_index]
            temperature = self._gpu_metrics['temp'][gpu_index]
            memory = self._gpu_metrics['mem'][gpu_index]

            # Update the metrics using PrometheusServer methods
            self.prometheus_server.update_metric(
                name=f'gpu_{gpu_index}_power_usage',
                value=power,
                gpu_index=gpu_str
            )
            self.prometheus_server.update_metric(
                name=f'gpu_{gpu_index}_utilization',
                value=utilization,
                gpu_index=gpu_str
            )
            self.prometheus_server.update_metric(
                name=f'gpu_{gpu_index}_temperature',
                value=temperature,
                gpu_index=gpu_str
            )
            self.prometheus_server.update_metric(
                name=f'gpu_{gpu_index}_memory_usage',
                value=memory,
                gpu_index=gpu_str
            )
        LOGGER.info("Prometheus metrics updated")

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
        LOGGER.info("Completion stats updated: %s", self._stats)

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
            f"{tabulate(self._gpu_metrics, headers=headers, tablefmt='grid')}"
        )

    def run(self, live_monitoring: bool = False) -> None:
        """
        Starts the GPU monitoring and Prometheus HTTP server with HTTPS.

        This method initializes and runs an HTTP server that exposes GPU metrics via
        Prometheus.
        """

        # Print the metrics as a formatted table
        try:
            # Initialize statistics and metrics
            self.__setup_stats()

            # Start the monitoring loop
            while True:
                # Update GPU metrics
                self.__update_gpu_metrics()

                # Update Prometheus metrics with the latest GPU data
                self.__update_prometheus_metrics()

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

if __name__ == "__main__":
    # Create an argument parser
    PARSER = argparse.ArgumentParser(description='Monitor GPU metrics')

    # Argument for enabling live monitoring
    PARSER.add_argument('--live_monitor', action='store_true',
                        help='Enable live monitoring of GPU metrics.')

    # Argument for setting the monitoring interval
    PARSER.add_argument('--interval', type=int, default=10,
                        help='Interval in seconds for collecting GPU metrics'
                             '(default is 10 seconds).')

    # Argument for specifying the carbon region
    PARSER.add_argument(
        '--carbon_region',
        type=str,
        default='South England',
        help='Region shorthand for The National Grid ESO Regional Carbon Intensity API '
             '(default is "South England").'
    )

    # Parse the command-line arguments
    ARGS = PARSER.parse_args()

    # Validate the interval argument
    if ARGS.interval <= 0:
        ERROR_MESSAGE = f"Monitoring interval must be a positive integer. " \
                        f"Provided value: {ARGS.interval}"
        print(ERROR_MESSAGE)
        LOGGER.error(ERROR_MESSAGE)
        sys.exit(1)  # Exit with error code 1

    # Validate the carbon region argument
    VALID_REGIONS = fetch_carbon_region_names()
    if ARGS.carbon_region not in VALID_REGIONS:
        ERROR_MESSAGE = (f"Invalid carbon region. Provided value: '{ARGS.carbon_region}'. "
                         f"Valid options are: {', '.join(VALID_REGIONS)}")
        print(ERROR_MESSAGE)
        LOGGER.error(ERROR_MESSAGE)
        sys.exit(1)  # Exit with error code 1

    # Initialize the GPUMonitor with parsed arguments
    MONITOR = GPUMonitor(
        monitor_interval=ARGS.interval,
        carbon_region_shorthand=ARGS.carbon_region
    )

    # Start monitoring with the specified options
    MONITOR.run(live_monitoring=ARGS.live_monitor)

    # Save collected metrics to a YAML file
    MONITOR.save_stats_to_yaml(METRICS_FILE_PATH)
