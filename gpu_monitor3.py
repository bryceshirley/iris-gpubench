"""
Collects live GPU metrics using nvidia-smi and carbon produced by GPU using
The nationalgridESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python gpu_monitor.py

Parameters:
    CARBON_INSTENSITY_REGION_SHORTHAND: The region for the The nationalgridESO
                                        Regional Carbon Intensity API
"""

import time
from typing import Dict, Optional, List
from datetime import datetime

import pynvml
import requests
import yaml
from prometheus_client import CollectorRegistry, Gauge, start_http_server

# Configuration for the Prometheus HTTP server
PROMETHEUS_SERVER_CONFIG = {
    'port': 8000, # Port for the Prometheus metrics HTTP server
    'addr': '0.0.0.0',
    'certfile': 'server.cert', # Path to your certificate file
    'keyfile': 'server.key' # Path to your private key file
}

# Constants
CARBON_INTENSITY_URL = "https://api.carbonintensity.org.uk/regional"
SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = './results/metrics.yml'

class PrometheusServer:
    """
    Manages the setup and operation of a Prometheus HTTP server.
    """

    DEFAULT_CONFIG = {
        'port': 8000,
        'addr': '0.0.0.0',
        'certfile': 'server.cert',
        'keyfile': 'server.key'
    }

    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initializes the PrometheusServer class.

        Args:
            config (dict, optional): Configuration for the Prometheus server.
        """
        self.config = config or self.DEFAULT_CONFIG
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
            registry=self.registry,
            certfile=self.config['certfile'],
            keyfile=self.config['keyfile']
        )

    def add_metric(self, name: str, description: str, labels: List[str]) -> None:
        """
        Adds a new metric to the Prometheus server.

        Args:
            name (str): The name of the metric.
            description (str): A description of the metric.
            labels (list of str): List of labels for the metric.
        """
        self.gauges[name] = Gauge(name, description, labelnames=labels, registry=self.registry)

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
        else:
            raise ValueError(f"Metric '{name}' does not exist.")

    def __validate_config(self) -> None:
        """
        Validates the configuration settings for the Prometheus server.
        """
        if not isinstance(self.config['port'], int) or not (1 <= self.config['port'] <= 65535):
            raise ValueError("Port must be an integer between 1 and 65535.")
        if not isinstance(self.config['addr'], str) or not self.config['addr']:
            raise ValueError("Address must be a non-empty string.")
        if not isinstance(self.config['certfile'], str) or not self.config['certfile']:
            raise ValueError("Certificate file path must be a non-empty string.")
        if not isinstance(self.config['keyfile'], str) or not self.config['keyfile']:
            raise ValueError("Key file path must be a non-empty string.")


class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML.
    """

    def __init__(self,
                 collect_interval=1,
                 carbon_region_shorthand="South England",
                 prometheus_config=None):
        """
        Initializes the GPUMonitor class.

        Args:
            collect_interval (int): Interval in seconds for collecting GPU metrics.
            port (int): Port for the Prometheus metrics HTTP server.
        """
        self.collect_interval = collect_interval/ SECONDS_IN_HOUR # Convert to hours
        self.carbon_region_shorthand = carbon_region_shorthand

        # Initialize private GPU metrics as a dict of Lists
        self._gpu_metrics = {
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': []
        }

        # Iniatialize Previous Power
        self.previous_power = []

        # Initialize stats
        self._stats = {}

        # Initialize pynvml
        pynvml.nvmlInit()

        # Set up Prometheus server
        self.prometheus_server = PrometheusServer(config=prometheus_config)
        self.__setup_prometheus_metrics()
        self.prometheus_server.start()

    def __setup_stats(self) -> None:
        """
        Initializes and returns a dictionary with GPU statistics and default values.

        The dictionary has the following keys and default values:

        - 'av_load': Average GPU load as a percentage (default 0.0).
        - 'av_power': Average power consumption in watts (default 0.0).
        - 'av_temp': Average GPU temperature in degrees Celsius (default 0.0).
        - 'av_mem': Average GPU memory usage in MiB (default 0.0).
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
        power_limit = power_limit/ 1000.0  # Convert from mW to W
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

    def __setup_prometheus_metrics(self):
        """
        Sets up Prometheus metrics.
        """
        self.gpu_metrics_gauges = []  # List to hold metrics gauges for each GPU

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            # Create gauges for each GPU
            self.gpu_metrics_gauges.append({
                'power_gauge': Gauge(f'gpu_{i}_power_usage',
                                     'GPU Power Usage (W)',
                                     labelnames=['gpu_index'],
                                     registry=self.registry),
                'utilization_gauge': Gauge(f'gpu_{i}_utilization',
                                           'GPU Utilization (%)',
                                           labelnames=['gpu_index'],
                                           registry=self.registry),
                'temperature_gauge': Gauge(f'gpu_{i}_temperature',
                                           'GPU Temperature (C)',
                                           labelnames=['gpu_index'],
                                           registry=self.registry),
                'memory_gauge': Gauge(f'gpu_{i}_memory_usage',
                                      'GPU Memory Usage (MiB)',
                                      labelnames=['gpu_index'],
                                      registry=self.registry)
            })

    def __update_gpu_metrics(self) -> None:
        """
        Retrieves the current GPU metrics for all GPUs and updates the internal dictionary.

        This method updates `self._gpu_metrics` with the following information:
            - 'gpu_idx': List of GPU indices
            - 'util': List of GPU utilization percentages
            - 'power': List of GPU power usage in watts
            - 'temp': List of GPU temperatures in degrees Celsius
            - 'mem': List of used GPU memory in MiB

        It does not return any value.

        Raises:
            pynvml.NVMLError: If there is an error accessing GPU metrics.
        """
        try:
            device_count = pynvml.nvmlDeviceGetCount()

            # Store previous power readings
            previous_power_readings = self._gpu_metrics['power']

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
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                # Append metrics directly to the dictionary lists
                self._gpu_metrics['gpu_idx'].append(i)
                self._gpu_metrics['util'].append(utilization)
                self._gpu_metrics['power'].append(power)
                self._gpu_metrics['temp'].append(temperature)
                self._gpu_metrics['mem'].append(memory)

            # Call a method to perform calculations based on updated metrics
            self.__update_total_energy(previous_power_readings)

        except pynvml.NVMLError as error_message:
            print(f"NVML Error: {error_message}")

    def __update_carbon_forecast(self) -> Optional[float]:
        """
        Uses The nationalgridESO Regional Carbon Intensity API to collect current carbon emissions.

        Returns:
            Tuple[Optional[float], Optional[str]]: Current carbon intensity, carbon index.
        """
        timeout_seconds = 30
        try:
            response = requests.get(CARBON_INTENSITY_URL,
                                    headers={'Accept': 'application/json'},
                                    timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            regions = data['data'][0]['regions']

            for region in regions:
                if region['shortname'] == self.carbon_region_shorthand:
                    intensity = region['intensity']
                    carbon_forecast = float(intensity['forecast'])

                    return carbon_forecast

        except requests.exceptions.RequestException as error_message:
            print(f"Error request timed out (30s): {error_message}")

        return None

    def __update_total_energy(self) -> None:
        """
        Computes the total energy consumed by all GPUs.

        Args:
            previous_power (List[float]): A list of previous power readings in watts.
        """
        device_count = pynvml.nvmlDeviceGetCount()
        current_power = self._gpu_metrics['power']

        # Check if previous_power and current_power have the same length
        if len(self.previous_power) != device_count or len(current_power) != device_count:
            raise ValueError(
                "Length of previous_power or current_power does not match the number of devices."
            )

        # Calculate total energy consumed in kWh
        energy_wh = sum(
            ((prev + curr) / 2) * self.collect_interval
            for prev, curr in zip(self.previous_power, current_power)
        )
        energy_kwh = energy_wh/1000.0  # Convert Wh to kWh

        # Update total energy in stats
        self._stats["total_energy"] += energy_kwh

    def __update_prometheus_metrics(self):
        """
        Updates Prometheus metrics with the latest GPU data.
        """
        # Prepare a list of labels for Prometheus gauges
        gpu_indices = [str(i) for i in self._gpu_metrics['gpu_idx']]

        # Update all gauges in a single loop
        for gpu_index, gpu_str in enumerate(gpu_indices):
            # Directly access metrics for the current GPU index
            utilization = self._gpu_metrics['util'][gpu_index]
            power = self._gpu_metrics['power'][gpu_index]
            temperature = self._gpu_metrics['temp'][gpu_index]
            memory = self._gpu_metrics['mem'][gpu_index]

            # Update the power gauge
            power_gauge = self.gpu_metrics_gauges[gpu_index]['power_gauge']
            power_gauge.labels(gpu_index=gpu_str).set(power)

            # Update the utilization gauge
            utilization_gauge = self.gpu_metrics_gauges[gpu_index]['utilization_gauge']
            utilization_gauge.labels(gpu_index=gpu_str).set(utilization)

            # Update the temperature gauge
            temperature_gauge = self.gpu_metrics_gauges[gpu_index]['temperature_gauge']
            temperature_gauge.labels(gpu_index=gpu_str).set(temperature)

            # Update the memory usage gauge
            memory_gauge = self.gpu_metrics_gauges[gpu_index]['memory_gauge']
            memory_gauge.labels(gpu_index=gpu_str).set(memory)

    def __completion_stats(self) -> None:
        """
        Calculates and Updates completiont metrics.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        # First Fill Carbon Forecast End time and End Forecast
        self.stats["end_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["end_carbon_forecast"] = self.__update_carbon_forecast()

        # Fill Total Energy and Total Carbon Estimations
        self.stats["av_carbon_forecast"] = (self.stats["start_carbon_forecast"] +
                                            self.stats["end_carbon_forecast"])/2
        self.stats["total_carbon"] = (self.stats["total_energy"] *
                                      self.stats["av_carbon_forecast"])

    def save_stats_to_yaml(self, file_path: str):
        """
        Saves stats to a YAML file.

        Args:
            file_path (str): Path to the YAML file.
        """
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(self._stats, yaml_file, default_flow_style=False)

    def run(self):
        """
        Starts the GPU monitoring and Prometheus HTTP server with HTTPS.

        This method initializes and runs an HTTP server that exposes GPU metrics via
        Prometheus. It sets up SSL/TLS encryption for secure communication.
        """

        try:
            # Initialize statistics and metrics
            self.__setup_stats()

            # Start the monitoring loop
            while True:
                # Update GPU metrics
                self.__update_gpu_metrics()

                # Update Prometheus metrics with the latest GPU data
                self.__update_prometheus_metrics()

                # Print the current GPU metrics to the console
                print(f"Current GPU Metrics: {self._gpu_metrics}")

                # Wait for the defined collection interval before the next iteration
                time.sleep(self.collect_interval)

        except KeyboardInterrupt:
            # Handle interruption (e.g., Ctrl+C) by storing completion stats
            self.__completion_stats()

            print("Monitoring stopped by user.")

        finally:
            # Properly shut down pynvml
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    MONITOR = GPUMonitor()
    MONITOR.run()
    MONITOR.save_stats_to_yaml(METRICS_FILE_PATH)
