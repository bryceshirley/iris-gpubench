"""
Collects live GPU metrics using nvidia-smi and carbon produced by GPU using 
The nationalgridESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python gpu_monitor.py

Options:
    --plot
        Produces live plots of the collected GPU Metrics

Parameters:
    CARBON_INSTENSITY_REGION_SHORTHAND: The region for the The nationalgridESO 
                                        Regional Carbon Intensity API 
"""

import argparse
import time
from typing import Tuple, Optional, List
from datetime import datetime
import ssl
from http.server import HTTPServer, BaseHTTPRequestHandler

import pynvml
import requests
import yaml
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from tabulate import tabulate

# Constants
CARBON_INTENTSITY_URL = "https://api.carbonintensity.org.uk/regional"
PORT = 8000  # Port for the Prometheus metrics HTTP server
SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = './results/metrics.yml'
FORMATTED_METRICS_PATH = './results/formatted_metrics.txt'
CERTFILE = 'server.cert'  # Path to your certificate file
KEYFILE = 'server.key'  # Path to your private key file


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(generate_latest(monitor.registry))


class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML.
    """

    def __init__(self,
                 collect_interval=1,
                 carbon_region_shorthand="South England",
                 port=PORT):
        """
        Initializes the GPUMonitor class.

        Args:
            collect_interval (int): Interval in seconds for collecting GPU metrics.
            port (int): Port for the Prometheus metrics HTTP server.
        """
        self.collect_interval = collect_interval/ SECONDS_IN_HOUR # Convert to hours
        self.carbon_region_shorthand = carbon_region_shorthand
        self.port = port

        # Initialize pynvml
        pynvml.nvmlInit()

        # Set up Prometheus metrics
        self.__setup_prometheus_metrics()

        # Initialize private GPU metrics attribute
        self._gpu_metrics = []

        # Initialize stats
        self.__setup_stats()

    @property
    def stats(self) -> dict:
        """
        Gets the current statistics dictionary.

        The dictionary has the following keys and default values:

        - 'av_load': Average GPU load as a percentage (default 0.0).
        - 'av_power': Average power consumption in watts (default 0.0).
        - 'av_temp': Average GPU temperature in degrees Celsius (default 0.0).
        - 'av_mem': Average GPU memory usage in MiB (default 0.0).
        - 'av_carbon_forcast': Average carbon forecast in grams of CO2 per kWh (default 0.0).
        - 'end_datetime': End date and time of the measurement period (default '').
        - 'end_carbon_forcast': Forecasted carbon intensity in grams of CO2 per
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

        Returns:
            dict: The dictionary containing GPU statistics.
        """
        return self._stats

    @stats.setter
    def stats(self, value: dict) -> None:
        """
        Sets the statistics dictionary using __setup_stats.

        Args:
            value (dict): The dictionary containing GPU statistics.
        """
        # We can perform some validation or other logic here if needed
        self._stats = value

    def __setup_stats(self) -> dict:
        """
        Initializes and returns a dictionary with GPU statistics and default values.

        Returns:
            dict: Dictionary with initialized metrics and default values.
        """
        # Find The First GPU's Name and Max Power
        first_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(first_handle)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(first_handle)
        power_limit = power_limit/ 1000.0  # Convert from mW to W
        total_memory_info = pynvml.nvmlDeviceGetMemoryInfo(first_handle)
        total_memory = total_memory_info.total / (1024 ** 2)  # Convert bytes to MiB

        # Collect initial carbon forecast
        carbon_forcast = self.__collect_carbon_forecast()

        self.stats = {
            "av_load": 0.0,
            "av_power": 0.0,
            "av_temp": 0.0,
            "av_mem": 0.0,
            "av_carbon_forcast": 0.0,
            "end_datetime": '',
            "end_carbon_forcast": 0.0,
            "max_power_limit": power_limit,
            "name": gpu_name,
            "start_carbon_forecast": carbon_forcast,
            "start_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_carbon": 0.0,
            "total_energy": 0.0,
            "total_mem": total_memory,
        }

    def __setup_prometheus_metrics(self):
        """
        Sets up Prometheus metrics.
        """
        self.registry = CollectorRegistry()
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

    def update_gpu_metrics(self) -> None:
        """
        Retrieves the current GPU metrics for all GPUs.

        Returns:
            List[Tuple[int, float, float, int,float]]: A list of tuples with GPU
              metrics (index, utilization, power, temperature, memory).
        """
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            current_gpu_metrics = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                # Get GPU memory metrics
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                current_gpu_metrics.append((i, utilization, power, temperature, memory))

            self.current_gpu_metrics = current_gpu_metrics

        except pynvml.NVMLError as error_message:
            print(f"NVML Error: {error_message}")
            self.current_gpu_metrics = []

    def __collect_carbon_forecast(self) -> float:
        """
        Uses The nationalgridESO Regional Carbon Intensity API to collect current carbon emissions.

        Returns:
            Tuple[Optional[float], Optional[str]]: Current carbon intensity, carbon index.
        """
        timeout_seconds = 30
        try:
            response = requests.get(CARBON_INTENTSITY_URL,
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

    def __calculate_total_energy(self) -> float:
        """
        Computes the total energy consumed by all GPUs.

        Returns:
            float: Total energy in kWh.
        """
        total_energy_wh = 0.0

        # Iterate over each GPU's power readings
        # And transpose the list of lists to iterate over each gpu
        for gpu_readings in zip(*self.metrics["gpu_power"]):

            for i in range(1, len(gpu_readings)):
                # Calculate average power between two consecutive readings
                avg_power = (gpu_readings[i - 1] + gpu_readings[i]) / 2
                # Increment energy for the time interval
                energy_increment_wh = avg_power * self.collect_interval
                total_energy_wh += energy_increment_wh

        total_energy_kwh = total_energy_wh / 1000  # Convert Wh to kWh

        return total_energy_kwh

    def __completion_stats(self) -> None:
        """
        Calculates and Updates completiont metrics.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        # First Fill Carbon Forecast End time and End Forecast
        self.stats["end_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["end_carbon_forcast"] = self.__collect_carbon_forecast()

        # Fill Total Energy and Total Carbon Estimations
        self.stats["total_energy"] = self.__calculate_total_energy()
        self.stats["av_carbon_forcast"] = (self.stats["start_carbon_forcast"] +
                                           self.stats["end_carbon_forcast"])/2
        self.stats["total_carbon"] = (self.stats["total_energy"] *
                                      self.stats["av_carbon_forecast"])

        # Add code for "av_load", "av_power", "av_temp", "av_mem"

    def __save_stats_to_yaml(self, file_path: str):
        """
        Saves stats to a YAML file.

        Args:
            file_path (str): Path to the YAML file.
        """
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(self.stats, yaml_file, default_flow_style=False)

    def update_prometheus_metrics(self):
        """
        Updates Prometheus metrics with the latest GPU data.
        """
        for gpu_index, utilization, power, temperature, memory in self._gpu_metrics:
            # Update the power gauge
            self.gpu_metrics_gauges[gpu_index]['power_gauge'].labels(gpu_index=str(gpu_index)).set(power)
            
            # Update the utilization gauge
            self.gpu_metrics_gauges[gpu_index]['utilization_gauge'].labels(gpu_index=str(gpu_index)).set(utilization)
            
            # Update the temperature gauge
            self.gpu_metrics_gauges[gpu_index]['temperature_gauge'].labels(gpu_index=str(gpu_index)).set(temperature)
            
            # Update the memory usage gauge
            self.gpu_metrics_gauges[gpu_index]['memory_gauge'].labels(gpu_index=str(gpu_index)).set(memory)

    def run(self):
        """
        Starts the GPU monitoring and Prometheus server.
        """
        server_address = ('', self.port)
        httpd = HTTPServer(server_address, MetricsHandler)
        httpd.socket = ssl.wrap_socket(httpd.socket, keyfile=KEYFILE, certfile=CERTFILE, server_side=True)
        print(f"Prometheus HTTPS exporter running on port {self.port}")

        # Initialize counters for GPU metrics
        av_count = av_util = av_power = av_temp = av_mem = 0

        try:
            while True:
                self.update_gpu_metrics()
                self.update_prometheus_metrics()

                # Aggregate metrics for GPUs with utilization > 0
                for _, util, power, temp, mem in self.current_gpu_metrics:
                    if util > 0.0:
                        av_util += util
                        av_power += power
                        av_temp += temp
                        av_mem += mem
                        av_count += 1

                # Print current GPU metrics
                print(f"Current GPU Metrics: {self.current_gpu_metrics}")

                # Sleep for the defined collection interval
                time.sleep(self.collect_interval)

        except KeyboardInterrupt:
            # Store Completion Stats
            self.__completion_stats()

            # Compute and store final statistics
            if av_count > 0:  # Prevent division by zero
                self.stats["av_load"] = av_util / av_count
                self.stats["av_power"] = av_power / av_count
                self.stats["av_temp"] = av_temp / av_count
                self.stats["av_mem"] = av_mem / av_count
            else:
                # Handle the case where no valid metrics were collected
                self.stats["av_load"] = self.stats["av_power"] = self.stats["av_temp"] = self.stats["av_mem"] = None

            print("Monitoring stopped by user.")
            self.__save_stats_to_yaml(METRICS_FILE_PATH)  # Save metrics to YAML file when interrupted



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU metrics and expose them via Prometheus.')
    parser.add_argument('--port', type=int, default=PORT, help='Port for the Prometheus metrics HTTP server.')
    args = parser.parse_args()

    monitor = GPUMonitor(port=args.port)
    monitor.run()
