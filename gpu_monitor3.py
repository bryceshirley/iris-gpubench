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
        self.metrics = {
            "gpu_power": [],  # List of lists, one per GPU
            "gpu_usage": [],  # List of lists, one per GPU
            "gpu_temp": [],  # List of lists, one per GPU
        }

        # Initialize pynvml
        pynvml.nvmlInit()

        # Set up Prometheus metrics
        self.setup_prometheus_metrics()

    def setup_prometheus_metrics(self):
        """
        Sets up Prometheus metrics.
        """
        self.registry = CollectorRegistry()
        self.gpu_metrics_gauges = []  # List to hold metrics gauges for each GPU

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
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
                                           registry=self.registry)
            })

    @staticmethod
    def __collect_gpu_metrics() -> List[Tuple[int, float, float, int]]:
        """
        Retrieves the current GPU metrics for all GPUs.

        Returns:
            List[Tuple[int, float, float, int]]: A list of tuples with GPU
              metrics (index, utilization, power, temperature).
        """
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            current_gpu_metrics = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                current_gpu_metrics.append((i, utilization, power, temperature))

            return current_gpu_metrics

        except pynvml.NVMLError as error_message:
            print(f"NVML Error: {error_message}")
        return []

    def __collect_carbon_metrics(self) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Uses The nationalgridESO Regional Carbon Intensity API to collect current carbon emissions.

        Returns:
            Tuple[Optional[float], Optional[str], Optional[str]]: Current carbon intensity, carbon index, and date/time.
        """
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

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
                    carbon_index = intensity['index']

                    carbon_metrics = (carbon_forecast, carbon_index, formatted_datetime])
                    return carbon_metrics

        except requests.exceptions.RequestException as error_message:
            print(f"Error request timed out (30s): {error_message}")

        return None, None, formatted_datetime

    def calculate_total_energy(self) -> float:
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

    def calculate_metrics(self) -> dict:
        """
        Calculates various metrics for all GPUs.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        total_energy = self.calculate_total_energy()
        avg_gpu_utilization = sum(sum(readings) for readings in zip(*self.metrics["gpu_utilizations"])) / len(
            self.metrics["gpu_utilizations"]) if self.metrics["gpu_utilizations"] else 0.0
        avg_gpu_power = sum(sum(readings) for readings in zip(*self.metrics["power_readings"])) / len(
            self.metrics["power_readings"]) if self.metrics["power_readings"] else 0.0

        try:
            max_powers = [pynvml.nvmlDeviceGetPowerManagementLimit(
                pynvml.nvmlDeviceGetHandleByIndex(i)) / 1000.0 for i in range(pynvml.nvmlDeviceGetCount())]
            max_power = max(max_powers) if max_powers else None
        except pynvml.NVMLError as error_message:
            print(f"Error getting max power limit from NVML: {error_message}")
            max_power = None

        carbon_forecast, carbon_index, _ = self.__collect_carbon_metrics()
        total_carbon = total_energy * carbon_forecast if carbon_forecast else None

        return {
            "total_GPU_Energy": total_energy,  # kWh (kilo-Watt-hour)
            "av_GPU_load": avg_gpu_utilization,  # % (percent)
            "av_GPU_power": avg_gpu_power,  # W (Watts)
            "max_GPU_power": max_power,  # W (Watts)
            "carbon_forecast": carbon_forecast,  # gCO2/kWh
            "carbon_index": carbon_index,  # Very Low to Very High
            "total_GPU_carbon": total_carbon,  # gCO2 (grams of CO2)
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Carbon Reading Date-Time 
        }

    def save_to_yaml(self, file_path: str):
        """
        Saves calculated metrics to a YAML file.

        Args:
            file_path (str): Path to the YAML file.
        """
        metrics = self.calculate_metrics()
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(metrics, yaml_file, default_flow_style=False)

    def print_and_save_formatted_metrics(self):
        """
        Formats the metrics from the YAML file, prints to the command line, and saves to a text file.
        """
        try:
            with open(METRICS_FILE_PATH, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)

            # Extract and convert values to appropriate types
            time_value = data.get('time', 0)
            av_gpu_load = data.get('av_GPU_load', 0)
            av_gpu_power = data.get('av_GPU_power', 0)
            max_gpu_power = int(data.get('max_GPU_power', 0))
            total_gpu_energy = data.get('total_GPU_Energy', 0)
            carbon_forecast = data.get('carbon_forecast', 0)
            carbon_index = data.get('carbon_index', 'Unknown')
            total_GPU_carbon = data.get('total_GPU_carbon', 0)
            date_time = data.get("date_time", "Unknown")

            # Prepare data for tabulate
            formatted_data_main = [
                ["Benchmark Score (s)", f"{time_value:.5f}"],
                ["Total GPU Energy Consumed (kWh)", f"{total_gpu_energy:.5f}"],
                ["Total GPU Carbon Emissions (gC02)", f"{total_GPU_carbon:.5f}"],
            ]

            formatted_data_extra = [
                ["Average GPU Utilization (%)", f"{av_gpu_load:.5f}"],
                ["Avg GPU Power (W)", f"{av_gpu_power:.5f} (max possible {max_gpu_power})"],
                ["Carbon Forecast (gCO2/kWh), Carbon Index", f"{carbon_forecast:.1f}, {carbon_index}"],
                ["Carbon Intensity Reading Date & Time", date_time]
            ]

            # Print as table
            output = []
            output.append("Benchmark Score and GPU Energy Performance")
            output.append("")
            output.append(tabulate(formatted_data_main, headers=["Metric", "Value"], tablefmt="grid"))
            output.append("")
            output.append("Additional Information")
            output.append("")
            output.append(tabulate(formatted_data_extra, headers=["Metric", "Value"], tablefmt="grid"))
            output.append("")

            # Print to console
            for line in output:
                print(line)

            # Save output to file
            with open(FORMATTED_METRICS_PATH, 'w', encoding='utf-8') as output_file:
                for line in output:
                    output_file.write(line + "\n")

        except FileNotFoundError:
            print(f"Metrics file {METRICS_FILE_PATH} not found. Please run the monitoring script first.")

    def update_prometheus_metrics(self):
        """
        Updates Prometheus metrics with the latest GPU data.
        """
        gpu_metrics = self.__collect_gpu_metrics()
        for gpu_index, utilization, power, temperature in gpu_metrics:
            self.gpu_metrics_gauges[gpu_index]['power_gauge'].labels(gpu_index=str(gpu_index)).set(power)
            self.gpu_metrics_gauges[gpu_index]['utilization_gauge'].labels(gpu_index=str(gpu_index)).set(utilization)
            self.gpu_metrics_gauges[gpu_index]['temperature_gauge'].labels(gpu_index=str(gpu_index)).set(temperature)

    def run(self):
        """
        Starts the GPU monitoring and Prometheus server.
        """
        server_address = ('', self.port)
        httpd = HTTPServer(server_address, MetricsHandler)
        httpd.socket = ssl.wrap_socket(httpd.socket, keyfile=KEYFILE, certfile=CERTFILE, server_side=True)
        print(f"Prometheus HTTPS exporter running on port {self.port}")

        try:
            while True:
                gpu_metrics = self.__collect_gpu_metrics()
                current_time = time.time()

                for i, (gpu_index, utilization, power, _) in enumerate(gpu_metrics):
                    self.metrics["timestamps"].append(current_time)
                    if i == 0:
                        # Initialize power and utilization lists for each timestamp
                        self.metrics["power_readings"].append([])
                        self.metrics["gpu_utilizations"].append([])

                    self.metrics["power_readings"][-1].append(power)
                    self.metrics["gpu_utilizations"][-1].append(utilization)

                print(f"Current GPU Metrics: {gpu_metrics}")

                self.update_prometheus_metrics()

                time.sleep(self.collect_interval)  # Check every `collect_interval` seconds

        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
            self.save_to_yaml(METRICS_FILE_PATH)  # Save metrics to YAML file when interrupted
            self.print_and_save_formatted_metrics()  # Print and save formatted metrics when interrupted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU metrics and expose them via Prometheus.')
    parser.add_argument('--port', type=int, default=PORT, help='Port for the Prometheus metrics HTTP server.')
    args = parser.parse_args()

    monitor = GPUMonitor(port=args.port)
    monitor.run()
