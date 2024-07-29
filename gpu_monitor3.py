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


import ssl  # SSL/TLS encryption for the HTTP server
import time  # Time functions like sleep
from http.server import BaseHTTPRequestHandler, HTTPServer  # HTTP server and request handling

import yaml  # A library for parsing and writing YAML files, used here for saving GPU metrics

 # Prometheus client library for Python; used to define and expose metrics
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from pynvml import *


# Initialize NVIDIA Management Library (NVML)
nvmlInit()

# Define Prometheus metrics
REGISTRY = CollectorRegistry()
GPU_UTILIZATION_GAUGE = Gauge('gpu_utilization', 'GPU Utilization', ['gpu'], registry=REGISTRY)
GPU_POWER_GAUGE = Gauge('gpu_power', 'GPU Power Draw', ['gpu'], registry=REGISTRY)
GPU_TEMPERATURE_GAUGE = Gauge('gpu_temperature', 'GPU Temperature', ['gpu'], registry=REGISTRY)

# Initialize data storage
GPU_METRICS_OVER_TIME = []
GPU_NAMES = {}


# Interval for data collection
COLLECT_INTERVAL = 1  # in seconds

# Choose the Carbon Intensity Region
# (ie "GB" or if in Oxford "South England")
CARBON_INSTENSITY_REGION_SHORTHAND = "South England"
CARBON_INTENTSITY_URL = "https://api.carbonintensity.org.uk/regional"

def get_gpu_metrics():
    """
    Retrieves metrics for all available NVIDIA GPUs.

    This function initializes the NVML library, queries each GPU for its utilization, power usage, and temperature,
    and stores the information in a list of tuples. It also updates the global `GPU_NAMES` dictionary with the names
    of the GPUs.

    Returns:
        list: A list of tuples, where each tuple contains:
              - GPU index (int)
              - GPU utilization percentage (int)
              - GPU power usage in watts (float)
              - GPU temperature in Celsius (int)
    """
    global GPU_NAMES  # Reference to the global dictionary holding GPU names

    # Get the number of NVIDIA GPU devices
    device_count = nvmlDeviceGetCount()
    gpu_metrics = []  # List to store metrics for each GPU

    # Iterate over each GPU device
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)  # Get handle for the GPU at index i
        name = nvmlDeviceGetName(handle).decode('utf-8')  # Get GPU name and decode from bytes
        utilization = nvmlDeviceGetUtilizationRates(handle).gpu  # Get GPU utilization percentage
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert power usage from milliWatts to watts
        temperature = nvmlDeviceGetTemperature(handle, 0)  # Get GPU temperature in Celsius

        # Append metrics for the current GPU to the list
        gpu_metrics.append((i, utilization, power, temperature))
        GPU_NAMES[i] = name  # Update global dictionary with GPU name

    return gpu_metrics


def collect_and_expose_gpu_metrics():
    """
    Collects GPU metrics and updates Prometheus gauges.

    This function retrieves GPU metrics (utilization, power usage, temperature) using
    `get_gpu_metrics()`, stores them in `gpu_metrics_over_time`, and updates the
    corresponding Prometheus gauges.

    Metrics:
        - Utilization percentage
        - Power usage in watts
        - Temperature in Celsius
    """
    # Retrieve the latest GPU metrics
    gpu_metrics = get_gpu_metrics()

    # Store the metrics for historical data
    gpu_metrics_over_time.append(gpu_metrics)

    # Iterate through the collected GPU metrics
    for gpu_metric in gpu_metrics:
        gpu_index, utilization, power, temperature = gpu_metric  # Unpack metrics
        gpu_name = GPU_NAMES[gpu_index]  # Get the GPU name from the index

        # Update Prometheus gauges with the current metrics
        gpu_utilization_gauge.labels(gpu=gpu_name).set(utilization)
        gpu_power_gauge.labels(gpu=gpu_name).set(power)
        gpu_temperature_gauge.labels(gpu=gpu_name).set(temperature)


def calculate_metrics():
    """
    Calculates average and total GPU metrics from historical data.

    This function computes the following metrics for each GPU:
        - Total energy consumption in kilowatt-hours (kWh)
        - Average power usage in watts (W)
        - Average utilization percentage
        - Average temperature in Celsius

    Returns:
        - total_energy_kwh: List of total energy consumption (kWh) for each GPU
        - avg_power: List of average power usage (W) for each GPU
        - avg_utilization: List of average utilization percentage for each GPU
        - avg_temperature: List of average temperature (Celsius) for each GPU
    """
    # Number of GPUs based on the metrics collected
    num_gpus = len(gpu_metrics_over_time[0])

    # Initialize lists to accumulate metrics
    total_energy_kwh = [0.0] * num_gpus
    total_utilization = [0.0] * num_gpus
    total_power = [0.0] * num_gpus
    total_temperature = [0.0] * num_gpus

    # Accumulate metrics for each GPU over all recorded data
    for gpu_metrics in gpu_metrics_over_time:
        for gpu_index in range(num_gpus):
            utilization = gpu_metrics[gpu_index][1]
            power_w = gpu_metrics[gpu_index][2]
            temperature = gpu_metrics[gpu_index][3]

            # Update totals
            total_utilization[gpu_index] += utilization
            total_power[gpu_index] += power_w
            total_temperature[gpu_index] += temperature

            # Calculate energy consumption in kWh
            total_energy_kwh[gpu_index] += power_w * COLLECT_INTERVAL / 3600 / 1000

    # Compute average metrics for each GPU
    avg_utilization = [
        total_utilization[gpu_index] / len(gpu_metrics_over_time)
        for gpu_index in range(num_gpus)
    ]
    avg_power = [
        total_power[gpu_index] / len(gpu_metrics_over_time)
        for gpu_index in range(num_gpus)
    ]
    avg_temperature = [
        total_temperature[gpu_index] / len(gpu_metrics_over_time)
        for gpu_index in range(num_gpus)
    ]

    return total_energy_kwh, avg_power, avg_utilization, avg_temperature


def write_metrics_to_yaml():
    total_energy_kwh, avg_power, avg_utilization, avg_temperature = calculate_metrics()
    metrics = {}
    for gpu_index in range(len(total_energy_kwh)):
        gpu_name = GPU_NAMES[gpu_index]
        metrics[gpu_name] = {
            'total_energy_kwh': total_energy_kwh[gpu_index],
            'average_power_w': avg_power[gpu_index],
            'average_utilization_percent': avg_utilization[gpu_index],
            'average_temperature_celsius': avg_temperature[gpu_index]
        }
    with open('gpu_metrics.yml', 'w') as file:
        yaml.dump(metrics, file, default_flow_style=False)

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.end_headers()
            self.wfile.write(generate_latest(registry))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    # Start HTTPS server to expose the metrics
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, MetricsHandler)
    httpd.socket = ssl.wrap_socket(httpd.socket, keyfile='server.key', certfile='server.cert', server_side=True)
    print("Prometheus HTTPS exporter running on port 8000")

    try:
        # Periodically collect and expose GPU metrics
        while True:
            collect_and_expose_gpu_metrics()
            time.sleep(COLLECT_INTERVAL)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving metrics to YAML file...")
        write_metrics_to_yaml()
        httpd.server_close()
        print("Server stopped and metrics saved.")

if __name__ == '__main__':
    main()
