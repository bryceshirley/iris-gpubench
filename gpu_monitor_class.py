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
from typing import Tuple, Optional, List
from http.server import BaseHTTPRequestHandler, HTTPServer  # HTTP server and request handling

import yaml  # A library for parsing and writing YAML files, used here for saving GPU metrics

 # Prometheus client library for Python; used to define and expose metrics
from prometheus_client import CollectorRegistry, Gauge, generate_latest
import pynvml


class GPUMetrics:
    """
    Manages NVIDIA GPU metrics using NVML.

    Attributes:
        collect_interval (int): Interval in seconds for metric collection.
        gpu_names (dict): Maps GPU indices to names.
        gpu_metrics_over_time (list): List of GPU metrics collected over time.

    Methods:
        __init__(collect_interval=1):
            Initializes NVML and sets up metrics collection.

        get_gpu_metrics():
            Retrieves GPU metrics (utilization, power, temperature).

        collect_metrics():
            Collects and stores GPU metrics.

        calculate_statistics():
            Computes average metrics and total energy consumption.

        shutdown():
            Shuts down NVML.
    """
    def __init__(self, collect_interval=1):
        """
        Initializes the GPUMetrics class and NVML.

        Args:
            collect_interval (int): Interval for data collection in seconds.
        """
        self.collect_interval = collect_interval
        self.gpu_names = {}
        self.gpu_metrics_over_time = []
        
        # Initialize NVML
        pynvml.nvmlInit()

        # Define Prometheus metrics
        registry = CollectorRegistry()
        self.gpu_utilization_gauge = Gauge('gpu_utilization',
                                           'GPU Utilization',
                                           ['gpu'],
                                           registry=registry)

        self.gpu_power_gauge = Gauge('gpu_power',
                                     'GPU Power Draw',
                                     ['gpu'],
                                     registry=registry)

        self.gpu_temperature_gauge = Gauge('gpu_temperature',
                                           'GPU Temperature',
                                           ['gpu'],
                                           registry=registry)

        # Internal attribute for metrics
        self._gpu_metrics = []

    def __set_gpu_metrics(self):
        """
        Retrieves and sets metrics for all available NVIDIA GPUs.
        """
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_metrics = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                gpu_metrics.append((i, utilization, power, temperature))
                self.gpu_names[i] = name

            self._gpu_metrics = gpu_metrics

        except pynvml.NVMLError as error_message:
            print(f"NVML Error: {error_message}")

    @property
    def gpu_metrics(self) -> List[Tuple[int, int, float, int]]:
        """
        Retrieves the currently stored GPU metrics, refreshing them if necessary.

        Returns:
            List[Tuple[int, int, float, int]]: A list of tuples with GPU metrics.
        """
        self.__set_gpu_metrics()  # Refresh metrics before returning
        return self._gpu_metrics

    def collect_and_expose_gpu_metrics(self):
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
        current_metrics = self.gpu_metrics  # This will call the property getter and refresh metrics

        # Store the metrics for historical data
        self.gpu_metrics_over_time.append(current_metrics)
        

        for gpu_index, utilization, power, temperature in current_metrics:
            gpu_name = self.gpu_names[gpu_index]

            # Update Prometheus gauges with the current metrics
            self.gpu_utilization_gauge.labels(gpu=gpu_name).set(utilization)
            self.gpu_power_gauge.labels(gpu=gpu_name).set(power)
            self.gpu_temperature_gauge.labels(gpu=gpu_name).set(temperature)

    def collect_metrics(self):
        """
        Collects and stores GPU metrics over time.
        """
        gpu_metrics = self.get_gpu_metrics()
        self.gpu_metrics_over_time.append(gpu_metrics)

    def calculate_statistics(self):
        """
        Calculates average metrics and total energy consumption.

        Returns:
            tuple: Total energy in kWh, average power, utilization, and temperature.
        """
        num_gpus = len(self.gpu_metrics_over_time[0])
        total_energy_kwh = [0.0] * num_gpus
        total_utilization = [0.0] * num_gpus
        total_power = [0.0] * num_gpus
        total_temperature = [0.0] * num_gpus

        for gpu_metrics in self.gpu_metrics_over_time:
            for gpu_index in range(num_gpus):
                utilization = gpu_metrics[gpu_index][1]
                power_w = gpu_metrics[gpu_index][2]
                temperature = gpu_metrics[gpu_index][3]
                total_utilization[gpu_index] += utilization
                total_power[gpu_index] += power_w
                total_temperature[gpu_index] += temperature
                total_energy_kwh[gpu_index] += power_w * self.collect_interval / 3600 / 1000

        avg_utilization = [total_utilization[gpu_index] / len(self.gpu_metrics_over_time)
                           for gpu_index in range(num_gpus)]
        avg_power = [total_power[gpu_index] / len(self.gpu_metrics_over_time)
                     for gpu_index in range(num_gpus)]
        avg_temperature = [total_temperature[gpu_index] / len(self.gpu_metrics_over_time)
                           for gpu_index in range(num_gpus)]

        return total_energy_kwh, avg_power, avg_utilization, avg_temperature

    def shutdown(self):
        """
        Shuts down the NVML library.
        """
        pynvml.nvmlShutdown()


# Example usage
if __name__ == "__main__":
    GPU_METRICS = GPUMetrics(collect_interval=1)
    GPU_METRICS.collect_metrics()
    print(GPU_METRICS.calculate_statistics())
    GPU_METRICS.shutdown()

