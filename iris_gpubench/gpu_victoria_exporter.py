"""
gpu_victoria_exporter.py

This module provides functionality for exporting time series data to a
VictoriaMetrics instance.
It includes a class for converting time series data into a CSV format compatible
with VictoriaMetrics
and for sending this data via HTTP POST requests.

Dependencies:
- requests: For sending HTTP requests.
- logging: For logging information and errors.

Usage:
1. Initialize the `VictoriaMetricsExporter` class with time series data and
optional configuration parameters.
2. Call the `send_to_victoria` method to format the data and send it to
VictoriaMetrics.

Configuration:
- The VictoriaMetrics instance URL and CSV format are configurable via the class
constructor.
- Logging is configured to capture and store logs in a specified file.

Example:
    from victoria_exporter import VictoriaMetricsExporter

    # Example time series data
    time_series_data = {
        'timestamp': ['2024-08-05 12:00:00', '2024-08-05 12:05:00'],
        'gpu_idx': [[0, 1], [0, 1]],
        'util': [[75.5, 80.0], [76.0, 81.0]],
        'power': [[120.0, 130.0], [121.0, 131.0]],
        'temp': [[60.0, 65.0], [61.0, 66.0]],
        'mem': [[8192, 8192], [8192, 8192]]
    }

    # Initialize the exporter
    exporter = VictoriaMetricsExporter(time_series_data)

    # Send data to VictoriaMetrics
    exporter.send_to_victoria()
"""
import os
from typing import Dict, List

import requests
from requests.auth import HTTPBasicAuth
import base64

# GLOBAL VARIABLES
from .utils.globals import LOGGER, TIMEOUT_SECONDS, RESULTS_DIR
MEERKAT_USERNAME = 'your_db_username'
MEERKAT_PASSWORD = 'your_db_password'
MEERKAT_URL = 'https://172.16.101.182:8247/write'


class VictoriaMetricsExporter:
    """
    A class for exporting GPU metrics to a VictoriaMetrics instance.

    This class handles the formatting and sending of GPU metric data to a specified
    VictoriaMetrics instance via HTTP POST requests. It supports multiple GPUs and
    various metric types.

    Attributes:
        benchmark_info (str): A formatted string containing GPU name and benchmark information.
        headers (Dict[str, str]): HTTP headers for authentication.
        db_url (str): URL of the VictoriaMetrics import endpoint.

    Methods:
        export_metric_readings(current_gpu_metrics: Dict[str, List]) -> None:
            Exports the current GPU metrics to the VictoriaMetrics instance.
        export_stats() -> None:
            Exports completion results to the VictoriaMetrics instance (not implemented yet).

    Usage:
        exporter = VictoriaMetricsExporter("NVIDIA A100", "BERT-Large")
        current_gpu_metrics = {
            'gpu_idx': [0, 1],
            'util': [80, 90],
            'power': [250.5, 260.2],
            'temp': [70.0, 72.5],
            'mem': [16384, 16384]
        }
        exporter.export_metric_readings(current_gpu_metrics)
    """

    def __init__(self, gpu_name: str, benchmark: str):
        """
        Initializes the VictoriaMetricsExporter with GPU and benchmark information.

        Args:
            gpu_name (str): The name or model of the GPU being monitored.
            benchmark (str): The name of the benchmark being run.
        """
        self.benchmark_info = "gpu_name={gpu_name},benchmark={benchmark}""
        self.headers = self._create_auth_header()
        self.db_url = MEERKAT_URL

    def _create_auth_header(self) -> Dict[str, str]:
        """
        Creates the authorization header for VictoriaMetrics requests.

        Returns:
            Dict[str, str]: A dictionary containing the authorization header.
        """
        auth_str = f"{MEERKAT_USERNAME}:{MEERKAT_PASSWORD}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        return {'Authorization': f'Basic {auth_b64}'}

    def export_metric_readings(self, current_gpu_metrics: Dict[str, List]) -> None:
        """
        Exports the current GPU metrics to the VictoriaMetrics instance.

        This method formats the GPU metrics into the required string format and
        sends a POST request to the VictoriaMetrics endpoint for each metric type.

        Args:
            current_gpu_metrics (Dict[str, List]): A dictionary containing the current
                GPU metrics. Each key (except 'gpu_idx') represents a metric type,
                and its value is a list of metric values for each GPU.

        Raises:
            ValueError: If the input metrics are invalid or missing required data.
            requests.RequestException: If there's an error in sending the request.
        """
        try:
            num_gpus = len(current_gpu_metrics['gpu_idx'])
            metric_keys = [key for key in current_gpu_metrics.keys() if key != 'gpu_idx']

            for metric_key in metric_keys:
                gpu_results = self._format_gpu_results(current_gpu_metrics[metric_key], num_gpus)
                data = f"{metric_key},{self.benchmark_info} {gpu_results}"
                self._send_metric_data(data)

            LOGGER.info(f"Successfully exported metrics for {self.gpu_name} - {self.benchmark}")
        except KeyError as e:
            LOGGER.error(f"Missing key in GPU metrics: {e}")
            raise ValueError(f"Invalid GPU metrics data: missing key {e}")
        except Exception as e:
            LOGGER.error(f"Error in exporting metrics: {e}")
            raise

    def _format_gpu_results(self, metric_values: List, num_gpus: int) -> str:
        """
        Formats the GPU results into a string for VictoriaMetrics.

        Args:
            metric_values (List): List of metric values for each GPU.
            num_gpus (int): Number of GPUs.

        Returns:
            str: Formatted string of GPU results.

        Raises:
            ValueError: If the number of metric values doesn't match the number of GPUs.
        """
        if len(metric_values) != num_gpus:
            raise ValueError(f"Mismatch in number of GPUs ({num_gpus}) and metric values ({len(metric_values)})")

        return ','.join(f"gpu{i}={value}" for i, value in enumerate(metric_values))

    def _send_metric_data(self, data: str) -> None:
        """
        Sends the formatted metric data to VictoriaMetrics.

        Args:
            data (str): Formatted metric data string.

        Raises:
            requests.RequestException: If there's an error in sending the request.
        """
        try:
            response = requests.post(self.db_url, headers=self.headers, data=data, verify=False)
            response.raise_for_status()
        except requests.RequestException as e:
            LOGGER.error(f"Failed to send data to VictoriaMetrics: {e}")
            raise

    def export_stats(self) -> None:
        """
        Exports Completion Results to Meerkat Database
        """
        print('This Does not work yet')

    def export_carbon_index(self, carbon_forecast) -> None:
        """
        Exports Carbon index"
        """
        print('This currently does nothing')


