"""
gpu_meerkat_exporter.py

This module provides functionality for exporting time series data to a
MeerkatDB instance.
It includes a class for converting time series data into a CSV format compatible
with MeerkatDB
and for sending this data via HTTP POST requests.

Dependencies:
- requests: For sending HTTP requests.
- logging: For logging information and errors.

Usage:
1. Initialize the `MeerkatDBExporter` class with time series data and
optional configuration parameters.
2. Call the `send_to_meerkat` method to format the data and send it to
MeerkatDB.

Configuration:
- The MeerkatDB instance URL and CSV format are configurable via the class
constructor.
- Logging is configured to capture and store logs in a specified file.

Example:
    from meerkat_exporter import MeerkatDBExporter

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
    exporter = MeerkatDBExporter(time_series_data)

    # Send data to MeerkatDB
    exporter.send_to_meerkat()
"""
import os
from typing import Dict, List
import requests
from requests.auth import HTTPBasicAuth
import base64
import urllib3

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .carbon_metrics import get_carbon_forecast

# GLOBAL VARIABLES
from .utils.globals import LOGGER, TIMEOUT_SECONDS, RESULTS_DIR, DEFAULT_REGION
MEERKAT_USERNAME = 'Meerkat'
MEERKAT_PASSWORD = 'mXe4t3etL2W2v3k'
MEERKAT_URL = 'https://172.16.101.182:8247/write'

class MeerkatExporter:
    """
    A class for exporting GPU metrics to a MeerkatDB instance.

    WARNING: This class currently uses insecure connections (SSL verification disabled).
    This should be addressed before use in a production environment.

    This class handles the formatting and sending of GPU metric data to a specified
    MeerkatDB instance via HTTP POST requests. It supports multiple GPUs and
    various metric types.

    Attributes:
        benchmark_info (str): A formatted string containing GPU name and benchmark information.
        headers (Dict[str, str]): HTTP headers for authentication.
        db_url (str): URL of the MeerkatDB import endpoint.
        verify_ssl (bool): Whether to verify SSL certificates.

    Methods:
        export_metric_readings(current_gpu_metrics: Dict[str, List]) -> None:
            Exports the current GPU metrics to the MeerkatDB instance.
        export_stats() -> None:
            Exports completion results to the MeerkatDB instance (not implemented yet).

    Usage:
        exporter = MeerkatDBExporter("NVIDIA A100", "BERT-Large")
        current_gpu_metrics = {
            'gpu_idx': [0, 1],
            'util': [80, 90],
            'power': [250.5, 260.2],
            'temp': [70.0, 72.5],
            'mem': [16384, 16384]
        }
        exporter.export_metric_readings(current_gpu_metrics)
    """

    def __init__(self, gpu_name: str, benchmark: str, verify_ssl: bool = False):
        """
        Initializes the MeerkatDBExporter with GPU and benchmark information.

        Args:
            gpu_name (str): The name or model of the GPU being monitored.
            benchmark (str): The name of the benchmark being run.
            verify_ssl (bool): Whether to verify SSL certificates. Defaults to False.
        """
        self.gpu_name = gpu_name.split(' ')[-1]
        self.benchmark = benchmark
        self.benchmark_info = f"gpu_name={self.gpu_name},benchmark={self.benchmark}"
        self.headers = self._create_auth_header()
        self.db_url = MEERKAT_URL
        self.verify_ssl = verify_ssl

        # TODO: Implement proper SSL certificate verification for production use.
        if not self.verify_ssl:
            LOGGER.warning("SSL certificate verification is disabled. This is insecure and should be addressed.")

    def _create_auth_header(self) -> Dict[str, str]:
        """
        Creates the authorization header for MeerkatDB requests.

        Returns:
            Dict[str, str]: A dictionary containing the authorization header.
        """
        auth_str = f"{MEERKAT_USERNAME}:{MEERKAT_PASSWORD}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        return {'Authorization': f'Basic {auth_b64}'}

    def export_metric_readings(self, current_gpu_metrics: Dict[str, List],
                            reset_meerkat: bool = False) -> None:
        """
        Exports the current GPU metrics to the MeerkatDB instance.

        This method formats the GPU metrics into the required string format and
        sends a POST request to the MeerkatDB endpoint for each metric type.

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
                if reset_meerkat:
                    gpu_results = self._format_gpu_results_reset(current_gpu_metrics[metric_key], num_gpus)
                else:
                    gpu_results = self._format_gpu_results(current_gpu_metrics[metric_key], num_gpus)
                
                data = f"{metric_key},{self.benchmark_info} {gpu_results}"
                self._send_metric_data(data)

            LOGGER.info(f"Successfully exported metrics for {self.benchmark_info}")
        except KeyError as e:
            LOGGER.error(f"Missing key in GPU metrics: {e}")
            raise ValueError(f"Invalid GPU metrics data: missing key {e}")
        except Exception as e:
            LOGGER.error(f"Error in exporting metrics: {e}")
            raise

    def _format_gpu_results(self, metric_values: List, num_gpus: int) -> str:
        """
        Formats the GPU results into a string for MeerkatDB.

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

    def _format_gpu_results_reset(self, metric_values: List, num_gpus: int) -> str:
        """
        Formats the GPU results into a string for MeerkatDB.

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

        return ','.join(f"gpu{i}={0}" for i in range(num_gpus))

    def _send_metric_data(self, data: str) -> None:
        """
        Sends the formatted metric data to MeerkatDB.

        Args:
            data (str): Formatted metric data string.

        Raises:
            requests.RequestException: If there's an error in sending the request.
        """
        try:
            # NOTE: Using verify=False is required for our current setup.
            # This disables SSL certificate verification and should be addressed in a production environment.
            response = requests.post(self.db_url, headers=self.headers, data=data, 
                                     verify=self.verify_ssl, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as e:
            LOGGER.error(f"Failed to send data to MeerkatDB: {e}")
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"An error occurred: {e}")

    def export_stats(self,stats) -> None:
        """
        Exports Completion Results to Meerkat Database
        """
        stats_values = f"av_carbon_forecast={stats['av_carbon_forecast']:.5f},total_carbon={stats['total_carbon']:.5f},total_energy={stats['total_energy']:.5f},av_temp={stats['av_temp']:.5f},av_util={stats['av_util']:.5f},av_mem={stats['av_mem']:.5f},av_power={stats['av_power']:.5f}"
        gpu_info = f"gpu_name={self.gpu_name}"
        data = f"{self.benchmark},{gpu_info},{stats_values} time={stats['elapsed_time']:.2f}"
        self._send_metric_data(data)

        LOGGER.info(f"Exported Stats. data:{data}")
        

    def export_carbon_forcast(self, carbon_region_shorthand: str = DEFAULT_REGION) -> None:
        """
        Exports Carbon forcast
        """
        carbon_forcast = get_carbon_forecast(carbon_region_shorthand)

        # Remove spaces
        carbon_region_shorthand=carbon_region_shorthand.replace(" ","_")

        data = f"carbon,region={carbon_region_shorthand} forecast={carbon_forcast}"

        self._send_metric_data(data)

        LOGGER.info(f"Exported Carbon Forecast. data:{data}")