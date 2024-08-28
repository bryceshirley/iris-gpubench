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
    A class for exporting time series data to a VictoriaMetrics instance.

    This class handles the conversion of time series data into CSV format
    compatible with VictoriaMetrics
    and sends the data to a specified VictoriaMetrics instance via HTTP POST
    request.

    Attributes:
        time_series_data (Dict[str, List]): Time series data to be exported.
            - 'timestamp': List of timestamp strings.
            - 'gpu_idx': List of lists, where each sublist contains GPU indices.
            - 'util': List of lists, where each sublist contains utilization metrics.
            - 'power': List of lists, where each sublist contains power metrics.
            - 'temp': List of lists, where each sublist contains temperature metrics.
            - 'mem': List of lists, where each sublist contains memory metrics.
        victoria_metrics_url (str): URL to the VictoriaMetrics import endpoint.
        csv_header (str): CSV format string specifying the format for VictoriaMetrics.

    Methods:
        _convert_to_csv() -> str:
            Converts the time series data into a CSV formatted string.

        send_to_victoria(timeout: int = 30) -> None:
            Sends the formatted CSV data to the VictoriaMetrics instance with a 
            specified timeout duration.
    """

    def __init__(self,
                 gpu_name: str, benchmark: str):
        """
        Initializes the VictoriaMetricsExporter class.
        """
        self.gpu_name = gpu_name
        self.benchmark = benchmark

    def export_metric_readings(self,metric_readings) -> None:
        """
        Exports a metric to Meerkat Database
        """

        # Create the authorization header
        auth_str = f"{MEERKAT_USERNAME}:{MEERKAT_PASSWORD}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        headers = {
            'Authorization': f'Basic {auth_b64}'
        }

        # Create the data payload
        for metric in metric_readings:
            gpu_results = "Parse results into this format: gpu0={metric_recording[0]},gpu1={metric_recording[1]}"
            data = f"{metric},gpu_name={self.gpu_name},benchmark={self.benchmark} {gpu_results}"

            # Send the POST request
            response = requests.post(MEERKAT_URL, headers=headers, data=data, verify=False)  # verify=False to skip SSL verification

            
    def export_carbon_and_metric_results(self) -> None:
        """
        Exports Completion Results to Meerkat Database
        """

    def _convert_to_csv(self) -> None:
        """
        Converts time series data to CSV data required by VictoriaMetrics.

        Raises:
            Exception: If an error occurs during the conversion process.
        """
        try:
            # Initialize a list to store CSV lines
            csv_lines = []

            # Extract data from the input dictionary
            timestamps = self.time_series_data['timestamp']
            gpu_indices = self.time_series_data['gpu_idx']
            util = self.time_series_data['util']
            power = self.time_series_data['power']
            temp = self.time_series_data['temp']
            mem = self.time_series_data['mem']

            # Flatten the data and format it as CSV
            for reading_index, timestamp in enumerate(timestamps):
                for gpu_index in gpu_indices[reading_index]:
                    # Format each line as a CSV entry
                    csv_line = (
                        f"{timestamp},"
                        f"{gpu_index},"
                        f"{util[reading_index][gpu_index]},"
                        f"{power[reading_index][gpu_index]},"
                        f"{temp[reading_index][gpu_index]},"
                        f"{mem[reading_index][gpu_index]}"
                    )
                    # Append the formatted line to the list
                    csv_lines.append(csv_line)

            # Join the lines with newline characters to create the final CSV string
            self.csv_data = "\n".join(csv_lines)
            LOGGER.info("Successfully converted time series data to CSV format.")

        except Exception as error_message:
            # Log any errors that occur during conversion
            LOGGER.error("Error converting time series data to CSV format: %s", error_message)
            raise

    def send_to_victoria(self) -> None:
        """
        Sends the formatted CSV data to VictoriaMetrics.

        This method performs a POST request to the VictoriaMetrics endpoint with the CSV data.

        Args:
            timeout (int): The timeout duration for the HTTP request in seconds.

        Raises:
            requests.RequestException: If an error occurs during the HTTP request.
        """
        try:
            # Perform the POST request to send the data
            response = requests.post(
                self.victoria_metrics_url,
                data=self.csv_data,
                headers={'Content-Type': 'text/csv'},
                timeout=TIMEOUT_SECONDS  # Set the timeout for the request
            )

            # Check the response status code
            if response.status_code == 200:
                LOGGER.info("Data successfully sent to VictoriaMetrics.")
            else:
                # Log the error if the request was not successful
                LOGGER.error(
                    "Failed to send data to VictoriaMetrics. Status code: %d, Response: %s",
                    response.status_code,
                    response.text
                )
        except requests.RequestException as error_message:
            # Log any request exceptions that occur
            LOGGER.error("An error occurred while sending data to VictoriaMetrics: %s",
                         error_message)

    def save_csv_to_file(self, file_dir: str = RESULTS_DIR) -> None:
        """
        Saves the CSV formatted data, including the header, to a specified file.

        Args:
            file_dir (str): The directory where the CSV data should be saved.
            The file will be named 'timeseries.csv'.

        Raises:
            IOError: If an error occurs while writing to the file.
        """
        try:
            # Ensure the target directory exists and create if not
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # Construct the full file path
            file_path = os.path.join(file_dir, 'timeseries.csv')

            # Prepare CSV data by combining the header and the data
            csv_to_write = f"{self.csv_header}\n{self.csv_data}"

            # Open the file in write mode with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(csv_to_write)

            # Log a success message with the file path
            LOGGER.info("CSV data successfully saved to %s", file_path)
        except IOError as error_message:
            # Log an error message if an exception occurs during file writing
            LOGGER.error("Error saving CSV data to file %s: %s", file_path, error_message)
            raise
