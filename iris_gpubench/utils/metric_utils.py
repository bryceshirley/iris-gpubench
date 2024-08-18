"""
Utility functions for handling and formatting metrics in the iris-gpubench package.

This module provides functionality to format metrics from a YAML file into 
human-readable tables and save them to a file.

Functions:
    format_metrics(results_dir: str, metrics_file_path: str, formatted_metrics_path: str) -> None:
        Formats metrics from a YAML file into human-readable tables, prints them to the console,
        and saves them to a text file.

Dependencies:
- `os`: For directory operations.
- `yaml`: For parsing YAML files.
- `tabulate`: For formatting data into tables.
"""

import os
import yaml
from tabulate import tabulate

from .globals import RESULTS_DIR, LOGGER

def format_metrics(results_dir: str = RESULTS_DIR,
                   metrics_file_path: str = 'metrics.yml',
                   formatted_metrics_path: str = 'formatted_metrics.txt') -> None:
    """
    Formats the metrics from a YAML file and saves the formatted metrics to a text file.

    Reads the GPU metrics from the specified YAML file, formats the data into tables,
    and outputs the formatted metrics both to the console and to a text file.

    Args:
        results_dir (str): Directory where the metrics and output files are located.
        metrics_file_path (str): Path to the metrics YAML file.
        formatted_metrics_path (str): Path where the formatted metrics will be saved.

    Returns:
        None
    """
    try:
        # Set up logging with specific configuration

        metrics_file_path = os.path.join(results_dir, metrics_file_path)
        formatted_metrics_path = os.path.join(results_dir, formatted_metrics_path)

        # Read YAML file
        with open(metrics_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Prepare data for tabulate
        main_data = []

        # Add benchmark image name if it exists
        if 'benchmark_image' in data:
            main_data.extend([
                ["Benchmark Image Name", f"{data.get('benchmark_image')}"],
                ["Elapsed Monitor Time of Container (s)", f"{data.get('elapsed_time'):.5f}"],
            ])

        # Add benchmark command run if it exists
        if 'benchmark_command' in data:
            main_data.extend([
                ["Benchmark Command Run", f"{data.get('benchmark_command')}"],
                ["Elapsed Monitor Time of Command (s)", f"{data.get('elapsed_time'):.5f}"],
            ])

        # Add other data
        main_data.extend([
            ["Total GPU Energy Consumed (kWh)", f"{data.get('total_energy'):.5f}"],
            ["Total GPU Carbon Emissions (gCO2)", f"{data.get('total_carbon'):.5f}"],
        ])

        if data.get('time') is not None:
            main_data.insert(0, ["Benchmark Score (s)", f"{data.get('time'):.5f}"])

        carbon_data = [
            ["Average Carbon Forecast (gCO2/kWh)", f"{data.get('av_carbon_forecast'):.1f}"],
            ["Carbon Forecast Start Time", data.get("start_datetime")],
            ["Carbon Forecast End Time", data.get("end_datetime")]
        ]

        gpu_data = [
            ["GPU Name", data.get("name")],
            ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{data.get('av_util'):.5f}"],
            ["Avg GPU Power (for >0.00% GPU Util.) (W)",
             f"{data.get('av_power'):.5f} (Power Limit: {int(data.get('max_power_limit', 0))})"],
            ["Avg GPU Temperature (for >0.00% GPU Util.) (C)",
             f"{data.get('av_temp'):.5f}"],
            ["Avg GPU Memory (for >0.00% GPU Util.) (MiB)",
             f"{data.get('av_mem'):.5f} (Total Memory: {data.get('total_mem')})"]
        ]

        # Create output list with formatted tables
        output = [
            "GPU and Carbon Performance Results",
            "",
            tabulate(main_data, headers=["Metric", "Value"], tablefmt="grid"),
            "",
            "Carbon Information",
            "",
            tabulate(carbon_data, headers=["Metric", "Value"], tablefmt="grid"),
            "",
            "GPU Information",
            "",
            tabulate(gpu_data, headers=["Metric", "Value"], tablefmt="grid"),
            ""
        ]

        # Print formatted data to console
        print("\n".join(output))

        # Save formatted data to a file
        with open(formatted_metrics_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n".join(output))

        # Log success message
        LOGGER.info("Metrics formatted and saved successfully.")

    except FileNotFoundError as fnf_error:
        # Log file not found error
        LOGGER.error("Metrics file not found: %s", fnf_error)
    except yaml.YAMLError as yaml_error:
        # Log YAML parsing error
        LOGGER.error("Error parsing YAML file: %s", yaml_error)
    except KeyError as key_error:
        # Log missing data error
        LOGGER.error("Missing expected data in metrics file: %s", key_error)
