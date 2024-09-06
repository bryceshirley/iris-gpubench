"""
Utility functions for metrics handling in iris-gpubench.

Provides functionality to format, convert, and visualize GPU metrics.

Functions:
    format_metrics: Format YAML metrics into human-readable tables.
    save_metrics_to_csv: Convert time series data to CSV format.

Dependencies: os, yaml, tabulate, matplotlib
"""

import os
import yaml
from tabulate import tabulate

from typing import Optional

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

        # Add other data
        main_data= [
            ["Benchmark: ", f"{data.get('benchmark')}"],
            ["Elapsed Monitor Time (s)", f"{data.get('elapsed_time'):.5f}"],
            ["Total GPU Energy Consumed (kWh)", f"{data.get('total_energy'):.5f}"],
            ["Total GPU Carbon Emissions (gCO2)", f"{data.get('total_carbon'):.5f}"],
        ]

        carbon_data = [
            ["Average Carbon Forecast (gCO2/kWh)", f"{data.get('av_carbon_forecast'):.1f}"],
            ["Carbon Forecast Start Time", data.get("start_datetime")],
            ["Carbon Forecast End Time", data.get("end_datetime")]
        ]

        gpu_data = [
            ["GPU Type", data.get("name")],
            ["No. of GPUs ", data.get("device_count")],
            ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{data.get('av_util'):.5f}"],
            ["Avg GPU Power (for >0.00% GPU Util.) (W)",
             f"{data.get('av_power'):.5f} (Power Limit: {int(data.get('max_power_limit', 0))})"],
            ["Avg GPU Temperature (for >0.00% GPU Util.) (C)",
             f"{data.get('av_temp'):.5f}"],
            ["Avg GPU Memory (for >0.00% GPU Util.) (MiB)",
             f"{data.get('av_mem'):.5f} (Total Memory: {data.get('total_mem')})"]
        ]

        # Collect Benchmark Score if Exists;
        if data.get('score') is not None:
            main_data.insert(1, ["Benchmark Score (s)", f"{data.get('score'):.5f}"])


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