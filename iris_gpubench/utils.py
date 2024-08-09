"""
utils.py

This module provides utility functions for the application, including logging 
setup and metrics formatting.

Functions:
    setup_logging(results_dir: str) -> logging.Logger:
        Configures and returns a logger instance for the application.

    format_metrics(results_dir: str, metrics_file_path: str, formatted_metrics_path: str) -> None:
        Formats metrics from a YAML file into human-readable tables, prints them to the console,
        and saves them to a text file.

Dependencies:
- `os`: For directory operations.
- `logging`: For configuring logging.
- `yaml`: For parsing YAML files.
- `tabulate`: For formatting data into tables.

Example:
    logger = setup_logging(results_dir='../results')
    logger.info("Logging is set up.")

    format_metrics(
        results_dir='../results',
        metrics_file_path='metrics.yml',
        formatted_metrics_path='formatted_metrics.txt'
    )
"""

import os
import subprocess
import logging
import yaml
from tabulate import tabulate
import docker

RESULTS_DIR = '../results'
# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# def get_base_directory():
#     # Try to locate the config file within the package directory
#     try:
#         package_dir = os.path.dirname(os.path.abspath(__file__))
#         config_path = os.path.join(package_dir, 'config', 'default_config.yml')

#         # Read the configuration file
#         with open(config_path, 'r', encoding='utf-8') as config_file:
#             config = yaml.safe_load(config_file)
#             return config.get('base_directory', os.path.dirname(os.path.abspath(__file__)))
#     except Exception as e:
#         print(f"Error loading configuration: {e}")
#         # Fallback to current directory if config is not available
#         return os.path.dirname(os.path.abspath(__file__))

def setup_logging(results_dir: str = RESULTS_DIR) -> logging.Logger:
    """
    Configures logging for the application.

    Sets up a logger that writes to 'gpu_monitor.log' in the specified directory.
    If the directory does not exist, it is created.

    Parameters:
        results_dir (str): Directory for saving the log file. Defaults to '../results'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Create the directory if it does not exist

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(results_dir, 'gpu_monitor.log'),
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger

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
        logger = setup_logging()

        metrics_file_path = os.path.join(results_dir, metrics_file_path)
        formatted_metrics_path = os.path.join(results_dir, formatted_metrics_path)

        # Read YAML file
        with open(metrics_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Prepare data for tabulate
        main_data = [
            ["Total GPU Energy Consumed (kWh)", f"{data.get('total_energy'):.5f}"],
            ["Total GPU Carbon Emissions (gCO2)", f"{data.get('total_carbon'):.5f}"],
        ]

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
            "Benchmark Score and GPU Energy Performance",
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
        logger.info("Metrics formatted and saved successfully.")

    except FileNotFoundError as fnf_error:
        # Log file not found error
        logger.error("Metrics file not found: %s", fnf_error)
    except yaml.YAMLError as yaml_error:
        # Log YAML parsing error
        logger.error("Error parsing YAML file: %s", yaml_error)
    except KeyError as key_error:
        # Log missing data error
        logger.error("Missing expected data in metrics file: %s", key_error)
    except Exception as ex:
        # Log any unexpected error
        logger.error("Unexpected error occurred: %s", ex)

def image_exists(image_name: str) -> bool:
    """
    Check if a specific Docker image exists on the local system.

    Args:
        image_name (str): The name of the Docker image to check. This can be in
        the format 'repository:tag'.

    Returns:
        bool: True if the Docker image exists, False otherwise.
    """
    logger = setup_logging()  # Set up logging
    try:
        # Initialize Docker client
        client = docker.from_env()

        # List images with the specified name
        images = client.images.list(name=image_name)

        # Check if any image matches the specified name
        return any(image_name in tag for img in images for tag in img.tags)
    except docker.errors.DockerException as e:
        # Log Docker-related errors
        logger.error("Error checking image existence: %s", e)
        return False

def list_available_images(exclude_none: bool = False, exclude_base: bool = False) -> list:
    """
    Retrieve and list Docker images available on the local system.

    Args:
        exclude_none (bool): If True, filter out images with '<none>' in their tag. Defaults to False.
        exclude_base (bool): If True, filter out images with 'base' in their name. Defaults to False.

    Returns:
        list: A list of Docker image tags that match the specified filters. Each tag is a string in the format 'repository:tag'.
    """
    logger = setup_logging()  # Set up logging
    try:
        # Initialize Docker client
        client = docker.from_env()

        # Retrieve all Docker images
        images = client.images.list()

        # Extract image tags
        image_tags = [tag for img in images for tag in img.tags]

        # Filter out images with 'base' in their name if specified
        if exclude_base:
            image_tags = [img for img in image_tags if 'base' not in img.lower()]

        # Filter out images with '<none>' in their name if specified
        if exclude_none:
            image_tags = [img for img in image_tags if '<none>' not in img.lower()]

        return image_tags
    except docker.errors.DockerException as e:
        # Log Docker-related errors
        logger.error("Error listing images: %s", e)
        return []
