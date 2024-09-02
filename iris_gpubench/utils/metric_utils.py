"""
Utility functions for metrics handling in iris-gpubench.

Provides functionality to format, convert, and visualize GPU metrics.

Functions:
    format_metrics: Format YAML metrics into human-readable tables.
    save_metrics_to_csv: Convert time series data to CSV format.
    plot_metric: Helper function for plotting individual GPU metrics.
    plot_metrics: Plot and save multiple GPU metrics to a file.

Dependencies: os, yaml, tabulate, matplotlib
"""

import os
import yaml
from tabulate import tabulate
import matplotlib.backends.backend_agg as agg
from matplotlib import figure
from matplotlib import ticker

from typing import Optional

from .globals import RESULTS_DIR, LOGGER

METRIC_PLOT_PATH = os.path.join(RESULTS_DIR, 'metric_plot.png')

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
            ["Elapsed Monitor Time of Command (s)", f"{data.get('elapsed_time'):.5f}"],
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


def save_metrics_to_csv(time_series_data: dict, results_dir: str = RESULTS_DIR,) -> None:
    """
    Converts time series data to CSV format and saves it to a file.

    Args:
        time_series_data (dict): A dictionary containing time series data with keys
                                 'timestamp', 'gpu_idx', 'util', 'power', 'temp', 'mem'.
        results_dir (str): The directory where the CSV file should be saved.
                        Defaults to RESULTS_DIR.

    Raises:
        ValueError: If the input data is invalid or missing required keys.
        IOError: If an error occurs while writing to the file.
    """
    try:
        # Validate input data
        required_keys = ['timestamp', 'gpu_idx', 'util', 'power', 'temp', 'mem']
        if not all(key in time_series_data for key in required_keys):
            raise ValueError("Input data is missing required keys")

        # Extract data from the input dictionary
        timestamps = time_series_data['timestamp']
        gpu_indices = time_series_data['gpu_idx']
        util = time_series_data['util']
        power = time_series_data['power']
        temp = time_series_data['temp']
        mem = time_series_data['mem']

        # Validate data lengths
        data_length = len(timestamps)
        if not all(len(data) == data_length for data in [gpu_indices, util, power, temp, mem]):
            raise ValueError("All data arrays must have the same length")

        # Initialize a list to store CSV lines
        csv_lines = []

        # Add header to CSV
        csv_header = "timestamp,gpu_index,utilization,power,temperature,memory"
        csv_lines.append(csv_header)

        # Flatten the data and format it as CSV
        for reading_index in range(data_length):
            for gpu_index, gpu_util in enumerate(util[reading_index]):
                # Format each line as a CSV entry
                csv_line = (
                    f"{timestamps[reading_index]},"
                    f"{gpu_indices[reading_index][gpu_index]},"
                    f"{gpu_util},"
                    f"{power[reading_index][gpu_index]},"
                    f"{temp[reading_index][gpu_index]},"
                    f"{mem[reading_index][gpu_index]}"
                )
                # Append the formatted line to the list
                csv_lines.append(csv_line)

        # Join the lines with newline characters to create the final CSV string
        csv_data = "\n".join(csv_lines)

        # Ensure the target directory exists and create if not
        os.makedirs(results_dir, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(results_dir, 'timeseries.csv')

        # Open the file in write mode with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(csv_data)

        LOGGER.info("CSV data successfully converted and saved to %s", file_path)

    except ValueError as error_message:
        LOGGER.error("Invalid input data: %s", error_message)
        raise
    except IOError as error_message:
        LOGGER.error("Error saving CSV data to file %s: %s", file_path, error_message)
        raise
    except Exception as error_message:
        LOGGER.error("Unexpected error: %s", error_message)
        raise

def plot_metric(axis, data: tuple, line_info: Optional[tuple] = None,
                ylim: Optional[tuple] = None) -> None:
    """
    Helper function to plot a GPU metric on a given axis.

    Args:
        axis (matplotlib.axes.Axes): The axis to plot on.
        data (tuple): Tuple containing (timestamps, y_data, title, ylabel, xlabel).
        line_info (Optional[tuple]): Tuple containing a horizontal line's y value and label.
        ylim (Optional[tuple]): y-axis limits.
    """
    # Unpack the data tuple into individual components
    timestamps, y_data, title, ylabel, xlabel = data

    # Plot the metric data for each GPU
    for i, gpu_data in enumerate(y_data):
        axis.plot(timestamps, gpu_data, label=f"GPU {i}", marker="*")

    # Optionally plot a horizontal line for a specific threshold
    if line_info:
        yline, yline_label = line_info
        axis.axhline(y=yline, color="r", linestyle="--", label=yline_label)

    # Set plot title and labels
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(ylabel, fontweight="bold")
    if xlabel:
        axis.set_xlabel(xlabel, fontweight="bold")

    # Add legend, grid, and format x-axis
    axis.legend()
    axis.grid(True)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis.tick_params(axis="x", rotation=45)

    # Optionally set y-axis limits
    if ylim:
        axis.set_ylim(ylim)


def plot_metrics(time_series_data: dict, plot_path: str = METRIC_PLOT_PATH) -> None:
    """
    Plot and save GPU metrics to a file.

    Creates plots for power usage, utilization, temperature, and memory usage,
    and saves them to the specified file path.
    """
    try:
        # Retrieve timestamps for plotting
        timestamps = time_series_data["timestamp"]

        # Prepare data for plotting for each metric and GPU
        power_data = [
            [p[i] for p in time_series_data["power"]]
            for i in range(self._stats['device_count'])
        ]
        util_data = [
            [u[i] for u in time_series_data["util"]]
            for i in range(self._stats['device_count'])
        ]
        temp_data = [
            [t[i] for t in time_series_data["temp"]]
            for i in range(self._stats['device_count'])
        ]
        mem_data = [
            [m[i] for m in time_series_data["mem"]]
            for i in range(self._stats['device_count'])
        ]

        # Create a new figure with a 2x2 grid of subplots
        fig = figure.Figure(figsize=(20, 15))
        axes = fig.subplots(nrows=2, ncols=2)

        # Create a backend for rendering the plot
        canvas = agg.FigureCanvasAgg(fig)

        # Plot each metric using the helper function
        plot_metric(
            axes[0, 0],
            (
                timestamps,
                power_data,
                f"GPU Power Usage, Total Energy: {self._stats['total_energy']:.3g}kWh",
                "Power (W)",
                "Timestamp",
            ),
            (self._stats["max_power_limit"], "Power Limit"),
        )
        plot_metric(
            axes[0, 1],
            (timestamps, util_data, "GPU Utilization", "Utilization (%)", "Timestamp"),
            ylim=(0, 100),  # Set y-axis limits for utilization
        )
        plot_metric(
            axes[1, 0],
            (timestamps, temp_data, "GPU Temperature", "Temperature (C)", "Timestamp"),
        )
        plot_metric(
            axes[1, 1],
            (timestamps, mem_data, "GPU Memory Usage", "Memory (MiB)", "Timestamp"),
            (self._stats["total_mem"], "Total Memory"),
        )

        # Adjust layout to prevent overlap
        fig.tight_layout(pad=3.0)

        # Render the figure to the canvas and save it as a PNG file
        canvas.draw()  # Ensure the figure is fully rendered
        canvas.figure.savefig(plot_path, bbox_inches="tight")  # Save the plot as PNG

        # Free memory by deleting the figure
        del fig  # Remove reference to figure to free memory

    except (FileNotFoundError, IOError) as plot_error:
        # Log specific error if the file cannot be found or opened
        LOGGER.error("Error during plotting: %s", plot_error)
    except Exception as ex:
        # Log any unexpected errors during plotting
        LOGGER.error("Unexpected error during plotting: %s", ex)