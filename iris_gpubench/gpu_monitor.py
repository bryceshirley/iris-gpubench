"""
gpu_monitor.py

This module provides a GPU monitoring class using NVIDIA's NVML library to
track various metrics, including utilization, power usage, temperature, and memory.
It also integrates with the Carbon Intensity API to fetch carbon intensity data
and saves metrics and plots for analysis.

Dependencies:
- pynvml: NVIDIA Management Library for GPU monitoring.
- requests: For HTTP requests to the Carbon Intensity API.
- yaml: For saving metrics to YAML format.
- matplotlib: For plotting GPU metrics.
- tabulate: For tabular data representation.

Usage:
- Create an instance of GPUMonitor with desired intervals and region settings.
- Use the `run` method to start monitoring, with options for live monitoring,
  plotting, and live plotting.
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, List

import matplotlib.backends.backend_agg as agg
from matplotlib import figure
from matplotlib import ticker
import docker
import pynvml
import yaml
from tabulate import tabulate

from .utils import setup_logging
from .carbon_metrics import get_carbon_forecast

# Set up logging with specific configuration
LOGGER = setup_logging()

RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = os.path.join(RESULTS_DIR, 'metrics.yml')
METRIC_PLOT_PATH = os.path.join(RESULTS_DIR, 'metric_plot.png')
TIMEOUT_SECONDS = 30

class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML and collects carbon metrics from the
    National Grid ESO Regional Carbon Intensity API.
    """

    def __init__(self, monitor_interval: int = 1, carbon_region_shorthand: str = "South England"):
        """
        Initializes the GPUMonitor class.

        Args:
            monitor_interval (int): Interval in seconds for collecting GPU metrics.
            carbon_region_shorthand (str): Region shorthand for carbon intensity API.
        """
        self.monitor_interval = monitor_interval
        self.carbon_region_shorthand = carbon_region_shorthand

        # Initialize time series data for GPU metrics
        self._time_series_data: Dict[str, List] = {
            'timestamp': [],
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': [],
        }

        # Initialize private GPU metrics as a dict of Lists
        self.current_gpu_metrics: Dict[str, List] = {
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': []
        }

        # Initialize Previous Power
        self.previous_power: List[float] = []

        # Initialize stats
        self._stats: Dict[str, float] = {}

        # Initialize pynvml
        try:
            pynvml.nvmlInit()
            LOGGER.info("NVML initialized")
        except pynvml.NVMLError as nvml_error:
            LOGGER.error("Failed to initialize NVML: %s", nvml_error)
            raise

        # Number of GPUs
        self.device_count = pynvml.nvmlDeviceGetCount()

        # Initialize Docker client
        self.client = docker.from_env()

    def __setup_stats(self) -> None:
        """
        Initializes GPU statistics and records initial carbon forecast.

        Sets up initial statistics including GPU name, power limits, total memory,
        and initial carbon forecast.
        """
        try:
            # Get handle for the first GPU
            first_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Retrieve GPU properties
            gpu_name = pynvml.nvmlDeviceGetName(first_handle)
            power_limit = (
                pynvml.nvmlDeviceGetPowerManagementLimit(first_handle) / 1000.0
            ) # Convert mW to W
            total_memory = (
                pynvml.nvmlDeviceGetMemoryInfo(first_handle).total / (1024 ** 2)
            )  # Convert bytes to MiB

            # Get initial carbon forecast
            carbon_forecast = get_carbon_forecast(self.carbon_region_shorthand)

            # Initialize statistics
            self._stats = {
                "av_temp": 0.0,
                "av_util": 0.0,
                "av_mem": 0.0,
                "av_power": 0.0,
                "av_carbon_forecast": 0.0,
                "end_datetime": '',
                "end_carbon_forecast": 0.0,
                "max_power_limit": power_limit,
                "name": gpu_name,
                "start_carbon_forecast": carbon_forecast,
                "start_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_carbon": 0.0,
                "total_energy": 0.0,
                "total_mem": total_memory,
            }

            LOGGER.info("Statistics initialized: %s", self._stats)

        except pynvml.NVMLError as nvml_error:
            LOGGER.error("Failed to setup GPU stats: %s", nvml_error)
            raise


    def __update_gpu_metrics(self) -> None:
        """
        Updates the GPU metrics and appends new data to the time series.

        Retrieves current metrics for each GPU and updates internal data structures.
        """
        try:
            # Store previous power readings
            self.previous_power = self.current_gpu_metrics['power']

            # Retrieve the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Reset current GPU metrics
            self.current_gpu_metrics = {
                'gpu_idx': [],
                'util': [],
                'power': [],
                'temp': [],
                'mem': []
            }

            # Collect metrics for each GPU
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Retrieve metrics for the current GPU
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                # Append metrics to current GPU metrics
                self.current_gpu_metrics['gpu_idx'].append(i)
                self.current_gpu_metrics['util'].append(utilization)
                self.current_gpu_metrics['power'].append(power_usage)
                self.current_gpu_metrics['temp'].append(temperature)
                self.current_gpu_metrics['mem'].append(memory)

            # Append new data to time series data, including the timestamp
            self._time_series_data['timestamp'].append(current_time)
            for metric, values in self.current_gpu_metrics.items():
                self._time_series_data[metric].append(values)

            # Update total energy and append to time series if previous power data exists
            if self.previous_power:
                self.__update_total_energy()
                LOGGER.info("Updated GPU metrics: %s", self._time_series_data)

        except pynvml.NVMLError as nvml_error:
            LOGGER.error("NVML Error: %s", nvml_error)

    def __update_total_energy(self) -> None:
        """
        Computes and updates the total energy consumption based on GPU power readings.

        Calculates energy consumption in kWh and updates the total energy in stats.
        """
        try:
            # Get current power readings
            current_power = self.current_gpu_metrics['power']

            # Ensure power readings match the number of devices
            if len(self.previous_power) != self.device_count or len(current_power) != self.device_count:
                raise ValueError("Length of previous_power or current_power does not match the number of devices.")

            # Convert monitoring interval from seconds to hours
            collection_interval_h = self.monitor_interval / SECONDS_IN_HOUR

            # Calculate energy consumption in Wh using the trapezoidal rule
            energy_wh = sum(((prev + curr) / 2) * collection_interval_h for prev, curr in zip(self.previous_power, current_power))

            # Convert energy consumption to kWh
            energy_kwh = energy_wh / 1000.0

            # Update total energy consumption in stats
            self._stats["total_energy"] += energy_kwh
            LOGGER.info("Updated total energy: %f kWh", self._stats['total_energy'])

            # Update previous power readings to current
            self.previous_power = current_power

        except ValueError as value_error:
            # Log specific error when power reading lengths are mismatched
            LOGGER.error("ValueError in total energy calculation: %s", value_error)

        except Exception as ex:
            # Log unexpected errors during energy calculation
            LOGGER.error("Unexpected error in total energy calculation: %s", ex)


    def __completion_stats(self) -> None:
        """
        Calculates and updates completion statistics including average metrics
        and total carbon emissions.

        Updates the final stats with completion details and average values.
        """
        try:
            # Record the end time of the stats collection
            self._stats["end_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get the carbon forecast at the end time
            self._stats["end_carbon_forecast"] = get_carbon_forecast(self.carbon_region_shorthand)

            # Calculate the average carbon forecast over the duration
            self._stats["av_carbon_forecast"] = (self._stats["start_carbon_forecast"] + self._stats["end_carbon_forecast"]) / 2

            # Calculate the total carbon emissions based on total energy consumed
            self._stats["total_carbon"] = self._stats["total_energy"] * self._stats["av_carbon_forecast"]

            # Extract time series data for utility, power, temperature, and memory
            util = self._time_series_data["util"]
            power = self._time_series_data["power"]
            temp = self._time_series_data["temp"]
            mem = self._time_series_data["mem"]

            n_utilized = 0  # Counter for utilized GPU readings

            # Iterate through GPU readings to calculate averages
            for gpu_idx, gpu_util in enumerate(util):
                for reading_idx, util_reading in enumerate(gpu_util):
                    if util_reading > 0:  # Consider only utilized GPU readings
                        self._stats["av_util"] += util_reading
                        self._stats["av_power"] += power[gpu_idx][reading_idx]
                        self._stats["av_mem"] += mem[gpu_idx][reading_idx]
                        self._stats["av_temp"] += temp[gpu_idx][reading_idx]
                        n_utilized += 1

            # Compute the averages for utilization, power, memory, and temperature
            if n_utilized > 0:
                self._stats["av_util"] /= n_utilized
                self._stats["av_power"] /= n_utilized
                self._stats["av_mem"] /= n_utilized
                self._stats["av_temp"] /= n_utilized

            # Log the updated statistics
            LOGGER.info("Completion stats updated: %s", self._stats)

        except KeyError as key_error:
            # Handle missing key errors in time series data
            LOGGER.error("Missing key in time series data: %s", key_error)
        except ValueError as value_error:
            # Handle value errors during statistics calculation
            LOGGER.error("Value error during stats calculation: %s", value_error)
        except Exception as ex:
            # Handle any unexpected errors
            LOGGER.error("Unexpected error in completion stats calculation: %s", ex)


    def save_stats_to_yaml(self, file_path: str = METRICS_FILE_PATH) -> None:
        """
        Saves the collected statistics to a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Raises:
            IOError: If there is an issue writing to the file.
        """
        try:
            # Open the specified file in write mode with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as yaml_file:
                # Dump the statistics dictionary into the YAML file
                yaml.dump(self._stats, yaml_file, default_flow_style=False)

            # Log success message with file path
            LOGGER.info("Stats saved to YAML file: %s", file_path)

        except IOError as io_error:
            # Log error message if file writing fails
            LOGGER.error("Failed to save stats to YAML file: %s. Error: %s", file_path, io_error)

    @property
    def time_series_data(self) -> Dict[str, List]:
        """
        Returns the collected time series data for GPU metrics.

        Returns:
            Dict[str, List]: A dictionary with lists of time series data for various GPU metrics.
        """
        return self._time_series_data

    @staticmethod
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


    def plot_metrics(self, plot_path: str = METRIC_PLOT_PATH) -> None:
        """
        Plot and save GPU metrics to a file.

        Creates plots for power usage, utilization, temperature, and memory usage,
        and saves them to the specified file path.
        """
        try:
            # Retrieve timestamps for plotting
            timestamps = self._time_series_data["timestamp"]

            # Prepare data for plotting for each metric and GPU
            power_data = [
                [p[i] for p in self._time_series_data["power"]]
                for i in range(self.device_count)
            ]
            util_data = [
                [u[i] for u in self._time_series_data["util"]]
                for i in range(self.device_count)
            ]
            temp_data = [
                [t[i] for t in self._time_series_data["temp"]]
                for i in range(self.device_count)
            ]
            mem_data = [
                [m[i] for m in self._time_series_data["mem"]]
                for i in range(self.device_count)
            ]

            # Create a new figure with a 2x2 grid of subplots
            fig = figure.Figure(figsize=(20, 15))
            axes = fig.subplots(nrows=2, ncols=2)

            # Create a backend for rendering the plot
            canvas = agg.FigureCanvasAgg(fig)

            # Plot each metric using the helper function
            self.plot_metric(
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
            self.plot_metric(
                axes[0, 1],
                (timestamps, util_data, "GPU Utilization", "Utilization (%)", "Timestamp"),
                ylim=(0, 100),  # Set y-axis limits for utilization
            )
            self.plot_metric(
                axes[1, 0],
                (timestamps, temp_data, "GPU Temperature", "Temperature (C)", "Timestamp"),
            )
            self.plot_metric(
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

    def _live_monitor(self) -> None:
        """
        Clears the terminal and prints the current GPU metrics in a formatted table.

        Clears the terminal screen, fetches the current date and time,
        and prints the GPU metrics as a grid table with headers.
        """
        try:
            # Clear the terminal screen for fresh output
            os.system('clear')
            
            # Get the current date and time as a formatted string
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Define table headers with GPU stats information
            headers = [
                f'GPU Index ({self._stats["name"]})',
                'Utilization (%)',
                f"Power (W) / Max {self._stats['max_power_limit']} W",
                'Temperature (C)',
                f"Memory (MiB) / Total {self._stats['total_mem']} MiB"
            ]

            # Format GPU metrics
            gpu_metrics_str = tabulate(self.current_gpu_metrics, headers=headers, tablefmt='grid')
            
            # Collect container logs
            container_logs = self.container.logs(follow=False,tail).decode('utf-8')

            # Build the full message
            message = (
                f"\nCurrent GPU Metrics as of {current_time}:\n"
                f"{gpu_metrics_str}\n\n"
                f"Container Logs as of {current_time}:\n"
                f"\n{container_logs}"
            )

            # Print the current GPU metrics in a grid format
            print(message)

        except OSError as os_error:
            # Log any OS-related errors during live monitoring
            LOGGER.error("OS error in live monitoring: %s", os_error)
        except ValueError as value_error:
            # Log value errors that occur during processing
            LOGGER.error("Value error in live monitoring: %s", value_error)
        except Exception as ex:
            # Log any unexpected errors
            LOGGER.error("Unexpected error in live monitoring: %s", ex)

    def run(self, benchmark_image: str, live_monitoring: bool = False, plot: bool = False,
            live_plot: bool = False) -> None:
        """
        Runs the GPU monitoring and plotting process while executing a container.

        Args:
            benchmark_image (str): The Docker container image to run.
            live_monitoring (bool): If True, enables live monitoring display.
            plot (bool): If True, saves the metrics plot at the end.
            live_plot (bool): If True, updates the plot in real-time.
        """
        # Initialize GPU statistics
        self.__setup_stats() 

        try:
            # Run the container in the background
            self.container = self.client.containers.run(
                benchmark_image,
                detach=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
            )
            
            # Reload to update status from created to running
            self.container.reload()

            while self.container.status == 'running':
                try:
                    # Reload container status to check if it is still running
                    self.container.reload()

                    # Update the current GPU metrics
                    self.__update_gpu_metrics()

                    # Plot metrics if live plotting is enabled
                    if live_plot:
                        try:
                            self.plot_metrics()
                        except (FileNotFoundError, IOError) as plot_error:
                            # Log any errors during plotting and continue monitoring
                            LOGGER.error("Error during plotting: %s", plot_error)
                            continue  # Skip plotting and continue monitoring

                    # Display live monitoring output if enabled
                    if live_monitoring:
                        self._live_monitor()

                    # Wait for the specified interval before the next update
                    time.sleep(self.monitor_interval)

                except (KeyboardInterrupt, SystemExit):
                    # Break the loop if user interrupts or system exits
                    LOGGER.info("Monitoring interrupted by user.")
                    print(
                        f"Monitoring interrupted by user.\n"
                        f"Stopping gracefully, please wait..."
                    )
                    break
                except Exception as ex:
                    # Log any unexpected errors that occur during the monitoring loop
                    LOGGER.error("Unexpected error during monitoring: %s", ex)

        except docker.errors.DockerException as docker_error:
            # Handle Docker-specific errors, such as connection issues or container errors
            LOGGER.error("Docker error: %s", docker_error)
        except Exception as ex:
            # Handle any unexpected errors that occur during container setup
            LOGGER.error("Unexpected error: %s", ex)
        finally:
            # Remove the container
            self.container.remove()

            # Finalize statistics and perform cleanup
            self.__completion_stats()
            LOGGER.info("Monitoring stopped.")
            print("\nMonitoring stopped.")

            # Save the metrics plot if requested
            if plot:
                try:
                    self.plot_metrics()
                except (FileNotFoundError, IOError) as plot_error:
                    # Log any errors during plotting
                    LOGGER.error("Error during plotting: %s", plot_error)

            # Ensure NVML resources are released on exit
            pynvml.nvmlShutdown()
            LOGGER.info("NVML shutdown")

            # Stop and wait for the container to finish if it is still running
            if self.container.status == 'running':
                self.container.stop()  # Stop the container
                self.container.wait()  # Wait for the container to finish
                LOGGER.info("Container stopped.")
