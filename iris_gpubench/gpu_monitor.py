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
import pynvml
import yaml
from tabulate import tabulate

from .carbon_metrics import get_carbon_forecast
from .gpu_victoria_exporter import VictoriaMetricsExporter

# Global Variables
from .utils.globals import RESULTS_DIR, LOGGER, MONITOR_INTERVAL
SECONDS_IN_HOUR = 3600  # Number of seconds in an hour
METRICS_FILE_PATH = os.path.join(RESULTS_DIR, 'metrics.yml')
METRIC_PLOT_PATH = os.path.join(RESULTS_DIR, 'metric_plot.png')

# Attempt to import docker and subprocess
try:
    import docker
    DOCKER_AVAILABLE = True
    LOGGER.info("Docker module available.")
except ImportError:
    DOCKER_AVAILABLE = False
    LOGGER.warning("Docker module not available. Docker functionality will be disabled.")

try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
    LOGGER.info("Subprocess module available.")
except ImportError:
    SUBPROCESS_AVAILABLE = False
    LOGGER.warning("Subprocess module not available. Tmux functionality will be disabled.")


class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML and collects carbon metrics from the
    National Grid ESO Regional Carbon Intensity API.
    """

    def __init__(self, monitor_interval: int = MONITOR_INTERVAL,
                 carbon_region_shorthand: str = "South England"):
        """
        Initializes the GPUMonitor class.

        Args:
            monitor_interval (int): Interval in seconds for collecting GPU metrics.
            carbon_region_shorthand (str): Region shorthand for carbon intensity API.
        """
        # General configuration
        self.config = {
            'monitor_interval': monitor_interval,
            'carbon_region_shorthand': carbon_region_shorthand
        }

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

        # Initialize pynvml
        try:
            pynvml.nvmlInit()
            LOGGER.info("NVML initialized")
        except pynvml.NVMLError as nvml_error:
            LOGGER.error("Failed to initialize NVML: %s", nvml_error)
            raise

        # Initialize Docker client if Docker is available
        self.client = docker.from_env() if DOCKER_AVAILABLE else None

        # Initialise parameter for Benchmark Container
        self.container = None

        # Initialize stats
        self.__setup_stats()

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

            # Number of GPUs
            device_count = pynvml.nvmlDeviceGetCount()

            # Initialize statistics
            self._stats = {
                "elapsed_time": 0.0,
                "av_temp": 0.0,
                "av_util": 0.0,
                "av_mem": 0.0,
                "av_power": 0.0,
                "av_carbon_forecast": 0.0,
                "end_datetime": '',
                "end_carbon_forecast": 0.0,
                "max_power_limit": power_limit,
                "name": gpu_name,
                "start_carbon_forecast": 0.0,
                "start_datetime": '',
                "total_carbon": 0.0,
                "total_energy": 0.0,
                "total_mem": total_memory,
                "device_count": device_count,
                "benchmark": ''
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
            for i in range(self._stats['device_count']):
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
                LOGGER.info("Timestamp: %s", current_time)
                self.__update_total_energy()
                LOGGER.info("Updated GPU metrics: %s", self.current_gpu_metrics)

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
            if len(self.previous_power) != self._stats['device_count'] or len(current_power) != self._stats['device_count']:
                raise ValueError("Length of previous_power or current_power does not match the number of devices.")

            # Convert monitoring interval from seconds to hours
            collection_interval_h = self.config['monitor_interval'] / SECONDS_IN_HOUR

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
            self._stats["end_carbon_forecast"] = get_carbon_forecast(self.config['carbon_region_shorthand'])

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
                for i in range(self._stats['device_count'])
            ]
            util_data = [
                [u[i] for u in self._time_series_data["util"]]
                for i in range(self._stats['device_count'])
            ]
            temp_data = [
                [t[i] for t in self._time_series_data["temp"]]
                for i in range(self._stats['device_count'])
            ]
            mem_data = [
                [m[i] for m in self._time_series_data["mem"]]
                for i in range(self._stats['device_count'])
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

    def _live_monitor_metrics(self) -> str:
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

            # Build the full message
            message = (
                f"\nCurrent GPU Metrics as of {current_time}:\n"
                f"{gpu_metrics_str}\n"
            )

            # Print the current GPU metrics in a grid format
            return message

        except OSError as os_error:
            # Log any OS-related errors during live monitoring
            LOGGER.error("OS error in live monitoring: %s", os_error)
        except ValueError as value_error:
            # Log value errors that occur during processing
            LOGGER.error("Value error in live monitoring: %s", value_error)
        except Exception as ex:
            # Log any unexpected errors
            LOGGER.error("Unexpected error in live monitoring: %s", ex)

    def _live_monitor_container(self) -> str:
        """
        Monitors and retrieves benchmark metrics and container logs in real-time.

        This method collects live metrics from the benchmarking process, retrieves
        logs from the container, and formats them into a complete message.

        It captures any potential errors during processing and logs them accordingly.

        Returns:
            str: A formatted string containing the metrics message and container logs.

        Raises:
            ValueError: If there is a value error during live monitoring.
            Exception: For any other unexpected errors during live monitoring.
        """
        try:
            # Collect live metrics
            metrics_message = self._live_monitor_metrics()

            # Collect container logs
            container_log = self.container.logs(follow=False).decode('utf-8')

            # Initialize the complete message with metrics and container logs header
            complete_message = f"{metrics_message}\nContainer Logs:\n"

            # Process the logs
            container_log = container_log.replace('\\r', '\r')
            lines = container_log.split('\n')  # Split the entire log into lines

            # Process each line to handle log loading bars
            for line in lines:
                if '\r' in line:
                    # Handle the last segment after '\r'
                    line = line.split('\r')[-1]
                # Append the processed line to the complete message
                complete_message += f"\n {line.strip()}"

            return complete_message

        except ValueError as value_error:
            # Log value errors that occur during processing
            LOGGER.error("Value error in live monitoring: %s", value_error)
            raise  # Re-raise the exception if you want it to propagate

        except Exception as ex:
            # Log any unexpected errors
            LOGGER.error("Unexpected error in live monitoring: %s", ex)
            raise  # Re-raise the exception if you want it to propagate
    
    def _live_monitor_tmux(self, session_name: str) -> str:
        """
        Monitors and retrieves benchmark metrics and tmux logs in real-time.

        This method collects live metrics from the benchmarking process, retrieves
        logs from the tmux, and formats them into a complete message.

        It captures any potential errors during processing and logs them accordingly.

        Returns:
            str: A formatted string containing the metrics message and tmux logs.

        Raises:
            ValueError: If there is a value error during live monitoring.
            Exception: For any other unexpected errors during live monitoring.
        """
        try:
            # Collect live metrics
            metrics_message = self._live_monitor_metrics()

            # Capture and display logs from tmux
            logs_command = ["tmux", "capture-pane", "-t", session_name, "-p"]
            try:
                logs = subprocess.check_output(logs_command).decode()
                LOGGER.info("Captured logs from tmux session.")
            except subprocess.CalledProcessError as e:
                LOGGER.error("Failed to capture logs from tmux session: %s", e)
            
            # Return complete message with metrics and Tmux logs header
            return f"{metrics_message}\nTmux Logs:\n\n{logs}"

        except ValueError as value_error:
            # Log value errors that occur during processing
            LOGGER.error("Value error in live monitoring: %s", value_error)
            raise  # Re-raise the exception if you want it to propagate

        except Exception as ex:
            # Log any unexpected errors
            LOGGER.error("Unexpected error in live monitoring: %s", ex)
            raise  # Re-raise the exception if you want it to propagate

    def _run_benchmark_in_docker(self, benchmark_image: str,
                                 live_monitoring: bool = True,
                                 plot: bool = True,
                                 live_plot: bool = False,
                                 monitor_logs: bool = False,
                                 victoria_exporter: bool = False) -> None:
        """
        Runs the GPU monitoring and plotting process while executing a container.

        Args:
            benchmark_image (str): The Docker container image to run.
            live_monitoring (bool): If True, enables live monitoring display.
            Defaults to True.
            plot (bool): If True, saves the metrics plot at the end. Defaults to True.
            live_plot (bool): If True, updates the plot in real-time. Defaults to False.
            monitor_logs (bool): If True, monitors both metrics and container logs.
            Defaults to False.
        """
        # Check if docker is available
        if not DOCKER_AVAILABLE:
            LOGGER.error("Docker functionality is not available. Please install Docker.")
            raise RuntimeError("The 'docker' module is required but not available. Please install it.")

        # Initialize GPU statistics
        self._stats["start_carbon_forecast"] = get_carbon_forecast(self.config['carbon_region_shorthand'])
        start_time = datetime.now() # Start timing
        self._stats["start_datetime"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        self._stats["benchmark"] = benchmark_image

        # Activate the Exporter
        victoria_exporter=True
        if victoria_exporter:
            exporter = VictoriaMetricsExporter(
                gpu_name=self._stats["name"],
                benchmark=self._stats["benchmark"]                
            )
            
        LOGGER.info("Initialized benchmark runner for tmux session.")

        try:
            # Run the container in the background
            self.container = self.client.containers.run(
                benchmark_image,
                detach=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
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
                        if monitor_logs:
                            # Monitor both metrics and container logs
                            print(self._live_monitor_container())
                        else:
                            # Monitor only metrics
                            print(self._live_monitor_metrics())
                            print(f"\n Benchmark Status: {self.container.status}")
                    
                    # Export to Victoria Metrics if enabled
                    if victoria_exporter:
                        try:
                            LOGGER.info("Export to meerkat")
                            exporter.export_metric_readings(self.current_gpu_metrics)
                        except ValueError as ve:
                            LOGGER.error("Invalid data for VictoriaMetrics export: %s", ve)
                            break
                        except requests.RequestException as re:
                            LOGGER.error("Failed to send data to VictoriaMetrics: %s", re)
                            break
                    # Wait for the specified interval before the next update
                    time.sleep(self.config['monitor_interval'])

                except (KeyboardInterrupt, SystemExit):
                    LOGGER.info("Monitoring interrupted by user.")
                    print(
                        "\nMonitoring interrupted by user."
                        "\nStopping gracefully, please wait..."
                    )
                    break
                except Exception as ex:
                    LOGGER.error("Unexpected error during monitoring: %s", ex)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("Monitoring interrupted by user.")
            print(
                "\nMonitoring interrupted by user."
                "\nStopping gracefully, please wait..."
            )
        except docker.errors.DockerException as docker_error:
            LOGGER.error("Docker error: %s", docker_error)
        except Exception as ex:
            LOGGER.error("Unexpected error: %s", ex)
        finally:
            # End timing
            end_time = datetime.now()
            self._stats['elapsed_time'] = (end_time - start_time).total_seconds()

            # Safe shutdown
            self._shutdown(plot)

            # Clean up Docker container
            if self.container:
                try:
                    self.container.remove(force=True)
                    LOGGER.info("Docker container removed.")
                except docker.errors.APIError as docker_error:
                    LOGGER.error("Failed to remove Docker container: %s", docker_error)

    def _run_benchmark_in_tmux(self, benchmark_command: str, 
                               live_monitoring: bool = True, plot: bool = True,
                               live_plot: bool = False, monitor_logs: bool = False,
                               victoria_exporter: bool = False) -> None:
        """
        Executes a benchmark command in a tmux session.

        Args:
            benchmark_command (str): The command to run in the tmux session.
            live_monitoring (bool): If True, enables live monitoring display. Defaults to True.
            plot (bool): If True, saves the metrics plot at the end. Defaults to True.
            live_plot (bool): If True, updates the plot in real-time. Defaults to False.
            monitor_logs (bool): If True, monitors both metrics and session logs. Defaults to False.
        """
        # Check if tmux is available
        if not SUBPROCESS_AVAILABLE:
            raise RuntimeError("The 'subprocess' module is required but not available. Please install it.")
        
        # Initialize GPU statistics
        self._stats["start_carbon_forecast"] = get_carbon_forecast(self.config['carbon_region_shorthand'])
        start_time = datetime.now() # Start timing
        self._stats["start_datetime"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        self._stats["benchmark"] = benchmark_command

        # Activate the Exporter
        if victoria_exporter:
            exporter = VictoriaMetricsExporter(
                gpu_name=self._stats["name"],
                benchmark=self._stats["benchmark"]                
            )

        LOGGER.info("Initialized benchmark runner for tmux session.")

        try:
            # Create a new tmux session and Run Benchmark Command
            session_name = "benchmark_session"
            tmux_command = [
                "tmux", "new-session", "-d", "-s", session_name,
                "bash -c 'cd \"$(pwd)\" && " + benchmark_command + "'"
            ]
            LOGGER.info("Starting tmux session with command: %s", benchmark_command)
            try:
                subprocess.run(tmux_command, check=True)
                LOGGER.info("Tmux session started successfully.")
            except subprocess.CalledProcessError as e:
                LOGGER.error("Failed to start tmux session: %s", e)
                raise RuntimeError(f"Failed to start tmux session: {e}") from e


            while True:
                try:
                    # Update the current GPU metrics
                    self.__update_gpu_metrics()

                    # Plot Metrics if live plotting is enabled
                    if live_plot:
                        try:
                            self.plot_metrics()
                            LOGGER.info("Live plot updated.")
                        except (FileNotFoundError, IOError) as plot_error:
                            LOGGER.error("Error during plotting: %s", plot_error)
                            continue

                    # Display live monitoring output if enabled
                    if live_monitoring:
                        if monitor_logs:
                            # Monitor both metrics and container logs
                            try:
                                print(self._live_monitor_tmux(session_name=session_name))
                            except subprocess.CalledProcessError:
                                LOGGER.info("Tmux session has ended.")
                                break
                        else:
                            print(self._live_monitor_metrics())
                            print("\nBenchmark Status: Running")
                            LOGGER.info("Live monitoring metrics displayed.")
                    
                    # Export to Victoria Metrics if enabled
                    if victoria_exporter:
                        try:
                            exporter.export_metric_readings(self.current_gpu_metrics)
                            LOGGER.info("Export to meerkat")
                        except ValueError as ve:
                            LOGGER.error("Invalid data for VictoriaMetrics export: %s", ve)
                        except requests.RequestException as re:
                            LOGGER.error("Failed to send data to VictoriaMetrics: %s", re)

                     # Check if the tmux session is still running
                    status_command = ["tmux", "has-session", "-t", session_name]
                    try:
                        subprocess.run(status_command, check=True)
                    except subprocess.CalledProcessError:
                        LOGGER.info("Tmux session has ended.")
                        break
                    # Check if tmux session is still running via logs
                    logs_command = ["tmux", "capture-pane", "-t", session_name, "-p"]
                    try:
                        subprocess.check_output(logs_command)
                        LOGGER.info("Captured logs from tmux session - still running.")
                    except subprocess.CalledProcessError as e:
                        LOGGER.error("Failed to capture logs from tmux session: %s", e)
                        LOGGER.info("Tmux session has ended.")
                        break
                    
                    # Wait for the specified interval before the next update
                    time.sleep(self.config['monitor_interval'])

                except (KeyboardInterrupt, SystemExit):
                    LOGGER.info("Monitoring interrupted by user.")
                    print("\nMonitoring interrupted by user.\nStopping gracefully, please wait...")
                    break
                except subprocess.CalledProcessError:
                        LOGGER.info("Tmux session has ended.")
                        break
                except Exception as ex:
                    LOGGER.error("Unexpected error during monitoring: %s", ex)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("Monitoring interrupted by user.")
            print("\nMonitoring interrupted by user.\nStopping gracefully, please wait...")
        except subprocess.CalledProcessError as subprocess_error:
            LOGGER.error("Subprocess error: %s", subprocess_error)
        except Exception as ex:
            LOGGER.error("Unexpected error: %s", ex)
        finally:
            # End timing
            end_time = datetime.now()
            self._stats['elapsed_time'] = (end_time - start_time).total_seconds()

            # Safe shutdown
            self._shutdown(plot)

            # Clean up tmux session
            try:
                subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
                LOGGER.info("Tmux session '%s' terminated.", session_name)
            except subprocess.CalledProcessError as e:
                LOGGER.error("Failed to clean up tmux session '%s': %s", session_name, e)

    def _shutdown(self, plot: bool) -> None:
        """
        Perform a safe and complete shutdown of the monitoring process.

        This method is responsible for finalizing the monitoring process by:
        - Finalizing and cleaning up statistics.
        - Saving the metrics plot if applicable.
        - Handling the removal of the Docker container if it exists.
        - Shutting down NVML (NVIDIA Management Library) resources.

        It logs detailed information about each step and handles potential errors
        that might occur during container management and NVML shutdown.

        Exceptions:
            - docker.errors.NotFound: If the container is not found during cleanup.
            - docker.errors.APIError: For errors related to Docker API operations
            such as stopping, removing, or force-removing containers.
            - pynvml.NVMLError: For errors related to NVML operations.
            - Exception: For any other unexpected errors during the shutdown process.
        """
        self.__completion_stats()  # Finalize and clean up statistics
        LOGGER.info("Monitoring stopped.")

        # Save the metrics plot if requested
        if plot:
            try:
                self.plot_metrics()
                LOGGER.info("Metrics plot saved.")
            except (FileNotFoundError, IOError) as plot_error:
                LOGGER.error("Error during plotting: %s", plot_error)

        # Handle NVML shutdown
        try:
            LOGGER.info("Attempting to shutdown NVML.")
            pynvml.nvmlShutdown()
            LOGGER.info("NVML shutdown successfully.")
        except pynvml.NVMLError as nvml_error:
            # Handle NVML-specific errors
            LOGGER.error("NVML error during shutdown: %s", nvml_error)
        except Exception as ex:
            # Handle any unexpected exceptions
            LOGGER.error("Unexpected error during NVML shutdown: %s", ex)

    def run(self, benchmark_command: str = None, benchmark_image: str = None,
            live_monitoring: bool = True,
            plot: bool = True, live_plot: bool = False,
            monitor_logs: bool = False,
            victoria_exporter: bool = False) -> None:
        """
        Runs the benchmark process either in a tmux session or Docker container based on provided arguments.

        This method determines whether to execute the benchmark using a tmux session or a Docker container,
        depending on the provided input. It will raise an error if both or neither `benchmark_command` and 
        `benchmark_image` are provided.

        Args:
            benchmark_command (str): The shell command to run in a tmux session.
            benchmark_image (str): The Docker container image to run.
            live_monitoring (bool): If True, enables live monitoring display during execution. Defaults to True.
            plot (bool): If True, saves the metrics plot at the end of execution. Defaults to True.
            live_plot (bool): If True, updates the metrics plot in real-time while the benchmark is running. Defaults to False.
            monitor_logs (bool): If True, monitors both GPU metrics and logs from the tmux session or Docker container. Defaults to False.
        """
        
        # Ensure that either a benchmark command or a Docker image is specified, but not both
        if benchmark_command and benchmark_image:
            # Log the error of conflicting input arguments
            LOGGER.error("Both 'benchmark_command' and 'benchmark_image' provided. Please use only one.")
            
            # Raise an error indicating that only one method of benchmark execution can be chosen
            raise ValueError("You must specify either 'benchmark_command' or 'benchmark_image', not both.")

        # Run the benchmark in a tmux session if a command is provided
        if benchmark_command:
            # Check if tmux is available
            if not SUBPROCESS_AVAILABLE:
                raise RuntimeError("The 'subprocess' module is required but not available. Please install it.")
        
            # Call the private method to handle tmux session execution
            self._run_benchmark_in_tmux(benchmark_command, live_monitoring,
                                        plot, live_plot, monitor_logs,
                                        victoria_exporter)
        
        # Run the benchmark in a Docker container if a Docker image is provided
        elif benchmark_image:
            # Check if Docker is available
            if not DOCKER_AVAILABLE:
                LOGGER.error("Docker functionality is not available. Please install Docker.")
                
                # Raise a runtime error indicating that Docker is required but not available
                raise RuntimeError("Docker functionality is not available. Please install Docker.")
            
            # Call the private method to handle Docker container execution
            self._run_benchmark_in_docker(benchmark_image, live_monitoring, plot, live_plot, monitor_logs)
        
        # If neither a benchmark command nor a Docker image is provided, raise an error
        else:
            LOGGER.error("Neither 'benchmark_command' nor 'benchmark_image' provided.")
            
            # Raise an error indicating that one of the two options must be provided to run the benchmark
            raise ValueError("You must specify either 'benchmark_command' or 'benchmark_image'.")
