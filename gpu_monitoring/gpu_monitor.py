# gpu_monitor.py

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import matplotlib.figure as figure
import matplotlib.backends.backend_agg as agg
import matplotlib.ticker as ticker

import pynvml
import requests
import yaml
from tabulate import tabulate



class GPUMonitor:
    """
    Manages NVIDIA GPU metrics using NVML and collects carbon metrics from the
    National Grid ESO Regional Carbon Intensity API.
    """

    def __init__(self,
                 monitor_interval=1,
                 carbon_region_shorthand="South England"):
        """
        Initializes the GPUMonitor class.

        Args:
            monitor_interval (int): Interval in seconds for collecting GPU metrics.
            carbon_region_shorthand (str): Region for carbon intensity API.
        """
        self.monitor_interval = monitor_interval
        self.carbon_region_shorthand = carbon_region_shorthand

        # Initialize time series data for GPU metrics
        self._time_series_data = {
            'timestamp': [],
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': [],
            # 'total_energy': []  # Added for storing total energy time series
        }

        # Initialize private GPU metrics as a dict of Lists
        self.current_gpu_metrics = {
            'gpu_idx': [],
            'util': [],
            'power': [],
            'temp': [],
            'mem': []
        }

        # Initialize Previous Power
        self.previous_power = []

        # Initialize Previous Power
        self.previous_power = []

        # Initialize stats
        self._stats = {}

        # Initialize pynvml
        pynvml.nvmlInit()
        LOGGER.info("NVML initialized")

        # Number of GPUS
        self.device_count = pynvml.nvmlDeviceGetCount()

    def __setup_stats(self) -> None:
        """
        Initializes and returns a dictionary with GPU statistics and default values.
        """
        # Find The First GPU's Name and Max Power
        first_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(first_handle)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(first_handle)
        power_limit = power_limit / 1000.0  # Convert from mW to W
        total_memory_info = pynvml.nvmlDeviceGetMemoryInfo(first_handle)
        total_memory = total_memory_info.total / (1024 ** 2)  # Convert bytes to MiB

        # Collect initial carbon forecast
        carbon_forecast = self.__update_carbon_forecast()

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

    def __update_gpu_metrics(self) -> None:
        """
        Retrieves the current GPU metrics for all GPUs and updates the internal time series data.

        This method updates `self._time_series_data` with the following information:
            - 'timestamp': List of timestamps
            - 'gpu_idx': List of GPU indices
            - 'util': List of GPU utilization percentages
            - 'power': List of GPU power usage in watts
            - 'temp': List of GPU temperatures in degrees Celsius
            - 'mem': List of used GPU memory in MiB
            - 'total_energy': List of total energy consumed in kWh
        """
        try:

            # Store previous power readings
            # Store previous power readings
            self.previous_power = self.current_gpu_metrics['power']

            # Retrieve metrics for each GPU
            timestamps = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_gpu_metrics['gpu_idx'] = []
            self.current_gpu_metrics['util'] = []
            self.current_gpu_metrics['power'] = []
            self.current_gpu_metrics['temp'] = []
            self.current_gpu_metrics['mem'] = []

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory = memory_info.used / (1024 ** 2)  # Convert bytes to MiB

                # Append metrics directly to the dictionary lists
                self.current_gpu_metrics['gpu_idx'].append(i)
                self.current_gpu_metrics['util'].append(utilization)
                self.current_gpu_metrics['power'].append(power_usage)
                self.current_gpu_metrics['temp'].append(temperature)
                self.current_gpu_metrics['mem'].append(memory)

            # Append new data to time series
            self._time_series_data['timestamp'].append(timestamps)
            self._time_series_data['gpu_idx'].append(self.current_gpu_metrics['gpu_idx'])
            self._time_series_data['util'].append(self.current_gpu_metrics['util'])
            self._time_series_data['power'].append(self.current_gpu_metrics['power'])
            self._time_series_data['temp'].append(self.current_gpu_metrics['temp'])
            self._time_series_data['mem'].append(self.current_gpu_metrics['mem'])

            # Update total energy and append to time series
            if len(self.previous_power) > 0:
                self.__update_total_energy()
                # self._time_series_data['total_energy'].append(self._stats['total_energy'])
            LOGGER.info("Updated GPU metrics: %s", self._time_series_data)

        except pynvml.NVMLError as error_message:
            LOGGER.error("NVML Error: %s", error_message)


    def __update_carbon_forecast(self) -> Optional[float]:
        """
        Uses The nationalgridESO Regional Carbon Intensity API to collect current carbon emissions.

        Returns:
            Optional[float]: Current carbon intensity.
        """
        try:
            response = requests.get(CARBON_INTENSITY_URL,
                                    headers={'Accept': 'application/json'},
                                    timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            regions = data['data'][0]['regions']

            for region in regions:
                if region['shortname'] == self.carbon_region_shorthand:
                    intensity = region['intensity']
                    carbon_forecast = float(intensity['forecast'])
                    LOGGER.info("Carbon forecast for '%s': %f",
                                self.carbon_region_shorthand, carbon_forecast)
                    return carbon_forecast

        except requests.exceptions.RequestException as error_message:
            LOGGER.error("Error request timed out (30s): %s", error_message)

        return None

    def __update_total_energy(self) -> None:
        """
        Computes the total energy consumed by all GPUs.

        Updates the total energy consumed and stores it in `self._stats`.
        """
        
        current_power = self.current_gpu_metrics['power']

        # Check if previous_power and current_power have the same length
        if len(self.previous_power) != self.device_count or len(current_power) != self.device_count:
            LOGGER.error(
                "Length of previous_power or current_power does not match the number of devices."
            )
            raise ValueError(
                "Length of previous_power or current_power does not match the number of devices."
            )

        # Convert Collection Interval from Seconds to Hours
        collection_interval_h = self.monitor_interval / SECONDS_IN_HOUR

        # Calculate total energy consumed in kWh
        energy_wh = sum(
            ((prev + curr) / 2) * collection_interval_h
            for prev, curr in zip(self.previous_power, current_power)
        )
        energy_kwh = energy_wh / 1000.0  # Convert Wh to kWh

        # Update total energy in stats
        self._stats["total_energy"] += energy_kwh
        LOGGER.info("Updated total energy: %f kWh", self._stats['total_energy'])

        # Update previous power for the next interval
        self.previous_power = current_power


    def __completion_stats(self) -> None:
        """
        Calculates and Updates completion metrics.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        # First Fill Carbon Forecast End time and End Forecast
        self._stats["end_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stats["end_carbon_forecast"] = self.__update_carbon_forecast()

        # Fill Total Energy and Total Carbon Estimations
        self._stats["av_carbon_forecast"] = (
            self._stats["start_carbon_forecast"] +
            self._stats["end_carbon_forecast"]
        ) / 2
        self._stats["total_carbon"] = (
            self._stats["total_energy"] * self._stats["av_carbon_forecast"]
        )

        # Number of Readings where GPUs are being utilized
        n_utilized = 0

        # Compute average power/ utilization/ memory and temperature whilst gpus
        # are being utilized
        util = self._time_series_data["util"]
        power = self._time_series_data["power"]
        temp = self._time_series_data["temp"]
        mem = self._time_series_data["mem"]

        for gpu_idx, gpu_util in enumerate(util):  # Iterate over each GPU index and its utilization
            for reading_idx, util_reading in enumerate(gpu_util):  # Iterate over each reading
                if util_reading > 0:
                    self._stats["av_util"] += util_reading
                    self._stats["av_power"] += power[gpu_idx][reading_idx]
                    self._stats["av_mem"] += mem[gpu_idx][reading_idx]
                    self._stats["av_temp"] += temp[gpu_idx][reading_idx]
                    n_utilized += 1

        # Avoid division by zero
        if n_utilized > 0:
            self._stats["av_util"] /= n_utilized
            self._stats["av_power"] /= n_utilized
            self._stats["av_mem"] /= n_utilized
            self._stats["av_temp"] /= n_utilized

        LOGGER.info("Completion stats updated: %s", self._stats)

        LOGGER.info("Completion statistics updated: %s", self._stats)

    def save_stats_to_yaml(self, file_path: str):
        """
        Saves stats to a YAML file.

        Args:
            file_path (str): Path to the YAML file.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as yaml_file:
                yaml.dump(self._stats, yaml_file, default_flow_style=False)
            LOGGER.info("Stats saved to YAML file: %s", file_path)
        except IOError as io_error:
            LOGGER.error("Failed to save stats to YAML file: %s. Error: %s",
                         file_path, io_error)

    @staticmethod
    def plot_metric(ax, data, line_info=None, ylim=None):
        """
        Helper function to plot a GPU metric on a given axis.
        """
        timestamps, y_data, title, ylabel, xlabel = data

        for i, gpu_data in enumerate(y_data):
            ax.plot(timestamps, gpu_data, label=f'GPU {i}', marker='*')

        if line_info:
            yline, yline_label = line_info
            ax.axhline(y=yline, color='r', linestyle='--', label=yline_label)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(axis='x', rotation=45)

        if ylim:
            ax.set_ylim(ylim)

    def plot_metrics(self):
        """
        Plot and save the GPU metrics to a file.
        """
        timestamps = self._time_series_data['timestamp']

        # Prepare data for plotting
        power_data = [[p[i] for p in self._time_series_data['power']] for i in range(self.device_count)]
        util_data = [[u[i] for u in self._time_series_data['util']] for i in range(self.device_count)]
        temp_data = [[t[i] for t in self._time_series_data['temp']] for i in range(self.device_count)]
        mem_data = [[m[i] for m in self._time_series_data['mem']] for i in range(self.device_count)]

        # Create a new figure and axes
        fig = figure.Figure(figsize=(20, 15))
        axes = fig.subplots(nrows=2, ncols=2)

        # Create a backend for rendering the plot
        canvas = agg.FigureCanvasAgg(fig)

        # Plot each metric using the helper function
        self.plot_metric(
            axes[0, 0],
            (timestamps,
             power_data,
             f'GPU Power Usage, Total Energy: {self._stats["total_energy"]:.3g}kWh',
             'Power (W)', 'Timestamp'),
            (self._stats["max_power_limit"], 'Power Limit')
        )
        self.plot_metric(
            axes[0, 1],
            (timestamps, util_data, 'GPU Utilization', 'Utilization (%)', 'Timestamp'),
            ylim=(0, 100)  # Set y-axis limits for utilization
        )
        self.plot_metric(
            axes[1, 0],
            (timestamps, temp_data, 'GPU Temperature', 'Temperature (C)', 'Timestamp')
        )
        self.plot_metric(
            axes[1, 1],
            (timestamps, mem_data, 'GPU Memory Usage', 'Memory (MiB)', 'Timestamp'),
            (self._stats["total_mem"], 'Total Memory')
        )

        # Ensure tight_layout is applied
        fig.tight_layout(pad=3.0)

        # Save the 2x2 grid plot
        canvas.draw()  # Render the figure to the canvas
        canvas.figure.savefig(METRIC_PLOT_PATH, bbox_inches='tight')  # Save the plot as PNG


        # Close the figure to free memory
        canvas.draw()  # Ensure the figure is fully rendered
        del fig  # Remove reference to figure to free memory

    def _live_monitor(self) -> None:
        """
        Clears the terminal and prints the current GPU metrics in a formatted table.

        This method clears the terminal screen, fetches the current date and time,
        and then prints the GPU metrics as a grid table with headers.
        """
        os.system('clear')
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Define headers for the table
        headers = [
            f'GPU Index ({self._stats["name"]})',
            'Utilization (%)',
            f"Power (W) / Max {self._stats['max_power_limit']} W",
            'Temperature (C)',
            f"Memory (MiB) / Total {self._stats['total_mem']} MiB"
        ]

        # Print current GPU metrics with date and time
        print(
            f"Current GPU Metrics as of {current_time}:\n"
            f"{tabulate(self.current_gpu_metrics, headers=headers, tablefmt='grid')}"
        )



    def run(self, live_monitoring: bool = False, plot: bool = False,
            live_plot: bool = False) -> None:
        """
        Runs the GPU monitoring and plotting process.
        """
        # Print the metrics as a formatted table
        try:
            # Initialize statistics and metrics
            self.__setup_stats()

            # Start the monitoring loop
            while True:
                # Update GPU metrics
                self.__update_gpu_metrics()

                # Update the plot every iteration or after a certain interval
                if live_plot:
                    try:
                        self.plot_metrics()
                    except (FileNotFoundError, IOError) as plot_error:
                        LOGGER.error("Error during plotting: %s", plot_error)
                        continue  # Skip plotting and continue monitoring

                if live_monitoring:
                    self._live_monitor()

                # Wait for the defined collection interval before the next iteration
                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            # Handle interruption (e.g., Ctrl+C) by storing completion stats
            self.__completion_stats()
            LOGGER.info("Monitoring stopped.")
            print("\nMonitoring stopped.")

            if plot:
                try:
                    self.plot_metrics()
                except (FileNotFoundError, IOError) as plot_error:
                    LOGGER.error("Error during plotting: %s", plot_error)

        finally:
            # Properly shut down pynvml
            pynvml.nvmlShutdown()
            LOGGER.info("NVML shutdown")
