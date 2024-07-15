"""
Collects live GPU metrics using nvidia-smi and carbon produced by GPU using 
The nationalgridESO Regional Carbon Intensity API:
https://api.carbonintensity.org.uk/regional

Usage:
    python gpu_monitor.py

Options:
    --plot
        Produces live plots of the collected GPU Metrics

Parameters:
    CARBON_INSTENSITY_REGION_SHORTHAND: The region for the The nationalgridESO 
                                        Regional Carbon Intensity API 
"""
import argparse
import subprocess
import time
from typing import Tuple, Optional
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import requests
import yaml

# Constants
SECONDS_IN_HOUR = 3600  # 60 seconds * 60 minutes

# Absolute path to save plots and metrics
UTILIZATION_PLOT_PATH = './results/gpu_utilization_plot.png'
POWER_USAGE_PLOT_PATH = './results/gpu_power_usage_plot.png'
METRICS_PATH = './results/metrics.yml'

# Choose the Carbon Intensity Region
# (ie "GB" or if in Oxford "South England")
CARBON_INSTENSITY_REGION_SHORTHAND = "South England"
CARBON_INTENTSITY_URL = "https://api.carbonintensity.org.uk/regional"

def get_gpu_metrics() -> Tuple[Optional[float], Optional[float]]:
    '''
    Uses nvidia-smi to return the GPU's power draw amd utilization at the time
    the command it run.

        Returns: 
            power_draw (float): gpu power usage in W.
            gpu_utilization(float)
    '''
    try:
        cmd = ['nvidia-smi', '--query-gpu=power.draw,utilization.gpu',
               '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                check=True, text=True)
        metrics = result.stdout.strip().split('\n')[0].split(', ')

        power_draw = float(metrics[0])
        gpu_utilization = float(metrics[1])

        return power_draw, gpu_utilization

    except subprocess.CalledProcessError as error_message:
        print(f"Error running nvidia-smi: {error_message}")
        return None, None
    except ValueError as error_message:
        print(f"Error converting value to float: {error_message}")
        return None, None
    except IndexError as error_message:
        print(f"Error accessing list elements: {error_message}")
        return None, None

def get_carbon_forcast_and_index() -> Tuple[Optional[float], Optional[str],
                                            Optional[str]]:
    '''
    Uses The nationalgridESO Regional Carbon Intensity API to collect the
    current carbon emissions for regions in GB. These are updated the values 
    every 30 minutes so the date and time is also recorded for the benchmark.

        Returns: 
            carbon_forcast(float): current carbon intensity (gC02/kWh)
            carbon_index(str): Very high/ High/ Moderate/ Low/ Very Low
            date_and_time(str): The date and time of the reading.
    '''
    # Record the current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Timeout to prevent program hanging indefinitley on request
    timeout_seconds = 30
    try:
        # Send GET request to the API
        response = requests.get(CARBON_INTENTSITY_URL,
                                headers={'Accept': 'application/json'},
                                timeout=timeout_seconds)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        regions = data['data'][0]['regions']
        for region in regions:
            if region['shortname'] == CARBON_INSTENSITY_REGION_SHORTHAND:
                intensity = region['intensity']
                carbon_forcast = float(intensity['forecast'])
                carbon_index = intensity['index']

                return carbon_forcast, carbon_index, formatted_datetime
        return None, None, formatted_datetime


    except requests.exceptions.RequestException as error_message:
        print(f"Error request timed out (30s): {error_message}")
        return None, None

def calculate_total_energy(timestamps: list, power_readings: list) -> float:
    '''
    Computes the total energy consumed by the GPU.

        Parameters:
            timestamps(list): A list of timestamps of each power reading (~1s 
                              apart)
            power_readings(list): A list of power readings (W)
        Returns: 
            total_energy_kwh(float): total energy (kWh)
    '''
    total_energy_wh = 0.0
    for i in range(1, len(timestamps)):
        delta_time = timestamps[i] - timestamps[i - 1]
        avg_power = (power_readings[i - 1] + power_readings[i]) / 2.0
        energy_increment_wh = avg_power * delta_time / SECONDS_IN_HOUR  # Convert to Wh
        total_energy_wh += energy_increment_wh
    total_energy_kwh = total_energy_wh / 1000  # Convert Wh to kWh
    return total_energy_kwh

def plot_timeseries(timestamps: list, data_reading: list, ylabel: str,
                    save_path: str) -> None:
    '''
    Plots the timeseries for a given data reading.

        Returns: 
            Saves the plot as a image file.
    '''
    if not timestamps or not data_reading:
        print(f"No data available for {ylabel} plot.")
        return

    # Convert timestamps to datetime objects
    datetime_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]

    # Plot timeseries
    axes = plt.subplots(figsize=(12, 6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[1].plot(datetime_timestamps, data_reading, marker='o', linestyle='-')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'GPU {ylabel} Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)  # Save plot as image file
    plt.close()  # Close plot to free memory

def main(plot_flag):
    '''
    The main function. GPU data is collected and saved to a yaml file. 

    Parameters:
        plot_flag: allows users to decide whether to plot the figures.
    '''
    metrics = {
        "timestamps": [],
        "power_readings": [],
        "gpu_utilizations": [],
        "gpu_power_when_utilizing": [],
        "gpu_util_when_utilizing": []
    }

    try:
        while True:
            power_usage, gpu_utilization = get_gpu_metrics()
            if power_usage is not None and gpu_utilization is not None:
                # Log metrics to lists
                current_time = time.time()
                metrics["timestamps"].append(current_time)
                metrics["power_readings"].append(power_usage)
                metrics["gpu_utilizations"].append(gpu_utilization)

                # Print for real-time monitoring
                print(f"Current GPU Power Usage: {power_usage:.2f} W, GPU Utilization: {gpu_utilization:.2f} %")

                # Plot timeseries if --plot flag is set
                if plot_flag:
                    plot_timeseries(metrics["timestamps"],
                                    metrics["gpu_utilizations"],
                                    'Utilization (%)', UTILIZATION_PLOT_PATH)
                    plot_timeseries(metrics["timestamps"],
                                    metrics["power_readings"],
                                    'Power Usage (W)', POWER_USAGE_PLOT_PATH)

                # Record power readings only when GPU utilization > 0.00%
                if gpu_utilization > 0.00:
                    metrics["gpu_power_when_utilizing"].append(power_usage)
                    metrics["gpu_util_when_utilizing"].append(gpu_utilization)

            time.sleep(1)  # Check every 1 second

    except KeyboardInterrupt:
        # Calculate total energy consumed on exit
        total_energy = calculate_total_energy(metrics["timestamps"], metrics["power_readings"])

        # Calculate average GPU utilization (ignore 0.00% values)
        avg_gpu_utilization = sum(metrics["gpu_util_when_utilizing"]) / len(metrics["gpu_util_when_utilizing"]) if metrics["gpu_util_when_utilizing"] else 0.0

        # Calculate average GPU power consumption when utilization > 0.00%
        avg_gpu_power = sum(metrics["gpu_power_when_utilizing"]) / len(metrics["gpu_power_when_utilizing"]) if metrics["gpu_power_when_utilizing"] else 0.0

        # Fetch max power from nvidia-smi
        try:
            cmd = ['nvidia-smi', '--query-gpu=power.limit',
                   '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
            max_power = float(result.stdout.strip())
        except subprocess.CalledProcessError as error_message:
            print(f"Error running nvidia-smi to get max power: {error_message}")
            max_power = None

        # Collect the Current Carbon Emission Rate and time/date
        carbon_forcast, carbon_index, data_time = get_carbon_forcast_and_index()

        # Compute the Carbon Emit by process gC02
        total_carbon = total_energy*carbon_forcast

        # Save to YAML file
        metrics_to_save = {
            "total_GPU_Energy": total_energy,   # kWh (kilo-Watt-hour)
            "av_GPU_load": avg_gpu_utilization, # % (percent)
            "av_GPU_power": avg_gpu_power,      # W (Watts)
            "max_GPU_power": max_power,         # W (Watts)
            "carbon_forcast": carbon_forcast,   # gCO2/kWh
            "carbon_index": str(carbon_index),       # Very Low to Very High
            "total_GPU_carbon": total_carbon,   # gC02 (grams of CO2)
            "date_time": data_time              # Carbon Reading Date-Time 
        }

        # Save to YAML file
        with open(METRICS_PATH, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(metrics_to_save, yaml_file, default_flow_style=False)

        # Optionally save plots based on --plot flag
        if plot_flag:
            print(f"Plots saved as {UTILIZATION_PLOT_PATH} and {POWER_USAGE_PLOT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU metrics and optionally plot timeseries.')
    parser.add_argument('--plot', action='store_true',
                        help='Enable plotting of GPU metrics over time.')
    args = parser.parse_args()

    main(args.plot)
