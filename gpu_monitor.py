import subprocess
import csv
import time
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

# Constants
SECONDS_IN_HOUR = 3600  # 60 seconds * 60 minutes

# Absolute path to save plots
utilization_plot_path = 'gpu_utilization_plot.png'
power_usage_plot_path = 'gpu_power_usage_plot.png'

def get_gpu_metrics():
    try:
        cmd = ['nvidia-smi', '--query-gpu=power.draw,utilization.gpu', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
        metrics = result.stdout.strip().split('\n')[0].split(', ')
        
        power_draw = float(metrics[0])
        gpu_utilization = float(metrics[1])
        
        return power_draw, gpu_utilization
        
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def calculate_total_energy(timestamps, power_readings):
    total_energy_wh = 0.0
    for i in range(1, len(timestamps)):
        delta_time = timestamps[i] - timestamps[i - 1]
        avg_power = (power_readings[i - 1] + power_readings[i]) / 2.0
        energy_increment_wh = avg_power * delta_time / SECONDS_IN_HOUR  # Convert to Wh
        total_energy_wh += energy_increment_wh
    total_energy_kwh = total_energy_wh / 1000  # Convert Wh to kWh
    return total_energy_kwh

def plot_timeseries(timestamps, data, ylabel, save_path):
    if not timestamps or not data:
        print(f"No data available for {ylabel} plot.")
        return
    
    # Convert timestamps to datetime objects
    datetime_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Plot timeseries
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.plot(datetime_timestamps, data, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(f'GPU {ylabel} Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)  # Save plot as image file
    plt.close()  # Close plot to free memory

def main(plot_flag):
    timestamps = []
    power_readings = []
    gpu_utilizations = []
    gpu_power_when_utilizing = []
    gpu_ultil_when_utilizing = []
    
    try:
        while True:
            power_usage, gpu_utilization = get_gpu_metrics()
            if power_usage is not None and gpu_utilization is not None:
                # Log metrics to lists
                current_time = time.time()
                timestamps.append(current_time)
                power_readings.append(power_usage)
                gpu_utilizations.append(gpu_utilization)
                
                # Print for real-time monitoring
                print(f"Current GPU Power Usage: {power_usage:.2f} W, GPU Utilization: {gpu_utilization:.2f} %")
                
                # Plot timeseries if --plot flag is set
                if plot_flag:
                    plot_timeseries(timestamps, gpu_utilizations, 'Utilization (%)', utilization_plot_path)
                    plot_timeseries(timestamps, power_readings, 'Power Usage (W)', power_usage_plot_path)
                
                # Record power readings only when GPU utilization > 0.00%
                if gpu_utilization > 0.00:
                    gpu_power_when_utilizing.append(power_usage)
                    gpu_ultil_when_utilizing.append(gpu_utilization)
                
            time.sleep(1)  # Check every 1 second
            
    except KeyboardInterrupt:
        # Calculate total energy consumed on exit
        total_energy = calculate_total_energy(timestamps, power_readings)
        
        # Calculate average GPU utilization (ignore 0.00% values)
        if len(gpu_ultil_when_utilizing) > 0:
            avg_gpu_utilization = sum(gpu_ultil_when_utilizing) / len(gpu_ultil_when_utilizing)
        else:
            avg_gpu_utilization = 0.0
        
        # Calculate average GPU power consumption when utilization > 0.00%
        if len(gpu_power_when_utilizing) > 0:
            avg_gpu_power = sum(gpu_power_when_utilizing) / len(gpu_power_when_utilizing)
        else:
            avg_gpu_power = 0.0
        
        # Fetch max power from nvidia-smi
        try:
            cmd = ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
            max_power = float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi to get max power: {e}")
            max_power = None
        
        # Prepare data for tabulate
        data = [
            ["Total GPU Energy Consumed (kWh)", f"{total_energy:.5f}"],
            ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{avg_gpu_utilization:.2f}"],
            ["Avg GPU Power (for >0.00% GPU Util.) (W)", f"{avg_gpu_power:.2f} (max {max_power:.2f})"]
        ]
        
        # Print as table
        print("")
        print("")
        print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Optionally save plots based on --plot flag
        if plot_flag:
            print(f"Plots saved as {utilization_plot_path} and {power_usage_plot_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU metrics and optionally plot timeseries.')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of GPU metrics over time.')
    args = parser.parse_args()
    
    main(args.plot)
