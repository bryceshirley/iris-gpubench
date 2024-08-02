"""
Formats the metric.yml file, ouputs in time to command line and saves to a txt
file.
"""
import yaml
from tabulate import tabulate

# Path to metric.yml file
METRICS_FILE_PATH = './results/metrics.yml'

# Path to formatted metrics
FORMATTED_METRICS_PATH = './results/formatted_metrics.txt'

# Read YAML file
with open(METRICS_FILE_PATH, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Extract and convert values to appropriate types

if 'time' in data: # Check if time has be retrieved
    time_value = data['time']
av_util = data['av_util']
av_temp = data['av_temp']
av_power = data['av_power']
av_mem = data['av_mem']
max_gpu_power = int(data['max_power_limit'])
total_gpu_energy = data['total_energy']
av_carbon_forecast = data['av_carbon_forecast']
total_carbon = data['total_carbon']
total_mem = data['total_mem']
end_datetime = data["end_datetime"]
start_datetime = data["start_datetime"]
name = data["name"]

# Prepare data for tabulate
formatted_data_main = [
    ["Total GPU Energy Consumed (kWh)", f"{total_gpu_energy:.5f}"],
    ["Total GPU Carbon Emissions (gC02)",f"{total_carbon:.5f}"],
]

# Check if time has be retrieved
if 'time' in data:
    formatted_data_main.insert(0,["Benchmark Score (s)", f"{time_value:.5f}"])

formatted_data_extra = [
    ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{av_util:.5f}"],
    ["Avg GPU Power (for >0.00% GPU Util.) (W)",
      f"{av_power:.5f} (Power Limit: {max_gpu_power})"],
    ["Avg GPU Temperature (for >0.00% GPU Util.) (C)",
      f"{av_temp:.5f}"],
    ["Avg GPU Memory (for >0.00% GPU Util.) (MiB)",
      f"{av_mem:.5f} (Total Memory: {total_mem})"],
    ["Average Carbon Forcast from start/end (gCO2/kWh)",
      f"{av_carbon_forecast:.1f}"]
]
# Print as table
output = []
output.append("Benchmark Score and GPU Energy Performance")
output.append("")
output.append(tabulate(formatted_data_main, headers=["Metric", "Value"],
                       tablefmt="grid"))
output.append("")
output.append("Additional Information")
output.append("")
output.append(tabulate(formatted_data_extra, headers=["Metric", "Value"],
                       tablefmt="grid"))
output.append("")


# Print to console
for line in output:
    print(line)

# Save output to file
with open(FORMATTED_METRICS_PATH, 'w', encoding='utf-8') as output_file:
    for line in output:
        output_file.write(line + "\n")
