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
time_value = data['time']
av_gpu_load = data['av_GPU_load']
av_gpu_power = data['av_GPU_power']
av_gpu_power = data['av_GPU_power']
av_gpu_power = data['av_GPU_power']
max_gpu_power = int(data['max_GPU_power'])
total_gpu_energy = data['total_GPU_Energy']
carbon_forcast = data['carbon_forcast']
carbon_index = data['carbon_index']
total_GPU_carbon = data['total_GPU_carbon']
date_time = data["date_time"]

# Prepare data for tabulate
formatted_data_main = [
    ["Benchmark Score (s)", f"{time_value:.5f}"],
    ["Total GPU Energy Consumed (kWh)", f"{total_gpu_energy:.5f}"],
    ["Total GPU Carbon Emissions (gC02)",f"{total_GPU_carbon:.5f}"],
]

formatted_data_extra = [
    ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{av_gpu_load:.5f}"],
    ["Avg GPU Power (for >0.00% GPU Util.) (W)",
      f"{av_gpu_power:.5f} (max possible {max_gpu_power})"],
    ["Carbon Forcast (gCO2/kWh), Carbon Index",
      f"{carbon_forcast:.1f}, {carbon_index}"],
    ["Carbon Intensity Reading Date & Time",date_time]
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
