from tabulate import tabulate
import yaml

# Path to your metric.yml file
file_path = './results/metrics.yml'

# Read YAML file
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract and convert values to appropriate types
time_value = data['time']
av_gpu_load = data['av_GPU_load']
av_gpu_power = data['av_GPU_power']
max_gpu_power = data['max_GPU_power']
total_gpu_energy = data['total_GPU_Energy']

# Prepare data for tabulate
data = [
    ["Benchmark Score (s)", f"{time_value:.5f}"],
    ["Total GPU Energy Consumed (kWh)", f"{total_gpu_energy:.5f}"],
    ["Average GPU Util. (for >0.00% GPU Util.) (%)", f"{av_gpu_load:.5f}"],
    ["Avg GPU Power (for >0.00% GPU Util.) (W)", f"{av_gpu_power:.5f} (max possible {int(max_gpu_power)})"]
]

# Print as table
output = []
output.append("Benchmark Score and GPU Performance")
output.append("")
output.append(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
output.append("")

# Print to console
for line in output:
    print(line)

# Save output to file
output_file_path = './results/formatted_scores.txt'
with open(output_file_path, 'w') as output_file:
    for line in output:
        output_file.write(line + "\n")

print(f"Output saved to {output_file_path}")
