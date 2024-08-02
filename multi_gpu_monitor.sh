#!/bin/bash

# Script name: multi_gpu_monitor.sh

# Ensure a benchmark command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <benchmark_command>"
    echo
    echo "Parameters:"
    echo "    <benchmark_command>       Command to execute the benchmark."
    echo
    echo "Example:"
    echo "    $0 <benchmark_command>"
    exit 1
fi

# Temporary file to indicate benchmark completion
COMPLETION_FILE="/tmp/benchmark_complete"

# Extract the benchmark command
BENCHMARK_COMMAND="sciml-bench run --output_dir=/tmp/Results/ $1"
CONDA_ACTIVATE="conda activate bench"
COMPLETION_FILE_COMMAND="touch $COMPLETION_FILE"

# Power monitor script
POWER_MONITOR_SCRIPT="python3 multi_gpu_monitor.py --live_monitor --plot --interval 1"

# Names of the tmux sessions
BENCHMARK_SESSION="benchmark_session"
MONITOR_SESSION="monitor_session"

# Clean up any existing completion file
rm -f "$COMPLETION_FILE"

# Create a new tmux session for the benchmark
tmux new-session -d -s "$BENCHMARK_SESSION" -x 145 -y 10

# Activate the conda environment and run the benchmark command in the benchmark session
tmux send-keys -t "$BENCHMARK_SESSION:0.0" "$CONDA_ACTIVATE" C-m
tmux send-keys -t "$BENCHMARK_SESSION:0.0" "$BENCHMARK_COMMAND" C-m
tmux send-keys -t "$BENCHMARK_SESSION:0.0" "$COMPLETION_FILE_COMMAND" C-m

# Create a new tmux session for the power monitor
tmux new-session -d -s "$MONITOR_SESSION" -x 145 -y 10

# Run the power monitor script in the monitor session
tmux send-keys -t "$MONITOR_SESSION:0.0" "$POWER_MONITOR_SCRIPT" C-m

# Function to check if the benchmark completion file exists
is_benchmark_running() {
    [ ! -f "$COMPLETION_FILE" ]
}

# Wait for the benchmark to complete
while is_benchmark_running; do
    # Read and display output from the power monitor script pane
    clear 
    echo -e "\nLive Monitor: GPU Metrics\n"
    tmux capture-pane -p -t "$MONITOR_SESSION"
    echo -e "\nLive Monitor: Benchmark Output\n"
    tmux capture-pane -p -t "$BENCHMARK_SESSION:0.0"
    sleep 1
done

# Kill the power monitor script
tmux send-keys -t "$MONITOR_SESSION:0.0" C-c

# Wait a moment for processes to properly terminate
sleep 5

# Live Monitoring has Finished
echo -e "\nLive Monitor: Finished.\n"

# Function to read value from YAML file
read_yaml_value() {
    local yaml_file="$1"
    local key="$2"

    if [ -f "$yaml_file" ]; then
        yq -r ".$key" "$yaml_file"
    else
        echo "Error: File $yaml_file does not exist."
    fi
}

# Paths to YAML files
time_file="/tmp/Results/metrics.yml"
metrics_file="./results/metrics.yml"

# Check if the `time` key exists in the YAML file
if key_exists "$time_file" "time"; then
    # Read time value from time.yml
    time_value=$(read_yaml_value "$time_file" "time")

    if [ -z "$time_value" ]; then
        echo "Error: Failed to read time value from $time_file"
        exit 1
    fi

    # Prepend time value to metrics.yml
    temp_file=$(mktemp)
    echo "time: $time_value" > "$temp_file"
    cat "$metrics_file" >> "$temp_file"
    mv "$temp_file" "$metrics_file"
else
    echo "Error: Key 'time' does not exist in $time_file"
fi

# Clean up temporary files
if [ -d "./results/benchmark_specific" ]; then
    rm -r ./results/benchmark_specific/*
else
    echo "Directory './results/benchmark_specific' does not exist."
fi

if [ -d "/tmp/Results" ]; then
    mv /tmp/Results/* ./results/benchmark_specific/
    rm -r /tmp/Results
else
    echo "Directory '/tmp/Results' does not exist."
fi

# Kill the tmux sessions
tmux kill-session -t "$BENCHMARK_SESSION"
tmux kill-session -t "$MONITOR_SESSION"

# Output Results
python3 ./format_results.py
