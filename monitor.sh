#!/bin/bash

# Script name: monitor.sh

# Ensure a benchmark command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <benchmark_command> [--plot]"
    exit 1
fi

# Temporary file to indicate benchmark completion
COMPLETION_FILE="/tmp/benchmark_complete"

# Get the benchmark command from input and other commands
BENCHMARK_COMMAND="sciml-bench run --output_dir=/tmp/Results/ $1"
shift
EXTRA_ARGS=("$@")
POWER_MONITOR_SCRIPT="python ./gpu_monitor.py"
CONDA_ACTIVATE="conda activate bench"
COMPLETION_FILE_COMMAND="touch $COMPLETION_FILE"

# Check for -plot option
PLOT_OPTION=false
if [[ " ${EXTRA_ARGS[@]} " =~ " --plot " ]]; then
    PLOT_OPTION=true
fi

# Name of the tmux session
SESSION_NAME="gpu_benchmark_monitor"

# Clean up any existing completion file
rm -f "$COMPLETION_FILE"

# Create a new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Split into two panes
tmux split-window -v -t "$SESSION_NAME:0.0"

# Run the benchmark in the first pane
tmux send-keys -t "$SESSION_NAME:0.0" "$CONDA_ACTIVATE" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "$BENCHMARK_COMMAND" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "$COMPLETION_FILE_COMMAND" C-m

# Run the power monitor script in the second pane
#tmux send-keys -t "$SESSION_NAME:0.1" "$CONDA_ACTIVATE" C-m
if [ "$PLOT_OPTION" = true ]; then
    tmux send-keys -t "$SESSION_NAME:0.1" "$POWER_MONITOR_SCRIPT --plot" C-m
else
    tmux send-keys -t "$SESSION_NAME:0.1" "$POWER_MONITOR_SCRIPT" C-m
fi

# Function to check if the benchmark completion file exists
is_benchmark_running() {
    [ ! -f "$COMPLETION_FILE" ]
}

# Temporary file to indicate benchmark completion
POWER_MONITOR_FILE="/tmp/power_monitor_output.txt"
BENCHMARK_MONITOR_FILE="/tmp/benchmark_output.txt"

# Wait for the benchmark to complete
while is_benchmark_running; do
    # Read and display output from the power monitor script pane
    tmux capture-pane -p -t "$SESSION_NAME:0.1" > "$POWER_MONITOR_FILE"
    tmux capture-pane -p -t "$SESSION_NAME:0.0" > "$BENCHMARK_MONITOR_FILE"
    clear  # Optional: Clear the terminal for a cleaner output view
    echo -e "\nLive Monitor: Power and Utilization\n"
    tail -n 2 "$POWER_MONITOR_FILE"
    echo -e "\nLive Monitor: Benchmark Output\n"
    tail -n 5 "$BENCHMARK_MONITOR_FILE"
    sleep 1
    rm "$BENCHMARK_MONITOR_FILE"
    rm "$POWER_MONITOR_FILE"
done

# Kill the power monitor script
tmux send-keys -t "$SESSION_NAME:0.1" C-c

# Wait a moment for processes to properly terminate
sleep 5

# Live Monitoring has Finished
echo -e "\nLive Monitor: Finished. \n"

# Function to read value from YAML file
read_yaml_value() {
    local yaml_file="$1"
    local key="$2"
    yq -r ".$key" "$yaml_file"

    if [ -f "$yaml_file" ]; then
        yq -r ".$key" "$yaml_file"
    else
        echo "Error: File $yaml_file does not exist."
    fi
}

# Paths to YAML files
time_file="/tmp/Results/metrics.yml"
metrics_file="./results/metrics.yml"

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


# Clean up temporary files
rm -r ./results/benchmark_specific/*
mv /tmp/Results/* ./results/benchmark_specific/
rm -r /tmp/Results

# Kill the tmux session
tmux kill-session -t "$SESSION_NAME"

# Output Results
python ./format_results.py
