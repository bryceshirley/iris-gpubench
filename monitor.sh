#!/bin/bash

# Script name: run_benchmark_and_monitor.sh

# Ensure a benchmark command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <benchmark_command> [--plot]"
    exit 1
fi

# Temporary file to indicate benchmark completion
COMPLETION_FILE="/tmp/benchmark_complete"

# Get the benchmark command from input and other commands
BENCHMARK_COMMAND="$1"
shift
EXTRA_ARGS=("$@")
POWER_MONITOR_SCRIPT="python /home/dnz75396/gpu_monitor.py"
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

# Wait for the benchmark to complete
while is_benchmark_running; do
    # Read and display output from the power monitor script pane
    tmux capture-pane -p -t "$SESSION_NAME:0.1" > /tmp/power_monitor_output.txt
    tmux capture-pane -p -t "$SESSION_NAME:0.0" > /tmp/benchmark_output.txt
    clear  # Optional: Clear the terminal for a cleaner output view
    echo -e "\nLive Monitor: Power and Utilization\n"
    tail -n 2 /tmp/power_monitor_output.txt
    echo -e "\nLive Monitor: Benchmark Output\n"
    tail -n 5 /tmp/benchmark_output.txt
    sleep 1
    rm /tmp/benchmark_output.txt
    rm /tmp/power_monitor_output.txt
done

# Kill the power monitor script
tmux send-keys -t "$SESSION_NAME:0.1" C-c

# Wait a moment for processes to properly terminate
sleep 5

# Capture the output of both commands
tmux capture-pane -p -t "$SESSION_NAME:0.0" > /tmp/benchmark_output.txt
tmux capture-pane -pJ -t "$SESSION_NAME:0.1" > /tmp/power_monitor_output.txt

# Display the results
{
  echo -e "\n+--------------------------------------------------------------------+"
  echo -e "\nBenchmark Results\n"
  tail -n 5 /tmp/benchmark_output.txt | head -n -4
  echo -e "\nPower and Utilization\n"
  tail -n 11 /tmp/power_monitor_output.txt | head -n 10
  echo " "
} > ./results/benchmark_scores.txt
cat ./results/benchmark_scores.txt

# Clean up temporary files
rm /tmp/benchmark_output.txt
rm /tmp/power_monitor_output.txt

# Kill the tmux session
tmux kill-session -t "$SESSION_NAME"
