#TO ADD:
- Should we add carbon produced to the benchmark power calculation? Ask Deniza
	- In 2023 the average emissions of United Kingdom were 217 g CO2eq/kWh. From <https://www.nowtricity.com/country/united-kingdom/>

The Command
===========
This bash script has been written for sciml-benchmarks and will be generalized to any runnable GPU benchmark:

```
./run_benchmark_and_monitor.sh <sciml benchmark command>
```

Or 

```
./run_benchmark_and_monitor.sh <sciml benchmark command> --plot
```

# How to use:
### 1. Before first run activate the bash script by:

```
chmod +x run_benchmark_and_monitor.sh
```

### 2. Run in terminal:

 ```
./run_benchmark_and_monitor.sh <sciml benchmark command> 
```

### 3.  Live Monitoring
	a. The Output of the Benchmark and Power/Utilization Are Tracked Live By Copying over The Tmux Outputs

	b. (Optional)Timeseries Using the --plot option
  
```
./run_benchmark_and_monitor.sh <sciml benchmark command> --plot
```

Gives you a live timeseries for GPU power consumption and GPU utilization. Just open the png files created gpu_utilization_plot.png and gpu_power_usage_plot.png. They can be found there afterwards too

### 4. If you need to terminate the tool for any reason (ie press CTRL+C) then you must kill the tmux session by running:
tmux kill-session


# Requirements:
gpu_monitor.py 
run_benchmark_and_monitor.sh


* **Python Script (gpu_monitor.py):**
	* Python interpreter.
	* Required Python modules: subprocess, csv, time, os, datetime, argparse, matplotlib, tabulate.
	* Dependency on nvidia-smi for GPU metrics.
* **Bash Script (run_benchmark_and_monitor.sh):**
	* Bash shell.
	* External commands: tmux, conda (optional).
	* Ensure correct paths for scripts (gpu_monitor.py) and temporary files.
* **Permissions:**
	* Both scripts should have execution permissions (chmod +x script_name.sh).
* **Dependencies:**
	* Python dependencies (matplotlib, tabulate) must be installed.
	* Availability of nvidia-smi for GPU metrics.
* **Configuration:**
	* Set environment variables (PATH, PYTHONPATH) appropriately.
	* Verify paths and environment configurations to prevent command not found errors.
