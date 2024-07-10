# TO ADD:
- Should we add carbon produced to the benchmark power calculation?
	- In 2023 the average emissions of United Kingdom were 217 g CO2eq/kWh. From <https://www.nowtricity.com/country/united-kingdom/>

The Command
===========
This bash script has been written for sciml-benchmarks and will be generalized to any runnable GPU benchmark:

```
./monitor.sh <sciml benchmark command>
```

Or 

```
./monitor.sh <sciml benchmark command> --plot
```

# How to use:
### 1. Before first run activate the bash script by:

```
chmod +x monitor.sh
```

### 2. Run in terminal (from folder):

```
./monitor.sh <sciml benchmark command> 
```

### 3.  Live Monitoring
	a. The Output of the Benchmark and Power/Utilization Are Tracked Live By Copying over The Tmux Outputs. Example:
```
(bench) dnz75396@bs-scimlbench-a4000:~/gpu_benchmark_metrics$ ./monitor.sh "sciml-bench run synthetic_regression"
Live Monitor: Power and Utilization

Current GPU Power Usage: 132.76 W, GPU Utilization: 98.00 %


Live Monitor: Benchmark Output

Epoch 0: 100%|███████████████████| 8000/8000 [00:40<00:00, 197.41it/s, v_num=10]
....<ENDED> Training model [ELAPSED = 40.945453 sec]
....<BEGIN> Saving training metrics to a file
....<ENDED> Saving training metrics to a file [ELAPSED = 0.000295 sec]
```   

	b. (Optional)Timeseries Using the --plot option
  
```
./monitor.sh <sciml benchmark command> --plot
```

Gives you a live timeseries for GPU power consumption and GPU utilization. Just open the png files created gpu_utilization_plot.png and gpu_power_usage_plot.png. They can be found there afterwards too

### 4. If you need to terminate the tool for any reason (ie press CTRL+C) then you must kill the tmux session by running:

```
tmux kill-session
```
### 5. Results 

* Results are saved to gpu_benchmark_metrics/Results (these include):
	* The plots: gpu_power_usage.png and gpu_utilization.png
 	* benchmark_scores.txt (also output in termal like so)
```
Benchmark Results

....<ENDED> Training model [ELAPSED = 40.945453 sec]
....<BEGIN> Saving training metrics to a file
....<ENDED> Saving training metrics to a file [ELAPSED = 0.000295 sec]

Power and Utilization


+----------------------------------------------+---------------------+          
| Metric                                       | Value               |          
+==============================================+=====================+          
| Total GPU Energy Consumed (kWh)              | 0.00155             |          
+----------------------------------------------+---------------------+          
| Average GPU Util. (for >0.00% GPU Util.) (%) | 89.38               |          
+----------------------------------------------+---------------------+          
| Avg GPU Power (for >0.00% GPU Util.) (W)     | 129.99 (max 140.00) |          
+----------------------------------------------+---------------------+ 
```
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
