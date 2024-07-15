# TO ADD:
- Should we add carbon produced to the benchmark power calculation?
	- In 2023 the average emissions of United Kingdom were 217 g CO2eq/kWh. From <https://www.nowtricity.com/country/united-kingdom/>
 	- National grid carbon loading (national grid give api's that give regional live metrics for carbon) to get carbon cost. From <https://carbon-intensity.github.io/api-definitions/#regional>
  	- There is also an IRIS contact for this kind of thing
- Make it work for Benchmarks using more than one GPU
- using shell check from bash script (similar to pylint) on bash script
- use pylint and isort
- add a requirements.txt file for setup
- Add CI tests for python script
- Make monitor.sh collect errors from sciml-bench command

The Command
===========
This bash script has been written for sciml-benchmarks and will be generalized to any runnable GPU benchmark:

```
./monitor.sh <--run_options sciml_benchmark>
```

Or 

```
./monitor.sh <--run_options sciml_benchmark> --plot
```

# How to use:
### 1. Run in terminal (from folder):

```
./monitor.sh <sciml benchmark command> 
```

### 3.  Live Monitoring

####		a. The Output of the Benchmark and Power/Utilization Are Tracked Live By Copying over The Tmux Outputs. Example:

This example uses the "synthetic_regression" benchmark with the "-b epochs 2" option for two epochs and "-b hidden_size 9000" options (see sciml-bench docs for more options)
```
(bench) dnz75396@bs-scimlbench-a4000:~/gpu_benchmark_metrics$ ./monitor.sh "-b epochs 2 -b hidden_size 9000 synthetic_regression" --plot

Live Monitor: Power and Utilization

Current GPU Power Usage: 89.62 W, GPU Utilization: 31.00 %


Live Monitor: Benchmark Output

Epoch 1: 100%|█████████████████████| 8000/8000 [02:17<00:00, 58.13it/s, v_num=0]
....<ENDED> Training model [ELAPSED = 276.374506 sec]
....<BEGIN> Saving training metrics to a file
....<ENDED> Saving training metrics to a file [ELAPSED = 0.000279 sec]
```   

#####		b. (Optional)Timeseries Using the --plot Option
  
```
./monitor.sh <sciml benchmark command> --plot
```

Gives you a live timeseries for GPU power consumption and GPU utilization. Just open the png files created gpu_utilization_plot.png and gpu_power_usage_plot.png. They can be found there afterwards too

### 4. If you need to terminate the tool for any reason (ie press CTRL+C) then you must kill the tmux session by running:

```
tmux kill-session
```
### 5. Results 

* Results are saved to gpu_benchmark_metrics/results (these include):
	* gpu_power_usage.png and gpu_utilization.png: Time series plots for gpu utilization and power usage
  	* metrics.yml: yml with the Benchmark Score and GPU Energy Performance results.
  	* benchmark_specific/: directory containing all the results output by the sciml-bench benchmark. 
 	* formatted_scores.txt : Formatted version of metrics.yml, see example below.
```
Benchmark Score and GPU Energy Performance

+-----------------------------------+-----------+
| Metric                            |     Value |
+===================================+===========+
| Benchmark Score (s)               | 276.367   |
+-----------------------------------+-----------+
| Total GPU Energy Consumed (kWh)   |   0.02166 |
+-----------------------------------+-----------+
| Total GPU Carbon Emissions (gC02) |   4.76572 |
+-----------------------------------+-----------+

Additional Information

+----------------------------------------------+------------------------------+
| Metric                                       | Value                        |
+==============================================+==============================+
| Average GPU Util. (for >0.00% GPU Util.) (%) | 96.86854                     |
+----------------------------------------------+------------------------------+
| Avg GPU Power (for >0.00% GPU Util.) (W)     | 278.99610 (max possible 300) |
+----------------------------------------------+------------------------------+
| Carbon Forcast (gCO2/kWh), Carbon Index      | 220.0, high                  |
+----------------------------------------------+------------------------------+
| Carbon Intensity Reading Date & Time         | 2024-07-15 14:02:05          |
+----------------------------------------------+------------------------------+

Data is collected from the National Grid ESO Regional Carbon Intensity API: 
https://api.carbonintensity.org.uk/regional
The Carbon Forcast and Index Readings are updated every 30 minutes.
```

# Requirements:
gpu_monitor.py 
monitor.sh


* **Python Script (gpu_monitor.py):**
	* Python interpreter.
	* Required Python modules: subprocess, csv, time, os, datetime, argparse, matplotlib, tabulate.
	* Dependency on nvidia-smi for GPU metrics.
* **Bash Script (monitor.sh):**
	* Bash shell.
 	* pip install jq
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
