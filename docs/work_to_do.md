## Priority Order
- Fix victoria_metrics exporter (username and password needed) and Test with grafana (add grafana link to docs)
- Allow users to save there own container results from their benchmarks running with docker.
- Update Readme with an explanation of the repos organisation and folder structure.
- Fix bug when collecting from Tmux logs.
- Save container logs and Tmux logs into a results file.
- Write CI tests and add them to github actions
- Profiling has been done for iris_gpubench of GPU resource but CPU resources are also utilized, though profiling tests to determine exact CPU usage by the monitor system have not yet been conducted (if they are high this is a limitation of the monitor system and optimization will be needed. Given the monitor pauses between intervals there is opportunity to use pythons asynco).
- using shell check from bash script (similar to pylint) on bash script
- Add Highly Optimized NVIDIA HPL Benchmarks: [NVIDIA HPC Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks) see [NVIDIA HPL Benchmark docs](https://docs.nvidia.com/nvidia-hpc-benchmarks/HPL_benchmark.html)
- Investigate how the sciml-bench outputs the types of cores in use (ie Tensor cores) and include this in the collected metrics.
- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- Add dependabot to github actions

---

[Previous Page](considerations_on_accuracy.md)
