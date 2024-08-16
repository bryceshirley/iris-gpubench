- Allow users to save container results
- Fix victoria_metrics exporter (username and password needed) and Test with grafana (add grafana link to docs)
- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- using shell check from bash script (similar to pylint) on bash script
- Add CI tests for python scripts
- Add [NVIDIA HPC Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks) see [NVIDIA HPL Benchmark docs](https://docs.nvidia.com/nvidia-hpc-benchmarks/HPL_benchmark.html)
- Investigate how the sciml-bench outputs the types of cores in use (ie Tensor cores) and include this in the collected metrics.
- 
---

[Previous Page](considerations_on_accuracy.md)
