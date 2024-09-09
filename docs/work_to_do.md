## Priority Order
- Add in docs how users can save data from containers
- Update Readme with an explanation of the repos organisation and folder structure.
- Use best practice for naming dockerfiles.
- Include logging levels (basically nothing - everything) - Log tagging, info, error
- Add shell check workflow: https://github.com/stfc/SCD-OpenStack-Utils/blob/master/.github/workflows/gpu_benchmark.yaml
- Fix bug when collecting from Tmux logs.
- Save container logs and Tmux logs into a results file.
- Write CI tests and add them to github actions
- Profiling has been done for iris_gpubench of GPU resource but CPU resources are also utilized, though profiling tests to determine exact CPU usage by the monitor system have not yet been conducted (if they are high this is a limitation of the monitor system and optimization will be needed. Given the monitor pauses between intervals there is opportunity to use pythons asynco).
- using shell check from bash script (similar to pylint) on bash script
- Add Highly Optimized NVIDIA HPL Benchmarks: [NVIDIA HPC Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks) see [NVIDIA HPL Benchmark docs](https://docs.nvidia.com/nvidia-hpc-benchmarks/HPL_benchmark.html) (GraceHopper Tests)
- Normalization of results.
- Integrate RFI GPU benchmarks
- Investigate how the sciml-bench outputs the types of cores in use (ie Tensor cores) and include this in the collected metrics.
- Edit The way "Carbon Forcast (gCO2/kWh)" is computed so that the program checks the Forcast every 30 mins (or less) and computes an average at the end. (Another way to do this would be to multiply the energy consumed each 30 mins (or time interval) by the Forecast for that time and then add them together for a more accurate result. This way we could also give live power updates)
- Add dependabot to github actions
- An idea for the future would be for Iris Users to integrate this benchmark into
there continuous integration and on pull request test there codes gpu's performance
if the performance is reduced by say 5% the test would fail and the pull request
rejected.

---

[Previous Page](considerations_on_accuracy.md)
