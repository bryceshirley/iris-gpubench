## Priority Order
- Use best practice for naming dockerfiles.
- Include logging levels (basically nothing - everything) - Log tagging, info, error
- Add shell check workflow: https://github.com/stfc/SCD-OpenStack-Utils/blob/master/.github/workflows/gpu_benchmark.yaml
- Write CI tests and add them to github actions
- using shell check from bash script (similar to pylint) on bash script
- Add dependabot to github actions

---

## Future Work

### Additional Benchmarks to Integrate
- **RFI GPU Benchmarks**: Explore and integrate relevant RFI GPU benchmarks to enhance the breadth of performance evaluations.
- **Highly Optimized NVIDIA HPL Benchmarks**: Add the [NVIDIA HPC Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks), particularly the HPL benchmarks designed for Grace Hopper tests. Refer to the [NVIDIA HPL Benchmark documentation](https://docs.nvidia.com/nvidia-hpc-benchmarks/HPL_benchmark.html) for details on implementation and optimization.

### Improvements and Enhancements on the GitHub Repository
- **Continuous Integration (CI) Tests**: Develop and integrate comprehensive CI tests into GitHub Actions to ensure code reliability and maintainability.
- **CPU Profiling**: While GPU resource profiling has been done for the `iris_gpubench`, profiling the CPU resources utilized by the monitor system has not been conducted yet. Future work should include tests to measure and understand CPU usage.
- **Carbon Index Calculation**: Calculate the carbon index as an average throughout the entire run rather than only at the start and end, for a more accurate representation of environmental impact.

### Enhancing Benchmark Results
- **Normalization of Results**: Implement normalization methods to better compare results across different GPU models and workloads.
- **FLOP Estimations and Efficiency Metrics**: Calculate FLOPs to determine performance per watt, providing a meaningful estimate of computational efficiency.
- **Detailed GPU Memory Utilization Metrics**: Include comprehensive measurements of GPU memory utilization, not just the total memory used.
- **Integration into Meerkat**: Complete the integration of benchmarks into Meerkat to facilitate periodic test execution. This would allow benchmark bar charts to include error bars by computing the mean and standard deviation across multiple tests.

### Other Ideas for Future Implementation
- **Consistent Hardware Configurations**: Ensure all GPUs being tested use the same hardware configurations (memory, CPUs, etc.) to control for variables and produce more consistent results.
- **Continuous Integration for Performance Testing**: Encourage Iris users to integrate these benchmarks into their CI workflows. On every pull request, a GPU performance test could be run; if performance drops by a specified percentage (e.g., 5%), the pull request would fail, ensuring code changes do not degrade performance.

### Investigating Clock Speeds and Flops
- Use `nvidia-smi` to collect and analyze clock speeds. Note that comparing GPUs based solely on clock speed can be misleading due to differences in CUDA cores, tensor cores, and the operations per clock cycle for different generations.
- **Factors Affecting Clock Speed:**
  - **Workload**: Heavier workloads may require higher clock speeds.
  - **Temperature**: Overheating can lead to reduced clock speeds.
  - **Power Supply**: The available power may limit the maximum achievable clock speed.

### Experimenting with Precision
- **Tensor Cores**: For GPUs with tensor cores, leverage the ability to perform matrix operations in lower precisions (e.g., FP16) to achieve significant performance gains where appropriate. This will be highly dependent on workload requirements for precision.

### Estimating Floating Point Operations Per Second (FLOPs)
To estimate the FLOPs of a GPU, use the formula:

\[
\text{FLOPs} = \text{CUDA Cores} \times \text{Clock Speed (Hz)} \times \text{Operations per Clock Cycle}
\]

To calculate this:
- Determine the number and type of cores (e.g., CUDA cores, tensor cores) and which cores are being used.
- Find the operations per clock cycle, specific to the GPU generation.
- Investigate how utilization is calculated in `nvidia-smi` and whether raw utilization data (e.g., core or clock cycle-based) can be accessed.
- Consider how tools like `sciml-bench` report core usage (e.g., tensor cores) and incorporate these details into metrics collection.

### Leveraging NVIDIA Nsight
- **Explore NVIDIA Nsight Systems**: Investigate the use of NVIDIA Nsight Systems for detailed workload profiling. This tool offers system-wide performance analysis, visualizing application algorithms, identifying optimization opportunities, and tuning performance across various CPU and GPU configurations. Determine if Nsight can provide insights into which cores/SMs (streaming multiprocessors) are activated during benchmarks as well as any other interesting metrics.

### Additional Metrics And Areas For Measurement That Could Be Interesting
- **Utilization Time**: Measure the total time the GPU is actively utilized (i.e., when utilization is not 0).
- **Stability: Crash Frequency**: Track and document any GPU crashes or visual artifacts appearing during benchmarks.
- **Throttling Events**: Monitor if and when GPU clock speeds are reduced due to high temperatures or power limitations.
- **Memory Bandwidth**: Measure the rate at which data is transferred between the GPU and system memory to understand potential bottlenecks.

---

Feel free to let me know if you want further adjustments or additions!



---

[Previous Page](considerations_on_accuracy.md)
