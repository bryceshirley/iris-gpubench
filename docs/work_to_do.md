### Future Work and Expansion

#### Additional Benchmarks to Integrate

- **McStas Neutron Monte Carlo Ray-Tracing**

- **RFI GPU Benchmarks**: Explore and integrate relevant Radio Frequency Interference (RFI) GPU benchmarks to cover a wider range of scientific performance evaluations.

- **Highly Optimized NVIDIA HPL Benchmarks**: Add the NVIDIA HPC Benchmarks, focusing on the HPL benchmarks designed for Grace Hopper tests. This integration will require referencing the NVIDIA HPL Benchmark documentation to ensure proper implementation and optimization.

#### Enhancing Benchmark Results

- **Integration into Meerkat (HIGH PRIORITY)**: Complete the integration with Meerkat to facilitate regular benchmark testing. This will allow the generation of more robust statistical data, such as error bars, by calculating mean and standard deviation across multiple tests.

- **Normalization of Results**: Implement normalization techniques to standardize benchmark results across different GPU models and workloads, facilitating meaningful comparisons.

- **FLOP Estimations and Efficiency Metrics**: Calculate Floating Point Operations per Second (FLOPs) to determine performance per watt, providing insights into computational efficiency and energy consumption.

##### Estimating Floating Point Operations Per Second (FLOPs)

Comparisons based solely on clock speed can be misleading due to differences in GPU architectures, CUDA cores, tensor cores, and operations per clock cycle. Note clock speeds vary for various reasons such as workload size, temperature, and power supply.

To estimate the FLOPs of a GPU:

```
FLOPs = CUDA Cores × Clock Speed (Hz) × Operations per Clock Cycle
```

**Steps to calculate FLOPs:**

- Determine the number and type of cores (e.g., CUDA cores, tensor cores) and if they are being utilized.
- Clock speed can be found using IRIS BENCH.
- Identify the operations per clock cycle, specific to the GPU's architecture (e.g., core generation). "The A100 SM includes new third-generation Tensor Cores that each perform 256 FP16/FP32 FMA operations per clock."
- Investigate how `nvidia-smi` calculates utilization and whether more granular utilization data can be obtained.
- Review how tools like `sciml-bench` and `pytorch` report core usage and incorporate these insights into metrics collection.

**Data Sheets Contain Information About the Number of Cores:**

- [V100 datasheet](#)
- [A4000 Datasheet](#)
- [NVIDIA A100](#)
- [RTX4000 DataSheet](#)

#### Explore NVIDIA Nsight Systems

Utilize NVIDIA Nsight Systems for detailed performance profiling, identifying optimization opportunities, and potentially gaining insight into the activated cores for FLOP calculations. This tool offers in-depth insights into GPU performance across various workloads and configurations.

#### Additional Metrics and Areas for Measurement in IRIS Bench

- **Utilization Time**: Measure the total time the GPU is actively utilized, which can provide insights into idle periods and workload efficiency.
- **Stability: Crash Frequency**: Track and report any GPU crashes or visual artifacts during benchmarks to assess stability.
- **Throttling Events**: Monitor instances of clock speed reductions due to high temperatures or power constraints.
- **Memory Bandwidth**: Measure the data transfer rate between the GPU and system memory to identify potential bottlenecks and optimize performance.

#### Other Ideas for Future Implementation

- **Consistent Hardware Configurations**: Ensure that all GPUs being tested use the same hardware configurations (e.g., memory, CPUs) to eliminate variability and produce consistent results.
- **Continuous Integration for Performance Testing**: Encourage IRIS users to integrate GPU benchmarks into their CI workflows. Implement automated performance tests on every pull request; if performance drops by a specified percentage, the pull request would fail, ensuring that code changes do not degrade performance.
- **Experimenting with Precision to Utilize Tensor Cores Fully**: For GPUs equipped with tensor cores, utilize lower precisions (e.g., FP16) for matrix operations where feasible. This can lead to significant performance gains, depending on the workload's precision requirements.


#### Improvements for GitHub Repository

- **Continuous Integration (CI) Tests**: Develop and integrate comprehensive CI tests using GitHub Actions to maintain code reliability, ensure consistent performance, and catch issues early in the development cycle.
- **Carbon Index Calculation**: Enhance the environmental impact analysis by calculating the carbon index throughout the entire benchmarking run, rather than just at the start and end, to provide a more accurate representation.
- **Use Best Practices for Naming Dockerfiles**: Ensure all Dockerfiles follow standard naming conventions for clarity and maintainability.
- **Include Logging Levels**: Implement various logging levels (e.g., debug, info, error) and log tagging to improve traceability and debugging.
- **Add Shell Check Workflow**: Integrate a shell check workflow, similar to the one used in the [SCD-OpenStack-Utils repository](https://github.com/stfc/SCD-OpenStack-Utils/blob/master/.github/workflows/gpu_benchmark.yaml), to catch errors in shell scripts.
- **Write CI Tests for GitHub Actions**: Develop and include additional CI tests within GitHub Actions to ensure consistent and error-free functionality.
- **Run Shell Check from Bash Scripts**: Use shell check (similar to pylint) to analyze bash scripts for potential issues and maintain code quality.
- **Add Dependabot to GitHub Actions**: Implement Dependabot for automated dependency updates, improving security and ensuring compatibility with new releases.

---

[Previous Page](considerations_on_accuracy.md)
