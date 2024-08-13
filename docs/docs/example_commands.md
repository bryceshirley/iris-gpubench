1. **Basic Monitoring with Completion Plot**:
   ```
   iris-gpubench --benchmark_image "synthetic_regression"
   ```

2. **Exporting Data to VictoriaMetrics**:
   ```
   iris-gpubench --benchmark_image "synthetic_regression" --export_to_victoria
   ```

3. **Full Command with All Options**:
   ```
   iris-gpubench --benchmark_image "synthetic_regression" --interval 10 --carbon_region "South England" --live_plot --export_to_victoria --monitor_benchmark_logs
   ```