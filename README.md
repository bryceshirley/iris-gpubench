<p align="center">
  <img src="docs/stfc_logo.png" alt="STFC Logo" height="100">
  <img src="docs/iris_logo.png" alt="IRIS Logo" height="100">
</p>


# IRIS GPU Bench
## A GPU Performance and Carbon Calculation Tool for Benchmarks
---

### Documentation

**Explore the IRIS GPU Bench documentation:**  
[![GitHub Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://bryceshirley.github.io/iris-gpubench/)

Developers are expected to keep the Documentation up-to-date.

---

### Brief Overview

The **IRIS GPU Bench** tool tracks GPU performance and carbon emissions during benchmarks and outputs final GPU Performance Benchmark Results. It provides:

- **Final GPU Performance Results**: See collecting results section of documentation.
- **Real-time GPU Metrics**: Monitors GPU performance in real-time.
- **Carbon Emission Estimates**: Estimates emissions using the National Grid ESO API.
- **Data Export**: Optionally exports data to a [Grafana Dashboard](http://172.16.112.145:3000/d/fdw7dv7phr0g0e/iris-bench?orgId=1) via VictoriaMetrics.
- **Flexible Benchmarking**:  
  - **Docker**: Run benchmarks in isolated containers for consistency.  
  - **Tmux**: Execute benchmarks directly on the host and keep them running in the background during monitoring.
- **Flexible Command-Line Interface**: Offers a customizable monitoring process with a variety of command-line arguments.
- **Real-time Logging**: Supports live prints of Docker container or Tmux logs.

This tool is ideal for evaluating GPU application performance, measuring environmental impact, optimizing GPU performance, and informing purchasing decisions by testing applications on different hardware configurations.

### Iris Bench Report - 12/09/24

For a report on the performance of the benchmarks available on this repo for on a range of GPU see [`Iris Bench Report - 12_09_24`](https://github.com/bryceshirley/iris-gpubench/blob/main/Iris%20Bench%20Report%20-%2012_09_24.pdf) file.

---

![Build Status](https://github.com/bryceshirley/iris-gpubench/actions/workflows/docker-build.yml/badge.svg)
