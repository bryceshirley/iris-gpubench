<p align="center">
  <img src="docs/stfc_logo.png" alt="STFC Logo" height="100">
  <img src="docs/iris_logo.png" alt="IRIS Logo" height="100">
</p>


# IRIS GPU Bench
## A GPU Monitoring and Carbon Calculation Tool for Benchmarks
---

### Documentation

**Explore the IRIS GPU Bench documentation:**  
[![GitHub Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://bryceshirley.github.io/iris-gpubench/)

---

### Brief Overview

The **IRIS GPU Bench** tool tracks GPU performance and carbon emissions during benchmarks. It provides:

- **Real-time GPU Metrics**: Monitors GPU performance in real-time.
- **Carbon Emission Estimates**: Estimates emissions using the National Grid ESO API.
- **Data Export**: Optionally exports data to Grafana via VictoriaMetrics.
- **Flexible Benchmarking**:  
  - **Docker**: Run benchmarks in isolated containers for consistency.  
  - **Tmux**: Execute benchmarks directly on the host and keep them running in the background during monitoring.
- **Flexible Command-Line Interface**: Offers a customizable monitoring process with a variety of command-line arguments.
- **Real-time Logging**: Supports live prints of Docker container or Tmux logs.

This tool is ideal for evaluating GPU application performance, measuring environmental impact, optimizing GPU performance, and informing purchasing decisions by testing applications on different hardware configurations.

---

![Build Status](https://github.com/bryceshirley/iris-gpubench/actions/workflows/docker-build.yml/badge.svg)
