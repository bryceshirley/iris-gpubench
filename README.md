---

# GPU Monitoring And Carbon Calculation Tool For Containerized Benchmarks

## Overview
The GPU Monitoring and Carbon Calculation Tool tracks GPU performance and carbon emissions during benchmarks.

The tool monitors GPU metrics in real-time, estimates carbon emissions using the National Grid ESO API, and optionally exports data to VictoriaMetrics. It integrates with Docker for containerized environments and offers a customizable command-line interface for monitoring and live plotting. Real-time logging is also supported.

Ideal for evaluating GPU application performance, measuring environmental impact, and optimizing GPU performance.

---

## Build Documentation

#### Build and Run the Docs Docker Container

To start the Docs container in the background, use the following command:

```sh
cd docs
./build_and_run.sh
```

Once running, you can access the documentation at: [http://localhost:8000/](http://localhost:8000/)

# Stop and Remove the Docs Docker Container

To stop and remove the Docs container, execute:

```sh
./stop_and_remove.sh
```
