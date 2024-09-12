# Live Monitoring

## Monitor GPU Metrics
A default option that can be switched off.

```sh
Current GPU Metrics (Tesla V100-PCIE-32GB) as of 2024-09-12 16:53:14:
+--------------------------+----------+---------+
| Metric                   |    GPU 0 |   GPU 1 |
+==========================+==========+=========+
| Utilization (%)          |   97     |   0     |
+--------------------------+----------+---------+
| Power (W)                |  178.133 |  27.284 |
+--------------------------+----------+---------+
| Temperature (C)          |   49     |  35     |
+--------------------------+----------+---------+
| Memory (MiB)             | 6793.62  | 275.625 |
+--------------------------+----------+---------+
| Clock Speed (MHz)        | 1380     | 135     |
+--------------------------+----------+---------+
| Memory Clock Speed (MHz) |  877     | 877     |
+--------------------------+----------+---------+
```

## Monitor Benchmark Container/Tmux Logs
  
```sh
gpu_monitor --benchmark_image "synthetic_regression" --monitor_logs

Container Logs:
<BEGIN> Running benchmark synthetic_regression in training mode
....<BEGIN> Parsing input arguments
....<ENDED> Parsing input arguments [ELAPSED = 0.000035 sec]
....<BEGIN> Creating dataset
....<ENDED> Creating dataset [ELAPSED = 12.436691 sec]
....<MESSG> Number of samples: 1024000
....<MESSG> Total number of batches: 8000, 8000.0
....<BEGIN> Training model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/root/anaconda3/envs/bench/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name | Type       | Params | Mode 
--------------------------------------------
0 | net  | Sequential | 250 M  | train
--------------------------------------------
250 M     Trainable params
0         Non-trainable params
250 M     Total params
1,000.404 Total estimated model params size (MB)
11        Modules in train mode
0         Modules in eval mode
/root/anaconda3/envs/bench/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
Epoch 0:   4%|▍         | 350/8000 [00:09<03:23, 37.56it/s, v_num=0]
```

(Long container logs containing "\r" to clear the line for progress bars are not a very efficiently processed, as it captures a snapshot of the entire container log at that moment.
Potiential solution: use asynico package to capture and process the logs whilst the monitor is paused between intervals)

## (Save png) Timeseries Plot Live

Gives you saves plot png during every reading so that the metrics can be viewed live **locally**.

**Example command and Results:**
```sh
gpu_monitor --benchmark_image "stemdl_classification_2gpu" --plot_live
```
This command was run on a VM with 2 V100 GPUs for the results in [Collecting Results Section](collecting_results.md#gpu-metrics-timeseries-plot-png).


## (Grafana) Timeseries Plot Live

If the `--export_to_meerkat` tag is used the results can be viewed live from the Grafana Dashboard. Data from multiple VMs can be collected all at once allowing for a live comparison of performance as well.

See example results [Collecting Results Section](collecting_results.md#gpu-metric-grafana-plots).

---

[Previous Page](collecting_results.md) | [Next Page](considerations_on_accuracy.md)
