# Live Monitoring

## 1. Monitor GPU Metrics

```bash
Current GPU Metrics as of 2024-08-01 23:32:47:
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|   GPU Index (Tesla V100-PCIE-32GB) |   Utilization (%) |   Power (W) / Max 250.0 W |   Temperature (C) |   Memory (MiB) / Total 32768.0 MiB |
+====================================+===================+===========================+===================+====================================+
|                                  0 |                60 |                    87.027 |                34 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
|                                  1 |                63 |                    87.318 |                40 |                            2075.62 |
+------------------------------------+-------------------+---------------------------+-------------------+------------------------------------+
```

## 2. Monitor Benchmark Container logs
  
```bash
gpu_monitor --benchmark_image "synthetic_regression" --monitor_benchmark_logs

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
## 3. Save png Timeseries Live
  
```bash
gpu_monitor --benchmark_image "synthetic_regression" --plot_live
```

Gives you saves plot png during every reading so that the metrics can be viewed live.