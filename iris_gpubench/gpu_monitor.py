"""
gpu_monitor.py

This module provides a high-level interface for GPU monitoring, supporting both
Docker and tmux-based benchmarks. The GPUMonitor class manages the creation of
appropriate monitor instances (either DockerGPUMonitor or TmuxGPUMonitor) and
delegates monitoring operations.

Dependencies:
- os: For handling file paths and environment variables.
- docker_gpu_monitor: Contains DockerGPUMonitor class for Docker-based benchmarks.
- tmux_gpu_monitor: Contains TmuxGPUMonitor class for tmux-based benchmarks.
"""

import os

from .docker_gpu_monitor import DockerGPUMonitor
from .tmux_gpu_monitor import TmuxGPUMonitor
# Global Variables
from .utils.globals import MONITOR_INTERVAL, RESULTS_DIR

METRICS_FILE_PATH = os.path.join(RESULTS_DIR, 'metrics.yml')
TIMESERIES_PLOT_PATH = os.path.join(RESULTS_DIR, 'timeseries_plot.png')
FINAL_MONITORING_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'final_monitoring_output.png')

class GPUMonitor:
    """
    High-level GPU monitoring interface.

    Provides a unified interface for monitoring GPU metrics,
    supporting both Docker and tmux-based benchmarks. Handles
    the creation of appropriate monitor instances and delegates
    monitoring operations.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the monitor.
        monitor (Optional[BaseMonitor]): Instance of DockerGPUMonitor or TmuxGPUMonitor.
    """
    def __init__(self, monitor_interval: int = MONITOR_INTERVAL,
                 carbon_region_shorthand: str = "South England") -> None:
        """
        Initialize GPUMonitor.

        Args:
            monitor_interval (int): Interval for collecting GPU metrics.
            carbon_region_shorthand (str): Region for carbon intensity data.
        """
        self.config = {
            'monitor_interval': monitor_interval,
            'carbon_region_shorthand': carbon_region_shorthand
        }
        self.monitor = None

    def run(self, benchmark_command: str = None, benchmark_image: str = None,
            live_monitoring: bool = True, plot: bool = True, live_plot: bool = False,
            monitor_logs: bool = False, export_to_meerkat: bool = False) -> None:
        """
        Run the benchmark and monitor GPU metrics.

        Creates appropriate monitor instance based on input and starts monitoring.

        Args:
            benchmark_command (Optional[str]): Command to run for tmux-based benchmarks.
            benchmark_image (Optional[str]): Docker image for Docker-based benchmarks.
            live_monitoring (bool): Enable live monitoring display.
            plot (bool): Save metrics plot at the end.
            live_plot (bool): Update metrics plot in real-time.
            monitor_logs (bool): Monitor both GPU metrics and logs.
            export_to_meerkat (bool): Export metrics to MeerkatDB.

        Raises:
            ValueError: If both or neither benchmark_command and benchmark_image are specified.
        """
        # Run as using Docker or Tmux
        if benchmark_command and benchmark_image:
            raise ValueError(
                "You must specify either 'benchmark_command' or 'benchmark_image', "
                "not both."
            )
        if benchmark_command:
            self.monitor = TmuxGPUMonitor(**self.config)
        elif benchmark_image:
            self.monitor = DockerGPUMonitor(**self.config)
        else:
            raise ValueError("You must specify either 'benchmark_command' or 'benchmark_image'.")

        # Run and Monitor the Benchmark
        self.monitor.run_benchmark(
            benchmark_command or benchmark_image,
            live_monitoring=live_monitoring,
            plot=plot,
            live_plot=live_plot,
            monitor_logs=monitor_logs,
            export_to_meerkat=export_to_meerkat
        )

    def save_stats_to_yaml(self, file_path: str = METRICS_FILE_PATH) -> None:
        """
        Save collected GPU statistics to a YAML file.

        Args:
            file_path (str): Path to save the YAML file.

        Raises:
            RuntimeError: If no monitoring has been performed yet.
        """
        if self.monitor:
            self.monitor.save_stats_to_yaml(file_path)
        else:
            raise RuntimeError("No monitoring has been performed yet.")

    def plot_timeseries(self, plot_path: str = TIMESERIES_PLOT_PATH):
        """Plot the time series data to a specified file path."""
        if self.monitor:
            self.monitor.plot_timeseries(plot_path)
        else:
            raise RuntimeError("No monitoring has been performed yet.")

    def save_timeseries_to_csv(self, results_dir: str = RESULTS_DIR):
        """Save the time series data to a CSV file in a specified directory."""
        if self.monitor:
            self.monitor.save_timeseries_to_csv(results_dir)
        else:
            raise RuntimeError("No monitoring has been performed yet.")
