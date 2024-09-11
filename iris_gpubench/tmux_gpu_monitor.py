"""
tmux_gpu_monitor.py

This module provides a GPU monitoring class for benchmarks executed within
tmux sessions. The TmuxGPUMonitor class extends the BaseMonitor class to
implement abstract methods for managing GPU metrics in tmux sessions. It
handles tmux session lifecycle management, log retrieval, and benchmark status checking.

Dependencies:
- subprocess: For running tmux commands and managing session execution.
- datetime: For handling timestamps in logs.

Usage:
- Create an instance of TmuxGPUMonitor to monitor GPU metrics for benchmarks
  executed in tmux sessions.
- Implementations of the following abstract methods from BaseMonitor:
  - `_start_benchmark`: Starts the benchmark in a tmux session.
  - `_is_benchmark_running`: Checks if the tmux session is still active.
  - `_live_monitor_logs`: Retrieves and formats live GPU metrics and tmux session logs.
  - `_cleanup_benchmark`: Terminates the tmux session.

Note:
   Most errors are logged but not raised, allowing the method to fail silently.
   Find them in runtime.log.
"""

import os
import subprocess
from datetime import datetime

from .base_monitor import BaseMonitor
# Global Variables
from .utils.globals import LOGGER

try:
    RESULT = subprocess.run(['tmux', '-V'], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True)
    TMUX_AVAILABLE = True
    LOGGER.info("Tmux is available.")
except subprocess.CalledProcessError:
    TMUX_AVAILABLE = False
    LOGGER.warning("Tmux command failed. Tmux functionality will be disabled.")
except FileNotFoundError:
    TMUX_AVAILABLE = False
    LOGGER.warning("Tmux is not installed. Tmux functionality will be disabled.")

class TmuxGPUMonitor(BaseMonitor):
    """
    GPU monitor for tmux-based benchmarks.

    Extends BaseMonitor to run and monitor GPU metrics for benchmarks
    executed within tmux sessions. Manages tmux session lifecycle
    and log retrieval.

    Attributes:
        session_name (str): Name of the tmux session running the benchmark.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize TmuxGPUMonitor.

        Raises:
            RuntimeError: If subprocess module is not available.
        """
        super().__init__(*args, **kwargs)
        if not TMUX_AVAILABLE:
            LOGGER.error("'Tmux' functionality is not available. Please install tmux.")
            raise RuntimeError("The 'tmux' is required but not available. Please install it.")
        self.session_name = "benchmark_session"
        self._benchmark_score_path = None

        self.benchmark_name = None

    def _start_benchmark(self, benchmark) -> None:
        """
        Start the benchmark in a tmux session.

        Args:
            benchmark (str): Command to run the benchmark.
        """
        tmux_command = [
            "tmux", "new-session", "-d", "-s", self.session_name,
            "bash -c 'cd \"$(pwd)\" && " + benchmark + "'"
        ]
        LOGGER.info("Starting tmux session with command: %s", benchmark)
        try:
            subprocess.run(
                tmux_command,
                check=True,
                stderr=subprocess.DEVNULL  # Redirect stderr to /dev/null
            )
            LOGGER.info("Tmux session started successfully.")
        except subprocess.CalledProcessError as error:
            LOGGER.error("Failed to start tmux session: %s", error)
            raise RuntimeError(f"Failed to start tmux session: {error}") from error

    def _is_benchmark_running(self) -> bool:
        """
        Check if the tmux session is still active.

        Returns:
            bool: True if the session is active, False otherwise.
        """
        # Check if the tmux session is still running
        status_command = ["tmux", "has-session", "-t", self.session_name]
        try:
            subprocess.run(
                status_command,
                check=True,
                stderr=subprocess.DEVNULL  # Redirect stderr to /dev/null
            )
            return True
        except subprocess.CalledProcessError:
            LOGGER.info("Tmux session has ended.")
            return False
        # # Check if tmux session is still running via logs
        # logs_command = ["tmux", "capture-pane", "-t", self.session_name, "-p"]
        # try:
        #     subprocess.check_output(logs_command)
        #     LOGGER.info("Captured logs from tmux session - still running.")
        #     return True
        # except subprocess.CalledProcessError as e:
        #     LOGGER.error("Failed to capture logs from tmux session: %s", e)
        #     LOGGER.info("Tmux session has ended.")
        #     return False

    def _live_monitor_logs(self, monitor_logs) -> str:
        """
        Retrieve and format live GPU metrics and tmux session logs.

        Returns:
            str: Formatted string containing GPU metrics and tmux session logs.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if monitor_logs:
            try:
                # Capture and display logs from tmux
                logs_command = ["tmux", "capture-pane", "-t", self.session_name, "-p"]
                try:
                    logs = subprocess.check_output(logs_command).decode()
                    LOGGER.info("Captured logs from tmux session.")
                except subprocess.CalledProcessError as error:
                    LOGGER.error("Failed to capture logs from tmux session: %s", error)
                    logs = ""

                # If logs are effectively empty
                if len(logs.strip()) == 0:
                    logs = "No logs."

                logs_message = (
                    f"\nContainer Log as of {current_time}:\n"
                    f" {logs}\n"
                )

            except ValueError as value_error:
                # Log value errors that occur during processing
                LOGGER.error("Value error in live monitoring: %s", value_error)
                raise  # Re-raise the exception if you want it to propagate

            except Exception as ex:
                # Log any unexpected errors
                LOGGER.error("Unexpected error in live monitoring: %s", ex)
                raise  # Re-raise the exception if you want it to propagate
        else:
            if self._is_benchmark_running():
                logs_message = f"\nBenchmark Status as of {current_time}:\n Running"
            else:
                logs_message = f"\nBenchmark Status as of {current_time}:\n Exited"

        return logs_message

    def _cleanup_benchmark(self) -> None:
        """Terminate the tmux session."""
        try:
            subprocess.run(["tmux", "kill-session", "-t", self.session_name],
                           check=True,
                           stderr=subprocess.DEVNULL)
            LOGGER.info("Tmux session '%s' terminated.", self.session_name)
        except subprocess.CalledProcessError as error:
            LOGGER.error("Failed to clean up tmux session '%s': %s",
                         self.session_name, error)
            
    @property
    def benchmark_score_path(self) -> str:
        """
        Returns the path to the benchmark score file.

        If a user-specified benchmark score path is set via the setter, this path is returned.
        Otherwise, the default path '../results/metrics.yml' is returned.

        This property allows for flexible management of the benchmark score file location,
        providing either a user-defined path or a standard default.

        Returns:
            str: The user-defined path to the benchmark score file if set, otherwise 
            the default path '../results/metrics.yml'.
        """
        # User specified benchmark score path
        if self._benchmark_score_path:
            return self._benchmark_score_path

        return '../results/metrics.yml'

    @benchmark_score_path.setter
    def benchmark_score_path(self, path: str) -> None:
        """
        Sets the path to the benchmark score file.

        Allows the user to manually specify the location of the benchmark score file.
        Ensures that the provided path is a valid string before setting it.

        Args:
            path (str): The full path to the benchmark score file.

        Raises:
            ValueError: If the provided path is not a valid string.
        """
        if not isinstance(path, str):
            raise ValueError("The benchmark score path must be a string.")

        self._benchmark_score_path = path
