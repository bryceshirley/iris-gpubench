"""
docker_gpu_monitor.py

This module provides a GPU monitoring class for benchmarks executed within
Docker containers. The DockerGPUMonitor class extends the BaseMonitor class
to implement abstract methods for managing GPU metrics in Docker containers.
It handles container lifecycle management, log retrieval, and result saving.

Dependencies:
- docker: Docker SDK for Python, used for managing Docker containers.
- subprocess: For running Docker commands to save results.
- datetime: For handling timestamps in logs.

Usage:
- Create an instance of DockerGPUMonitor to monitor GPU metrics for benchmarks
  executed in Docker containers.
- Implementations of the following abstract methods from BaseMonitor:
  - `_start_benchmark`: Starts the benchmark in a Docker container.
  - `_is_benchmark_running`: Checks if the Docker container is still running.
  - `_live_monitor_logs`: Retrieves and formats live GPU metrics and container logs.
  - `_cleanup_benchmark`: Removes the Docker container and saves results.

Note:
   Most errors are logged but not raised, allowing the method to fail silently.
   Find them in runtime.log.
"""

import os
import subprocess
from datetime import datetime

from .base_monitor import BaseMonitor
# Global Variables
from .utils.globals import LOGGER, RESULTS_DIR

# Ensure Docker is Installed
try:
    import docker
    DOCKER_AVAILABLE = True
    LOGGER.info("Docker module available.")
except ImportError as exc:
    DOCKER_AVAILABLE = False
    LOGGER.warning("Docker module not available. Docker functionality will be disabled.")
    raise RuntimeError("Docker functionality is not available. Please install Docker.") from exc


class DockerGPUMonitor(BaseMonitor):
    """
    GPU monitor for Docker-based benchmarks.

    Extends BaseMonitor to run and monitor GPU metrics for benchmarks
    executed within Docker containers. Handles container lifecycle
    and logs retrieval.

    Attributes:
        client (docker.client.DockerClient): Docker client for container management.
        container (docker.models.containers.Container): Docker container running the benchmark.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize DockerGPUMonitor.

        Raises:
            RuntimeError: If Docker functionality is not available.
        """
        super().__init__(*args, **kwargs)

        # Initialize Docker client
        self.client = docker.from_env()

        # Initialise parameter for Benchmark Container
        self.container = None

        self.benchmark_image = ""

    def _start_benchmark(self, benchmark) -> None:
        """
        Start the benchmark in a Docker container.

        Args:
            benchmark_image (str): Docker image name for the benchmark.
        """
        try:
            self.benchmark_image = benchmark
            # Run the container in the background
            self.container = self.client.containers.run(
                self.benchmark_image,
                detach=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                shm_size="1024G" # Set to large to ensure all that is needed is used
            )

            # Reload to update status from created to running
            self.container.reload()
        except docker.errors.DockerException as docker_error:
            LOGGER.error("Docker error: %s", docker_error)

    def _is_benchmark_running(self) -> bool:
        """
        Check if the Docker container is still running.

        Returns:
            bool: True if the container is running, False otherwise.
        """

        # Reload container status and check if it is still running
        try:
            self.container.reload()
            return self.container.status == 'running'
        except docker.errors.DockerException as docker_error:
            LOGGER.error("Docker error: %s", docker_error)

    def _live_monitor_logs(self, monitor_logs) -> str:
        """
        Retrieve and format live GPU metrics and container logs.

        Returns:
            str: Formatted string containing GPU metrics and container logs.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if monitor_logs:
            try:
                # Collect container logs
                container_log = self.container.logs(follow=False).decode('utf-8')

                # Initialize container message
                logs = ""

                # Process the logs
                container_log = container_log.replace('\\r', '\r')
                lines = container_log.split('\n')  # Split the entire log into lines

                # Process each line to handle log loading bars
                for line in lines:
                    if '\r' in line:
                        # Handle the last segment after '\r'
                        line = line.split('\r')[-1]
                    # Append the processed line to the complete message
                    logs += f"\n {line.strip()}"

                logs_message = (
                    f"\nContainer Log as of {current_time}:\n"
                    f"{logs}\n"
                )
            except OSError as os_error:
                LOGGER.error("Error clearing the terminal screen: %s", os_error)
                raise

            except KeyError as key_error:
                LOGGER.error("Missing key in GPU stats or metrics: %s", key_error)
                raise

            except ValueError as value_error:
                LOGGER.error("Error formatting GPU metrics: %s", value_error)
                raise
        else:
            logs_message = f"\nContainer Status as of {current_time}:\n {self.container.status}\n"

        return logs_message

    def _cleanup_benchmark(self) -> None:
        """Remove the Docker container and save results if possible."""
        if self.container:
            # Try to save results
            try:
                container_id = self.container.id
                container_results_path = os.path.join(
                    RESULTS_DIR,
                    f'results_{self.benchmark_image}'
                    )
                cmd = f'docker cp {container_id}:/root/results/ {container_results_path}'
                subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                LOGGER.info("Results from Docker saved successfully.")
            except subprocess.CalledProcessError as cp_error:
                LOGGER.error("Failed to save results: %s", cp_error)
                LOGGER.error("Stderr: %s", cp_error.stderr.decode())

            # Try to remove container
            try:
                self.container.remove(force=True)
                LOGGER.info("Docker container removed.")
            except docker.errors.APIError as docker_error:
                LOGGER.error("Failed to remove Docker container: %s", docker_error)
