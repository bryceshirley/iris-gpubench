"""
__init__.py

This module initializes the package by exposing key components for easy access
and providing package metadata.

Imports:
- `main`: The main function or entry point for the application.
- `setup_logging`: Utility function for configuring logging.
- `format_metrics`: Utility function for formatting and saving metrics.
- `GPUMonitor`: Class for monitoring GPU metrics.
- `get_carbon_forecast`: Function for retrieving carbon forecast data.
- `get_carbon_region_names`: Function for obtaining valid carbon region names.
- `VictoriaMetricsExporter`: Class for exporting data to VictoriaMetrics.

Package Metadata:
- __version__: The current version of the package.
- __author__: The author of the package.
"""

from iris_gpubench.main import *
from iris_gpubench.gpu_monitor import *
from iris_gpubench.carbon_metrics import *
from iris_gpubench.gpu_victoria_exporter import *

# Import utilities from the utils sub-package
from .utils.logging_utils import *
from .utils.globals import *
from .utils.metric_utils import *
from .utils.docker_utils import *

# Define package-level variables
__version__ = '0.1.0-dev1'
__author__ = 'Bryce Shirley'
