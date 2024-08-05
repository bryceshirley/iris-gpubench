"""
__init__.py

This module initializes the package by exposing key components for easy access.

Imports:
- `main`: Main function or entry point for the application.
- `setup_logging`: Utility function for configuring logging.
- `GPUMonitor`: Class for monitoring GPU metrics.

Package Metadata:
- __version__: Current version of the package.
- __author__: Author of the package.
"""

# Import specific classes or functions to simplify access
from .main import main
from .utils import setup_logging
from .gpu_monitor import GPUMonitor

# Define package-level variables
__version__ = '1.0.0'
__author__ = 'Bryce Shirley'
