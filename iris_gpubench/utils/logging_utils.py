"""
Logging configuration for the iris-gpubench package.

This module provides a setup_logging function to configure logging for the package.
"""

import logging
import os


def setup_logging(results_dir: str) -> logging.Logger:
    """
    Configures logging for the application.

    Sets up a logger that writes to 'gpu_monitor.log' in the specified directory.
    If the directory does not exist, it is created.

    Parameters:
        results_dir (str): Directory for saving the log file. Defaults to '../results'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Create the directory if it does not exist

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(results_dir, 'runtime.log'),
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger
