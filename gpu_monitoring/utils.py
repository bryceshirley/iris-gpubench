"""
utils.py

This module provides utility functions for the application, including logging setup.

Functions:
    setup_logging(results_dir: str) -> logging.Logger:
        Configures and returns a logger instance for the application.

Dependencies:
- `os`: For directory operations.
- `logging`: For configuring logging.

Example:
    logger = setup_logging(results_dir='../results')
    logger.info("Logging is set up.")
"""

import os
import logging

def setup_logging(results_dir: str = '../results') -> logging.Logger:
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
        filename=os.path.join(results_dir, 'gpu_monitor.log'),
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger
