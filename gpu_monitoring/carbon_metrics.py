"""
carbon_metrics.py

This module provides functions for interacting with the Carbon Intensity API to fetch
and handle carbon intensity data for different regions.

Functions:
- fetch_carbon_region_names: Retrieves and returns the short names of all regions
  from the Carbon Intensity API.
- get_carbon_forecast: Retrieves the current carbon intensity forecast for a specified
  region using the Carbon Intensity API.

Constants:
- CARBON_INTENSITY_URL: URL endpoint for accessing the Carbon Intensity API.
- TIMEOUT_SECONDS: Timeout duration for API requests.

Dependencies:
- requests: For making HTTP requests to the Carbon Intensity API.
- setup_logging: Utility function for configuring logging.

Logging:
- The module uses logging to capture information and errors related to API requests.
"""

from typing import List, Optional
import requests

from .utils import setup_logging

# Set up logging with specific configuration
LOGGER = setup_logging()

# Constants
CARBON_INTENSITY_URL = "https://api.carbonintensity.org.uk/regional"
DEFAULT_REGION = "South England"
TIMEOUT_SECONDS = 30

def fetch_carbon_region_names() -> List[str]:
    """
    Retrieves and returns the short names of all regions from the Carbon Intensity API.

    Returns:
        List[str]: Short names of the regions.
    """
    try:
        response = requests.get(CARBON_INTENSITY_URL,
                                headers={'Accept': 'application/json'},
                                timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        # Extract the list of regions
        regions = data['data'][0]['regions']

        # Extract short names of all regions
        region_names = [region['shortname'] for region in regions]

        LOGGER.info("Extracted region names: %s", region_names)
        return region_names

    except requests.exceptions.RequestException as error_message:
        LOGGER.error("Error occurred during request (timeout %ds): %s",
                     TIMEOUT_SECONDS, error_message)
        return []

def get_carbon_forecast(carbon_region_shorthand: str = DEFAULT_REGION) -> Optional[float]:
    """
    Retrieves the current carbon intensity forecast for a specified region using
    the Carbon Intensity API.

    Args:
        carbon_region_shorthand (str): The short name of the region to get the forecast for.

    Returns:
        Optional[float]: Current carbon intensity forecast, or None if an error occurs.
    """
    try:
        response = requests.get(CARBON_INTENSITY_URL,
                                headers={'Accept': 'application/json'},
                                timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        regions = data['data'][0]['regions']

        for region in regions:
            if region['shortname'] == carbon_region_shorthand:
                intensity = region['intensity']
                carbon_forecast = float(intensity['forecast'])
                LOGGER.info("Carbon forecast for '%s': %f",
                            carbon_region_shorthand, carbon_forecast)
                return carbon_forecast

    except requests.exceptions.RequestException as error_message:
        LOGGER.error("Error request timed out (30s): %s", error_message)

    return None
