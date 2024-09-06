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
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout

from .utils.globals import DEFAULT_REGION, LOGGER, TIMEOUT_SECONDS

# Constants
CARBON_INTENSITY_URL = "https://api.carbonintensity.org.uk/regional"

def get_carbon_region_names() -> List[str]:
    """
    Retrieves and returns the short names of all regions from the Carbon Intensity API.

    Returns:
        List[str]: Short names of the regions.
    """
    try:
        response = requests.get(
            CARBON_INTENSITY_URL,
            headers={'Accept': 'application/json'},
            timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()

        # Extract the list of regions
        regions = data['data'][0]['regions']

        # Extract short names of all regions
        region_names = [region['shortname'] for region in regions]

        LOGGER.info("Extracted region names: %s", region_names)
        return region_names

    except (HTTPError, RequestsConnectionError) as network_error:
        LOGGER.error("Network error occurred: %s", network_error)
    except Timeout:
        LOGGER.error("Request timed out after %d seconds.", TIMEOUT_SECONDS)
    except ValueError as json_error:
        LOGGER.error("Failed to decode JSON response: %s", json_error)

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
        response = requests.get(
            CARBON_INTENSITY_URL,
            headers={'Accept': 'application/json'},
            timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()
        regions = data['data'][0]['regions']

        for region in regions:
            if region['shortname'] == carbon_region_shorthand:
                intensity = region['intensity']
                carbon_forecast = float(intensity['forecast'])
                LOGGER.info("Carbon forecast for '%s': %f",
                            carbon_region_shorthand,
                            carbon_forecast)
                return carbon_forecast

        LOGGER.warning("Region '%s' not found in the response.", carbon_region_shorthand)

    except (HTTPError, RequestsConnectionError) as network_error:
        LOGGER.error("Network error occurred: %s", network_error)
    except Timeout:
        LOGGER.error("Request timed out after %d seconds.", TIMEOUT_SECONDS)
    except ValueError as json_error:
        LOGGER.error("Failed to decode JSON response: %s", json_error)

    return None
