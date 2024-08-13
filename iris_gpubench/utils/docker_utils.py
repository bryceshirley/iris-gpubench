# utils/docker_utils.py

"""
Utility functions for handling Docker operations in the iris-gpubench package.

This module provides functions to check for Docker image existence and list 
available Docker images on the local system.

Functions:
    image_exists(image_name: str) -> bool:
        Checks if a specific Docker image exists on the local system.

    list_available_images(exclude_none: bool, exclude_base: bool) -> list:
        Retrieves and lists Docker images available on the local system, with options
        to exclude images with '<none>' or 'base' in their tags.

Dependencies:
- `docker`: For interacting with the Docker API.
- `logging`: For logging operations.
"""

import docker
from .globals import LOGGER


def image_exists(image_name: str) -> bool:
    """
    Check if a specific Docker image exists on the local system.

    Args:
        image_name (str): The name of the Docker image to check. This can be in
        the format 'repository:tag'.

    Returns:
        bool: True if the Docker image exists, False otherwise.
    """
    try:
        # Initialize Docker client
        client = docker.from_env()

        # List images with the specified name
        images = client.images.list(name=image_name)

        # Check if any image matches the specified name
        return any(image_name in tag for img in images for tag in img.tags)
    except docker.errors.DockerException as e:
        # Log Docker-related errors
        LOGGER.error("Error checking image existence: %s", e)
        return False

def list_available_images(exclude_none: bool = False, exclude_base: bool = False) -> list:
    """
    Retrieve and list Docker images available on the local system.

    Args:
        exclude_none (bool): If True, filter out images with '<none>' in their
        tag. Defaults to False.
        exclude_base (bool): If True, filter out images with 'base' in their
        name. Defaults to False.

    Returns:
        list: A list of Docker image tags that match the specified filters.
        Each tag is a string in the format 'repository:tag'.
    """
    try:
        # Initialize Docker client
        client = docker.from_env()

        # Retrieve all Docker images
        images = client.images.list()

        # Extract image tags
        image_tags = [tag for img in images for tag in img.tags]

        image_tags = [img for img in image_tags if 'docs' not in img.lower()]

        # Filter out images with 'base' in their name if specified
        if exclude_base:
            image_tags = [img for img in image_tags if 'base' not in img.lower()]

        # Filter out images with '<none>' in their name if specified
        if exclude_none:
            image_tags = [img for img in image_tags if '<none>' not in img.lower()]

        return image_tags
    except docker.errors.DockerException as e:
        # Log Docker-related errors
        LOGGER.error("Error listing images: %s", e)
        return []
