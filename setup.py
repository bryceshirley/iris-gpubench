"""
setup.py

This script sets up the package distribution for the GPU monitoring application.
Dependencies are directly included in this file.

Python Version:
- This script and module were designed to work with Python 3.8.10.
"""

from setuptools import setup, find_packages

setup(
    name='iris_gpubench',
    version='0.1.0-dev1',  # Development version
    packages=find_packages(),  # Automatically find packages in the current directory
    install_requires=[
        'pynvml==11.5.3',
        'requests==2.32.3',
        'pyyaml==6.0.2',
        'tabulate==0.9.0',
        'matplotlib==3.7.5',
        'docker==7.1.0',
    ],
    entry_points={
        'console_scripts': [
            'iris-gpubench=iris_gpubench.main:main',
        ],
    },
    python_requires='==3.8.10',  # Ensure compatibility with Python 3.8.10
)
