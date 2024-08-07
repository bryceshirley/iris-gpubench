"""
setup.py

This script sets up the package distribution for the GPU monitoring application.
Dependencies are directly included in this file.

Python Version:
- This script and module were designed to work with Python 3.8.10.

System Dependencies:
- You may need to install system-level dependencies using: sudo apt-get install yq tmux
"""

from setuptools import setup, find_packages

setup(
    name='gpu_monitor',
    version='0.1.0-dev1',  # Development version
    packages=find_packages(),  # Automatically find packages in the current directory
    install_requires=[
        'pynvml==11.5.3',
        'requests==2.31.0',
        'pyyaml==6.0.1',
        'tabulate==0.9.0',
        'matplotlib==3.7.1',
    ],
    entry_points={
        'console_scripts': [
            'gpu_monitor=gpu_monitor.main:main',
        ],
    },
    python_requires='==3.8.10',  # Ensure compatibility with Python 3.8.10
    package_data={
        'gpu_monitor': ['config/default_config.yml'],
    },
)

