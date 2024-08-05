#!/bin/bash

# Install Python package and dependencies
pip install .

# Install non-Python dependencies
sudo apt-get update
sudo apt-get install -y yq tmux

echo "All dependencies installed."

