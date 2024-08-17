#!/bin/bash

#### INSTALL IrisBench ###

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nInstall git, python3-pip, and Venv/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Install Git, Python, and venv
sudo apt-get install -y git
sudo apt-get install -y wget python3-pip
sudo apt-get install -y python3-venv

#### INSTALL TMUX


echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nINSTALL Tmux/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

sudo apt update
sudo apt install tmux


##### INSTALL NVIDIA DRIVERS #####

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nINSTALL NVIDIA DRIVERS/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Install NVIDIA drivers (if needed)
sudo apt install -y nvidia-driver-535

##### REBOOT VM #####

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nREBOOT/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Reboot the system to finish setup
sudo reboot