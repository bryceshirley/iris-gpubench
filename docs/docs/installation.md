To set up the project, follow these steps:

## Installation Instructions

Start with an Ubuntu VM equipped with Nvidia GPUs and ensure the following dependencies are pre-installed: Docker, Nvidia Docker (for GPU support), Python with Virtual Environment and Nvidia drivers. If these are not installed, you can run ./setup_vm.sh from the cloned Git repository to set them up automatically. (Note: Git is required to clone the repository—see step 1).

Follow these steps to set up the project:

1. **Clone the Repository**  
   Start by cloning the project repository:
```sh
git clone https://github.com/bryceshirley/iris-gpubench.git
cd iris-gpubench
```

2. **Set Up a Virtual Environment**  
   Next, create and activate a virtual environment:
```sh
python3 -m venv env
source env/bin/activate
```

3. **Install Dependencies and iris-gpubench Package**  
   Finally, install the package along with necessary dependencies:
```sh
pip install wheel
pip install .
```

3. **(For Developers)**
```sh
pip install wheel
pip install -e .
```
* `-e` for editable mode, lets you install Python packages in a way that allows immediate reflection of any changes you make to the source code without needing to reinstall the package.

4. **Next Build Docker Images for the Benchmarks**

[Previous Page](overview.md) | [Next Page](building_docker_images.md)