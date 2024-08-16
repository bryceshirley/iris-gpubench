To set up the project, follow these steps:

## Pre-Installation Instructions

**If you are using Docker Images for Benchmarks:**

Start with an Ubuntu VM that has Nvidia GPUs. Ensure the following dependencies are pre-installed: Docker, Nvidia Docker (for GPU support), Python with Virtual Environment, and the Nvidia drivers. If these dependencies are not installed, you can easily set them up by running `./setup_vm_docker.sh` from the cloned Git repository. *(Note: Git is required to clone the repository—refer to step 1 for instructions).*

**If you plan to run Benchmarks directly on your machine (for benchmarks that aren't suitable for Docker or if you need quick results):**

Begin with an Ubuntu VM with Nvidia GPUs, ensuring that Tmux, Python with Virtual Environment, and the Nvidia drivers are pre-installed. If these dependencies are missing, you can automatically set them up by executing `./setup_vm_tmux.sh` from the cloned Git repository. *(Note: Git is required to clone the repository—see step 1 for details).*

## Pre-Installation Instructions

Follow these steps to set up iris-gpubench:

### 1.**Clone the Repository**  
   Start by cloning the project repository:
```sh
git clone https://github.com/bryceshirley/iris-gpubench.git
cd iris-gpubench
```

### 2.**Set Up a Virtual Environment**  
   Next, create and activate a virtual environment:
```sh
python3 -m venv env
source env/bin/activate
```

### 3.**Install Dependencies and iris-gpubench Package**  
####   a. Finally, install the package along with necessary dependencies:
```sh
pip install wheel
pip install .
```
####   b. **(For Developers)**
```sh
pip install wheel
pip install -e .
```
   -  `-e` for editable mode, lets you install Python packages in a way that
   allows immediate reflection of any changes you make to the source code
   without needing to reinstall the package.

### 4.**Next Build Docker Images for the Benchmarks**

---

[Previous Page](overview.md) | [Next Page](building_docker_images.md)