## Pre-Installation Instructions

### 1. **Set Up Ubuntu VM**: 
- Start with an Ubuntu VM equipped with Nvidia GPUs.

### 2. **Install Dependencies**:

   - **For Docker Benchmarks**: Install Docker, Nvidia Docker (for GPU support), git, Python with Virtual Environment, and Nvidia drivers.

   - **For Direct Machine Benchmarks**: Install Tmux, git, Python with Virtual Environment, and Nvidia drivers.

### 3. **Setup VM With Script (requires git to clone repo)**:

   - **Docker Benchmarks**: Execute `./setup_vm_docker.sh` from the cloned Git repository.

   - **Direct Machine Benchmarks**: Execute `./setup_vm_tmux.sh` from the cloned Git repository.

---

## Installation Instructions

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

---

[Previous Page](overview.md) | [Next Page](building_docker_images.md)
