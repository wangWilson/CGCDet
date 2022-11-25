## Installation

### Requirements

- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
- PyTorch 1.1
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3

### Install CGC

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n CGC python=3.7 -y
source activate CGC

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the AerialDetection repository.

```shell
unzip CGC_code
cd CGC_code
```

d. Compile cuda extensions.

```shell
./compile.sh
```

e. Install CGC (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```

### Install DOTA_devkit
```
    sudo apt-get install swig
    cd DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

