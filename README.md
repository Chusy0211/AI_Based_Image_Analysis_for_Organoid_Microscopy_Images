# AI_Based_Image_Analysis_for_Organoid_Microscopy_Images

## Overview
This repository provides the dataset and source code developed for our study on automated abnormality detection in organoid microscopy images.

---

## Directory Structure
```shell
datasets/
├── train/
├───── *.[jpg|png]
src/
├── checkpoints/
├───── *.pth
├── model_pool.py
├── train.py
├── validate.py
```

---

## Dataset Description

The dataset consists of grayscale images in JPG and PNG format.

- Total samples: 254
- Normal samples: 203
- Abnormal samples: 51 (including false positives)
- Image format: JPG, PNG
- Resolution: [e.g., 224*224]


### Data Access

All data should be included in this repository under the `datasets/train` directory.

If you want to access the full dataset, please refer to the paper.

---

## Code Description

This repository includes Python code for training and inference.

### Main Files
- `train.py` – training pipeline
- `validate.py` – validate script
- `model_pool.py` – model_pool definition

---

### Installation

#### step1 install *conda [e.g., Miniconda]

- linux installer
```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
- other installers please refer to [https://www.anaconda.com/docs/getting-started/miniconda/install/overview](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

#### step2 install lib
```bash
conda create -n cv python=3.12 -y
conda activate cv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python
pip install matplotlib
pip install scikit-learn
```
---

### Usage

#### Training
```shell
cd src && python train.py
```

#### Inference
```shell
cd src && python validate.py
```

---
## Reproducibility

To ensure reproducibility:

- Validation threshold: 0.5
- Inference threshold: 0.70

Hardware: [optional, e.g., NVIDIA 4090, NVIDIA 4060ti]

---
## License
This project is licensed under the MIT License.

---

## Contact

For questions, please contact: [xiangyf@shanghaitech.edu.cn]