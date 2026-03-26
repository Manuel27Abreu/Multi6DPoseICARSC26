# Attention-Based Multimodal Fusion for Robust 6D Pose Estimation in Cluttered Industrial Environments

**ICARSC 2026 – 26th IEEE International Conference on Autonomous Robot Systems and Competitions**

---

## Overview
This repository contains the official code for the paper:

> **Attention-Based Multimodal Fusion for Robust 6D Pose Estimation in Cluttered Industrial Environments**  
> _ICARSC 2026 – 26th IEEE International Conference on Autonomous Robot Systems and Competitions_  
> [DOI: to be added]

The project provides tools for training, evaluating, and running inference for 6D pose estimation using multimodal data in challenging industrial scenarios.

## Table of Contents
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd 6DPose
   ```
2. (Optional) Install [TorchSparse](https://github.com/mit-han-lab/torchsparse) for accelerated sparse convolution support:
   ```bash
   cd torchsparsemaster
   pip install -r requirements.txt
   python setup.py install
   cd ..
   ```
   > If you do not wish to use TorchSparse, comment out the relevant imports in the code.
3. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Structure

Datasets should be organized as follows:

```
DATASETS/
└── Dataset 6DManuel/
    └── results/
```
- Place your RGB images, depth maps, masks, and point clouds in the `results/` folder.
- You may need to customize the dataset path in the dataloader files (see `dataloader/dataloader_Manuel.py`).

---

## Usage

The main entry point is `tools/Pose6D.py`. Example commands:

- **Train a model:**
  ```bash
  python tools/Pose6D.py --train --option 8 --dataset model/full
  ```
- **Train with specific model and modality:**
  ```bash
  python tools/Pose6D.py --model ResNet18 --option 8 --modalities 0 --train
  ```
- **Run inference:**
  ```bash
  python tools/Pose6D.py --run
  ```
- **Evaluate metrics:**
  ```bash
  python tools/Pose6D.py --metrics
  ```
- **Annotate dataset:**
  ```bash
  python tools/Pose6D.py --annotate
  ```
- **Train for a specific class:**
  ```bash
  python tools/Pose6D.py --train --class_id <ID>
  ```

See `tools/README` for more command options and details.

---

## Citation

Please cite our work if you use this code or dataset in your research:

> [Citation to be added]

---

## Acknowledgments
This work has been supported by the Portuguese Foundation for Science and Technology (FCT) through grant ISR-UC UID/00048/2025 (DOI: 10.54499/UID/00048/2025) and by Agenda "GreenAuto: Green innovation for the Automotive Industry", with reference 02/C05-i01.01/2022.PC644867037-00000013.
