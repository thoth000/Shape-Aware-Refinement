# Shape-Aware Refinement
This research proposes a **PDE-based trainable refinement module** for **tubular-structure segmentation** to enhance shape continuity.
![image](https://github.com/user-attachments/assets/3865efd2-732f-470a-9b96-36f107b2e5c5)

## Paper / Slides

## Overview
- PDE-based refinement improves mask continuity **while maintaining thin shape**
- Our approach is applied to **various architectures** for tubular-structure segmentation because of **post-process**

## Dataset
- [DRIVE dataset](https://github.com/zhengyuan-liu/Retinal-Vessel-Segmentation/tree/master/DRIVE)

## Environment
- Ubuntu 24.04.2 LTS
- NVIDIA TITAN RTX(Memory: 24GB) * 4

## How to run
1. Download DRIVE dataset to `/dataset`.

2. Run `/dataset/data_process.py` to fit dataset for training models.
```bash
cd dataset
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```

3. train and test model
```bash
cd pde-shape-refiner
main_ddp.sh
```