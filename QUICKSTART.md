# 6DPose – Quick Start Guide

## Main Script
- **tools/Pose6D.py**: Main entry point for training, inference, metrics, and annotation.

## Example Commands

- **Train:**
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
- **Train for a specific class:**
  ```bash
  python tools/Pose6D.py --train --class_id <ID>
  ```

## Arguments
- `--train`: Train the model
- `--run`: Run inference
- `--metrics`: Show evaluation metrics
- `--annotate`: Annotate dataset using the model
- `--class_id`: Train for a specific class
- `--option`: Selects training option
- `--modalities`: Selects active modalities
- `--dataset`: Choose dataset variant
- `--model`: Selects model architecture

## Dataset
- Edit dataset path in `dataloader/dataloader_Manuel.py` if needed.
- Structure:
  ```
  DATASETS/
  └── Dataset 6DManuel/
      └── results/
  ```

## Notes
- TorchSparse is optional. Comment out its imports if not used.
- For more details, see DOCUMENTATION.md.
