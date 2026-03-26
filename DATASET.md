# Dataset Organization for 6DPose

Your dataset should be organized as follows:

```
DATASETS/
└── Dataset 6DManuel/
    └── results/
```

- **results/**: Contains RGB images, depth maps, masks, and point clouds. Naming conventions should match those expected by the dataloader (see `dataloader/dataloader_Manuel.py`).

> **Note:**
> You may need to customize the dataset path in the dataloader file. See the `dataset_path` variable in `dataloader/dataloader_Manuel.py`.

## Example File Names in `results/`
- `RGB_<id>.png`
- `DEPTH_<id>.png`
- `mask_<id>.png`
- `PC_DEPTH_<id>.ply`
- `PC_VELODYNE_<id>.ply`
- `PC_MODEL_<id>.ply`
- `<id>.txt` (pose/label file)

## Train/Test Split
- The dataloader automatically splits the dataset into train/test based on a fixed random seed.

## Customization
- Adjust the dataloader if your dataset structure or file naming differs.
