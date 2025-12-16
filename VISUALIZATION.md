# Prediction Visualization

The `plot_prediction_comparison` function creates comprehensive visual comparisons of predictions against ground truth data.

## Function Overview

```python
from analysis_pipeline import plot_prediction_comparison

plot_prediction_comparison(
    prediction_paths: List[Union[str, Path]],
    target_dir: Union[str, Path],
    target_channel_paths: List[str],
    method_names: List[str],
    frame_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    crop_factor: float = 4.0,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    figsize: Tuple[int, int] = (20, 12),
    dpi: int = 300,
)
```

## What It Creates

The function generates a grid layout showing:
- **Full ground truth images** for each channel with crop region indicators
- **Full input image** (concatenated GT channels)
- **Zoomed/cropped regions** from the center of the image
- **Predictions from multiple methods** for each channel (full + cropped)
- **Difference maps** between prediction methods

## Example Usage

### Basic 2-Channel Comparison

```python
from pathlib import Path
from analysis_pipeline import plot_prediction_comparison

plot_prediction_comparison(
    prediction_paths=[
        "results/pred_og.tiff",
        "results/pred_sw.tiff",
        "results/pred_swt.tiff"
    ],
    target_dir="/group/jug/aman/Datasets/PAVIA_ATN/data",
    target_channel_paths=[
        "channel_1/channel_1.tiff",
        "channel_2/channel_2.tiff"
    ],
    method_names=["OuterTiling", "SlidingWindow", "SWT"],
    frame_idx=0,
    save_path="./comparison_pavia.png",
    crop_factor=4.0
)
```

### 3D Data Visualization

For 3D data, specify the frame index to visualize:

```python
plot_prediction_comparison(
    prediction_paths=[
        "results/pred_z5.tiff",
        "results/pred_z9.tiff"
    ],
    target_dir="/group/jug/aman/Datasets/Care3D/data",
    target_channel_paths=["target.tiff"],
    method_names=["Z5-32-32", "Z9-32-32"],
    frame_idx=10,  # Middle frame
    save_path="./comparison_care3d_frame10.png"
)
```

### Custom Crop and Normalization

```python
plot_prediction_comparison(
    prediction_paths=["pred1.tiff", "pred2.tiff"],
    target_dir="/path/to/data",
    target_channel_paths=["channel_1/ch1.tiff"],
    method_names=["Method1", "Method2"],
    crop_factor=6.0,  # Smaller crop (1/6 of image)
    vmin_percentile=0.5,  # Tighter normalization
    vmax_percentile=99.5,
    figsize=(24, 14),  # Larger figure
    dpi=150  # Lower DPI for faster rendering
)
```

## Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `prediction_paths` | List of paths to prediction files (.tiff, .pkl, .dill) | `["pred1.tiff", "pred2.tiff"]` |
| `target_dir` | Base directory containing target channel subdirectories | `"/path/to/dataset/data"` |
| `target_channel_paths` | Relative paths from target_dir to channel files | `["channel_1/channel_1.tiff"]` |
| `method_names` | Names for each prediction method | `["Original", "Modified"]` |
| `frame_idx` | Frame index for multi-frame data (default: 0) | `0` |
| `save_path` | Optional path to save figure | `"./output.png"` |
| `crop_factor` | Center crop as 1/crop_factor of image size | `4.0` (1/4 of image) |
| `vmin_percentile` | Lower percentile for normalization | `1.0` |
| `vmax_percentile` | Upper percentile for normalization | `99.0` |
| `figsize` | Figure size (width, height) in inches | `(20, 12)` |
| `dpi` | DPI for saved figure | `300` |

## Output Layout

For 2 channels and 3 methods, the layout is:

```
Row 0:  [GT Ch1] [GT Ch2] [Input] [GT Ch1 Crop] [GT Ch2 Crop]
Row 1:  [Method1 Ch1] [Method2 Ch1] [Method3 Ch1] [M1 Crop] [M2 Crop] [M3 Crop] [Diff]
Row 2:  [Method1 Ch2] [Method2 Ch2] [Method3 Ch2] [M1 Crop] [M2 Crop] [M3 Crop] [Diff]
```

- Yellow rectangles indicate the crop region
- Difference maps show: Method2 - Method1, Method3 - Method1
- All images use the same normalization (computed from GT)

## Python API Usage

```python
from pathlib import Path
from analysis_pipeline import plot_prediction_comparison

# Create comparison visualization
fig = plot_prediction_comparison(
    prediction_paths=[
        Path("/path/to/pred_og_128_OT.tiff"),
        Path("/path/to/pred_og.pkl"),
        Path("/path/to/pred_swt.pkl")
    ],
    target_dir=Path("/group/jug/aman/Datasets/PAVIA_ATN/data"),
    target_channel_paths=[
        "channel_1/channel_1.tiff",
        "channel_2/channel_2.tiff"
    ],
    method_names=["OuterTiling", "InnerTiling", "SWT"],
    save_path=Path("./results/comparison.png")
)

# Or display interactively
import matplotlib.pyplot as plt
plt.show()
```

## Tips

1. **Crop Factor**: Use 4.0 for a good balance between detail and context. Larger values (6.0, 8.0) show more detail.

2. **Normalization**: Default percentiles (1, 99) work well for most data. Adjust if images appear washed out or too dark.

3. **File Formats**: The function automatically handles:
   - TIFF files (.tif, .tiff)
   - Pickle files (.pkl)
   - Dill files (.dill)

4. **Channel Organization**:
   - Targets should be in separate subdirectories per channel
   - Predictions can be multi-channel or single-channel files
   - The function automatically detects and handles channel dimensions

5. **Memory**: For large images or many methods, reduce `dpi` or `figsize` to save memory.

## Common Use Cases

### Compare Tiling Strategies
```python
plot_prediction_comparison(
    prediction_paths=[
        "outer_tiling.tiff",
        "inner_tiling.tiff",
        "sliding_window.tiff"
    ],
    target_dir="/data/experiment",
    target_channel_paths=["target.tiff"],
    method_names=["Outer", "Inner", "Sliding"],
    save_path="tiling_comparison.png"
)
```

### Analyze Different Network Depths
```python
plot_prediction_comparison(
    prediction_paths=[
        "depth_3_layers.tiff",
        "depth_5_layers.tiff",
        "depth_7_layers.tiff"
    ],
    target_dir="/data/denoising",
    target_channel_paths=["clean.tiff"],
    method_names=["3-Layer", "5-Layer", "7-Layer"],
    save_path="depth_comparison.png"
)
```

### Evaluate Hyperparameters
```python
for lr in [1e-4, 5e-4, 1e-3]:
    plot_prediction_comparison(
        prediction_paths=[f"pred_lr{lr}.tiff"],
        target_dir="/data/training",
        target_channel_paths=["target.tiff"],
        method_names=[f"LR={lr}"],
        save_path=f"eval_lr{lr}.png"
    )
```
