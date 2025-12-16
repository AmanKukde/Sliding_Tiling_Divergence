#!/usr/bin/env python3
"""
Example script demonstrating the prediction comparison visualization.

This script shows how to use plot_prediction_comparison() to create
visual comparisons of predictions against ground truth data.
"""

from pathlib import Path
from analysis_pipeline import plot_prediction_comparison


def main():
    """Run example visualization."""

    # Example 1: PAVIA_ATN dataset with multiple methods
    print("Creating PAVIA_ATN comparison visualization...")

    try:
        plot_prediction_comparison(
            prediction_paths=[
                "/path/to/pred_og_128_OT.tiff",
                "/path/to/pred_og.pkl",
                "/path/to/pred_swt.pkl"
            ],
            target_dir="/group/jug/aman/Datasets/PAVIA_ATN/data",
            target_channel_paths=[
                "channel_1/channel_1.tiff",
                "channel_2/channel_2.tiff"
            ],
            method_names=["OuterTiling", "InnerTiling", "SWT"],
            frame_idx=0,
            save_path="./comparison_pavia_atn.png",
            crop_factor=4.0,
            figsize=(20, 12),
            dpi=300
        )
        print("✅ Saved: comparison_pavia_atn.png")
    except FileNotFoundError as e:
        print(f"⚠️  Skipping PAVIA_ATN example: {e}")

    # Example 2: 3D data (Care3D)
    print("\nCreating Care3D comparison visualization...")

    try:
        plot_prediction_comparison(
            prediction_paths=[
                "/path/to/pred_swt_G5-32-32.tiff",
                "/path/to/pred_og_G9-32-32.tif"
            ],
            target_dir="/group/jug/aman/Datasets/Care3D/data",
            target_channel_paths=["target.tiff"],
            method_names=["SWT_Z5", "OG_Z9"],
            frame_idx=10,  # Middle frame
            save_path="./comparison_care3d.png",
            crop_factor=6.0,  # More detailed crop
            figsize=(16, 10),
            dpi=150
        )
        print("✅ Saved: comparison_care3d.png")
    except FileNotFoundError as e:
        print(f"⚠️  Skipping Care3D example: {e}")

    print("\n🎉 Done! Check the output PNG files for visual comparisons.")


if __name__ == "__main__":
    main()
