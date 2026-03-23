#!/usr/bin/env python3
"""
Prepare CROSS-SUBJECT test data for FPGA inference.
Extracts MR from one subject and CT from DIFFERENT subject (matches training setup).
Saves volumes as .npy files for FPGA testing.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import argparse

def load_and_preprocess_volume(nifti_path):
    """
    Load a NIfTI volume and preprocess it the same way as training.
    Returns numpy array with shape (D, H, W) normalized to [-1, 1].
    """
    print(f"Loading {nifti_path}...")
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata().astype(np.float32)

    print(f"  Original shape: {volume.shape}")
    print(f"  Original range: [{volume.min():.2f}, {volume.max():.2f}]")

    # Normalize to [-1, 1] (same as training)
    vol_min = volume.min()
    vol_max = volume.max()

    if vol_max > vol_min:
        volume = 2 * (volume - vol_min) / (vol_max - vol_min) - 1
    else:
        volume = np.zeros_like(volume)

    print(f"  Normalized range: [{volume.min():.2f}, {volume.max():.2f}]")

    return volume

def load_segmentation(nifti_path):
    """
    Load a NIfTI segmentation volume.
    """
    print(f"Loading {nifti_path}...")
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata().astype(np.int16)
    print(f"  Original shape: {volume.shape}")
    print(f"  Labels: {np.unique(volume)}")
    return volume


def main():
    parser = argparse.ArgumentParser(description='Prepare test data for FPGA inference')
    parser.add_argument('--data_root', type=str, default='../../Data/Tuco',
                        help='Path to Tuco dataset')
    parser.add_argument('--mr_subject', type=str, default='1BA001',
                        help='Subject ID for moving MR volume (e.g., 1BA001)')
    parser.add_argument('--ct_subject', type=str, default='1BA002',
                        help='Subject ID for fixed CT volume (e.g., 1BA002) - MUST be different from mr_subject')
    parser.add_argument('--output_dir', type=str, default='./test_data',
                        help='Output directory for test data')

    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    volumes_dir = data_root / 'volumes_center'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate cross-subject requirement
    if args.mr_subject == args.ct_subject:
        print(f"ERROR: mr_subject and ct_subject must be DIFFERENT!")
        print(f"  The model was trained on cross-subject registration.")
        print(f"  Using the same subject for both MR and CT will give incorrect results.")
        return

    # Find MR and CT files for DIFFERENT subjects (cross-subject)
    mr_file = volumes_dir / f"{args.mr_subject}_mr.nii.gz"
    ct_file = volumes_dir / f"{args.ct_subject}_ct.nii.gz"
    mr_seg_file = data_root / 'seg_center' / f"{args.mr_subject}.nii.gz"
    ct_seg_file = data_root / 'seg_center' / f"{args.ct_subject}.nii.gz"

    if not mr_file.exists():
        print(f"ERROR: MR file not found: {mr_file}")
        print(f"\nAvailable subjects:")
        subjects = sorted(set([f.stem.split('_')[0] for f in volumes_dir.glob('*_mr.nii.gz')]))
        for s in subjects[:10]:
            print(f"  - {s}")
        return

    if not ct_file.exists():
        print(f"ERROR: CT file not found: {ct_file}")
        print(f"\nAvailable subjects:")
        subjects = sorted(set([f.stem.split('_')[0] for f in volumes_dir.glob('*_ct.nii.gz')]))
        for s in subjects[:10]:
            print(f"  - {s}")
        return

    print(f"\n{'='*60}")
    print(f"Preparing CROSS-SUBJECT test data")
    print(f"  Moving (MR): {args.mr_subject}")
    print(f"  Fixed (CT):  {args.ct_subject}")
    print(f"{'='*60}\n")

    # Load and preprocess volumes
    mr_volume = load_and_preprocess_volume(mr_file)
    ct_volume = load_and_preprocess_volume(ct_file)

    # Save as .npy files with cross-subject naming
    mr_output = output_dir / f"{args.mr_subject}_mr.npy"
    ct_output = output_dir / f"{args.ct_subject}_ct.npy"

    print(f"\nSaving volumes...")
    np.save(mr_output, mr_volume)
    np.save(ct_output, ct_volume)

    print(f"  MR: {mr_output} ({mr_volume.nbytes / 1024 / 1024:.2f} MB)")
    print(f"  CT: {ct_output} ({ct_volume.nbytes / 1024 / 1024:.2f} MB)")

    # Load and save segmentations from DIFFERENT subjects
    if mr_seg_file.exists() and ct_seg_file.exists():
        mr_seg_volume = load_segmentation(mr_seg_file)
        ct_seg_volume = load_segmentation(ct_seg_file)

        mr_seg_output = output_dir / f"{args.mr_subject}_mr_seg.npy"
        ct_seg_output = output_dir / f"{args.ct_subject}_ct_seg.npy"

        print(f"\nSaving segmentations...")
        np.save(mr_seg_output, mr_seg_volume)
        np.save(ct_seg_output, ct_seg_volume)
        print(f"  MR Seg: {mr_seg_output} ({mr_seg_volume.nbytes / 1024 / 1024:.2f} MB)")
        print(f"  CT Seg: {ct_seg_output} ({ct_seg_volume.nbytes / 1024 / 1024:.2f} MB)")
    else:
        print(f"\nWARNING: Segmentation files not found:")
        if not mr_seg_file.exists():
            print(f"  Missing: {mr_seg_file}")
        if not ct_seg_file.exists():
            print(f"  Missing: {ct_seg_file}")

    print(f"Test data prepared successfully.")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nFiles to upload to PYNQ:")
    print(f"  1. {mr_output.name}")
    print(f"  2. {ct_output.name}")
    if mr_seg_file.exists() and ct_seg_file.exists():
        print(f"  3. {mr_seg_output.name}")
        print(f"  4. {ct_seg_output.name}")
    print(f"\nNote: This is cross-subject data from subjects {args.mr_subject} and {args.ct_subject}.")
    print(f"You can now upload these files to your PYNQ board.")


if __name__ == '__main__':
    main()
